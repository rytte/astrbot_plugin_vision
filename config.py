from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field
from pathlib import Path

LIVE_SOURCES = ("screen", "camera", "stream")
LIVE_MODES = ("demand", "interval", "change")
LIVE_GENERAL_FIELDS = (
    "live_enabled",
    "live_buffer_seconds",
    "live_session_seconds",
    "live_max_analyses",
    "live_speech_cooldown",
)
LIVE_DEVICE_FIELDS = ("live_camera_device", "live_audio_device")
LIVE_PRESET_FIELDS = ("sample_interval", "max_edge", "max_frames", "lookback_seconds")
LIVE_MODE_FIELDS = {
    "demand": (),
    "interval": ("observation_interval", "proactive"),
    "change": ("sensitivity", "min_analysis_interval"),
}


def bounded_number(value, name, minimum, maximum, *, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
        or (integer and int(value) != value)
    ):
        raise ValueError(
            f"{name} 必须是 {minimum}～{maximum} 之间的{'整数' if integer else '数字'}"
        )
    return int(value) if integer else float(value)


@dataclass(frozen=True)
class LivePreset:
    sample_interval: float = 1.0
    max_edge: int = 1280
    max_frames: int = 4
    lookback_seconds: float = 12.0
    observation_interval: float = 10.0
    proactive: bool = True
    sensitivity: str = "medium"
    min_analysis_interval: float = 5.0

    @classmethod
    def parse(cls, values: Mapping, default: LivePreset, mode: str, path: str):
        if not isinstance(values, Mapping):
            raise ValueError(f"{path} 必须是配置对象")
        unknown = set(values) - set(LIVE_PRESET_FIELDS + LIVE_MODE_FIELDS[mode])
        if unknown:
            raise ValueError(f"未知实时预设 {path}：" + ", ".join(sorted(unknown)))
        data = {
            f.name: values.get(f.name, getattr(default, f.name)) for f in fields(cls)
        }
        for name, minimum, maximum, integer in (
            ("sample_interval", 0.2, 30, False),
            ("max_edge", 64, 4096, True),
            ("max_frames", 1, 32, True),
            ("lookback_seconds", 0, 300, False),
            ("observation_interval", 0.2, 300, False),
            ("min_analysis_interval", 0.2, 300, False),
        ):
            data[name] = bounded_number(
                data[name], f"{path}.{name}", minimum, maximum, integer=integer
            )
        if type(data["proactive"]) is not bool:
            raise ValueError(f"{path}.proactive 必须是布尔值")
        if data["sensitivity"] not in ("low", "medium", "high"):
            raise ValueError(f"{path}.sensitivity 必须是 low、medium 或 high")
        return cls(**data)


def default_live_presets():
    sizes = {"screen": 1600, "camera": 960, "stream": 1280}
    frames = {"screen": (4, 2, 3), "camera": (4, 4, 5), "stream": (4, 4, 6)}
    windows = {"screen": (0, 10, 4), "camera": (2, 10, 4), "stream": (4, 10, 6)}
    return {
        source: {
            mode: LivePreset(
                sample_interval=0.5 if mode == "change" else 1.0,
                max_edge=sizes[source],
                max_frames=frames[source][i],
                lookback_seconds=windows[source][i],
            )
            for i, mode in enumerate(LIVE_MODES)
        }
        for source in LIVE_SOURCES
    }


def migrate_live_config(values: Mapping) -> dict:
    """将旧版实时配置归入分组；不恢复旧的命令参数。"""
    values = dict(values)
    live = values.get("live", {})
    if not isinstance(live, Mapping):
        raise ValueError("live 必须是配置对象")
    live = dict(live)
    general, devices = {}, {}
    removed = ("live_host", "live_port", "live_public_url", "live_stream_url")
    legacy_presets = {
        "live_max_edge": "max_edge",
        "live_sensitivity": "sensitivity",
        "live_observation_interval": "observation_interval",
        "live_interval_proactive": "proactive",
        "live_min_analysis_interval": "min_analysis_interval",
    }
    old = {}
    for key in (*LIVE_GENERAL_FIELDS, *LIVE_DEVICE_FIELDS, *legacy_presets, *removed):
        if key in values:
            old[key] = values.pop(key)
        if key in live:
            old[key] = live.pop(key)
    for group, names, result in (
        ("general", LIVE_GENERAL_FIELDS, general),
        ("devices", LIVE_DEVICE_FIELDS, devices),
    ):
        current = live.get(group, {})
        if not isinstance(current, Mapping):
            raise ValueError(f"live.{group} 必须是配置对象")
        result.update(current)
        for key in names:
            if key in old:
                result[key] = old[key]
        if result or group in live:
            live[group] = result
    for source in LIVE_SOURCES:
        if source not in live and not set(old).intersection(legacy_presets):
            continue
        section = live.get(source, {})
        if isinstance(section, list):
            indexed = {}
            for item in section:
                if not isinstance(item, Mapping):
                    raise ValueError(f"live.{source} 的预设条目必须是配置对象")
                mode = item.get("__template_key")
                if not isinstance(mode, str) or mode not in LIVE_MODES:
                    raise ValueError(f"live.{source} 的预设类型无效")
                if mode in indexed:
                    raise ValueError(
                        f"live.{source} 的 {mode} 预设重复，每种响应模式只能有一项"
                    )
                indexed[mode] = {k: v for k, v in item.items() if k != "__template_key"}
            section = indexed
        elif isinstance(section, Mapping):
            if set(section) - set(LIVE_MODES):
                raise ValueError(f"未知实时配置 live.{source}")
            # 旧的固定对象转换成默认折叠的原生模板列表。
            section = {mode: section.get(mode, {}) for mode in LIVE_MODES}
        else:
            raise ValueError(f"live.{source} 必须是预设列表")
        entries = []
        defaults = default_live_presets()[source]
        for mode in section:
            preset = section.get(mode, {})
            if not isinstance(preset, Mapping):
                raise ValueError(f"live.{source}.{mode} 必须是配置对象")
            preset = dict(preset)
            for old_key, new_key in legacy_presets.items():
                if (
                    old_key not in old
                    or new_key not in LIVE_PRESET_FIELDS + LIVE_MODE_FIELDS[mode]
                ):
                    continue
                preset.setdefault(new_key, old[old_key])
            validated = LivePreset.parse(
                preset, defaults[mode], mode, f"live.{source}.{mode}"
            )
            entries.append(
                {
                    "__template_key": mode,
                    **{
                        name: getattr(validated, name)
                        for name in LIVE_PRESET_FIELDS + LIVE_MODE_FIELDS[mode]
                    },
                }
            )
        live[source] = entries
    if live or "live" in values:
        values["live"] = live
    return values


def migrate_live_config_file(path: Path) -> bool:
    """在 AstrBot 按新 schema 加载配置前迁移原文件，保留备份并原子替换。"""
    if not path.is_file():
        return False
    original = path.read_bytes()
    values = json.loads(original.decode("utf-8-sig"))
    migrated = migrate_live_config(values)
    if migrated == values:
        return False
    Settings.from_mapping(migrated)
    digest = hashlib.sha256(original).hexdigest()[:12]
    backup = path.with_name(f"{path.name}.live-presets-{digest}.bak")
    if not backup.exists():
        with backup.open("xb") as file:
            os.chmod(backup, path.stat().st_mode & 0o777)
            file.write(original)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(migrated, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.chmod(name, path.stat().st_mode & 0o777)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return True


CONFIG_GROUPS = {
    "dependencies": ("ffmpeg_path", "ffprobe_path"),
    "video_limits": ("download_max_mb", "download_max_duration"),
    "user_quota": ("daily_analysis_limit",),
    "share_video": (
        "share_links_enabled",
        "download_cache_max_mb",
        "download_cookie_file",
        "download_allow_fake_ip",
    ),
    "model_enhancement": (
        "overview_model_enabled",
        "overview_provider_id",
        "asr_enabled",
        "asr_provider_id",
    ),
    "overview": (
        "overview_interval",
        "overview_output_mode",
        "overview_max_frames",
        "overview_max_edge",
        "scene_threshold",
    ),
    "inspect": ("inspect_max_frames", "inspect_max_edge", "inspect_max_calls"),
    "live": ("general", "devices", *LIVE_SOURCES),
    "runtime_cache": (
        "max_edge_limit",
        "tool_timeout",
        "max_concurrency",
        "cache_ttl",
        "cache_max_mb",
    ),
}


@dataclass(frozen=True)
class Settings:
    ffmpeg_path: str = ""
    ffprobe_path: str = ""
    share_links_enabled: bool = True
    # 大小与时长上限同时用于直接附件和分享视频。
    download_max_mb: int = 200
    download_max_duration: float = 1800.0
    daily_analysis_limit: int = 5
    download_cache_max_mb: int = 1024
    download_cookie_file: str = ""
    download_allow_fake_ip: bool = False
    overview_model_enabled: bool = False
    overview_provider_id: str = ""
    asr_enabled: bool = False
    asr_provider_id: str = ""
    overview_output_mode: str = "frames"
    overview_interval: float = 4.0
    overview_max_frames: int = 200
    overview_max_edge: int = 640
    scene_threshold: float = 0.3
    inspect_max_frames: int = 8
    inspect_max_edge: int = 768
    max_edge_limit: int = 1280
    inspect_max_calls: int = 3
    live_enabled: bool = False
    live_camera_device: str = ""
    live_audio_device: str = ""
    live_buffer_seconds: float = 60.0
    live_session_seconds: float = 1800.0
    live_speech_cooldown: float = 15.0
    live_max_analyses: int = 120
    live_presets: dict[str, dict[str, LivePreset]] = dataclass_field(
        default_factory=default_live_presets
    )
    tool_timeout: float = 300.0
    max_concurrency: int = 2
    cache_ttl: float = 3600.0
    cache_max_mb: int = 512

    def live_preset(self, source: str, mode: str) -> LivePreset:
        return self.live_presets[source][mode]

    @classmethod
    def from_mapping(cls, values: Mapping) -> Settings:
        defaults = cls()
        values = migrate_live_config(values)
        live = values.pop("live", {})
        unknown = set(live) - set(CONFIG_GROUPS["live"])
        if unknown:
            raise ValueError("未知实时配置：" + ", ".join(sorted(unknown)))
        for group, names in (
            ("general", LIVE_GENERAL_FIELDS),
            ("devices", LIVE_DEVICE_FIELDS),
        ):
            section = live.get(group, {})
            if set(section) - set(names):
                raise ValueError(f"未知实时配置 live.{group}")
            values.update(section)
        presets = default_live_presets()
        for source in LIVE_SOURCES:
            for entry in live.get(source, []):
                mode = entry["__template_key"]
                presets[source][mode] = LivePreset.parse(
                    {
                        key: value
                        for key, value in entry.items()
                        if key != "__template_key"
                    },
                    presets[source][mode],
                    mode,
                    f"live.{source}.{mode}",
                )
        for group, keys in CONFIG_GROUPS.items():
            if group not in values:
                continue
            section = values.pop(group)
            if not isinstance(section, Mapping):
                raise ValueError(f"{group} 必须是配置对象")
            unknown = set(section) - set(keys)
            if unknown:
                raise ValueError(
                    f"未知视频理解配置 {group}：" + ", ".join(sorted(unknown))
                )
            values.update(section)
        unknown = set(values) - {
            field.name for field in fields(cls) if field.name != "live_presets"
        }
        # 兼容旧配置中的已移除项，不再用它扩展读取范围。
        unknown.discard("allowed_video_roots")
        if unknown:
            raise ValueError("未知视频理解配置：" + ", ".join(sorted(unknown)))
        normalized = {}
        for field in fields(cls):
            if field.name == "live_presets":
                normalized[field.name] = presets
                continue
            value = values.get(field.name, getattr(defaults, field.name))
            if isinstance(getattr(defaults, field.name), bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{field.name} 必须是布尔值")
            elif isinstance(getattr(defaults, field.name), str):
                if not isinstance(value, str):
                    raise ValueError(f"{field.name} 必须是字符串")
                value = value.strip()
            else:
                default = getattr(defaults, field.name)
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ValueError(f"{field.name} 必须是数字")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{field.name} 必须是有限正数")
                if isinstance(default, int):
                    if int(value) != value:
                        raise ValueError(f"{field.name} 必须是整数")
                    value = int(value)
                normalized[field.name] = value
                continue
            normalized[field.name] = value
        settings = cls(**normalized)
        if settings.overview_output_mode not in {"frames", "grid"}:
            raise ValueError(
                "overview_output_mode 必须是 frames（独立帧）或 grid（九宫格）"
            )
        if not 2 <= settings.overview_max_frames <= 500:
            raise ValueError("overview_max_frames 必须在 2～500 之间")
        if not 0 < settings.scene_threshold < 1:
            raise ValueError("scene_threshold 必须在 0～1 之间")
        if not 64 <= settings.max_edge_limit <= 4096:
            raise ValueError("max_edge_limit 必须在 64～4096 之间")
        if not all(
            64 <= edge <= settings.max_edge_limit
            for edge in (settings.overview_max_edge, settings.inspect_max_edge)
        ):
            raise ValueError("默认图片长边必须在 64 与 max_edge_limit 之间")
        if settings.inspect_max_frames > 32 or settings.max_concurrency > 8:
            raise ValueError("单次回看最多 32 帧，并发数最多 8")
        if settings.live_buffer_seconds > 300 or settings.live_session_seconds > 14400:
            raise ValueError("实时缓冲最多 300 秒，单次会话最多 4 小时")
        return settings
