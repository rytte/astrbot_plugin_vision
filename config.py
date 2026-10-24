from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

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
    tool_timeout: float = 300.0
    max_concurrency: int = 2
    cache_ttl: float = 3600.0
    cache_max_mb: int = 512

    @classmethod
    def from_mapping(cls, values: Mapping) -> Settings:
        defaults = cls()
        values = dict(values)
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
        unknown = set(values) - {field.name for field in fields(cls)}
        # 兼容旧配置中的已移除项，不再用它扩展读取范围。
        unknown.discard("allowed_video_roots")
        if unknown:
            raise ValueError("未知视频理解配置：" + ", ".join(sorted(unknown)))
        normalized = {}
        for field in fields(cls):
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
        return settings
