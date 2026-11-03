import copy
import json
import stat
from dataclasses import asdict
from pathlib import Path

import pytest
from astrbot.api import AstrBotConfig
from astrbot_plugin_vision.config import Settings, migrate_config, migrate_config_file


@pytest.mark.parametrize("flat", [False, True])
def test_old_configuration_keeps_all_values_after_schema_reload(tmp_path, flat):
    old = {
        "dependencies": {"ffmpeg_path": "/opt/ffmpeg", "ffprobe_path": "/opt/ffprobe"},
        "model_enhancement": {
            "overview_model_enabled": True,
            "overview_provider_id": "vision-provider",
            "asr_enabled": True,
            "asr_provider_id": "speech-provider",
        },
        "overview": {
            "overview_interval": 6,
            "overview_output_mode": "grid",
            "overview_max_frames": 120,
            "overview_max_edge": 800,
            "scene_threshold": 0.4,
        },
        "inspect": {
            "inspect_max_frames": 12,
            "inspect_max_edge": 960,
            "inspect_max_calls": 6,
        },
        "video_limits": {"download_max_mb": 90, "download_max_duration": 600},
        "share_video": {
            "share_links_enabled": False,
            "download_cache_max_mb": 600,
            "download_cookie_file": "/opt/cookies.txt",
            "download_allow_fake_ip": True,
        },
        "user_quota": {"daily_analysis_limit": 12},
        "runtime_cache": {
            "max_edge_limit": 1920,
            "tool_timeout": 150,
            "max_concurrency": 3,
            "cache_ttl": 2400,
            "cache_max_mb": 300,
        },
    }
    expected = Settings(
        **{key: value for group in old.values() for key, value in group.items()}
    )
    if flat:
        original = asdict(expected)
        original.pop("live_presets")
    else:
        original = old
    original["live"] = {
        "screen": [{"__template_key": "change", "max_frames": 7, "sensitivity": "high"}]
    }
    snapshot = copy.deepcopy(original)
    migrated = migrate_config(original)
    assert original == snapshot
    assert migrate_config(migrated) == migrated
    assert migrated["dependencies"] == old["dependencies"]
    assert migrated["model_enhancement"] == old["model_enhancement"]
    assert migrated["general"] == {"tool_timeout": 150, "max_concurrency": 3}
    assert set(migrated["video_analysis"]["runtime_cache"]) == {
        "max_edge_limit",
        "cache_ttl",
        "cache_max_mb",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(original), encoding="utf-8-sig")
    path.chmod(0o640)
    original_bytes = path.read_bytes()
    assert migrate_config_file(path)
    assert not migrate_config_file(path)
    (backup,) = tmp_path.glob("*.config-groups-*.bak")
    assert backup.read_bytes() == original_bytes
    assert stat.S_IMODE(backup.stat().st_mode) == 0o640
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    config = AstrBotConfig(str(path), schema=schema)
    loaded = Settings.from_mapping(config)
    assert loaded == Settings(
        **{**asdict(expected), "live_presets": loaded.live_presets}
    )
    assert loaded.live_preset("screen", "change").max_frames == 7
    assert loaded.live_preset("screen", "change").sensitivity == "high"
    config.save_config()
    assert Settings.from_mapping(AstrBotConfig(str(path), schema=schema)) == loaded
    assert not migrate_config_file(path)


def test_current_groups_take_precedence_when_combined_with_legacy_values():
    original = {
        "tool_timeout": 60,
        "overview_interval": 2,
        "runtime_cache": {
            "tool_timeout": 100,
            "max_concurrency": 4,
            "cache_max_mb": 200,
        },
        "overview": {"overview_interval": 3, "overview_max_frames": 60},
        "general": {"tool_timeout": 120},
        "video_analysis": {
            "overview": {"overview_interval": 7},
            "runtime_cache": {"cache_max_mb": 400},
        },
    }
    snapshot = copy.deepcopy(original)
    settings = Settings.from_mapping(original)
    assert settings.tool_timeout == 120
    assert settings.max_concurrency == 4
    assert settings.cache_max_mb == 400
    assert settings.overview_interval == 7
    assert settings.overview_max_frames == 60
    assert original == snapshot


@pytest.mark.parametrize(
    "values",
    [
        {"video_analysis": []},
        {"general": None},
        {"video_analysis": {"overview": []}},
        {"overview": {"misspelled": 1}},
        {"video_analysis": {"unknown": {}}},
        {"general": {"tool_timeout": 0}},
        {"runtime_cache": {"max_concurrency": 9}},
    ],
)
def test_invalid_configuration_is_rejected_without_changing_file(tmp_path, values):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(values))
    original = path.read_bytes()
    with pytest.raises(ValueError):
        Settings.from_mapping(values)
    with pytest.raises(ValueError):
        migrate_config_file(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.bak"))
