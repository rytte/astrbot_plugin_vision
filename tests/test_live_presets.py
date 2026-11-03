import json
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from astrbot.api import AstrBotConfig
from astrbot_plugin_vision.config import (
    LIVE_MODE_FIELDS,
    LIVE_MODES,
    LIVE_PRESET_FIELDS,
    LIVE_SOURCES,
    Settings,
    migrate_config_file,
    migrate_live_config,
)
from astrbot_plugin_vision.live import (
    LiveSessions,
    decode_frame,
    parse_live_command,
    select_frames,
)
from test_live import picture


@pytest.mark.parametrize("source", LIVE_SOURCES)
@pytest.mark.parametrize("mode", LIVE_MODES)
async def test_each_preset_controls_capture_resolution_window_and_model_input(
    tmp_path, source, mode
):
    values = {
        "sample_interval": 0.2,
        "max_edge": 128,
        "max_frames": 2,
        "lookback_seconds": 2,
    }
    if mode == "interval":
        values.update(observation_interval=17, proactive=False)
    if mode == "change":
        values.update(sensitivity="high", min_analysis_interval=9)
    settings = Settings.from_mapping(
        {"live": {source: [{"__template_key": mode, **values}]}}
    )
    source_arg = {
        "screen": "屏幕",
        "camera": "摄像头",
        "stream": "https://example.com/live",
    }[source]
    mode_arg = {"demand": "提问", "interval": "定时", "change": "变化"}[mode]
    options, _ = parse_live_command(f"{source_arg} {mode_arg} 观察", settings)
    analyzer = AsyncMock(return_value={"summary": "近期画面", "speak": False})
    manager = LiveSessions(settings, tmp_path, analyzer, AsyncMock(), AsyncMock())
    session = await manager.create("chat", "alice")
    manager.configure(session, options)
    session.last_user = time.monotonic()
    try:
        for i in range(6):
            await manager.add_frame(session, picture((i * 45, 0, 0)), i)
        result = await manager.observe(session, direct=False)
        assert 1 <= len(result["evidence_times"]) <= 2
        assert min(result["evidence_times"]) >= 3
        assert result["evidence_times"][-1] == 5
        assert session.options.sample_interval == 0.2
        frames = analyzer.await_args.args[1]
        import io

        from PIL import Image

        assert Image.open(io.BytesIO(frames[-1].jpeg)).size == (128, 72)
        if mode == "interval":
            assert session.options.analysis_interval == 17
            assert not session.options.proactive
        if mode == "change":
            assert session.options.analysis_interval == 9
            assert session.options.threshold == 0.05
        assert settings.live_preset(source, mode).max_frames == 2
        other_source = next(s for s in LIVE_SOURCES if s != source)
        assert settings.live_preset(other_source, mode) == Settings().live_preset(
            other_source, mode
        )
    finally:
        await manager.close()


def test_nested_schema_round_trip_and_mode_specific_fields(tmp_path):
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    config_path = str(tmp_path / "config.json")
    config = AstrBotConfig(config_path, schema=schema)
    for i, source in enumerate(LIVE_SOURCES):
        assert schema["live"]["items"][source]["type"] == "template_list"
        for j, section in enumerate(config["live"][source]):
            mode = section["__template_key"]
            assert mode == LIVE_MODES[j]
            assert set(section) == {
                "__template_key",
                *LIVE_PRESET_FIELDS,
                *LIVE_MODE_FIELDS[mode],
            }
            section.update(
                sample_interval=0.5 + i,
                max_edge=640 + 100 * i,
                max_frames=1 + i + j,
                lookback_seconds=i + j,
            )
    config["live"]["devices"]["live_camera_device"] = "/dev/video2"
    config.save_config()
    reloaded = AstrBotConfig(config_path, schema=schema)
    assert reloaded == config
    settings = Settings.from_mapping(reloaded)
    assert settings.live_camera_device == "/dev/video2"
    for i, source in enumerate(LIVE_SOURCES):
        for j, mode in enumerate(LIVE_MODES):
            preset = settings.live_preset(source, mode)
            assert preset.sample_interval == 0.5 + i
            assert preset.max_frames == 1 + i + j
            assert preset.lookback_seconds == i + j


@pytest.mark.parametrize(
    "key,value",
    [
        ("sample_interval", 0),
        ("sample_interval", 31),
        ("sample_interval", True),
        ("max_frames", 0),
        ("max_frames", 33),
        ("max_frames", 2.5),
        ("lookback_seconds", -1),
        ("lookback_seconds", 301),
        ("max_edge", 63),
        ("max_edge", 4097),
        ("lookback_seconds", float("nan")),
    ],
)
def test_invalid_preset_is_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        Settings.from_mapping({"live": {"screen": {"demand": {key: value}}}})


@pytest.mark.parametrize(
    "live",
    [
        {"screen": "invalid"},
        {"screen": [None]},
        {"screen": [{}]},
        {"screen": [{"__template_key": "unknown"}]},
        {"screen": [{"__template_key": "demand"}, {"__template_key": "demand"}]},
        {"screen": {"demand": []}},
        {"devices": []},
        {"screen": {"unknown": {}}},
        {"camera": {"demand": {"proactive": True}}},
        {"stream": {"interval": {"sensitivity": "high"}}},
        {"screen": {"change": {"min_analysis_interval": 0}}},
        {"camera": {"interval": {"proactive": 1}}},
    ],
)
def test_invalid_nested_sections_and_inapplicable_options_are_rejected(live):
    with pytest.raises(ValueError):
        Settings.from_mapping({"live": live})


@pytest.mark.parametrize("grouped", [True, False])
def test_legacy_config_migrates_before_astrbot_loads_schema(tmp_path, grouped):
    path = tmp_path / "old.json"
    legacy = {
        "live_enabled": True,
        "live_max_edge": 1920,
        "live_camera_device": "/dev/video3",
        "live_sensitivity": "high",
        "live_observation_interval": 21,
        "live_min_analysis_interval": 7,
        "live_interval_proactive": False,
    }
    path.write_text(json.dumps({"live": legacy} if grouped else legacy))
    original = path.read_bytes()
    assert migrate_config_file(path)
    assert not migrate_config_file(path)
    assert next(tmp_path.glob("*.bak")).read_bytes() == original
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    config = AstrBotConfig(str(path), schema=schema)
    migrated = migrate_live_config(config)
    assert migrate_live_config(migrated) == migrated
    assert "live_max_edge" not in migrated
    assert "live_enabled" not in migrated
    settings = Settings.from_mapping(migrated)
    assert settings.live_enabled
    assert settings.live_camera_device == "/dev/video3"
    for source in LIVE_SOURCES:
        assert settings.live_preset(source, "demand").max_edge == 1920
        assert settings.live_preset(source, "interval").observation_interval == 21
        assert settings.live_preset(source, "interval").proactive is False
        assert settings.live_preset(source, "change").min_analysis_interval == 7
        assert settings.live_preset(source, "change").sensitivity == "high"


def test_invalid_legacy_config_is_not_overwritten(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"live": {"live_max_edge": 30}}')
    original = path.read_bytes()
    with pytest.raises(ValueError, match="max_edge"):
        migrate_config_file(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.bak"))


def test_fixed_preset_objects_migrate_to_native_template_lists(tmp_path):
    original = {
        "live": {
            source: {
                mode: {"sample_interval": 1 + i, "max_frames": 2 + j}
                for j, mode in enumerate(LIVE_MODES)
            }
            for i, source in enumerate(LIVE_SOURCES)
        }
    }
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(original))
    before = Settings.from_mapping(original)
    assert migrate_config_file(path)
    migrated = json.loads(path.read_text())
    assert migrate_live_config(migrated) == migrated
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    loaded = AstrBotConfig(str(path), schema=schema)
    assert Settings.from_mapping(loaded) == before
    for source in LIVE_SOURCES:
        assert [entry["__template_key"] for entry in loaded["live"][source]] == list(
            LIVE_MODES
        )


def test_removing_template_entry_uses_default_without_disabling_mode():
    settings = Settings.from_mapping(
        {
            "live": {
                "screen": [{"__template_key": "change", "max_frames": 7}],
                "camera": [],
            }
        }
    )
    assert settings.live_preset("screen", "change").max_frames == 7
    assert settings.live_preset("screen", "demand") == Settings().live_preset(
        "screen", "demand"
    )
    assert settings.live_presets["camera"] == Settings().live_presets["camera"]


async def test_long_sample_interval_is_not_mistaken_for_disconnection(tmp_path):
    settings = Settings.from_mapping(
        {"live": {"screen": {"demand": {"sample_interval": 20}}}}
    )
    manager = LiveSessions(settings, tmp_path, AsyncMock(), AsyncMock(), AsyncMock())
    session = await manager.create("chat", "alice")
    manager.configure(session, {"mode": "demand"})
    try:
        session.frames.append(
            replace(decode_frame(picture(), 0, 128), received=time.monotonic() - 21)
        )
        assert not session.stale
        session.frames[-1] = replace(session.frames[-1], received=time.monotonic() - 61)
        assert session.stale
    finally:
        await manager.close()


@pytest.mark.parametrize("source", LIVE_SOURCES)
@pytest.mark.parametrize("mode", LIVE_MODES)
def test_static_frames_collapse_to_latest_for_every_preset(source, mode):
    frames = [decode_frame(picture(), timestamp, 128) for timestamp in range(11)]
    selected = select_frames(
        frames, source=source, mode=mode, max_frames=4, lookback=10
    )
    assert selected == [frames[-1]]


def test_change_selection_keeps_brief_transition_and_latest():
    frames = [
        decode_frame(picture((255, 255, 255) if i == 3 else (i * 8,) * 3), i, 128)
        for i in range(11)
    ]
    selected = select_frames(
        frames, source="screen", mode="change", max_frames=3, lookback=10
    )
    times = [frame.timestamp for frame in selected]
    assert len(times) == 3
    assert 3 in times
    assert 2 in times or 4 in times
    assert times[-1] == 10


def test_interval_selection_covers_only_current_observation_period():
    frames = [decode_frame(picture((i * 20,) * 3), i, 128) for i in range(11)]
    selected = select_frames(
        frames,
        source="camera",
        mode="interval",
        max_frames=3,
        lookback=10,
        reference=frames[7],
    )
    assert len(selected) == 3
    assert selected[0].timestamp == 7
    assert selected[-1].timestamp == 10


@pytest.mark.parametrize("mode,interval", [("change", 9), ("interval", 17)])
async def test_automatic_analysis_respects_selected_preset_interval(
    tmp_path, mode, interval
):
    field = "min_analysis_interval" if mode == "change" else "observation_interval"
    settings = Settings.from_mapping({"live": {"camera": {mode: {field: interval}}}})
    analyzer = AsyncMock(return_value={"summary": "画面", "speak": False})
    manager = LiveSessions(settings, tmp_path, analyzer, AsyncMock(), AsyncMock())
    session = await manager.create("chat", "alice")
    manager.configure(session, {"source": "camera", "mode": mode})
    session.last_user = time.monotonic()
    try:
        await manager.add_frame(session, picture(), 0)
        session.last_analysis = time.monotonic() - interval + 1
        assert await manager.observe(session, automatic=True, direct=False) == {}
        analyzer.assert_not_awaited()
        session.last_analysis = time.monotonic() - interval - 1
        await manager.observe(session, automatic=True, direct=False)
        analyzer.assert_awaited_once()
        assert session.options.analysis_interval == interval
    finally:
        await manager.close()
