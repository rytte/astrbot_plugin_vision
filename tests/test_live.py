import asyncio
import io
import time
import wave
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_vision.config import LIVE_MODES, LIVE_SOURCES, Settings
from astrbot_plugin_vision.live import (
    MAX_FRAMES,
    LiveOptions,
    LiveSessions,
    decode_frame,
    frame_distance,
    parse_live_command,
    parse_observation,
    transcribe_pcm,
)
from astrbot_plugin_vision.media import VideoError
from PIL import Image


def picture(color="white", size=(320, 180)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, "JPEG")
    return output.getvalue()


@pytest.fixture
async def live(tmp_path):
    manager = LiveSessions(
        Settings(overview_model_enabled=True, overview_provider_id="vision"),
        tmp_path / "live",
        AsyncMock(
            return_value={
                "summary": "出现新的画面",
                "memory": "前情",
                "speak": True,
                "comment": "值得吐槽",
            }
        ),
        AsyncMock(),
        AsyncMock(return_value="新的台词"),
    )
    yield manager
    await manager.close()


async def setup_session(live, **options):
    session = await live.create("chat", "alice")
    live.configure(session, options)
    # 大多数测试直接驱动输入，单独的 worker 测试验证后台调度。
    session.last_user = time.monotonic()
    return session


async def push(live, session, timestamp, color):
    if session.frames:
        session.frames[-1] = replace(
            session.frames[-1], received=time.monotonic() - 0.3
        )
    await live.add_frame(session, picture(color), timestamp)


@pytest.mark.parametrize("max_edge", [64, 1920, 4096])
def test_live_resolution_config_is_independent(max_edge):
    settings = Settings.from_mapping(
        {
            "live": {"live_max_edge": max_edge},
            "runtime_cache": {"max_edge_limit": 1024},
        }
    )
    assert all(
        settings.live_preset(source, mode).max_edge == max_edge
        for source in LIVE_SOURCES
        for mode in LIVE_MODES
    )
    assert settings.overview_max_edge == 640
    assert settings.max_edge_limit == 1024


@pytest.mark.parametrize("max_edge", [63, 4097, True, 1280.5, "1280"])
def test_invalid_live_resolution_config_rejected(max_edge):
    with pytest.raises(ValueError, match="max_edge"):
        Settings.from_mapping({"live": {"live_max_edge": max_edge}})


@pytest.mark.parametrize(
    "mode,max_edge", [("demand", 1920), ("interval", 640), ("change", 960)]
)
async def test_live_resolution_reaches_buffer_and_model(live, mode, max_edge):
    presets = Settings.from_mapping(
        {"live": {"screen": {mode: {"max_edge": max_edge}}}}
    ).live_presets
    live.settings = replace(live.settings, live_presets=presets, max_edge_limit=1024)
    session = await setup_session(live, mode=mode)
    await live.add_frame(session, picture(size=(2560, 1440)), 0)
    expected = (max_edge, max_edge * 9 // 16)
    with Image.open(io.BytesIO(session.frames[-1].jpeg)) as image:
        assert image.size == expected
    await live.observe(session)
    frames = live.analyze.await_args.args[1]
    with Image.open(io.BytesIO(frames[-1].jpeg)) as image:
        assert image.size == expected


@pytest.mark.parametrize("mode", ["change", "interval"])
async def test_direct_background_passes_frames_without_observer_call(live, mode):
    live.settings = replace(
        live.settings, overview_model_enabled=False, overview_provider_id=""
    )
    session = await setup_session(live, mode=mode)
    session.last_user = -float("inf")
    await push(live, session, 0, "red")
    session.transcripts.append({"start_sec": 0, "end_sec": 1, "text": "注意刚才的红色"})
    result = await live.observe(session, automatic=True)
    live.analyze.assert_not_awaited()
    live.speak.assert_awaited_once()
    assert live.speak.await_args.kwargs["image_urls"] == result["image_urls"]
    assert result["mode"] == "direct"
    assert result["audio"][0]["text"] == "注意刚才的红色"
    assert "image_urls" not in session.observation
    assert not session.reasons


async def test_direct_background_keeps_changes_while_cooling_down(live):
    live.settings = replace(live.settings, overview_model_enabled=False)
    session = await setup_session(live)
    session.last_user = -float("inf")
    session.last_speech = time.monotonic()
    await push(live, session, 0, "red")
    assert await live.observe(session, automatic=True) == {}
    assert session.reasons == {"initial"}
    assert session.analysis_count == 0
    live.analyze.assert_not_awaited()
    live.speak.assert_not_awaited()


async def test_direct_setting_still_updates_silent_interval_summary(live):
    live.settings = replace(live.settings, overview_model_enabled=False)
    session = await setup_session(live, mode="interval", proactive=False)
    session.last_user = -float("inf")
    await push(live, session, 0, "red")
    result = await live.observe(session, automatic=True)
    assert result["summary"] == "出现新的画面"
    assert "image_urls" not in result
    live.analyze.assert_awaited_once()
    live.speak.assert_not_awaited()


@pytest.mark.parametrize(
    "mode,requested,expected",
    [
        ("demand", True, False),
        ("change", False, True),
        ("interval", False, False),
        ("interval", True, True),
    ],
)
def test_mode_controls_speaking(mode, requested, expected):
    assert (
        LiveOptions.parse({"mode": mode, "proactive": requested}, Settings()).proactive
        is expected
    )


@pytest.mark.parametrize(
    "values",
    [
        {"source": []},
        {"mode": None},
        {"sensitivity": {}},
        {"audio": 1},
        {"interval": float("nan")},
        {"origin": "other-chat"},
    ],
)
def test_invalid_options_rejected(values):
    with pytest.raises((ValueError, TypeError)):
        LiveOptions.parse(values, Settings())


def test_frame_validation_and_color_difference():
    red = decode_frame(picture("red"), 0, 160)
    blue = decode_frame(picture("blue"), 1, 160)
    assert frame_distance(red, blue) > 0.5
    assert frame_distance(red, red) == 0
    with Image.open(io.BytesIO(red.jpeg)) as image:
        assert max(image.size) == 160
    with pytest.raises(ValueError):
        decode_frame(b"invalid", 0, 160)


async def test_owner_expiry_and_replacement(live):
    session = await live.create("chat", "alice")
    assert live.for_chat("chat", "bob") is None
    assert live.for_chat("other-chat", "alice") is None
    replacement = await live.create("chat", "alice")
    assert session.stopped
    assert live.for_chat("chat", "alice") is replacement
    replacement.expires = time.monotonic() - 1
    assert live.for_chat("chat", "alice") is None


async def test_changes_include_accumulation_not_only_adjacent_frames(live):
    session = await setup_session(live)
    await push(live, session, 0, (0, 0, 0))
    session.reference = session.frames[-1]
    session.reasons.clear()
    await push(live, session, 1, (20, 20, 20))
    assert not session.reasons
    await push(live, session, 2, (40, 40, 40))
    assert session.reasons == {"accumulated_change"}
    session.reasons.clear()
    await push(live, session, 3, "white")
    assert session.reasons == {"visual_change"}


async def test_frame_buffer_bounded_and_timestamps_monotonic(live):
    session = await setup_session(live, mode="demand")
    for index in range(MAX_FRAMES + 3):
        await push(live, session, index, "black")
    assert len(session.frames) == MAX_FRAMES
    assert session.frames[0].timestamp == 3
    with pytest.raises(ValueError):
        await live.add_frame(session, picture(), 0)
    assert not session.reasons


async def test_no_speech_for_demand_cooldown_or_user_interrupt(live):
    session = await setup_session(live, mode="demand")
    await push(live, session, 0, "white")
    result = await live.observe(session, automatic=False)
    assert result["summary"] == "出现新的画面"
    live.speak.assert_not_awaited()
    session.options = LiveOptions()
    session.last_user = -float("inf")
    session.last_analysis = -float("inf")
    session.last_speech = time.monotonic()
    await live.observe(session, automatic=True)
    live.speak.assert_not_awaited()
    session.last_speech = -float("inf")
    session.last_analysis = -float("inf")

    async def interrupt(*args):
        live.user_turn(session)
        return {"summary": "新事件", "speak": True, "comment": "不要抢话"}

    live.analyze.side_effect = interrupt
    await live.observe(session, automatic=True)
    live.speak.assert_not_awaited()


async def test_speech_occurs_only_after_semantic_decision(live):
    session = await setup_session(live)
    await push(live, session, 0, "white")
    session.last_user = -float("inf")
    live.analyze.return_value["speak"] = False
    await live.observe(session, automatic=True)
    live.speak.assert_not_awaited()
    live.analyze.return_value["speak"] = True
    session.last_analysis = -float("inf")
    await live.observe(session, automatic=True)
    live.speak.assert_awaited_once()


async def test_slow_observation_can_speak_while_capture_is_fresh(live, monkeypatch):
    session = await setup_session(live)
    await push(live, session, 0, "white")
    session.last_user = -float("inf")
    clock = [time.monotonic()]
    monkeypatch.setattr(
        "astrbot_plugin_vision.live.time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )

    async def slow_analysis(*args):
        clock[0] += 21
        session.frames.append(
            replace(session.frames[-1], timestamp=21, received=clock[0])
        )
        return {"summary": "页面仍在等待操作", "speak": True, "comment": "先打开浏览器"}

    live.analyze.side_effect = slow_analysis
    await live.observe(session, automatic=True)
    assert not session.stale
    live.speak.assert_awaited_once()


async def test_stale_frames_rejected_and_resume_requires_new_frame(live):
    session = await setup_session(live)
    await push(live, session, 0, "white")
    session.frames[-1] = replace(session.frames[-1], received=time.monotonic() - 11)
    with pytest.raises(VideoError):
        await live.observe(session)
    live.analyze.assert_not_awaited()
    live.pause(session, True)
    live.pause(session, False)
    assert session.stale
    assert not session.frames


async def test_result_invalidated_by_pause_during_inference(live):
    session = await setup_session(live)
    await push(live, session, 0, "white")

    async def pause(*args):
        live.pause(session, True)
        return {"summary": "不能提交", "speak": True, "comment": "不能发送"}

    live.analyze.side_effect = pause
    with pytest.raises(VideoError):
        await live.observe(session)
    assert session.observation is None
    assert not session.busy
    live.speak.assert_not_awaited()


async def test_analysis_budget_and_failure_do_not_queue_unbounded_calls(live):
    live.settings = replace(live.settings, live_max_analyses=1)
    session = await setup_session(live)
    await push(live, session, 0, "white")
    live.analyze.side_effect = RuntimeError("provider secret")
    with pytest.raises(RuntimeError):
        await live.observe(session)
    assert not session.busy
    with pytest.raises(VideoError, match="预算"):
        await live.observe(session)
    assert live.analyze.await_count == 1


async def test_audio_transcript_triggers_change_and_preserves_chunk_times(live):
    live.settings = replace(live.settings, asr_enabled=True, asr_provider_id="asr")
    session = await setup_session(live, audio=True)
    live.add_audio(session, b"\0\0" * 16000, 0, 1)
    await live.flush_audio(session)
    assert session.transcripts[0] == {"start_sec": 0, "end_sec": 1, "text": "新的台词"}
    assert session.reasons == {"new_dialogue"}
    assert session.audio_status == "ready"
    with pytest.raises(ValueError):
        live.add_audio(session, b"\0\0" * 16000, 0, 1)


async def test_asr_failure_does_not_prevent_visual_analysis(live):
    live.settings = replace(live.settings, asr_enabled=True, asr_provider_id="asr")
    session = await setup_session(live, audio=True)
    live.add_audio(session, b"\0\0" * 16000, 0, 1)
    live.transcribe.side_effect = RuntimeError("secret")
    await live.flush_audio(session)
    assert session.audio_status == "failed"
    assert "secret" not in session.error
    await push(live, session, 1, "white")
    assert (await live.observe(session))["summary"]


async def test_realtime_asr_reuses_provider_and_removes_temporary_wav(
    tmp_path, asr_provider
):
    def transcribe(path):
        with wave.open(path) as audio:
            assert audio.getframerate() == 16000
            assert audio.getnchannels() == 1
            assert audio.getnframes() == 16000
        return "转写结果"

    asr_provider.get_text.side_effect = transcribe
    context = SimpleNamespace(get_provider_by_id=lambda _: asr_provider)
    root = tmp_path / "asr"
    settings = replace(Settings(), asr_enabled=True, asr_provider_id="asr")
    assert await transcribe_pcm(context, settings, root, b"\0\0" * 16000) == "转写结果"
    assert not list(root.glob("*.wav"))


async def test_worker_processes_audio_while_visual_model_is_busy(live):
    live.settings = replace(live.settings, asr_enabled=True, asr_provider_id="asr")
    session = await setup_session(live, audio=True)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked(*args):
        entered.set()
        await release.wait()
        return {"summary": "新画面", "speak": False}

    live.analyze.side_effect = blocked
    session.last_user = -float("inf")
    await push(live, session, 0, "white")
    await asyncio.wait_for(entered.wait(), 2)
    live.add_audio(session, b"\0\0" * 16000, 0, 1)
    for _ in range(10):
        if session.transcripts:
            break
        await asyncio.sleep(0.1)
    assert session.transcripts
    assert live.analyze.await_count == 1
    await live.stop(session)
    assert session.analysis_task.done()
    assert session.task.done()
    assert not session.frames


async def test_stop_cancels_on_demand_analysis_and_clears_private_memory(live):
    session = await setup_session(live, mode="demand")
    await push(live, session, 0, "white")
    entered = asyncio.Event()

    async def blocked(*args):
        entered.set()
        await asyncio.Event().wait()

    live.analyze.side_effect = blocked
    task = asyncio.create_task(live.observe(session))
    await entered.wait()
    session.memory = "private"
    session.last_audio_text = "private dialogue"
    session.reasons.add("new_dialogue")
    await live.stop(session)
    assert task.cancelled()
    assert not session.operations
    assert not session.memory
    assert not session.last_audio_text
    assert not session.reasons
    live.speak.assert_not_awaited()


async def test_new_frames_do_not_make_old_observation_current(live):
    session = await setup_session(live, mode="demand")
    await push(live, session, 0, "white")
    await live.observe(session)
    session.observed_received = time.monotonic() - 11
    await push(live, session, 1, "black")
    snapshot = live.snapshot(session)
    assert snapshot["stale"] is False
    assert snapshot["observation_stale"] is True


async def test_worker_expires_even_while_model_is_blocked(live):
    session = await setup_session(live)
    await push(live, session, 0, "white")
    entered = asyncio.Event()

    async def blocked(*args):
        entered.set()
        await asyncio.Event().wait()

    live.analyze.side_effect = blocked
    session.last_user = -float("inf")
    await asyncio.wait_for(entered.wait(), 2)
    session.expires = time.monotonic() - 1
    await asyncio.wait_for(session.task, 2)
    assert session.stopped
    assert session.analysis_task.cancelled()


@pytest.mark.parametrize("text", ["not json", "[]", "{}", '{"summary":null}'])
def test_invalid_model_output_rejected(text):
    with pytest.raises(VideoError):
        parse_observation(text)


def test_fenced_model_json_accepted():
    assert parse_observation('```json\n{"summary": "画面"}\n```')["summary"] == "画面"


@pytest.mark.parametrize(
    "source,expected",
    [
        ("屏幕", "screen"),
        ("摄像头", "camera"),
        ("https://media.example/live.m3u8?x=1", "stream"),
    ],
)
def test_command_uses_configuration_and_preserves_instruction(source, expected):
    settings = Settings.from_mapping(
        {
            "asr_enabled": True,
            "asr_provider_id": "existing-asr",
            "live": {
                expected: {
                    "interval": {
                        "observation_interval": 20,
                        "proactive": False,
                        "sample_interval": 2,
                        "max_edge": 1920,
                        "max_frames": 3,
                        "lookback_seconds": 8,
                    }
                }
            },
        }
    )
    options, url = parse_live_command(source + " 定时 look at this，不要剧透", settings)
    assert options["source"] == expected
    assert options["focus"] == "look at this，不要剧透"
    assert options["interval"] == 20
    assert options["audio"] is True
    assert options["proactive"] is False
    effective = LiveOptions.parse(options, settings)
    assert effective.sample_interval == 2
    assert effective.max_edge == 1920
    assert effective.max_frames == 3
    assert effective.lookback_seconds == 8
    assert url == (source if expected == "stream" else "")


@pytest.mark.parametrize(
    "command",
    [
        "",
        "屏幕",
        "屏幕 变化",
        "file:///etc/passwd 变化 观察",
        "https:// 变化 观察",
        "屏幕 不存在 观察",
        "屏幕 按需 观察",
        "屏幕 按需取帧 观察",
        "屏幕 定时观察 观察",
        "屏幕 变化触发 观察",
        "屏幕 demand 观察",
        "屏幕 interval 观察",
        "屏幕 change 观察",
    ],
)
def test_invalid_live_command_rejected(command):
    with pytest.raises(ValueError):
        parse_live_command(command, Settings())


def test_removed_web_settings_are_ignored_during_upgrade():
    settings = Settings.from_mapping(
        {
            "live": {
                "live_enabled": True,
                "live_port": 6198,
                "live_host": "127.0.0.1",
                "live_public_url": "",
                "live_stream_url": "https://example.com/old",
            }
        }
    )
    assert settings.live_enabled
    assert not hasattr(settings, "live_port")
