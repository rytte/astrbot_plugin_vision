import asyncio
import io
import os
import sys
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.live import LiveSessions
from astrbot_plugin_vision.live_capture import CaptureInput, capture_input, read_capture
from astrbot_plugin_vision.media import VideoError
from PIL import Image


@pytest.fixture
async def capture(tmp_path):
    sessions = LiveSessions(
        Settings(asr_enabled=True, asr_provider_id="asr"),
        tmp_path,
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
    )
    session = await sessions.create("chat", "alice")
    sessions.configure(session, {"source": "stream", "mode": "demand", "audio": True})
    session.source_url = "https://example.com/live"
    yield sessions, session
    await sessions.close()


async def test_actual_http_media_decodes_and_stops(capture, audio_video, ffmpeg):
    sessions, session = capture
    app = web.Application()

    async def media(request):
        return web.FileResponse(audio_video)

    app.router.add_get("/video.mp4", media)
    async with TestServer(app) as source:
        session.source_url = str(source.make_url("/video.mp4"))
        session.stream_task = asyncio.create_task(
            read_capture(sessions, session, ffmpeg)
        )
        await asyncio.wait_for(session.ready.wait(), 10)
        assert session.frames, session.error
        assert session.frames[-1].timestamp >= 0
        await sessions.stop(session)
        assert session.stream_task.done()
        assert not session.frames
        assert not session.source_url


@pytest.mark.parametrize(
    "source_size,max_edge,expected",
    [
        ((1920, 1080), 1920, (1920, 1080)),
        ((1080, 1920), 1600, (900, 1600)),
        ((1920, 1080), 640, (640, 360)),
        ((320, 180), 1920, (320, 180)),
    ],
)
async def test_actual_capture_respects_live_resolution(
    capture, ffmpeg, monkeypatch, source_size, max_edge, expected
):
    sessions, session = capture
    sessions.settings = replace(sessions.settings, max_edge_limit=1024)
    session.options = replace(session.options, audio=False, max_edge=max_edge)
    width, height = source_size
    monkeypatch.setattr(
        "astrbot_plugin_vision.live_capture.capture_input",
        AsyncMock(
            return_value=CaptureInput(
                ("-f", "lavfi", "-i", f"color=c=red:s={width}x{height}:r=1:d=2"),
                None,
            )
        ),
    )
    session.stream_task = asyncio.create_task(read_capture(sessions, session, ffmpeg))
    await asyncio.wait_for(session.ready.wait(), 10)
    assert session.frames, session.error
    with Image.open(io.BytesIO(session.frames[-1].jpeg)) as image:
        assert image.size == expected
    await sessions.stop(session)
    assert session.stream_task.done()


@pytest.mark.parametrize("sample_interval", [0.5, 2, 30])
async def test_capture_outputs_first_frame_promptly_and_uses_preset_cadence(
    capture, ffmpeg, monkeypatch, sample_interval
):
    sessions, session = capture
    session.options = replace(
        session.options, audio=False, sample_interval=sample_interval
    )
    second_frame = asyncio.Event()
    add_frame = sessions.add_frame

    async def receive(*args):
        await add_frame(*args)
        if len(session.frames) >= 2:
            second_frame.set()

    monkeypatch.setattr(sessions, "add_frame", receive)
    monkeypatch.setattr(
        "astrbot_plugin_vision.live_capture.capture_input",
        AsyncMock(
            return_value=CaptureInput(
                ("-re", "-f", "lavfi", "-i", "color=c=red:s=160x90:r=8"),
                None,
            )
        ),
    )
    started = time.monotonic()
    session.stream_task = asyncio.create_task(read_capture(sessions, session, ffmpeg))
    await asyncio.wait_for(session.ready.wait(), 5)
    assert session.frames, session.error
    assert session.frames[0].received - started < 5
    if sample_interval < 30:
        await asyncio.wait_for(second_frame.wait(), 6)
        gap = session.frames[1].received - session.frames[0].received
        assert gap == pytest.approx(sample_interval, abs=0.3)
    await sessions.stop(session)
    assert session.stream_task.done()


@pytest.mark.parametrize("audio_failure", [True, False])
async def test_audio_failure_or_final_partial_chunk(
    capture, monkeypatch, audio_failure
):
    sessions, session = capture
    waiting = asyncio.Event()

    async def wait_for_frame(_):
        await waiting.wait()

    video = SimpleNamespace(
        stdout=SimpleNamespace(read=AsyncMock(side_effect=wait_for_frame)),
        returncode=None,
        kill=Mock(),
        wait=AsyncMock(return_value=0),
    )
    pcm = b"\0\0" * 3200
    audio = SimpleNamespace(
        stdout=SimpleNamespace(
            readexactly=AsyncMock(side_effect=asyncio.IncompleteReadError(pcm, 160000))
        ),
        returncode=None,
        kill=Mock(),
        wait=AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        AsyncMock(
            side_effect=[video, OSError("unavailable") if audio_failure else audio]
        ),
    )
    session.stream_task = asyncio.create_task(read_capture(sessions, session, "ffmpeg"))
    for _ in range(100):
        if session.audio_status == "unavailable" or session.audio_chunks:
            break
        await asyncio.sleep(0.01)
    assert not session.stream_task.done()
    if audio_failure:
        assert session.audio_status == "unavailable"
    else:
        start, end, chunk = session.audio_chunks[0]
        assert chunk == pcm
        assert end - start == pytest.approx(0.2)
    await sessions.stop(session)
    video.kill.assert_called_once()
    if not audio_failure:
        audio.kill.assert_called_once()


async def test_linux_sources_use_host_and_existing_audio_config(capture, monkeypatch):
    sessions, session = capture
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":23")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setattr(
        "astrbot_plugin_vision.live_capture.shutil.which", lambda _: "/usr/bin/parec"
    )
    session.options = replace(session.options, source="screen")
    incoming = await capture_input(session, sessions.settings, "ffmpeg")
    assert incoming.video[-1] == ":23"
    assert "--device=@DEFAULT_MONITOR@" in incoming.audio
    session.options = replace(session.options, source="camera")
    incoming = await capture_input(session, sessions.settings, "ffmpeg")
    assert incoming.video[-1] == "/dev/video0"
    assert "--device=@DEFAULT_SOURCE@" in incoming.audio
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    session.options = replace(session.options, source="screen")
    with pytest.raises(VideoError, match="X11"):
        await capture_input(session, sessions.settings, "ffmpeg")


@pytest.mark.skipif(
    not os.environ.get("DISPLAY") or sys.platform != "linux",
    reason="requires local X11",
)
async def test_actual_x11_desktop_capture(capture, ffmpeg):
    sessions, session = capture
    session.options = replace(session.options, source="screen", audio=False)
    session.stream_task = asyncio.create_task(read_capture(sessions, session, ffmpeg))
    await asyncio.wait_for(session.ready.wait(), 10)
    assert session.frames, session.error
    await sessions.stop(session)
    assert session.stream_task.cancelled()
