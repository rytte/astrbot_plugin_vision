import asyncio
import os
import subprocess
import sys
import time
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.media import FFmpegMedia, VideoError
from astrbot_plugin_vision.service import VideoService


@pytest.fixture
def media(tmp_path):
    return FFmpegMedia(Settings(), tmp_path / "frames", [tmp_path])


async def test_probe_seek_scene_detection_and_cache(media, sample_video, monkeypatch):
    await media.check_dependencies()
    info = await media.probe(str(sample_video))
    assert info.duration == pytest.approx(3)
    assert (info.width, info.height, info.fps) == (320, 180, 10)
    scenes = await media.scenes(info)
    assert any(abs(scene.timestamp - 1) < 0.15 for scene in scenes)
    frame = await media.extract(info, 1.25, 128, "inspect")
    assert frame.timestamp_sec == pytest.approx(1.3, abs=0.001)
    assert frame.requested_sec == 1.25
    assert "requested_sec" not in frame.metadata()
    assert "frame_id" not in frame.metadata()
    assert frame.metadata(include_frame_id=True)["frame_id"] == frame.frame_id
    assert (frame.width, frame.height) == (128, 72)
    assert frame.path.read_bytes().startswith(b"\xff\xd8")

    async def unexpected_run(*args, **kwargs):
        raise AssertionError("缓存命中时不应再次运行 FFmpeg")

    monkeypatch.setattr(media, "_run", unexpected_run)
    cached = await media.extract(info, 1.25, 128, "scene_change")
    assert cached.frame_id == frame.frame_id
    assert cached.requested_sec == 1.25
    assert cached.reason == "scene_change"
    assert await media.scenes(info) == scenes


async def test_no_upscale_and_tail_frame(media, sample_video):
    info = await media.probe(str(sample_video))
    frame = await media.extract(info, 3, 640, "inspect")
    assert (frame.width, frame.height) == (320, 180)
    assert frame.timestamp_sec == pytest.approx(2.9, abs=0.001)


async def test_nonzero_source_timestamp(media, sample_video, ffmpeg, tmp_path):
    shifted = tmp_path / "shifted.mp4"
    await asyncio.to_thread(
        subprocess.run,
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(sample_video),
            "-c",
            "copy",
            "-output_ts_offset",
            "5",
            str(shifted),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    info = await media.probe(str(shifted))
    assert info.origin == pytest.approx(5)
    frame = await media.extract(info, 1.2, 128, "inspect")
    assert frame.timestamp_sec == pytest.approx(1.2, abs=0.001)


async def test_real_overview_and_inspect(media, sample_video):
    service = VideoService(media)
    overview = await service.overview(str(sample_video))
    assert len(overview.frames) <= 200
    assert overview.frames[0].timestamp_sec == 0
    assert overview.frames[-1].timestamp_sec >= 2.8
    assert any(frame.reason == "scene_change" for frame in overview.frames)
    result = await service.inspect(
        str(sample_video),
        mode="range",
        times_sec=[0.8, 1.2],
        max_frames=4,
        max_edge=128,
    )
    assert 1 <= len(result.frames) <= 4
    assert all(frame.width <= 128 for frame in result.frames)
    assert all(0.8 <= frame.timestamp_sec <= 1.3 for frame in result.frames)


def test_path_traversal_and_external_symlink_rejected(media, tmp_path):
    outside = tmp_path.parent / f"outside-{tmp_path.name}.mp4"
    outside.write_bytes(b"video")
    try:
        with pytest.raises(VideoError, match="允许"):
            media.resolve_path(str(outside))
        link = tmp_path / "linked.mp4"
        try:
            link.symlink_to(outside)
        except OSError:
            return
        with pytest.raises(VideoError, match="允许"):
            media.resolve_path(str(link))
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "path",
    ["https://example.com/a.mp4", "file:///tmp/a.mp4", "\\\\server\\a.mp4", "", True],
)
def test_nonlocal_or_invalid_paths_rejected(media, path):
    with pytest.raises(VideoError):
        media.resolve_path(path)


def test_video_at_size_limit_is_accepted(media, tmp_path):
    media.settings = replace(media.settings, download_max_mb=1)
    source = tmp_path / "at-limit.mp4"
    source.write_bytes(b"x" * 1024**2)
    assert media.resolve_path(str(source)) == source


async def test_oversize_video_is_rejected_before_ffprobe(media, tmp_path):
    media.settings = replace(media.settings, download_max_mb=1)
    source = tmp_path / "oversize.mp4"
    source.write_bytes(b"x" * (1024**2 + 1))
    media._run = AsyncMock()
    with pytest.raises(VideoError, match="大小上限.*1 MB"):
        await media.probe(str(source))
    media._run.assert_not_awaited()
    assert source.exists()


@pytest.mark.parametrize("limit", [2.99, 3.0])
async def test_video_duration_limit_includes_boundary(media, sample_video, limit):
    media.settings = replace(media.settings, download_max_duration=limit)
    if limit < 3:
        with pytest.raises(VideoError, match="时长上限.*2.99 秒"):
            await media.probe(str(sample_video))
    else:
        assert (await media.probe(str(sample_video))).duration == pytest.approx(3)
    assert sample_video.exists()


async def test_timeout_and_cancellation_kill_children(media):
    with pytest.raises(VideoError, match="超时"):
        await media._run(
            sys.executable, "-c", "import time; time.sleep(20)", timeout=0.1
        )
    assert not media.processes
    task = asyncio.create_task(
        media._run(sys.executable, "-c", "import time; time.sleep(20)")
    )
    for _ in range(100):
        if media.processes:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not media.processes


def test_cache_cleanup_does_not_delete_source(media, sample_video):
    stale = media.cache_dir / "stale.jpg"
    stale.write_bytes(b"cached")
    os.utime(stale, (time.time() - 7200, time.time() - 7200))
    media.cleanup_cache()
    assert not stale.exists()
    assert sample_video.exists()


async def test_model_requested_limits_are_clamped(media, sample_video):
    service = VideoService(media)
    result = await service.inspect(
        str(sample_video),
        mode="range",
        times_sec=[0, 1],
        max_frames=100,
        max_edge=10000,
    )
    assert result.data["effective_max_frames"] == 8
    assert result.data["effective_max_edge"] == 1280


async def test_missing_executable_gives_clear_error(media):
    media.ffmpeg = "nonexistent-vision-ffmpeg-executable"
    with pytest.raises(VideoError, match="启动"):
        await media.check_dependencies()


async def test_changed_source_invalidates_cache(media, sample_video):
    original = await media.probe(str(sample_video))
    stat = sample_video.stat()
    os.utime(sample_video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    updated = await media.probe(str(sample_video))
    assert updated.key != original.key
