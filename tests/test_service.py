from pathlib import Path

import pytest
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.media import Frame, VideoError, VideoInfo
from astrbot_plugin_vision.sampling import Candidate
from astrbot_plugin_vision.service import VideoService


class FakeMedia:
    settings = Settings(overview_interval=2, overview_max_frames=10)

    def __init__(self, *, different=False, low_quality=False, fail_scene=False):
        self.different = different
        self.low_quality = low_quality
        self.fail_scene = fail_scene

    async def probe(self, video_path):
        return VideoInfo(Path(video_path), "key", 10, 640, 360, 10, 0)

    async def scenes(self, info):
        if self.fail_scene:
            raise VideoError("test")
        return [Candidate(2.05, "scene_change", 0.9)]

    async def extract(self, info, timestamp, max_edge, reason):
        fingerprint = 100 if self.different and reason == "scene_change" else 0
        quality = 0 if self.low_quality and reason == "scene_change" else 10
        return Frame(
            str(timestamp),
            timestamp,
            timestamp,
            reason,
            640,
            360,
            Path("frame.jpg"),
            (fingerprint,) * 576,
            quality,
        )


async def test_nearby_similar_frames_prefer_keyframe():
    result = await VideoService(FakeMedia()).overview("video.mp4")
    times = [frame.timestamp_sec for frame in result.frames]
    assert 2.05 in times
    assert 2 not in times
    assert 0 in times and 9.9 in times
    assert result.data["sampling"]["actual_max_gap_sec"] <= 2.1


async def test_meaningful_nearby_difference_is_preserved():
    result = await VideoService(FakeMedia(different=True)).overview("video.mp4")
    times = [frame.timestamp_sec for frame in result.frames]
    assert 2 in times and 2.05 in times


async def test_low_quality_keyframe_does_not_replace_clear_frame():
    result = await VideoService(FakeMedia(low_quality=True)).overview("video.mp4")
    times = [frame.timestamp_sec for frame in result.frames]
    assert 2 in times and 2.05 not in times


async def test_failed_scene_scan_falls_back_to_uniform():
    result = await VideoService(FakeMedia(fail_scene=True)).overview("video.mp4")
    assert result.data["warnings"]
    assert len(result.frames) >= 5


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_frames": True},
        {"max_frames": 1.5},
        {"max_edge": float("inf")},
        {"max_edge": 10},
    ],
)
async def test_invalid_model_limits_rejected(arguments):
    with pytest.raises(ValueError):
        await VideoService(FakeMedia()).inspect(
            "video.mp4", start_sec=0, end_sec=1, **arguments
        )
