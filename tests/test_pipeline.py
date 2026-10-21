from dataclasses import replace

import pytest
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.media import FFmpegMedia
from astrbot_plugin_vision.pipeline import STAGES, OverviewPipeline
from astrbot_plugin_vision.service import VideoService


@pytest.mark.parametrize("output_mode", ["frames", "grid"])
@pytest.mark.parametrize("budget,interval", [(20, 0.5), (3, 0.5), (200, 4)])
async def test_step_pipeline_matches_chat_output(
    tmp_path, sample_video, budget, interval, output_mode
):
    settings = Settings(
        overview_max_frames=budget,
        overview_interval=interval,
        overview_max_edge=128,
        overview_output_mode=output_mode,
    )
    media = FFmpegMedia(settings, tmp_path / "frames", [tmp_path])
    normal = await VideoService(media).overview(str(sample_video))
    diagnostic = OverviewPipeline(media, str(sample_video), diagnostic=True)
    for index, _ in enumerate(STAGES):
        await diagnostic.advance()
        assert diagnostic.completed == index + 1
        if index < 4:
            assert diagnostic.frames == []
    assert diagnostic.result.data == normal.data
    assert [frame.metadata() for frame in diagnostic.kept] == [
        frame.metadata() for frame in normal.frames
    ]
    assert diagnostic.scene_trace["scanned_frames"] == 30
    assert diagnostic.scene_trace["threshold_hits"] >= 1
    assert len(diagnostic.dedup_trace) == len(diagnostic.frames) - 1
    assert len(diagnostic.merge_trace) == len(diagnostic.scenes)
    with pytest.raises(ValueError, match="完毕"):
        await diagnostic.advance()


async def test_scene_cache_includes_window_and_preserves_diagnostics(
    tmp_path, sample_video, monkeypatch
):
    media = FFmpegMedia(Settings(), tmp_path / "frames", [tmp_path])
    info = await media.probe(str(sample_video))
    coarse = await media.scenes(info)
    media.settings = replace(media.settings, overview_interval=0.5)
    fine = await media.scenes(info)
    assert len(fine) > len(coarse)
    trace = {}
    assert await media.scenes(info, trace=trace) == fine
    assert trace["scanned_frames"] == 30
    assert len(trace["curve"]) == 30
    assert trace["cache_hit"] is False

    async def unexpected(*args, **kwargs):
        raise AssertionError("相同参数应复用诊断缓存")

    monkeypatch.setattr(media, "_run", unexpected)
    cached = {}
    assert await media.scenes(info, trace=cached) == fine
    assert cached == {**trace, "cache_hit": True}
