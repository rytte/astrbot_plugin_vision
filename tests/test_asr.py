import array
import asyncio
import json
import os
import subprocess
import wave
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot_plugin_vision.asr import MAX_TEXT_CHARS, VideoASR, chunk_indices
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.media import FFmpegMedia
from astrbot_plugin_vision.pipeline import VideoResult


@pytest.fixture
def speech(tmp_path, asr_provider):
    media = FFmpegMedia(Settings(), tmp_path / "frames", [tmp_path])
    return SimpleNamespace(
        asr=VideoASR(media),
        media=media,
        settings=Settings(asr_enabled=True, asr_provider_id="test-asr"),
        context=SimpleNamespace(get_provider_by_id=Mock(return_value=asr_provider)),
    )


def video_result(path, duration=3, request=None):
    data = {"video_path": str(path), "duration_sec": duration}
    if request is not None:
        data["request"] = request
    return VideoResult(data, [])


async def enhance(speech, path, *, delivered=None, budget=10, **kwargs):
    return await speech.asr.enhance(
        speech.context,
        speech.settings,
        video_result(path, **kwargs),
        asyncio.get_running_loop().time() + budget,
        delivered or set(),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_asr_configuration(enabled):
    settings = Settings.from_mapping(
        {"asr_enabled": enabled, "asr_provider_id": " test-asr "}
    )
    assert settings.asr_enabled is enabled
    assert settings.asr_provider_id == "test-asr"


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None, [], {}])
def test_asr_switch_requires_boolean(value):
    with pytest.raises(ValueError, match="布尔"):
        Settings.from_mapping({"asr_enabled": value})


@pytest.mark.parametrize(
    "duration,selection,expected",
    [
        (65, None, [0, 1, 2]),
        (60, None, [0, 1]),
        (0.1, None, [0]),
        (65, {"mode": "range", "times_sec": [29, 30]}, [0]),
        (65, {"mode": "range", "times_sec": [30, 60]}, [1]),
        (65, {"mode": "range", "times_sec": [30, 30]}, [1]),
        (60, {"mode": "timestamps", "times_sec": [60]}, [1]),
        (65, {"mode": "timestamps", "times_sec": [60, 1, 2]}, [0, 2]),
    ],
)
def test_chunk_ranges_follow_video_timeline(duration, selection, expected):
    assert list(chunk_indices(duration, selection)) == expected


async def test_disabled_asr_never_probes_or_resolves_provider(speech, monkeypatch):
    speech.settings = replace(speech.settings, asr_enabled=False)
    monkeypatch.setattr(speech.media, "_run", AsyncMock())
    assert await enhance(speech, "missing.mp4") is None
    speech.context.get_provider_by_id.assert_not_called()
    speech.media._run.assert_not_awaited()


@pytest.mark.parametrize("availability", ["blank", "missing", "wrong_type", "disabled"])
async def test_unavailable_asr_does_not_extract_audio(
    speech, asr_provider, availability, monkeypatch
):
    if availability == "blank":
        speech.settings = replace(speech.settings, asr_provider_id="")
    elif availability == "missing":
        speech.context.get_provider_by_id.return_value = None
    elif availability == "wrong_type":
        speech.context.get_provider_by_id.return_value = SimpleNamespace(
            get_text=AsyncMock()
        )
    else:
        asr_provider.provider_config["enable"] = False
    monkeypatch.setattr(speech.media, "_run", AsyncMock())
    result = await enhance(speech, "missing.mp4")
    assert result["status"] == "unavailable"
    assert result["segments"] == []
    speech.media._run.assert_not_awaited()
    asr_provider.get_text.assert_not_awaited()


async def test_video_without_audio_does_not_call_provider(
    speech, sample_video, asr_provider
):
    result = await enhance(speech, sample_video)
    assert result["status"] == "no_audio"
    assert result["segments"] == []
    asr_provider.get_text.assert_not_awaited()
    assert not list(speech.media.cache_dir.glob("*.wav"))


async def test_real_audio_is_mono_wav_and_cached(speech, audio_video, asr_provider):
    def transcribe(path):
        with wave.open(path, "rb") as audio:
            assert audio.getnchannels() == 1
            assert audio.getframerate() == 16000
            assert audio.getsampwidth() == 2
            assert audio.getnframes() / 16000 == pytest.approx(3, abs=0.05)
        return "识别到的语音。"

    asr_provider.get_text.side_effect = transcribe
    first = await enhance(speech, audio_video)
    assert first["status"] == "success"
    assert first["time_precision"] == "chunk"
    assert first["segments"][0]["start_sec"] == 0
    assert first["segments"][0]["end_sec"] == 3
    assert first["segments"][0]["text"] == "识别到的语音。"
    second = await enhance(speech, audio_video)
    assert second["segments"] == first["segments"]
    assert second["cache_hit_count"] == 1
    asr_provider.get_text.assert_awaited_once()
    assert not list(speech.media.cache_dir.glob("*.wav"))
    cached = "".join(
        path.read_text(encoding="utf-8")
        for path in speech.media.cache_dir.glob("*.json")
    )
    assert "private-key" not in cached


async def test_concurrent_calls_share_cache(speech, audio_video, asr_provider):
    first, second = await asyncio.gather(
        enhance(speech, audio_video), enhance(speech, audio_video)
    )
    assert first["segments"] == second["segments"]
    assert sorted([first["cache_hit_count"], second["cache_hit_count"]]) == [0, 1]
    asr_provider.get_text.assert_awaited_once()


async def test_already_delivered_transcript_is_not_repeated(
    speech, audio_video, asr_provider
):
    first = await enhance(speech, audio_video)
    second = await enhance(
        speech,
        audio_video,
        delivered={segment["segment_id"] for segment in first["segments"]},
        request={"mode": "range", "times_sec": [1, 2]},
    )
    assert second["segments"] == []
    assert second["complete"]
    assert second["skipped_already_provided_count"] == 1
    asr_provider.get_text.assert_awaited_once()


@pytest.mark.parametrize("change", ["file", "provider", "model"])
async def test_transcript_cache_tracks_video_and_provider(
    speech, audio_video, asr_provider, change
):
    first = await enhance(speech, audio_video)
    if change == "file":
        stat = audio_video.stat()
        os.utime(audio_video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    elif change == "provider":
        asr_provider.provider_config["key"] = ["new-private-key"]
    else:
        asr_provider.get_model.return_value = "other-model"
    second = await enhance(speech, audio_video)
    assert second["segments"][0]["segment_id"] != first["segments"][0]["segment_id"]
    assert asr_provider.get_text.await_count == 2


async def test_failed_asr_is_sanitized_not_cached_and_can_retry(
    speech, audio_video, asr_provider
):
    asr_provider.get_text.side_effect = RuntimeError("private-key provider response")
    failed = await enhance(speech, audio_video)
    assert failed["status"] == "failed"
    assert "private-key" not in json.dumps(failed)
    assert not list(speech.media.cache_dir.glob("*.wav"))
    asr_provider.get_text.side_effect = None
    assert (await enhance(speech, audio_video))["status"] == "success"
    assert asr_provider.get_text.await_count == 2


async def test_partial_timeout_keeps_completed_segments(
    speech, sample_video, monkeypatch
):
    monkeypatch.setattr(speech.asr, "_has_audio", AsyncMock(return_value=True))
    monkeypatch.setattr(
        speech.asr,
        "_chunk",
        AsyncMock(side_effect=[("前半段转写", False), TimeoutError()]),
    )
    result = await enhance(speech, sample_video, duration=65)
    assert result["status"] == "partial"
    assert result["completed_chunk_count"] == 1
    assert result["requested_chunk_count"] == 3
    assert not result["complete"]
    assert result["segments"][0]["text"] == "前半段转写"


async def test_long_transcript_is_explicitly_truncated(
    speech, sample_video, monkeypatch
):
    monkeypatch.setattr(speech.asr, "_has_audio", AsyncMock(return_value=True))
    monkeypatch.setattr(
        speech.asr,
        "_chunk",
        AsyncMock(return_value=("文" * (MAX_TEXT_CHARS + 1), False)),
    )
    result = await enhance(speech, sample_video)
    assert result["status"] == "partial"
    assert result["truncated"]
    assert result["segments"][0]["truncated"]
    assert len(result["segments"][0]["text"]) == MAX_TEXT_CHARS


async def test_timeout_cancels_provider_and_removes_audio(
    speech, audio_video, asr_provider, monkeypatch
):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def transcribe(path):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    asr_provider.get_text.side_effect = transcribe
    monkeypatch.setattr("astrbot_plugin_vision.asr.MAX_ASR_SECONDS", 0.5)
    result = await enhance(speech, audio_video)
    assert started.is_set() and cancelled.is_set()
    assert result["status"] == "timeout"
    assert not speech.media.processes
    assert not list(speech.media.cache_dir.glob("*.wav"))


async def test_cancellation_propagates_and_removes_audio(
    speech, audio_video, asr_provider
):
    started = asyncio.Event()

    async def transcribe(path):
        started.set()
        await asyncio.Event().wait()

    asr_provider.get_text.side_effect = transcribe
    task = asyncio.create_task(enhance(speech, audio_video))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not speech.media.processes
    assert not list(speech.media.cache_dir.glob("*.wav"))


async def test_audio_with_nonzero_container_origin(
    speech, audio_video, ffmpeg, tmp_path, asr_provider
):
    shifted = tmp_path / "shifted audio.mp4"
    await asyncio.to_thread(
        subprocess.run,
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(audio_video),
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

    def transcribe(path):
        with wave.open(path, "rb") as audio:
            assert audio.getnframes() / audio.getframerate() == pytest.approx(
                3, abs=0.05
            )
        return "移位视频中的语音"

    asr_provider.get_text.side_effect = transcribe
    result = await enhance(speech, shifted)
    assert result["status"] == "success"
    assert result["segments"][0]["start_sec"] == 0


@pytest.mark.parametrize("offset", [0, 5])
async def test_inspect_seeks_correct_audio_chunk_then_overview_reuses_it(
    speech, asr_provider, tmp_path, ffmpeg, offset
):
    video = tmp_path / "two tones.mkv"
    await asyncio.to_thread(
        subprocess.run,
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x36:r=2:d=65",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=30[a];"
            "sine=frequency=880:sample_rate=16000:duration=35[b];"
            "[a][b]concat=n=2:v=0:a=1",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-c:a",
            "pcm_s16le",
            "-output_ts_offset",
            str(offset),
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    frequencies = []

    def transcribe(path):
        with wave.open(path, "rb") as audio:
            audio.setpos(8000)
            samples = array.array("h", audio.readframes(16000))
        # 一秒的正向过零次数可以区分分段前后的两种音频。
        frequency = sum(left <= 0 < right for left, right in zip(samples, samples[1:]))
        frequencies.append(frequency)
        return f"音调 {round(frequency / 10) * 10}"

    asr_provider.get_text.side_effect = transcribe
    inspected = await enhance(
        speech,
        video,
        duration=65,
        request={"mode": "range", "times_sec": [30, 31]},
    )
    assert inspected["status"] == "success"
    assert frequencies == pytest.approx([880], abs=2)
    assert inspected["segments"][0]["start_sec"] == 30
    assert inspected["segments"][0]["end_sec"] == 60
    overview = await enhance(speech, video, duration=65)
    assert overview["status"] == "success"
    assert overview["cache_hit_count"] == 1
    assert frequencies == pytest.approx([880, 440, 880], abs=2)
    assert [segment["start_sec"] for segment in overview["segments"]] == [0, 30, 60]
    assert overview["segments"][-1]["end_sec"] == 65


@pytest.mark.parametrize("response", ["", "  ", None])
async def test_empty_or_invalid_asr_output(speech, audio_video, asr_provider, response):
    asr_provider.get_text.return_value = response
    result = await enhance(speech, audio_video)
    assert result["status"] == ("failed" if response is None else "success")
    assert result["segments"] == []
    assert "不证明没有声音" in result["guidance"]
    assert not list(speech.media.cache_dir.glob("*.wav"))
