import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(PLUGIN_DIR.parent / "AstrBot"))


@pytest.fixture(scope="session")
def ffmpeg():
    executable = shutil.which("ffmpeg")
    if not executable or not shutil.which("ffprobe"):
        pytest.skip("真实视频测试需要 FFmpeg 和 ffprobe")
    return executable


@pytest.fixture
def sample_video(tmp_path, ffmpeg):
    output = tmp_path / "sample video.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:r=10:d=1[red];"
            "color=c=blue:s=320x180:r=10:d=1[blue];"
            "color=c=green:s=320x180:r=10:d=1[green];"
            "[red][blue][green]concat=n=3:v=1:a=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return output


@pytest.fixture
def asr_provider():
    from astrbot.api.provider import STTProvider

    provider = Mock(spec=STTProvider)
    provider.provider_config = {
        "id": "test-asr",
        "enable": True,
        "key": ["private-key"],
    }
    provider.get_model.return_value = "test-speech-model"
    provider.meta.return_value = SimpleNamespace(
        id="test-asr", model="test-speech-model"
    )
    provider.get_text = AsyncMock(return_value="测试语音转写：请观察画面。")
    return provider


@pytest.fixture
def audio_video(tmp_path, ffmpeg, sample_video):
    output = tmp_path / "video with audio.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(sample_video),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return output
