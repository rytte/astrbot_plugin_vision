import shutil
import subprocess
import sys
from pathlib import Path

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
