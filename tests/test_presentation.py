import asyncio
import threading
import time
from dataclasses import replace

import pytest
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.media import FFmpegMedia, Frame, VideoError
from astrbot_plugin_vision.presentation import (
    GUTTER,
    LABEL_HEIGHT,
    present_frames,
    presentation_metadata,
)
from PIL import Image


@pytest.fixture
def media(tmp_path):
    return FFmpegMedia(
        Settings(overview_output_mode="grid"), tmp_path / "frames", [tmp_path]
    )


def make_frames(media, count, uniform=False):
    frames = []
    for index in range(count):
        width, height = (
            (64, 48) if uniform else (64 + index % 3 * 11, 48 + index % 2 * 7)
        )
        path = media.cache_dir / f"frame_{index}.jpg"
        with Image.new("RGB", (width, height)) as image:
            image.putdata(
                [
                    ((x * 9 + index * 17) % 256, y * 3 % 256, (x + y) % 256)
                    for y in range(height)
                    for x in range(width)
                ]
            )
            image.save(path)
        frames.append(
            Frame(
                f"frame-{index}",
                index * 1.137,
                index * 1.13,
                "interval",
                width,
                height,
                path,
                (0,) * 576,
                1,
            )
        )
    return frames


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 8, 9, 10, 18, 19])
@pytest.mark.parametrize("uniform", [False, True])
async def test_grid_preserves_every_pixel_and_maps_time_in_order(media, count, uniform):
    frames = make_frames(media, count, uniform)
    images = await present_frames(media, frames)
    assert len(images) == (count + 8) // 9
    cells = []
    for output in images:
        assert output.mime_type == "image/png"
        assert 1 <= len(output.cells) <= 9
        assert [(cell["row"], cell["column"]) for cell in output.cells] == sorted(
            (cell["row"], cell["column"]) for cell in output.cells
        )
        metadata = output.metadata()
        sizes = {
            (
                frames[cell["frame_number"] - 1].width,
                frames[cell["frame_number"] - 1].height,
            )
            for cell in output.cells
        }
        if len(sizes) == 1:
            assert (metadata["frame_width"], metadata["frame_height"]) == next(
                iter(sizes)
            )
            assert all(
                "width" not in cell and "height" not in cell for cell in output.cells
            )
        else:
            assert "frame_width" not in metadata and "frame_height" not in metadata
            assert all("width" in cell and "height" in cell for cell in output.cells)
        with Image.open(output.path) as grid:
            assert grid.format == "PNG"
            assert grid.size == (output.width, output.height)
            columns = max(cell["column"] for cell in output.cells)
            rows = max(cell["row"] for cell in output.cells)
            cell_width = (grid.width - (columns + 1) * GUTTER) // columns
            cell_height = (grid.height - (rows + 1) * GUTTER) // rows
            for cell in output.cells:
                frame = frames[cell["frame_number"] - 1]
                assert cell["timestamp_sec"] == frame.timestamp_sec
                assert "requested_sec" not in cell
                assert cell["frame_id"] == frame.frame_id
                assert "x" not in cell and "y" not in cell
                assert cell.get("width", metadata.get("frame_width")) == frame.width
                assert cell.get("height", metadata.get("frame_height")) == frame.height
                x = GUTTER + (cell["column"] - 1) * (cell_width + GUTTER)
                y = GUTTER + (cell["row"] - 1) * (cell_height + GUTTER) + LABEL_HEIGHT
                crop = grid.crop(
                    (
                        x,
                        y,
                        x + frame.width,
                        y + frame.height,
                    )
                )
                with Image.open(frame.path) as source:
                    assert crop.size == source.size
                    assert crop.tobytes() == source.convert("RGB").tobytes()
                cells.append(cell)
    assert [cell["frame_number"] for cell in cells] == list(range(1, count + 1))
    assert presentation_metadata("grid", images)["frame_count"] == count


@pytest.mark.parametrize("uniform", [False, True])
async def test_independent_mode_returns_original_jpegs(media, uniform):
    frames = make_frames(media, 3, uniform=uniform)
    images = await present_frames(media, frames, mode="frames")
    assert [image.path for image in images] == [frame.path for frame in frames]
    assert all(image.mime_type == "image/jpeg" for image in images)
    assert all("requested_sec" not in image.cells[0] for image in images)
    metadata = presentation_metadata("frames", images)
    if uniform:
        assert (metadata["frame_width"], metadata["frame_height"]) == (64, 48)
        assert all(
            "width" not in image and "height" not in image
            for image in metadata["images"]
        )
    else:
        assert "frame_width" not in metadata and "frame_height" not in metadata
        assert [(image["width"], image["height"]) for image in metadata["images"]] == [
            (frame.width, frame.height) for frame in frames
        ]
    # 公共文本元数据的压缩不改变原图及内部图片尺寸。
    assert [
        (image.metadata()["width"], image.metadata()["height"]) for image in images
    ] == [(frame.width, frame.height) for frame in frames]


async def test_grid_cache_reuses_same_labels_but_changes_with_frame_order(
    media, monkeypatch
):
    frames = make_frames(media, 9)
    first = await present_frames(media, frames)
    reversed_images = await present_frames(media, list(reversed(frames)))
    assert reversed_images[0].path != first[0].path
    monkeypatch.setattr(Image, "new", lambda *a, **kw: pytest.fail("应复用拼图缓存"))
    cached = await present_frames(media, frames)
    assert cached == first


async def test_grid_rejects_oversized_canvas_without_resizing(media):
    frames = [
        replace(frame, width=4096, height=4096) for frame in make_frames(media, 9)
    ]
    with pytest.raises(VideoError, match="独立帧"):
        await present_frames(media, frames)
    assert not list(media.cache_dir.glob("*.png"))


async def test_grid_respects_debug_cache_capacity_and_removes_temporary_file(media):
    frames = make_frames(media, 9)
    media.cache_limit_bytes = sum(frame.path.stat().st_size for frame in frames) + 1
    with pytest.raises(VideoError, match="容量上限"):
        await present_frames(media, frames)
    assert all(frame.path.is_file() for frame in frames)
    assert not list(media.cache_dir.glob("*.png"))


async def test_cancel_waits_for_atomic_grid_write_before_returning(media, monkeypatch):
    import astrbot_plugin_vision.presentation as presentation

    frames = make_frames(media, 1)
    started, release = threading.Event(), threading.Event()
    original = presentation._grid

    def slow_grid(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(presentation, "_grid", slow_grid)
    task = asyncio.create_task(present_frames(media, frames))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(list(media.cache_dir.glob("grid_*.png"))) == 1
    assert not list(media.cache_dir.glob("*.tmp.png"))


async def test_grid_png_participates_in_expiry_cleanup(media, monkeypatch):
    image = (await present_frames(media, make_frames(media, 1)))[0]
    now = time.time()
    monkeypatch.setattr("astrbot_plugin_vision.media.time.time", lambda: now + 4000)
    media.cleanup_cache()
    assert not image.path.exists()


@pytest.mark.parametrize("value", ["", "nine", 9, True, None])
def test_invalid_output_mode_is_rejected(value):
    with pytest.raises(ValueError):
        Settings.from_mapping({"overview_output_mode": value})
