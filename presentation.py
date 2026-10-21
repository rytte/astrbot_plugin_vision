"""概览图片的呈现方式；拼图不改变采样、去重或单帧像素。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .media import FFmpegMedia, Frame, VideoError

MAX_GRID_PIXELS = 33_554_432
GUTTER = 8
LABEL_HEIGHT = 32


@dataclass
class OutputImage:
    path: Path
    mime_type: str
    width: int
    height: int
    cells: list[dict]

    def metadata(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "mime_type": self.mime_type,
            "cells": self.cells,
        }


def presentation_metadata(mode: str, images: list[OutputImage]) -> dict:
    return {
        "mode": mode,
        "reading_order": "left_to_right_then_top_to_bottom",
        "note": "九宫格保留抽帧后的像素尺寸，编号和实际时间戳位于画面外；按 cells 引用证据。"
        if mode == "grid"
        else "每张图片对应一帧，按时间排序。",
        "image_count": len(images),
        "frame_count": sum(len(image.cells) for image in images),
        "total_pixels": sum(image.width * image.height for image in images),
        "total_bytes": sum(
            image.path.stat().st_size for image in images if image.path.is_file()
        ),
        "images": [image.metadata() for image in images],
    }


def _grid(media: FFmpegMedia, frames: list[Frame], offset: int) -> OutputImage:
    count = len(frames)
    columns = count if count <= 3 else 2 if count == 4 else 3
    rows = math.ceil(count / columns)
    labels = [
        f"#{offset + index + 1:03d} | {frame.timestamp_sec:.3f} s"
        for index, frame in enumerate(frames)
    ]
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow 10.0 的默认字体没有 size 参数。
        font = ImageFont.load_default()
    cell_width = max(
        max(frame.width for frame in frames),
        max(math.ceil(font.getlength(label)) + 16 for label in labels),
    )
    cell_height = max(frame.height for frame in frames) + LABEL_HEIGHT
    width = columns * cell_width + (columns + 1) * GUTTER
    height = rows * cell_height + (rows + 1) * GUTTER
    if width * height > MAX_GRID_PIXELS:
        raise VideoError("九宫格画布超过 3200 万像素，请减小单帧长边或改用独立帧")
    cells = []
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        cells.append(
            {
                **frame.metadata(),
                "frame_number": offset + index + 1,
                "row": row + 1,
                "column": column + 1,
                "x": GUTTER + column * (cell_width + GUTTER),
                "y": GUTTER + row * (cell_height + GUTTER) + LABEL_HEIGHT,
                "width": frame.width,
                "height": frame.height,
            }
        )
    signature = json.dumps(["grid-v1", width, height, cells], sort_keys=True)
    token = hashlib.sha256(signature.encode()).hexdigest()
    target = media.cache_dir / f"grid_{token}.png"
    result = OutputImage(target, "image/png", width, height, cells)
    if target.is_file():
        target.touch()
        return result
    temporary = target.with_name(f"{target.stem}_{uuid.uuid4().hex}.tmp.png")
    try:
        with Image.new("RGB", (width, height), "#18202d") as canvas:
            draw = ImageDraw.Draw(canvas)
            for frame, cell, label in zip(frames, cells, labels):
                with Image.open(frame.path) as source:
                    if source.size != (cell["width"], cell["height"]):
                        raise VideoError("抽帧图片尺寸已变化，请重新运行概览")
                    canvas.paste(source, (cell["x"], cell["y"]))
                draw.text(
                    (cell["x"] + 6, cell["y"] - LABEL_HEIGHT + 6),
                    label,
                    font=font,
                    fill="white",
                )
            canvas.save(temporary, format="PNG")
        media._check_cache_capacity(temporary, target)
        temporary.replace(target)
        return result
    finally:
        temporary.unlink(missing_ok=True)


async def present_frames(
    media: FFmpegMedia, frames: list[Frame], *, mode: str | None = None
) -> list[OutputImage]:
    mode = mode or media.settings.overview_output_mode
    if mode == "frames":
        return [
            OutputImage(
                frame.path,
                "image/jpeg",
                frame.width,
                frame.height,
                [{**frame.metadata(), "frame_number": index + 1}],
            )
            for index, frame in enumerate(frames)
        ]
    if mode != "grid":
        raise ValueError("无效的图片输出模式")
    images = []
    for offset in range(0, len(frames), 9):
        async with media.slots:
            worker = asyncio.create_task(
                asyncio.to_thread(_grid, media, frames[offset : offset + 9], offset)
            )
            try:
                images.append(await asyncio.shield(worker))
            except asyncio.CancelledError:
                # 等待当前原子写入结束，避免重跑或卸载删除目录后线程仍写缓存。
                await asyncio.gather(worker, return_exceptions=True)
                raise
    return images
