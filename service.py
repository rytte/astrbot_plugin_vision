from __future__ import annotations

import math

from .media import FFmpegMedia, Frame, VideoInfo
from .pipeline import OverviewPipeline, VideoResult, extract_many, video_metadata
from .sampling import (
    Candidate,
    InspectArgumentError,
    inspect_timestamps,
)


class VideoService:
    def __init__(self, media: FFmpegMedia):
        self.media = media
        self.settings = media.settings

    async def _extract_many(
        self, info: VideoInfo, candidates: list[Candidate], max_edge: int
    ) -> list[Frame]:
        return await extract_many(self.media, info, candidates, max_edge)

    async def overview(self, video_path: str) -> VideoResult:
        return await OverviewPipeline(self.media, video_path).run()

    async def inspect(
        self,
        video_path: str,
        mode: str,
        times_sec: list[float],
        max_frames: int | None = None,
        max_edge: int | None = None,
    ) -> VideoResult:
        for name, value in (("max_frames", max_frames), ("max_edge", max_edge)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or int(value) != value
                or value <= 0
            ):
                raise InspectArgumentError(f"{name} 必须是正整数或 null（使用配置值）")
        frame_limit = min(
            max_frames or self.settings.inspect_max_frames,
            self.settings.inspect_max_frames,
        )
        edge_limit = min(
            max_edge or self.settings.inspect_max_edge, self.settings.max_edge_limit
        )
        if edge_limit < 64:
            raise InspectArgumentError("max_edge 不能小于 64")
        info = await self.media.probe(video_path)
        timestamps, request = inspect_timestamps(
            info.duration,
            info.last_timestamp,
            mode,
            times_sec,
            int(frame_limit),
        )
        frames = await self._extract_many(
            info,
            [Candidate(timestamp, "inspect") for timestamp in timestamps],
            int(edge_limit),
        )
        unique = {frame.frame_id: frame for frame in frames}
        data = self._metadata(info)
        data.update(
            {
                "request": request,
                "requested_max_frames": max_frames,
                "effective_max_frames": int(frame_limit),
                "requested_max_edge": max_edge,
                "effective_max_edge": int(edge_limit),
                "returned_count": len(unique),
                "guidance": "图片按实际时间顺序返回；请求时间不一定等于实际帧时间。不要从离散截图推断未经证实的连续动作。",
            }
        )
        return VideoResult(
            data, sorted(unique.values(), key=lambda frame: frame.timestamp_sec)
        )

    def _metadata(self, info: VideoInfo) -> dict:
        return video_metadata(info)
