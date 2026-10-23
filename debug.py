"""有界、按管理员隔离的临时流水线调试会话。"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from .config import Settings
from .debug_models import ModelState, ModelTests
from .media import FFmpegMedia, Frame, VideoError
from .pipeline import STAGES, OverviewPipeline, video_metadata
from .presentation import OutputImage

MAX_UPLOAD = 256 * 1024 * 1024
MAX_SESSIONS = 4
PARAMETERS = (
    "overview_interval",
    "overview_max_frames",
    "overview_max_edge",
    "scene_threshold",
    "overview_output_mode",
)
VIDEO_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".flv",
    ".m4v",
    ".ts",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".ogv",
    ".3gp",
}


@dataclass
class DebugSession:
    id: str
    owner: str
    filename: str
    directory: Path
    source: Path
    pipeline: OverviewPipeline
    size: int = 0
    revision: int = 0
    touched: float = field(default_factory=time.monotonic)
    elapsed: float = 0
    preview_elapsed: float = 0
    error: str | None = None
    task: asyncio.Task | None = None
    operation: str | None = None
    model: ModelState = field(default_factory=ModelState)
    page_state: dict = field(default_factory=dict)
    page_writers: dict[str, int] = field(default_factory=dict)
    assets: set[asyncio.Task] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class DebugSessions:
    def __init__(self, media: FFmpegMedia, directory: Path, slots: asyncio.Semaphore):
        self.media = media
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.slots = slots
        self.sessions: dict[str, DebugSession] = {}
        self.active_sessions: dict[str, str] = {}
        self.ttl = min(media.settings.cache_ttl, 3600)
        self.closed = False
        self.janitor: asyncio.Task | None = None
        self.models = ModelTests(self)

    def start(self) -> None:
        # 重载留下的临时目录已无可恢复的内存状态。
        for path in self.directory.iterdir():
            if len(path.name) == 32 and all(c in "0123456789abcdef" for c in path.name):
                self._remove(path)
        self.janitor = asyncio.create_task(self._cleanup_loop())

    def _remove(self, path: Path) -> None:
        if path.is_symlink() or path.resolve().parent != self.directory:
            return
        if path.is_dir():
            shutil.rmtree(path)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self.prune()

    async def prune(self) -> None:
        for session in list(self.sessions.values()):
            if (
                time.monotonic() - session.touched > self.ttl
                and session.task is None
                and not session.assets
                and not session.lock.locked()
            ):
                await self.delete(session)

    def settings(self, values: dict) -> Settings:
        if not isinstance(values, dict) or set(values) - set(PARAMETERS):
            raise ValueError(
                "仅支持本次概览间隔、帧数预算、输出尺寸、场景阈值和图片输出模式"
            )
        return Settings.from_mapping({**asdict(self.media.settings), **values})

    def _active(self, session: DebugSession) -> None:
        if self.closed or self.sessions.get(session.id) is not session:
            raise ValueError("调试会话已删除或正在关闭")

    def _pipeline(self, source: Path, directory: Path, settings: Settings):
        media = FFmpegMedia(settings, directory / "frames", [directory])
        media.slots = self.media.slots
        media.cache_limit_bytes = min(settings.cache_max_mb, 512) * 1024 * 1024
        return OverviewPipeline(media, str(source), diagnostic=True)

    async def upload(self, owner: str, upload) -> DebugSession:
        if self.closed:
            raise ValueError("调试台正在关闭")
        await self.prune()
        if len(self.sessions) >= MAX_SESSIONS:
            raise ValueError("最多保留 4 个调试会话，请先删除旧测试")
        filename = (upload.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise ValueError("请上传 MP4、MKV、MOV、WebM 等支持的视频文件")
        identifier = uuid.uuid4().hex
        directory = self.directory / identifier
        directory.mkdir()
        source = directory / ("source" + suffix)
        session = DebugSession(
            identifier,
            owner,
            filename[:160],
            directory,
            source,
            self._pipeline(source, directory, self.media.settings),
        )
        # 在第一次 await 前预留名额，避免并发上传突破会话数限制。
        self.sessions[identifier] = session
        try:
            async with session.lock:
                with source.open("wb") as output:
                    while chunk := await upload.read(1024 * 1024):
                        session.size += len(chunk)
                        if session.size > MAX_UPLOAD:
                            raise ValueError("视频不能超过 256 MiB")
                        output.write(chunk)
                if not session.size:
                    raise ValueError("上传的视频为空")
                session.touched = time.monotonic()
        except BaseException:
            self.sessions.pop(identifier, None)
            self._remove(directory)
            raise
        self.active_sessions[owner] = identifier
        return session

    def active_session(self, owner: str) -> str | None:
        identifier = self.active_sessions.get(owner)
        session = self.sessions.get(identifier)
        if session is not None and session.owner == owner:
            return identifier
        return next(
            (
                item.id
                for item in reversed(list(self.sessions.values()))
                if item.owner == owner
            ),
            None,
        )

    def save_page_state(
        self, session: DebugSession, revision, writer, sequence, state
    ) -> None:
        self._active(session)
        if type(revision) is not int or revision != session.revision:
            raise ValueError("测试版本已变化，请重新读取状态")
        if (
            not isinstance(writer, str)
            or len(writer) != 32
            or any(c not in "0123456789abcdef" for c in writer)
            or type(sequence) is not int
            or not 0 < sequence < 2**53
            or not isinstance(state, dict)
            or set(state)
            - {
                "view",
                "parameters",
                "model_draft",
                "pair_index",
                "pages",
                "selected_runs",
                "video_time",
            }
        ):
            raise ValueError("无效的页面状态")
        try:
            encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("无效的页面状态") from exc
        if len(encoded.encode("utf-8")) > 24 * 1024:
            raise ValueError("页面状态过大")
        if sequence <= session.page_writers.get(writer, 0):
            return
        if writer not in session.page_writers and len(session.page_writers) >= 16:
            del session.page_writers[next(iter(session.page_writers))]
        session.page_writers[writer] = sequence
        session.page_state = json.loads(encoded)
        self.active_sessions[session.owner] = session.id

    def get(self, owner: str, identifier: str) -> DebugSession:
        if self.closed:
            raise ValueError("调试台正在关闭")
        session = self.sessions.get(identifier) if isinstance(identifier, str) else None
        if session is None or session.owner != owner:
            raise ValueError("调试会话不存在或已过期，请重新上传")
        if (
            time.monotonic() - session.touched > self.ttl
            and session.task is None
            and not session.assets
            and not session.lock.locked()
        ):
            raise ValueError("调试会话已过期，请重新上传")
        session.touched = time.monotonic()
        return session

    async def next(self, session: DebugSession, expected: int, revision: int) -> None:
        async with session.lock:
            self._active(session)
            if (
                type(expected) is not int
                or type(revision) is not int
                or expected != session.pipeline.completed
                or revision != session.revision
            ):
                raise ValueError("阶段已变化，请刷新状态后再继续")
            if session.task is not None or session.pipeline.done:
                raise ValueError("当前阶段正在执行或流水线已完成")
            session.error = None
            session.operation = "pipeline"
            session.task = asyncio.create_task(self._step(session))

    async def _step(self, session: DebugSession) -> None:
        started = None
        try:
            async with self.slots:
                remaining = session.pipeline.settings.tool_timeout - session.elapsed
                if remaining <= 0:
                    raise VideoError("本次处理时间预算已耗尽，请调整参数后重跑")
                started = time.monotonic()
                await asyncio.wait_for(session.pipeline.advance(), remaining)
        except asyncio.CancelledError:
            session.error = "已取消当前阶段，可以重试或调参重跑"
            raise
        except asyncio.TimeoutError:
            session.error = "本次处理超时，请调整参数后重跑"
        except Exception as exc:
            session.error = (
                str(exc)
                if isinstance(exc, VideoError | ValueError)
                else "阶段执行失败，请查看 AstrBot 日志"
            )
            if not isinstance(exc, VideoError | ValueError):
                from astrbot.api import logger

                logger.exception("视频流水线调试阶段执行失败")
        finally:
            if started is not None:
                session.elapsed += time.monotonic() - started
            session.task = None
            session.operation = None
            session.touched = time.monotonic()

    async def _stop(self, session: DebugSession) -> None:
        tasks = [*session.assets]
        if session.task is not None:
            tasks.append(session.task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # create_task 后立刻取消时，协程体（包括 finally）可能完全没有运行。
        session.task = None
        session.operation = None
        self.models.stopped(session)
        session.assets.difference_update(tasks)
        await session.pipeline.media.close()

    async def cancel(self, session: DebugSession) -> None:
        async with session.lock:
            self._active(session)
            running = session.task is not None and session.operation not in {
                "model",
                "asr",
            }
            await self._stop(session)
            if running:
                session.error = "已取消当前阶段，可以重试或调参重跑"

    async def reset(self, session: DebugSession, values: dict) -> None:
        settings = self.settings(values)
        async with session.lock:
            self._active(session)
            await self._stop(session)
            cache = session.pipeline.media.cache_dir
            if (
                cache.is_symlink()
                or cache.resolve().parent != session.directory.resolve()
            ):
                raise ValueError("调试缓存目录无效")
            shutil.rmtree(cache)
            session.pipeline = self._pipeline(
                session.source, session.directory, settings
            )
            session.revision += 1
            session.elapsed = session.preview_elapsed = 0
            session.error = None
            session.model = ModelState()
            session.page_state.clear()
            session.page_writers.clear()

    async def delete(self, session: DebugSession) -> None:
        async with session.lock:
            await self._stop(session)
            self.sessions.pop(session.id, None)
            if self.active_sessions.get(session.owner) == session.id:
                self.active_sessions.pop(session.owner, None)
            self._remove(session.directory)

    @staticmethod
    def _frame(frame: Frame) -> dict:
        return {
            **frame.metadata(include_frame_id=True),
            "requested_sec": frame.requested_sec,
            "width": frame.width,
            "height": frame.height,
            "bytes": frame.path.stat().st_size if frame.path.is_file() else 0,
            "quality": frame.quality,
        }

    def snapshot(self, session: DebugSession) -> dict:
        pipeline = session.pipeline
        info = video_metadata(pipeline.info) if pipeline.info else None
        if info:
            info.pop("video_path")
        result = dict(pipeline.result.data) if pipeline.result else None
        if result:
            result.pop("video_path")
            result["total_image_bytes"] = sum(
                frame.path.stat().st_size for frame in pipeline.kept
            )
            result["output_image_bytes"] = pipeline.result.data["presentation"][
                "total_bytes"
            ]
        return {
            "id": session.id,
            "filename": session.filename,
            "size": session.size,
            "revision": session.revision,
            "page_state": session.page_state,
            "completed": pipeline.completed,
            "running": session.task is not None,
            "operation": session.operation,
            "model": self.models.summary(session),
            "done": pipeline.done,
            "error": session.error,
            "info": info,
            "parameters": {key: getattr(pipeline.settings, key) for key in PARAMETERS},
            "effective_interval_sec": pipeline.effective_interval,
            "stages": [
                {"id": key, "title": title, "seconds": pipeline.timings.get(key)}
                for key, title in STAGES
            ]
            + [{"id": "model", "title": "模型理解", "seconds": None}],
            "scene_trace": pipeline.scene_trace,
            "fixed": [asdict(item) for item in pipeline.fixed],
            "scenes": [asdict(item) for item in pipeline.scenes],
            "candidates": [asdict(item) for item in pipeline.candidates],
            "merge_trace": pipeline.merge_trace,
            "frames": [self._frame(frame) for frame in pipeline.frames],
            "dedup_trace": pipeline.dedup_trace,
            "kept_indices": [
                next(i for i, item in enumerate(pipeline.frames) if item is frame)
                for frame in pipeline.kept
            ],
            "warnings": pipeline.warnings,
            "result": result,
            "processing_seconds": session.elapsed,
            "preview_seconds": session.preview_elapsed,
        }

    async def image(
        self,
        session: DebugSession,
        kind: str,
        index: int,
        variant: str,
        revision: int,
        record: str | None = None,
    ) -> dict:
        async with session.lock:
            self._active(session)
            if revision != session.revision:
                raise ValueError("图片属于上一次测试，请刷新页面")
            task = asyncio.create_task(
                self._image(session, kind, index, variant, record)
            )
            session.assets.add(task)
        try:
            return await asyncio.wait_for(task, session.pipeline.settings.tool_timeout)
        finally:
            session.assets.discard(task)
            session.touched = time.monotonic()

    async def _image(
        self,
        session: DebugSession,
        kind: str,
        index: int,
        variant: str,
        record: str | None = None,
    ) -> dict:
        pipeline = session.pipeline
        if kind in {"grid", "model"}:
            started = time.monotonic()
            try:
                if kind == "model":
                    return await self.models.image(session, record, index, variant)
                if (
                    not pipeline.result
                    or pipeline.settings.overview_output_mode != "grid"
                    or not 0 <= index < len(pipeline.result.images)
                ):
                    raise ValueError("九宫格尚未生成或索引无效")
                return self._output_image(pipeline.result.images[index], variant)
            finally:
                session.preview_elapsed += time.monotonic() - started
        groups = {
            "fixed": pipeline.fixed,
            "scene": pipeline.scenes,
            "merged": pipeline.candidates,
            "frame": pipeline.frames,
        }
        if kind not in groups or index < 0 or index >= len(groups[kind]):
            raise ValueError("该阶段图片尚未生成或索引无效")
        if variant not in {"thumb", "full", "source", "fingerprint", "detector"}:
            raise ValueError("无效的图片类型")
        if variant in {"source", "fingerprint"} and kind != "frame":
            raise ValueError("请在正式抽帧后查看原图或指纹")
        started = time.monotonic()
        try:
            item = groups[kind][index]
            if kind == "frame":
                frame = item
                if variant == "source":
                    if pipeline.info.width * pipeline.info.height > 33_554_432:
                        raise ValueError("原分辨率超过 3200 万像素，请直接查看源视频")
                    frame = await pipeline.media.extract(
                        pipeline.info,
                        item.timestamp_sec,
                        max(pipeline.info.width, pipeline.info.height),
                        "debug_source",
                    )
            else:
                frame = await pipeline.media.extract(
                    pipeline.info,
                    item.timestamp,
                    160 if variant == "detector" else 240,
                    "debug_preview",
                )
            buffer = io.BytesIO()
            if variant == "fingerprint":
                image = Image.frombytes("L", (32, 18), bytes(frame.fingerprint))
                image.save(buffer, format="PNG")
                mime = "image/png"
            elif variant in {"thumb", "detector"}:
                with Image.open(frame.path) as image:
                    image.thumbnail((160, 160) if variant == "detector" else (240, 180))
                    image.convert("RGB").save(buffer, format="JPEG", quality=80)
                mime = "image/jpeg"
            else:
                buffer.write(frame.path.read_bytes())
                mime = "image/jpeg"
            return {
                "data_url": f"data:{mime};base64,"
                + base64.b64encode(buffer.getvalue()).decode(),
                "frame": self._frame(frame),
            }
        finally:
            session.preview_elapsed += time.monotonic() - started

    @staticmethod
    def _output_image(image: OutputImage, variant: str) -> dict:
        if variant not in {"thumb", "full"}:
            raise ValueError("无效的图片类型")
        buffer = io.BytesIO()
        if variant == "thumb":
            with Image.open(image.path) as source:
                source.thumbnail((480, 360))
                source.save(
                    buffer, format="PNG" if image.mime_type == "image/png" else "JPEG"
                )
        else:
            buffer.write(image.path.read_bytes())
        return {
            "data_url": f"data:{image.mime_type};base64,"
            + base64.b64encode(buffer.getvalue()).decode(),
            "image": {
                **image.metadata(include_frame_id=True),
                "bytes": image.path.stat().st_size,
            },
        }

    async def close(self) -> None:
        self.closed = True
        if self.janitor is not None:
            self.janitor.cancel()
            await asyncio.gather(self.janitor, return_exceptions=True)
        for session in list(self.sessions.values()):
            await self.delete(session)
