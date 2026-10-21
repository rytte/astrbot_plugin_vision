from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from .config import Settings
from .sampling import Candidate

FORMATS = "mov,matroska,webm,avi,mpegts,mpeg,flv,asf,ogg"
TIMESTAMP = re.compile(r"pts_time:\s*([-+\d.eE]+)")


class VideoError(Exception):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    key: str
    duration: float
    width: int
    height: int
    fps: float
    origin: float

    @property
    def last_timestamp(self) -> float:
        return max(0.0, self.duration - 1.0 / self.fps)


@dataclass
class Frame:
    frame_id: str
    timestamp_sec: float
    requested_sec: float
    reason: str
    width: int
    height: int
    path: Path
    fingerprint: tuple[int, ...]
    quality: float

    def metadata(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp_sec": self.timestamp_sec,
            "requested_sec": self.requested_sec,
            "reason": self.reason,
        }


class FFmpegMedia:
    def __init__(self, settings: Settings, cache_dir: Path, allowed_roots: list[Path]):
        self.settings = settings
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_roots = tuple(root.resolve(strict=True) for root in allowed_roots)
        if any(not root.is_dir() for root in self.allowed_roots):
            raise ValueError("允许读取的视频根目录必须是已存在的目录")
        self.ffmpeg = settings.ffmpeg_path or shutil.which("ffmpeg") or ""
        self.ffprobe = settings.ffprobe_path or shutil.which("ffprobe") or ""
        self.processes: set[asyncio.subprocess.Process] = set()
        self.slots = asyncio.Semaphore(settings.max_concurrency)
        self.cache_limit_bytes: int | None = None

    def resolve_path(self, video_path: str) -> Path:
        if not isinstance(video_path, str) or not video_path.strip():
            raise VideoError("video_path 必须是本地视频文件路径")
        if video_path.startswith(("http:", "https:", "file:", "\\\\")):
            raise VideoError("只接受本地路径，不接受网址、文件 URI 或网络共享路径")
        try:
            path = Path(video_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise VideoError("视频文件不存在或路径无效") from exc
        if not any(path.is_relative_to(root) for root in self.allowed_roots):
            raise VideoError("视频不在允许读取的目录中，请配置 allowed_video_roots")
        if not path.is_file() or path.stat().st_size == 0:
            raise VideoError("路径必须指向非空的普通视频文件")
        return path

    async def check_dependencies(self) -> None:
        if not self.ffmpeg or not self.ffprobe:
            raise VideoError(
                "未找到 FFmpeg/ffprobe，请安装并加入 PATH，或配置可执行文件路径"
            )
        await self._run(self.ffmpeg, "-version", timeout=10)
        await self._run(self.ffprobe, "-version", timeout=10)

    async def _run(
        self,
        executable: str,
        *arguments: str,
        timeout: float = 60,
        on_stderr: Callable[[str], None] | None = None,
    ) -> tuple[bytes, str]:
        if not executable:
            raise VideoError("未配置 FFmpeg/ffprobe 可执行文件")
        async with self.slots:
            options = (
                {"creationflags": subprocess.CREATE_NO_WINDOW}
                if os.name == "nt"
                else {}
            )
            try:
                process = await asyncio.create_subprocess_exec(
                    executable,
                    *arguments,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **options,
                )
            except OSError as exc:
                raise VideoError(
                    "无法启动 FFmpeg/ffprobe，请检查可执行文件配置"
                ) from exc
            self.processes.add(process)
            stdout = bytearray()
            stderr: deque[str] = deque(maxlen=128)

            async def read_stdout() -> None:
                while chunk := await process.stdout.read(65536):
                    stdout.extend(chunk)
                    if len(stdout) > 2 * 1024 * 1024:
                        raise VideoError("视频探测输出过大，已中止处理")

            async def read_stderr() -> None:
                while line := await process.stderr.readline():
                    text = line.decode("utf-8", errors="replace")
                    stderr.append(text)
                    if on_stderr is not None:
                        on_stderr(text)

            try:
                await asyncio.wait_for(
                    asyncio.gather(read_stdout(), read_stderr(), process.wait()),
                    timeout,
                )
                if process.returncode:
                    raise VideoError("视频处理失败：" + "".join(stderr)[-600:])
                return bytes(stdout), "".join(stderr)
            except asyncio.TimeoutError as exc:
                raise VideoError("FFmpeg/ffprobe 处理超时，已终止子进程") from exc
            finally:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
                self.processes.discard(process)

    async def probe(self, video_path: str) -> VideoInfo:
        path = self.resolve_path(video_path)
        stat = path.stat()
        key = hashlib.sha256(
            f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        ).hexdigest()[:24]
        stdout, _ = await self._run(
            self.ffprobe,
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            FORMATS,
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration,start_time:stream=width,height,duration,start_time,avg_frame_rate",
            "-of",
            "json",
            str(path),
            timeout=30,
        )
        try:
            payload = json.loads(stdout)
            stream = payload["streams"][0]
            container = payload.get("format", {})
            origin = float(container.get("start_time", stream.get("start_time", 0)))
            duration = float(stream.get("duration", container.get("duration", 0)))
            duration += max(0.0, float(stream.get("start_time", origin)) - origin)
            numerator, denominator = stream.get("avg_frame_rate", "25/1").split("/")
            fps = float(numerator) / float(denominator) if float(denominator) else 25.0
            if fps <= 0 or not math.isfinite(fps):
                fps = 25.0
            width, height = int(stream["width"]), int(stream["height"])
            if (
                not math.isfinite(duration)
                or duration <= 0
                or width <= 0
                or height <= 0
            ):
                raise ValueError("invalid video metadata")
            if not math.isfinite(origin):
                raise ValueError("invalid start time")
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise VideoError("文件没有可读取的视频流，或缺少有效时长") from exc
        return VideoInfo(path, key, duration, width, height, fps, origin)

    async def scenes(
        self, info: VideoInfo, *, trace: dict | None = None
    ) -> list[Candidate]:
        bucket_size = max(
            self.settings.overview_interval,
            info.duration / (self.settings.overview_max_frames * 4),
        )
        # 窗口取决于间隔和预算，调参时不能复用不同窗口的筛选结果。
        cache = self.cache_dir / (
            f"{info.key}_scene_v2_{self.settings.scene_threshold}_{bucket_size}"
            f"{'_trace' if trace is not None else ''}.json"
        )
        if cache.is_file():
            try:
                values = json.loads(cache.read_text(encoding="utf-8"))
                cache.touch()
                if trace is not None:
                    trace.update(values["trace"])
                    trace["cache_hit"] = True
                return [Candidate(**value) for value in values["candidates"]]
            except (OSError, TypeError, ValueError, KeyError):
                pass
        winners: dict[int, Candidate] = {}
        curve: dict[int, dict] = {}
        hits: list[dict] = []
        scanned = hit_count = 0
        current_time = None

        def collect(line: str) -> None:
            nonlocal current_time, scanned, hit_count
            if match := TIMESTAMP.search(line):
                current_time = float(match.group(1)) - info.origin
            if "lavfi.scene_score=" not in line or current_time is None:
                return
            score = float(line.split("lavfi.scene_score=", 1)[1].strip())
            if not 0 <= current_time <= info.duration:
                return
            scanned += 1
            if trace is not None:
                point = {"timestamp": round(current_time, 6), "score": score}
                curve_bucket = min(1999, int(current_time / info.duration * 2000))
                if curve_bucket not in curve or score > curve[curve_bucket]["score"]:
                    curve[curve_bucket] = point
            if score <= self.settings.scene_threshold:
                return
            hit_count += 1
            if trace is not None and len(hits) < 4000:
                hits.append(point)
            bucket = int(current_time / bucket_size)
            if bucket not in winners or score > winners[bucket].score:
                winners[bucket] = Candidate(
                    round(current_time, 6), "scene_change", score
                )

        await self._run(
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-threads",
            "1",
            "-copyts",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            FORMATS,
            "-i",
            str(info.path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            "scale='min(iw,160)':'min(ih,160)':force_original_aspect_ratio=decrease,"
            + (
                "select='gte(scene,0)',metadata=print"
                if trace is not None
                else f"select='gt(scene,{self.settings.scene_threshold})',metadata=print"
            ),
            "-fps_mode",
            "vfr",
            "-f",
            "null",
            "-",
            on_stderr=collect,
        )
        result = sorted(winners.values(), key=lambda item: item.timestamp)
        if trace is not None:
            selected = {candidate.timestamp for candidate in result}
            trace.update(
                {
                    "cache_hit": False,
                    "bucket_sec": bucket_size,
                    "scanned_frames": scanned,
                    "threshold_hits": hit_count,
                    "curve": sorted(curve.values(), key=lambda item: item["timestamp"]),
                    "curve_aggregated": scanned > len(curve),
                    "hits": [
                        {**hit, "selected": hit["timestamp"] in selected}
                        for hit in hits
                    ],
                    "omitted_hits": max(0, hit_count - len(hits)),
                }
            )
        self._write_json(
            cache,
            {"candidates": [asdict(candidate) for candidate in result], "trace": trace},
        )
        return result

    def _write_json(self, target: Path, payload: object) -> None:
        temporary = target.with_name(f"{target.stem}_{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            self._check_cache_capacity(temporary, target)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def _check_cache_capacity(self, temporary: Path, target: Path) -> None:
        if self.cache_limit_bytes is None:
            return
        total = temporary.stat().st_size
        for path in self.cache_dir.iterdir():
            if path not in (temporary, target) and path.is_file():
                total += path.stat().st_size
        if total > self.cache_limit_bytes:
            raise VideoError(
                "本次调试图片缓存已达到容量上限，请减小帧数或输出尺寸后重跑"
            )

    async def extract(
        self, info: VideoInfo, timestamp: float, max_edge: int, reason: str
    ) -> Frame:
        token = f"{info.key}_{timestamp:.6f}_{max_edge}"
        image_path = self.cache_dir / f"{token}.jpg"
        metadata_path = self.cache_dir / f"{token}.json"
        if image_path.is_file() and metadata_path.is_file():
            try:
                values = json.loads(metadata_path.read_text(encoding="utf-8"))
                values["path"] = image_path
                values["reason"] = reason
                values["fingerprint"] = tuple(values["fingerprint"])
                frame = Frame(**values)
                image_path.touch()
                metadata_path.touch()
                return frame
            except (OSError, TypeError, ValueError, KeyError):
                pass
        temporary = self.cache_dir / f"{token}_{uuid.uuid4().hex}.tmp.jpg"
        try:
            for retreat in (0.0, max(1 / info.fps, 0.1), 1.0):
                target = max(0.0, min(timestamp, info.last_timestamp) - retreat)
                _, logs = await self._run(
                    self.ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "info",
                    "-y",
                    "-threads",
                    "1",
                    "-copyts",
                    "-ss",
                    f"{target:.6f}",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-format_whitelist",
                    FORMATS,
                    "-i",
                    str(info.path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-sn",
                    "-dn",
                    "-vf",
                    f"scale='min(iw,{max_edge})':'min(ih,{max_edge})':"
                    "force_original_aspect_ratio=decrease,showinfo",
                    "-frames:v",
                    "1",
                    "-fps_mode",
                    "vfr",
                    "-threads",
                    "1",
                    "-q:v",
                    "3",
                    "-update",
                    "1",
                    str(temporary),
                    timeout=30,
                )
                if temporary.is_file() and temporary.stat().st_size:
                    break
            else:
                raise VideoError(f"无法从 {timestamp:.3f} 秒附近提取画面")
            match = TIMESTAMP.search(logs)
            if not match:
                raise VideoError("未能读取抽取帧的真实时间戳")
            actual_time = round(float(match.group(1)) - info.origin, 6)
            with Image.open(temporary) as image:
                width, height = image.size
                gray = image.convert("L")
                fingerprint = tuple(gray.resize((32, 18)).tobytes())
                statistics = ImageStat.Stat(gray)
                quality = (
                    statistics.stddev[0]
                    + ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0]
                )
            frame_id = hashlib.sha256(
                f"{info.key}:{actual_time}:{width}:{height}".encode()
            ).hexdigest()[:20]
            frame = Frame(
                frame_id,
                actual_time,
                timestamp,
                reason,
                width,
                height,
                image_path,
                fingerprint,
                quality,
            )
            self._check_cache_capacity(temporary, image_path)
            temporary.replace(image_path)
            metadata = asdict(frame)
            metadata.pop("path")
            self._write_json(metadata_path, metadata)
            return frame
        finally:
            temporary.unlink(missing_ok=True)

    def cleanup_cache(self) -> None:
        files = []
        now = time.time()
        for path in self.cache_dir.iterdir():
            if path.suffix not in (".jpg", ".json", ".tmp") or path.is_symlink():
                continue
            try:
                stat = path.stat()
                if path.is_file() and path.resolve().parent == self.cache_dir:
                    if now - stat.st_mtime > self.settings.cache_ttl:
                        path.unlink(missing_ok=True)
                    else:
                        files.append((stat.st_mtime, stat.st_size, path))
            except FileNotFoundError:
                continue
        total = sum(size for _, size, _ in files)
        for _, size, path in sorted(files):
            if total <= self.settings.cache_max_mb * 1024 * 1024:
                break
            path.unlink(missing_ok=True)
            total -= size

    async def close(self) -> None:
        processes = list(self.processes)
        for process in processes:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        await asyncio.gather(*(process.wait() for process in processes))
