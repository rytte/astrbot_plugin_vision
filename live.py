"""实时会话：有界缓冲、变化候选合并与独立后台观察，不持有旧消息事件。"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import io
import json
import math
import shutil
import time
import uuid
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from astrbot.api import logger
from PIL import Image, UnidentifiedImageError

from .config import LIVE_MODE_FIELDS, LIVE_PRESET_FIELDS, LivePreset, Settings
from .media import VideoError

MAX_FRAME_BYTES = 1024 * 1024
MAX_BUFFER_BYTES = 24 * 1024 * 1024
MAX_AUDIO_BYTES = 32000 * 10
MAX_FRAMES = 1500
FRESH_SECONDS = 10
MODEL_TIMEOUT_SECONDS = 45
REPLY_TIMEOUT_SECONDS = 45
COMMENT_MAX_AGE_SECONDS = MODEL_TIMEOUT_SECONDS + REPLY_TIMEOUT_SECONDS

MODE_NAMES = {
    "提问": "demand",
    "定时": "interval",
    "变化": "change",
}
MODE_LABELS = {"demand": "提问", "interval": "定时", "change": "变化"}
LIVE_USAGE = (
    "用法：/live <来源> <响应模式> <观察指令>\n"
    "来源：屏幕 / 摄像头 / 媒体URL\n"
    "响应模式：提问 / 定时 / 变化\n"
    "定时和变化模式会自动观察，需要回应时才发言。\n"
    "例如：/live 屏幕 变化 一起看视频，有意思时简短吐槽\n"
    "结束：/live end"
)


def parse_live_command(arguments: str, settings: Settings) -> tuple[dict, str]:
    parts = arguments.strip().split(maxsplit=2)
    if len(parts) != 3 or not parts[2].strip():
        raise ValueError(LIVE_USAGE)
    source, mode, focus = parts
    source = source.strip("\"'")
    mode = MODE_NAMES.get(mode.strip("\"'").lower())
    if mode is None:
        raise ValueError(LIVE_USAGE)
    url = ""
    if source in {"屏幕", "screen"}:
        source = "screen"
    elif source in {"摄像头", "camera"}:
        source = "camera"
    else:
        try:
            parsed = urlsplit(source)
            valid = parsed.scheme in {
                "http",
                "https",
                "rtsp",
                "rtmp",
                "rtmps",
            } and bool(parsed.hostname)
        except ValueError:
            valid = False
        if not valid or any(ord(char) < 32 for char in source):
            raise ValueError("来源须为屏幕、摄像头或 HTTP(S)/RTSP/RTMP 媒体直链")
        url, source = source, "stream"
    preset = settings.live_preset(source, mode)
    return {
        "source": source,
        "mode": mode,
        "focus": focus.strip(),
        "interval": preset.observation_interval,
        "sensitivity": preset.sensitivity,
        "proactive": preset.proactive,
        "audio": settings.asr_enabled and bool(settings.asr_provider_id),
    }, url


def number(value, name: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} 必须在 {minimum}～{maximum} 之间")
    return float(value)


@dataclass(frozen=True)
class LiveOptions:
    source: str = "screen"
    mode: str = "change"
    proactive: bool = True
    audio: bool = False
    interval: float = 10.0
    sensitivity: str = "medium"
    sample_interval: float = 1.0
    max_edge: int = 1280
    max_frames: int = 4
    lookback_seconds: float = 12.0
    min_analysis_interval: float = 5.0
    focus: str = "一起观看，遇到有意思的内容简短吐槽；不必每次都发言。"

    @classmethod
    def parse(cls, values: dict, settings: Settings) -> LiveOptions:
        if not isinstance(values, dict):
            raise ValueError("设置必须是对象")
        if set(values) - set(cls.__dataclass_fields__):
            raise ValueError("包含未知实时设置")
        data = {
            key: values.get(key, item.default)
            for key, item in cls.__dataclass_fields__.items()
        }
        if data["source"] not in ("screen", "camera", "stream") or data["mode"] not in (
            "demand",
            "interval",
            "change",
        ):
            raise ValueError("来源或响应模式无效")
        preset = settings.live_preset(data["source"], data["mode"])
        preset_values = {
            name: values.get(name, getattr(preset, name))
            for name in LIVE_PRESET_FIELDS + LIVE_MODE_FIELDS[data["mode"]]
        }
        # interval 是会话内部字段名，配置界面使用 observation_interval。
        preset_values.pop("observation_interval", None)
        validated = LivePreset.parse(preset_values, preset, data["mode"], "实时预设")
        for name in LIVE_PRESET_FIELDS + ("min_analysis_interval",):
            data[name] = getattr(validated, name)
        for name, preset_name in (
            ("interval", "observation_interval"),
            ("proactive", "proactive"),
            ("sensitivity", "sensitivity"),
        ):
            if name not in values:
                data[name] = getattr(preset, preset_name)
        if type(data["audio"]) is not bool or type(data["proactive"]) is not bool:
            raise ValueError("音频和主动发言开关必须是布尔值")
        data["proactive"] = data["mode"] == "change" or (
            data["mode"] == "interval" and data["proactive"]
        )
        data["interval"] = number(data["interval"], "观察间隔", 0.1, 300)
        if data["sensitivity"] not in ("low", "medium", "high"):
            raise ValueError("变化灵敏度无效")
        if not isinstance(data["focus"], str) or len(data["focus"]) > 1000:
            raise ValueError("观察重点最多 1000 字")
        return cls(**data)

    @property
    def threshold(self) -> float:
        return {"low": 0.22, "medium": 0.12, "high": 0.05}[self.sensitivity]

    @property
    def fresh_seconds(self) -> float:
        return max(FRESH_SECONDS, self.sample_interval * 3)

    @property
    def analysis_interval(self) -> float:
        return self.interval if self.mode == "interval" else self.min_analysis_interval


@dataclass(frozen=True)
class LiveFrame:
    timestamp: float
    received: float
    jpeg: bytes
    fingerprint: tuple[int, ...]


def decode_frame(data: bytes, timestamp: float, max_edge: int) -> LiveFrame:
    if not data or len(data) > MAX_FRAME_BYTES:
        raise ValueError("单帧不得为空或超过 1 MiB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if (
                image.format not in {"JPEG", "PNG"}
                or image.width * image.height > 16777216
            ):
                raise ValueError("仅支持不超过 1600 万像素的 JPEG/PNG")
            image = image.convert("RGB")
            fingerprint = tuple(image.resize((32, 18)).tobytes())
            image.thumbnail((max_edge, max_edge))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=80)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取画面") from exc
    return LiveFrame(timestamp, time.monotonic(), output.getvalue(), fingerprint)


def frame_distance(left: LiveFrame, right: LiveFrame) -> float:
    return sum(abs(a - b) for a, b in zip(left.fingerprint, right.fingerprint)) / (
        255 * len(left.fingerprint)
    )


def select_frames(
    frames: list[LiveFrame],
    *,
    source: str,
    mode: str,
    max_frames: int,
    lookback: float,
    reference: LiveFrame | None = None,
) -> list[LiveFrame]:
    """最新画面必选；定时覆盖本轮，变化保留显著变化前后的证据。"""
    latest = frames[-1]
    if lookback == 0 or max_frames == 1:
        return [latest]
    cutoff = latest.timestamp - lookback
    if mode == "interval" and reference is not None:
        cutoff = max(cutoff, reference.timestamp)
    candidates = [frame for frame in frames if frame.timestamp >= cutoff]
    # 合并相邻近重复画面，保留该段最后一张；不删除动作前后重现的同一状态。
    tolerance = {"screen": 0.002, "camera": 0.01, "stream": 0.015}[source]
    distinct = []
    for frame in candidates:
        if distinct and frame_distance(distinct[-1], frame) <= tolerance:
            distinct[-1] = frame
        else:
            distinct.append(frame)
    if len(distinct) <= max_frames:
        return distinct
    selected = {len(distinct) - 1}
    if mode == "change":
        changes = sorted(
            range(1, len(distinct)),
            key=lambda i: (frame_distance(distinct[i - 1], distinct[i]), i),
            reverse=True,
        )
        for index in changes:
            # 两帧上限时优先保留变化前状态，与最新状态对比。
            for candidate in (index - 1, index):
                if len(selected) < max_frames:
                    selected.add(candidate)
            if len(selected) == max_frames:
                break
    else:
        span = distinct[-1].timestamp - distinct[0].timestamp
        for index in range(max_frames - 1):
            target = distinct[0].timestamp + span * index / (max_frames - 1)
            nearest = min(
                (i for i in range(len(distinct)) if i not in selected),
                key=lambda i: abs(distinct[i].timestamp - target),
            )
            selected.add(nearest)
    return [distinct[index] for index in sorted(selected)]


@dataclass
class LiveSession:
    id: str
    origin: str
    user: str
    created: float
    expires: float
    options: LiveOptions | None = None
    frames: deque[LiveFrame] = field(default_factory=deque)
    transcripts: deque[dict] = field(default_factory=lambda: deque(maxlen=80))
    audio_chunks: deque[tuple[float, float, bytes]] = field(
        default_factory=lambda: deque(maxlen=12)
    )
    reasons: set[str] = field(default_factory=set)
    reference: LiveFrame | None = None
    memory: str = ""
    observation: dict | None = None
    recent_speech: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    last_analysis: float = -math.inf
    last_speech: float = -math.inf
    last_user: float = -math.inf
    last_audio_end: float = -1.0
    last_audio_text: str = ""
    asr_completed_end: float | None = None
    observed_received: float = -math.inf
    analysis_count: int = 0
    generation: int = 0
    user_generation: int = 0
    paused: bool = False
    stopped: bool = False
    busy: bool = False
    error: str = ""
    audio_status: str = "off"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ingest_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    audio_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task | None = None
    stream_task: asyncio.Task | None = None
    analysis_task: asyncio.Task | None = None
    audio_task: asyncio.Task | None = None
    operations: set[asyncio.Task] = field(default_factory=set)
    user_busy: int = 0
    source_url: str = ""
    reply_route: object | None = None
    pending_reply: dict | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def stale(self) -> bool:
        return not self.frames or time.monotonic() - self.frames[-1].received > (
            self.options.fresh_seconds if self.options else FRESH_SECONDS
        )


class LiveSessions:
    def __init__(
        self,
        settings: Settings,
        root: Path,
        analyze: Callable[..., Awaitable[dict]],
        speak: Callable[..., Awaitable[None]],
        transcribe: Callable[..., Awaitable[str]],
        *,
        ended: Callable[[LiveSession, str], Awaitable[None]] | None = None,
    ):
        self.settings = settings
        self.root = root
        self.analyze = analyze
        self.speak = speak
        self.transcribe = transcribe
        self.ended = ended
        self.sessions: dict[str, LiveSession] = {}
        self.asr_lock = asyncio.Lock()

    async def create(self, origin: str, user: str) -> LiveSession:
        for item in list(self.sessions.values()):
            if item.origin == origin or item.expires <= time.monotonic():
                await self.stop(
                    item, reason="replaced" if item.origin == origin else "expired"
                )
        if len(self.sessions) >= 4:
            raise ValueError("实时会话已达到 4 个上限")
        now = time.monotonic()
        session = LiveSession(
            uuid.uuid4().hex,
            origin,
            user,
            now,
            now + self.settings.live_session_seconds,
        )
        self.sessions[session.id] = session
        session.task = contextvars.Context().run(
            asyncio.create_task, self._worker(session)
        )
        return session

    def for_chat(self, origin: str, user: str) -> LiveSession | None:
        return next(
            (
                s
                for s in self.sessions.values()
                if s.origin == origin
                and s.user == user
                and not s.stopped
                and s.expires > time.monotonic()
            ),
            None,
        )

    def configure(self, session: LiveSession, values: dict) -> None:
        if session.stopped or session.expires <= time.monotonic():
            raise ValueError("会话已结束")
        if session.options is not None:
            raise ValueError("此会话已开始，请先 /live end 再更换来源或模式")
        options = LiveOptions.parse(values, self.settings)
        if options.audio and (
            not self.settings.asr_enabled or not self.settings.asr_provider_id
        ):
            raise ValueError("请先启用并配置插件的 ASR 提供商")
        session.options = options
        session.audio_status = "waiting" if options.audio else "off"

    async def add_frame(
        self, session: LiveSession, data: bytes, timestamp: float
    ) -> None:
        async with session.ingest_lock:
            if session.stopped or session.paused or not session.options:
                raise ValueError("会话未运行")
            timestamp = number(
                timestamp, "画面时间", 0, self.settings.live_session_seconds + 60
            )
            if session.frames and timestamp <= session.frames[-1].timestamp:
                raise ValueError("画面时间必须递增")
            if (
                session.frames
                and timestamp - session.frames[-1].timestamp
                < session.options.sample_interval * 0.5
            ):
                return
            frame = await asyncio.to_thread(
                decode_frame, data, timestamp, session.options.max_edge
            )
            if session.stopped or session.paused:
                return
            previous = session.frames[-1] if session.frames else None
            session.frames.append(frame)
            session.ready.set()
            total = sum(len(item.jpeg) for item in session.frames)
            while len(session.frames) > 1 and (
                len(session.frames) > MAX_FRAMES
                or total > MAX_BUFFER_BYTES
                or frame.received - session.frames[0].received
                > self.settings.live_buffer_seconds
            ):
                total -= len(session.frames.popleft().jpeg)
            if session.options.mode != "change":
                return
            if previous is None:
                session.reasons.add("initial")
            elif frame_distance(previous, frame) >= session.options.threshold:
                session.reasons.add("visual_change")
            elif (
                session.reference is not None
                and frame_distance(session.reference, frame)
                >= session.options.threshold
            ):
                session.reasons.add("accumulated_change")

    def add_audio(
        self, session: LiveSession, pcm: bytes, start: float, end: float
    ) -> None:
        if (
            not session.options
            or not session.options.audio
            or session.paused
            or session.stopped
        ):
            raise ValueError("会话未启用音频")
        start = number(
            start, "音频开始时间", 0, self.settings.live_session_seconds + 60
        )
        end = number(end, "音频结束时间", start + 0.05, start + 10)
        if (
            not pcm
            or len(pcm) > MAX_AUDIO_BYTES
            or len(pcm) % 2
            or abs(len(pcm) / 32000 - (end - start)) > 0.25
        ):
            raise ValueError("音频必须是时长匹配、最多 10 秒的 16 kHz 单声道 PCM16")
        if start < session.last_audio_end - 0.05:
            raise ValueError("音频片段不得倒序或重复")
        session.last_audio_end = end
        while (
            session.audio_chunks
            and end - session.audio_chunks[0][1] > self.settings.live_buffer_seconds
        ):
            session.audio_chunks.popleft()
        session.audio_chunks.append((start, end, pcm))
        session.audio_status = "buffering"

    async def flush_audio(self, session: LiveSession) -> None:
        async with session.audio_lock:
            count = len(session.audio_chunks)
            for _ in range(count):
                if not session.audio_chunks or session.stopped or session.paused:
                    break
                start, end, pcm = session.audio_chunks.popleft()
                if (
                    session.frames
                    and session.frames[-1].timestamp - end
                    > self.settings.live_buffer_seconds
                ):
                    continue
                generation = session.generation
                try:
                    session.audio_status = "transcribing"
                    async with asyncio.timeout(20):
                        async with self.asr_lock:
                            text = await self.transcribe(pcm)
                    if generation != session.generation or session.stopped:
                        continue
                    session.audio_status = "ready"
                    session.asr_completed_end = end
                    if (
                        not isinstance(text, str)
                        or not text.strip()
                        or text.strip() == session.last_audio_text
                    ):
                        continue
                    session.last_audio_text = text.strip()[:2000]
                    session.transcripts.append(
                        {
                            "start_sec": start,
                            "end_sec": end,
                            "text": session.last_audio_text,
                        }
                    )
                    if session.options and session.options.mode == "change":
                        session.reasons.add("new_dialogue")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    session.audio_status = "failed"
                    session.error = "实时语音转写失败或超时，继续分析画面"

    def pause(self, session: LiveSession, paused: bool) -> None:
        session.paused = paused
        session.generation += 1
        self.cancel_reply(session)
        session.reasons.clear()
        session.audio_chunks.clear()
        if not paused:
            session.frames.clear()
            session.reference = None
            if session.options and session.options.mode == "change":
                session.reasons.add("resume")

    def user_turn(self, session: LiveSession) -> None:
        session.last_user = time.monotonic()
        session.user_generation += 1
        self.cancel_reply(session)

    def cancel_reply(self, session: LiveSession) -> None:
        if session.pending_reply is not None:
            session.pending_reply["cancelled"] = True
            session.pending_reply.pop("image_urls", None)
            session.pending_reply = None

    def snapshot(self, session: LiveSession) -> dict:
        return {
            "id": session.id,
            "configured": session.options is not None,
            "paused": session.paused,
            "stale": session.stale,
            "frame_count": len(session.frames),
            "busy": session.busy,
            "latest_frame_sec": session.frames[-1].timestamp
            if session.frames
            else None,
            "frame_age_sec": round(time.monotonic() - session.frames[-1].received, 1)
            if session.frames
            else None,
            "observation": session.observation,
            "observation_age_sec": round(
                time.monotonic() - session.observed_received, 1
            )
            if session.observation
            else None,
            "observation_stale": not session.observation
            or time.monotonic() - session.observed_received
            > (session.options.fresh_seconds if session.options else FRESH_SECONDS),
            "recent_speech": list(session.recent_speech),
            "audio_status": session.audio_status,
            "audio_completed_sec": session.asr_completed_end,
            "error": session.error,
            "analysis_count": session.analysis_count,
            "remaining_seconds": max(0, int(session.expires - time.monotonic())),
        }

    async def observe(
        self,
        session: LiveSession,
        question: str = "",
        lookback: float | None = None,
        *,
        automatic: bool = False,
        direct: bool | None = None,
    ) -> dict:
        task = asyncio.current_task()
        session.operations.add(task)
        try:
            return await self._observe(
                session, question, lookback, automatic=automatic, direct=direct
            )
        finally:
            session.operations.discard(task)

    async def _observe(
        self,
        session: LiveSession,
        question: str = "",
        lookback: float | None = None,
        *,
        automatic: bool = False,
        direct: bool | None = None,
    ) -> dict:
        async with session.lock:
            if (
                session.paused
                or session.stopped
                or session.stale
                or time.monotonic() >= session.expires
            ):
                raise VideoError(
                    "实时画面已暂停、断开或超过采样有效期未更新，不能作为当前画面"
                )
            options = session.options
            if options is None:
                raise VideoError("请先使用 /live 开始观察")
            if direct is None:
                direct = not (
                    self.settings.overview_model_enabled
                    and self.settings.overview_provider_id
                ) and (not automatic or options.proactive)
            if (
                automatic
                and direct
                and (
                    session.user_busy
                    or time.monotonic() - session.last_user < 5
                    or time.monotonic() - session.last_speech
                    < self.settings.live_speech_cooldown
                    or session.pending_reply is not None
                )
            ):
                return {}
            if session.analysis_count >= self.settings.live_max_analyses:
                raise VideoError("本次实时会话的分析预算已用完，请重新开启会话")
            if (
                automatic
                and time.monotonic() - session.last_analysis < options.analysis_interval
            ):
                return {}
            generation, user_generation = session.generation, session.user_generation
            latest = session.frames[-1]
            lookback = min(
                options.lookback_seconds
                if lookback is None
                else number(lookback, "取帧时间范围", 0, 300),
                self.settings.live_buffer_seconds,
            )
            frames = select_frames(
                list(session.frames),
                source=options.source,
                mode=options.mode,
                max_frames=options.max_frames,
                lookback=lookback,
                reference=session.reference if automatic else None,
            )
            audio_start = latest.timestamp - max(lookback, 5 if options.audio else 0)
            audio = [
                item
                for item in session.transcripts
                if item["end_sec"] >= audio_start
                and item["start_sec"] <= latest.timestamp
            ]
            reasons = sorted(session.reasons)
            session.reasons.clear()
            session.busy = True
            session.last_analysis = time.monotonic()
            session.analysis_count += 1
            if automatic:
                logger.info(
                    "实时自动观察开始：session=%s，mode=%s，reasons=%s，count=%s",
                    session.id[:8],
                    options.mode,
                    reasons,
                    session.analysis_count,
                )
            try:
                if direct:
                    result = {
                        "summary": "本次采样图片直接提供给主模型，未生成独立观察摘要。",
                        "speak": True,
                        "comment": "",
                    }
                else:
                    result = await asyncio.wait_for(
                        self.analyze(session, frames, audio, question, reasons),
                        min(self.settings.tool_timeout, MODEL_TIMEOUT_SECONDS),
                    )
                if (
                    session.stopped
                    or session.paused
                    or generation != session.generation
                ):
                    raise VideoError("会话状态已改变，本次分析已作废")
                if not isinstance(result, dict) or not isinstance(
                    result.get("summary"), str
                ):
                    raise VideoError("实时看图模型未返回有效分析")
                session.reference = latest
                session.observed_received = latest.received
                if not direct:
                    session.memory = str(result.get("memory", result["summary"]))[:4000]
                session.observation = {
                    "mode": "direct" if direct else "delegated",
                    "start_sec": frames[0].timestamp,
                    "end_sec": latest.timestamp,
                    "summary": result["summary"][:4000],
                    "evidence_times": [f.timestamp for f in frames],
                    "sampling_window_sec": lookback,
                    "audio": audio,
                    "audio_status": session.audio_status,
                    "audio_completed_sec": session.asr_completed_end,
                    "trigger_reasons": reasons,
                }
                images = [image_url(frame) for frame in frames] if direct else []
                session.error = ""
                comment = result.get("comment", "")
                now = time.monotonic()
                # 采集是否断流与模型处理耗时分开判断；正常推理超过 10 秒
                # 不应被当作断流。原始证据时间仍保留，不伪装成最新画面。
                timely = now - latest.received <= options.fresh_seconds + min(
                    self.settings.tool_timeout, MODEL_TIMEOUT_SECONDS
                )
                if automatic:
                    blocked = []
                    if not options.proactive:
                        blocked.append("proactive_disabled")
                    if not direct and result.get("speak") is not True:
                        blocked.append("model_silent")
                    if not direct and (
                        not isinstance(comment, str) or not comment.strip()
                    ):
                        blocked.append("empty_candidate")
                    if not timely:
                        blocked.append("evidence_expired")
                    if session.stale:
                        blocked.append("capture_stale")
                    if user_generation != session.user_generation:
                        blocked.append("user_interrupted")
                    if now - session.last_user < 5 or session.user_busy:
                        blocked.append("user_active")
                    if now - session.last_speech < self.settings.live_speech_cooldown:
                        blocked.append("cooldown")
                    if not direct and (
                        isinstance(comment, str)
                        and comment.strip() in session.recent_speech
                    ):
                        blocked.append("duplicate")
                    logger.info(
                        "实时自动观察完成：session=%s，elapsed=%.1fs，主动发言=%s",
                        session.id[:8],
                        now - session.last_analysis,
                        ",".join(blocked) if blocked else "ready",
                    )
                    if not blocked:
                        kwargs = {"image_urls": images} if direct else {}
                        await asyncio.wait_for(
                            self.speak(
                                session,
                                comment.strip()[:500],
                                user_generation,
                                **kwargs,
                            ),
                            30,
                        )
                observation = {
                    **session.observation,
                    "stale": time.monotonic() - latest.received > options.fresh_seconds,
                    "memory": session.memory,
                }
                if direct:
                    observation.update(
                        image_urls=images,
                        focus=options.focus,
                        question=question,
                        recent_comments=list(session.recent_speech),
                    )
                return observation
            finally:
                session.busy = False

    async def _worker(self, session: LiveSession) -> None:
        reason = "cancelled"
        try:
            while not session.stopped:
                await asyncio.sleep(0.25)
                now = time.monotonic()
                if now >= session.expires:
                    reason = "expired"
                    break
                if (
                    session.pending_reply is not None
                    and now - session.pending_reply["created_at"]
                    >= REPLY_TIMEOUT_SECONDS
                ):
                    self.cancel_reply(session)
                # 额度耗尽后等最后一轮分析、工具或排队回复结束，再清理会话。
                # 此检查也适用于提问模式、暂停或暂时没有新画面的情况。
                if session.analysis_count >= self.settings.live_max_analyses:
                    if (
                        session.operations
                        or session.busy
                        or session.pending_reply is not None
                        or (session.analysis_task and not session.analysis_task.done())
                    ):
                        continue
                    reason = "analysis_limit"
                    break
                options = session.options
                if not options or session.paused:
                    continue
                if (
                    options.mode != "demand"
                    and session.audio_chunks
                    and (session.audio_task is None or session.audio_task.done())
                ):
                    session.audio_task = asyncio.create_task(self.flush_audio(session))
                if session.stale or options.mode == "demand":
                    continue
                if session.pending_reply is not None:
                    continue
                interval = options.analysis_interval
                if now - session.last_analysis < interval or (
                    options.mode == "change" and not session.reasons
                ):
                    continue
                if (
                    now - session.last_user < 5
                    or session.user_busy
                    or session.lock.locked()
                    or (session.analysis_task and not session.analysis_task.done())
                ):
                    continue
                session.analysis_task = asyncio.create_task(
                    self._automatic_observe(session)
                )
        except Exception:
            reason = "worker_failed"
            logger.exception("实时观察后台任务失败：session=%s", session.id[:8])
        finally:
            await self.stop(session, reason=reason)

    async def _automatic_observe(self, session: LiveSession) -> None:
        try:
            await self.observe(session, automatic=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("实时自动观察失败：session=%s", session.id[:8])
            session.error = "实时分析失败或超时；可检查模型与采集状态"
            session.last_analysis = time.monotonic()
            session.reasons.add("retry")

    async def stop(self, session: LiveSession, *, reason: str = "manual") -> None:
        if session.stopped:
            return
        session.stopped = True
        logger.info(
            "实时观察已结束：session=%s，reason=%s，elapsed=%.1fs，count=%s/%s",
            session.id[:8],
            reason,
            time.monotonic() - session.created,
            session.analysis_count,
            self.settings.live_max_analyses,
        )
        session.generation += 1
        self.cancel_reply(session)
        self.sessions.pop(session.id, None)
        tasks = [
            task
            for task in (
                session.task,
                session.stream_task,
                session.analysis_task,
                session.audio_task,
                *session.operations,
            )
            if task and task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        session.frames.clear()
        session.reference = None
        session.transcripts.clear()
        session.audio_chunks.clear()
        session.last_audio_text = ""
        session.reasons.clear()
        session.source_url = ""
        session.observation = None
        session.memory = ""
        session.recent_speech.clear()
        try:
            if self.ended and reason in {"expired", "analysis_limit"}:
                async with asyncio.timeout(5):
                    await self.ended(session, reason)
        except Exception:
            logger.exception("实时观察结束通知失败：session=%s", session.id[:8])
        finally:
            session.reply_route = None

    async def close(self) -> None:
        for session in list(self.sessions.values()):
            await self.stop(session, reason="plugin_unloaded")
        if self.root.exists():
            await asyncio.to_thread(shutil.rmtree, self.root, True)


async def transcribe_pcm(context, settings: Settings, root: Path, pcm: bytes) -> str:
    from astrbot.api.provider import STTProvider

    provider = context.get_provider_by_id(settings.asr_provider_id)
    if (
        not settings.asr_enabled
        or not isinstance(provider, STTProvider)
        or not provider.provider_config.get("enable", True)
    ):
        raise VideoError("实时 ASR 提供商不可用")
    path = root / f"live_{uuid.uuid4().hex}.wav"

    def write():
        root.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(pcm)

    try:
        await asyncio.to_thread(write)
        text = await provider.get_text(str(path))
        if not isinstance(text, str):
            raise VideoError("ASR 未返回文本")
        return text.strip()[:2000]
    finally:
        path.unlink(missing_ok=True)


def image_url(frame: LiveFrame) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode("ascii")


def parse_observation(text: str) -> dict:
    if not isinstance(text, str):
        raise VideoError("实时模型未返回文字")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise VideoError("实时模型未返回有效 JSON") from exc
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("summary"), str)
        or not value["summary"].strip()
    ):
        raise VideoError("实时模型缺少画面描述")
    return value
