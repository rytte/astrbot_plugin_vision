"""复用同一批帧的模型实验；预览不会调用模型。"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from astrbot.api.provider import STTProvider

from .asr import VideoASR
from .media import VideoError
from .overview_model import (
    OverviewModelInput,
    invoke_overview_model,
    prepare_overview_input,
    response_usage,
)
from .pipeline import VideoResult

if TYPE_CHECKING:
    from .debug import DebugSession, DebugSessions

MAX_MODEL_RUNS = 20
MAX_ASR_RUNS = 20


@dataclass
class PreparedInput:
    id: str
    provider: dict
    question: str
    mode: str
    request: OverviewModelInput
    asr: dict | None = None

    def summary(self) -> dict:
        return {
            "id": self.id,
            "provider": self.provider,
            "question": self.question,
            "mode": self.mode,
            "image_count": len(self.request.images),
            "frame_count": self.request.presentation["frame_count"],
            "asr": self.asr,
        }

    def detail(self) -> dict:
        return {
            **self.summary(),
            "system_prompt": self.request.system_prompt,
            "prompt": self.request.prompt,
            "presentation": self.request.presentation,
        }


@dataclass
class ModelRun:
    id: str
    prepared: PreparedInput
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "queued"
    seconds: float = 0
    answer: str | None = None
    usage: dict | None = None
    error: str | None = None

    def summary(self) -> dict:
        return {
            "id": self.id,
            "input": self.prepared.summary(),
            "created_at": self.created_at,
            "status": self.status,
            "seconds": self.seconds,
            "usage": self.usage,
            "error": self.error,
        }

    def detail(self) -> dict:
        return {
            **self.summary(),
            "input": self.prepared.detail(),
            "answer": self.answer,
        }


@dataclass
class ASRRun:
    id: str
    provider: dict
    status: str = "queued"
    seconds: float = 0
    result: dict | None = None
    error: str | None = None

    def summary(self) -> dict:
        result = self.result or {}
        return {
            "id": self.id,
            "provider": self.provider,
            "status": self.status,
            "seconds": self.seconds,
            "error": self.error,
            "transcription_status": result.get("status"),
            "complete": result.get("complete", False),
            "segment_count": len(result.get("segments", [])),
            "cache_hit_count": result.get("cache_hit_count", 0),
        }

    def detail(self) -> dict:
        return {**self.summary(), "result": self.result}


@dataclass
class ModelState:
    version: int = 0
    preparing: bool = False
    prepared: PreparedInput | None = None
    runs: list[ModelRun] = field(default_factory=list)
    asr_runs: list[ASRRun] = field(default_factory=list)
    asr_engine: VideoASR | None = None


class ModelTests:
    def __init__(self, sessions: DebugSessions):
        self.sessions = sessions

    @staticmethod
    def providers(context) -> list[dict]:
        getter = getattr(context, "get_all_providers", None)
        if getter is None:
            return []
        providers = []
        for provider in getter():
            meta = provider.meta()
            modalities = provider.provider_config.get("modalities")
            if (
                isinstance(modalities, list)
                and modalities
                and "image" not in modalities
            ):
                continue
            providers.append(
                {
                    "id": meta.id,
                    "model": meta.model,
                    "vision_confirmed": isinstance(modalities, list)
                    and "image" in modalities,
                }
            )
        return providers

    def _provider(self, context, provider_id) -> dict:
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("请选择已配置的看图模型")
        provider = next(
            (item for item in self.providers(context) if item["id"] == provider_id),
            None,
        )
        if provider is None:
            raise ValueError("模型不存在、未启用或未开启图片输入，请刷新模型列表")
        return provider

    @staticmethod
    def asr_providers(context) -> list[dict]:
        getter = getattr(context, "get_all_stt_providers", None)
        if getter is None:
            return []
        providers = []
        for provider in getter():
            if not isinstance(
                provider, STTProvider
            ) or not provider.provider_config.get("enable", True):
                continue
            meta = provider.meta()
            providers.append({"id": meta.id, "model": meta.model})
        return providers

    def _asr_provider(self, context, provider_id) -> dict:
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("请选择已配置的 ASR 提供商")
        provider = next(
            (item for item in self.asr_providers(context) if item["id"] == provider_id),
            None,
        )
        if provider is None:
            raise ValueError("ASR 提供商不存在或未启用，请刷新提供商列表")
        return provider

    def summary(self, session: DebugSession) -> dict:
        state = session.model
        return {
            "version": state.version,
            "preparing": state.preparing,
            "prepared": state.prepared.summary() if state.prepared else None,
            "runs": [run.summary() for run in state.runs],
            "max_runs": MAX_MODEL_RUNS,
            "asr": state.asr_runs[-1].summary() if state.asr_runs else None,
            "max_asr_runs": MAX_ASR_RUNS,
        }

    def _ready(self, session: DebugSession, revision) -> None:
        self.sessions._active(session)
        if type(revision) is not int or revision != session.revision:
            raise ValueError("测试版本已变化，请重新预览模型输入")
        if (
            not session.pipeline.done
            or session.task is not None
            or session.model.preparing
        ):
            raise ValueError("请先完成本地流水线，并等待当前操作结束")

    async def prepare(
        self,
        session: DebugSession,
        context,
        revision,
        provider_id,
        question,
        mode,
        *,
        asr_enabled=False,
        asr_provider_id="",
        asr_run_id=None,
    ) -> dict:
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > 4000
        ):
            raise ValueError("测试问题不能为空，且最多 4000 个字符")
        if mode not in ("frames", "grid"):
            raise ValueError("请选择独立帧或九宫格")
        if not isinstance(asr_enabled, bool):
            raise ValueError("ASR 选项必须是布尔值")
        provider = self._provider(context, provider_id)
        async with session.lock:
            self._ready(session, revision)
            state = session.model
            audio = None
            result = session.pipeline.result
            if asr_enabled:
                speech_provider = self._asr_provider(context, asr_provider_id)
                audio = state.asr_runs[-1] if state.asr_runs else None
                if (
                    audio is None
                    or audio.id != asr_run_id
                    or audio.provider != speech_provider
                    or audio.result is None
                ):
                    raise ValueError("请先完成所选 ASR 的音轨转写，再预览模型输入")
                # 使用快照，切换 ASR 或重试不修改本地结果与已有调用记录。
                result = VideoResult(
                    {**result.data, "asr": copy.deepcopy(audio.result)},
                    result.frames,
                    result.images,
                )
            state.preparing = True
            state.prepared = None
            state.version += 1
            task = asyncio.create_task(
                prepare_overview_input(
                    session.pipeline.media, result, question, mode=mode
                )
            )
            session.assets.add(task)
        started = time.monotonic()
        try:
            request = await asyncio.wait_for(
                task, session.pipeline.settings.tool_timeout
            )
            async with session.lock:
                self.sessions._active(session)
                if session.revision != revision or session.model is not state:
                    raise ValueError("测试版本已变化，请重新预览模型输入")
                state.prepared = PreparedInput(
                    uuid.uuid4().hex,
                    provider,
                    question,
                    mode,
                    request,
                    audio.summary() if audio else None,
                )
                return state.prepared.detail()
        finally:
            session.assets.discard(task)
            state.preparing = False
            state.version += 1
            if session.model is state:
                session.preview_elapsed += time.monotonic() - started
            session.touched = time.monotonic()

    async def start_asr(
        self, session: DebugSession, context, revision, provider_id, run_id
    ) -> None:
        if (
            not isinstance(run_id, str)
            or len(run_id) != 32
            or any(char not in "0123456789abcdef" for char in run_id)
        ):
            raise ValueError("转写编号无效，请刷新页面后重试")
        provider = self._asr_provider(context, provider_id)
        async with session.lock:
            self.sessions._active(session)
            if type(revision) is not int or revision != session.revision:
                raise ValueError("测试版本已变化，请重新执行音轨转写")
            state = session.model
            existing = next((run for run in state.asr_runs if run.id == run_id), None)
            if existing:
                if existing.provider != provider:
                    raise ValueError("转写编号已用于其他提供商")
                return
            self._ready(session, revision)
            if len(state.asr_runs) >= MAX_ASR_RUNS:
                raise ValueError(
                    "本次测试最多保留 20 次转写，请导出记录后重跑或新建测试"
                )
            run = ASRRun(run_id, provider)
            state.asr_runs.append(run)
            state.prepared = None
            state.version += 1
            if state.asr_engine is None:
                state.asr_engine = VideoASR(session.pipeline.media)
            session.error = None
            session.operation = "asr"
            session.task = asyncio.create_task(self._run_asr(session, context, run))

    async def _run_asr(self, session: DebugSession, context, run: ASRRun) -> None:
        started = None
        state = session.model
        try:
            async with self.sessions.slots:
                if self._asr_provider(context, run.provider["id"]) != run.provider:
                    raise VideoError("ASR 配置已变化，请重新选择提供商并转写")
                started = time.monotonic()
                run.status = "running"
                state.version += 1
                settings = replace(
                    session.pipeline.settings,
                    asr_enabled=True,
                    asr_provider_id=run.provider["id"],
                )
                run.result = await asyncio.wait_for(
                    state.asr_engine.enhance(
                        context,
                        settings,
                        session.pipeline.result,
                        asyncio.get_running_loop().time() + settings.tool_timeout,
                        set(),
                    ),
                    settings.tool_timeout,
                )
                status = run.result["status"]
                run.status = (
                    "succeeded"
                    if status in {"success", "no_audio"}
                    else "partial"
                    if status == "partial"
                    else "failed"
                )
                if run.status in {"failed", "partial"}:
                    run.error = run.result.get("message")
        except asyncio.CancelledError:
            run.status, run.error = "cancelled", "已取消音轨转写，可以重试"
            raise
        except TimeoutError:
            run.status, run.error = "failed", "音轨转写超时，可以重试；本地帧已保留"
        except Exception as exc:
            run.status = "failed"
            run.error = (
                str(exc)
                if isinstance(exc, VideoError | ValueError)
                else "音轨转写失败，请检查所选 ASR 提供商"
            )
        finally:
            if started is not None:
                run.seconds = time.monotonic() - started
            state.version += 1
            session.task = None
            session.operation = None
            session.touched = time.monotonic()

    async def start(
        self, session: DebugSession, context, revision, preview_id, run_id
    ) -> None:
        if (
            not isinstance(run_id, str)
            or len(run_id) != 32
            or any(char not in "0123456789abcdef" for char in run_id)
        ):
            raise ValueError("调用编号无效，请刷新页面后重试")
        async with session.lock:
            self.sessions._active(session)
            if type(revision) is not int or revision != session.revision:
                raise ValueError("测试版本已变化，请重新预览模型输入")
            existing = next(
                (run for run in session.model.runs if run.id == run_id), None
            )
            if existing:
                if existing.prepared.id != preview_id:
                    raise ValueError("调用编号已用于其他输入，请重新预览")
                return  # 网络重试不会重复调用模型。
            self._ready(session, revision)
            prepared = session.model.prepared
            if prepared is None or prepared.id != preview_id:
                raise ValueError("输入预览已变化，请重新预览后调用")
            if self._provider(context, prepared.provider["id"]) != prepared.provider:
                raise ValueError("模型配置已变化，请重新预览模型输入")
            if len(session.model.runs) >= MAX_MODEL_RUNS:
                raise ValueError(
                    "本次测试最多保留 20 次调用，请导出记录后重跑或新建测试"
                )
            run = ModelRun(run_id, prepared)
            session.model.runs.append(run)
            session.model.version += 1
            session.error = None
            session.operation = "model"
            session.task = asyncio.create_task(self._run(session, context, run))

    async def _run(self, session: DebugSession, context, run: ModelRun) -> None:
        started = None
        try:
            async with self.sessions.slots:
                if (
                    self._provider(context, run.prepared.provider["id"])
                    != run.prepared.provider
                ):
                    raise VideoError("模型配置已变化，请重新预览模型输入")
                started = time.monotonic()
                run.status = "running"
                session.model.version += 1
                response = await asyncio.wait_for(
                    invoke_overview_model(
                        context, run.prepared.provider["id"], run.prepared.request
                    ),
                    session.pipeline.settings.tool_timeout,
                )
                run.answer = response.completion_text.strip()
                run.usage = response_usage(response)
                run.status = "succeeded"
        except asyncio.CancelledError:
            run.status, run.error = "cancelled", "已取消本次调用，可以重新调用"
            raise
        except TimeoutError:
            run.status, run.error = "failed", "模型调用超时，可以重试；本地帧已保留"
        except Exception as exc:
            run.status = "failed"
            run.error = (
                str(exc)
                if isinstance(exc, VideoError)
                else "模型调用失败，请查看 AstrBot 日志"
            )
        finally:
            if started is not None:
                run.seconds = time.monotonic() - started
            session.model.version += 1
            session.task = None
            session.operation = None
            session.touched = time.monotonic()

    def stopped(self, session: DebugSession) -> None:
        # 在 create_task 后立即取消时，协程体可能还没有执行。
        for run in session.model.runs:
            if run.status in {"queued", "running"}:
                run.status, run.error = "cancelled", "已取消本次调用，可以重新调用"
                session.model.version += 1
        for run in session.model.asr_runs:
            if run.status in {"queued", "running"}:
                run.status, run.error = "cancelled", "已取消音轨转写，可以重试"
                session.model.version += 1

    def detail(self, session: DebugSession, identifier: str) -> dict:
        state = session.model
        if state.prepared and state.prepared.id == identifier:
            return {"input": state.prepared.detail()}
        audio = next((run for run in state.asr_runs if run.id == identifier), None)
        if audio is not None:
            return {"asr": audio.detail()}
        run = next((run for run in state.runs if run.id == identifier), None)
        if run is None:
            raise ValueError("模型记录不存在或已清理")
        return run.detail()

    async def image(
        self, session: DebugSession, identifier: str, index: int, variant: str
    ):
        prepared = session.model.prepared
        if prepared is None or prepared.id != identifier:
            run = next(
                (run for run in session.model.runs if run.id == identifier), None
            )
            prepared = run.prepared if run else None
        if prepared is None or not 0 <= index < len(prepared.request.images):
            raise ValueError("模型输入图片不存在或索引无效")
        return self.sessions._output_image(prepared.request.images[index], variant)
