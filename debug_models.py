"""复用同一批帧的模型实验；预览不会调用模型。"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .media import VideoError
from .overview_model import (
    OverviewModelInput,
    invoke_overview_model,
    prepare_overview_input,
    response_usage,
)

if TYPE_CHECKING:
    from .debug import DebugSession, DebugSessions

MAX_MODEL_RUNS = 20


@dataclass
class PreparedInput:
    id: str
    provider: dict
    question: str
    mode: str
    request: OverviewModelInput

    def summary(self) -> dict:
        return {
            "id": self.id,
            "provider": self.provider,
            "question": self.question,
            "mode": self.mode,
            "image_count": len(self.request.images),
            "frame_count": self.request.presentation["frame_count"],
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
class ModelState:
    version: int = 0
    preparing: bool = False
    prepared: PreparedInput | None = None
    runs: list[ModelRun] = field(default_factory=list)


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

    def summary(self, session: DebugSession) -> dict:
        state = session.model
        return {
            "version": state.version,
            "preparing": state.preparing,
            "prepared": state.prepared.summary() if state.prepared else None,
            "runs": [run.summary() for run in state.runs],
            "max_runs": MAX_MODEL_RUNS,
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
        self, session: DebugSession, context, revision, provider_id, question, mode
    ) -> dict:
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > 4000
        ):
            raise ValueError("测试问题不能为空，且最多 4000 个字符")
        if mode not in ("frames", "grid"):
            raise ValueError("请选择独立帧或九宫格")
        provider = self._provider(context, provider_id)
        async with session.lock:
            self._ready(session, revision)
            state = session.model
            state.preparing = True
            state.prepared = None
            state.version += 1
            task = asyncio.create_task(
                prepare_overview_input(
                    session.pipeline.media, session.pipeline.result, question, mode=mode
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
                    uuid.uuid4().hex, provider, question, mode, request
                )
                return state.prepared.detail()
        finally:
            session.assets.discard(task)
            state.preparing = False
            state.version += 1
            if session.model is state:
                session.preview_elapsed += time.monotonic() - started
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

    def detail(self, session: DebugSession, identifier: str) -> dict:
        state = session.model
        if state.prepared and state.prepared.id == identifier:
            return {"input": state.prepared.detail()}
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
