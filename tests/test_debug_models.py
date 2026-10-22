import asyncio
import base64
import json
import time
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import jwt
import pytest
from astrbot.core.provider.entities import TokenUsage
from astrbot.dashboard.api.plugins import _call_plugin_extension
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.debug import DebugSessions
from astrbot_plugin_vision.debug_web import DebugWeb
from astrbot_plugin_vision.media import FFmpegMedia
from astrbot_plugin_vision.overview_model import prepare_overview_input
from fastapi import FastAPI, Request


@pytest.fixture
def model_context():
    provider = SimpleNamespace(
        provider_config={"modalities": ["text", "image"], "key": "private-key"},
        meta=lambda: SimpleNamespace(id="vision-test", model="fake-model"),
    )
    text_provider = SimpleNamespace(
        provider_config={"modalities": ["text"]},
        meta=lambda: SimpleNamespace(id="text-only", model="no-images"),
    )
    context = SimpleNamespace(
        get_all_providers=lambda: [provider, text_provider],
        llm_generate=AsyncMock(
            return_value=SimpleNamespace(
                role="assistant",
                completion_text="0 秒红色；1 秒蓝色；2 秒绿色。",
                usage=TokenUsage(input_other=120, input_cached=30, output=20),
            )
        ),
        registered_web_apis=[],
        get_config=lambda: {
            "dashboard": {"jwt_secret": "local-test-secret-32-characters-long"}
        },
    )
    context.register_web_api = lambda *entry: context.registered_web_apis.append(entry)
    return context


@pytest.fixture
def speech_context(model_context, asr_provider):
    model_context.get_all_stt_providers = lambda: [asr_provider]
    model_context.get_provider_by_id = Mock(return_value=asr_provider)
    return model_context


@pytest.fixture
async def speech_experiment(experiment, audio_video):
    sessions, session = experiment
    session.source.write_bytes(audio_video.read_bytes())
    await sessions.reset(session, {})
    await session.pipeline.run()
    return sessions, session


async def transcribe(sessions, session, context, identifier=None):
    await sessions.models.start_asr(
        session, context, session.revision, "test-asr", identifier or uuid.uuid4().hex
    )
    if session.task is not None:
        await session.task
    return session.model.asr_runs[-1]


async def test_asr_is_explicit_and_same_transcript_is_used_for_both_image_modes(
    speech_experiment, speech_context, asr_provider
):
    sessions, session = speech_experiment
    plain = await prepare(sessions, session, speech_context)
    assert plain["asr"] is None
    assert "asr" not in json.loads(plain["prompt"].split("\n", 1)[1])
    asr_provider.get_text.assert_not_awaited()
    with pytest.raises(ValueError, match="先完成"):
        await prepare(
            sessions,
            session,
            speech_context,
            asr_enabled=True,
            asr_provider_id="test-asr",
        )
    audio = await transcribe(sessions, session, speech_context)
    assert audio.status == "succeeded"
    assert sessions.media.settings.asr_enabled is False
    assert session.pipeline.settings.asr_enabled is False
    asr_provider.get_text.assert_awaited_once()
    assert session.model.prepared is None
    assert "asr" not in session.pipeline.result.data
    inputs = []
    for mode in ("frames", "grid"):
        prepared = await prepare(
            sessions,
            session,
            speech_context,
            mode,
            asr_enabled=True,
            asr_provider_id="test-asr",
            asr_run_id=audio.id,
        )
        payload = json.loads(prepared["prompt"].split("\n", 1)[1])
        assert payload["asr"] == audio.result
        assert prepared["asr"]["provider"]["id"] == "test-asr"
        assert prepared["asr"]["id"] == audio.id
        record = await run(sessions, session, speech_context, prepared)
        assert (
            speech_context.llm_generate.call_args.kwargs["prompt"] == prepared["prompt"]
        )
        assert record.prepared.asr["segment_count"] == 1
        inputs.append(prepared)
    asr_provider.get_text.assert_awaited_once()
    # 状态轮询不重复传输完整转写和提示词。
    snapshot = sessions.snapshot(session)
    assert snapshot["model"]["asr"]["segment_count"] == 1
    assert asr_provider.get_text.return_value not in json.dumps(
        snapshot, ensure_ascii=False
    )
    pure = await prepare(sessions, session, speech_context, asr_enabled=False)
    assert "asr" not in json.loads(pure["prompt"].split("\n", 1)[1])
    assert inputs[0]["asr"] == inputs[1]["asr"]
    assert all(record.prepared.asr is not None for record in session.model.runs)


async def test_asr_retries_use_cache_and_old_model_records_are_immutable(
    speech_experiment, speech_context, asr_provider
):
    sessions, session = speech_experiment
    identifier = uuid.uuid4().hex
    audio = await transcribe(sessions, session, speech_context, identifier)
    await sessions.models.start_asr(
        session, speech_context, session.revision, "test-asr", identifier
    )
    assert len(session.model.asr_runs) == 1
    prepared = await prepare(
        sessions,
        session,
        speech_context,
        asr_enabled=True,
        asr_provider_id="test-asr",
        asr_run_id=audio.id,
    )
    record = await run(sessions, session, speech_context, prepared)
    original_prompt = record.prepared.request.prompt
    retried = await transcribe(sessions, session, speech_context)
    assert retried.result["cache_hit_count"] == 1
    assert session.model.prepared is None
    assert record.prepared.request.prompt == original_prompt
    assert record.prepared.asr["id"] == audio.id
    asr_provider.get_text.assert_awaited_once()
    with pytest.raises(ValueError, match="先完成"):
        await prepare(
            sessions,
            session,
            speech_context,
            asr_enabled=True,
            asr_provider_id="test-asr",
            asr_run_id=audio.id,
        )


@pytest.mark.parametrize("immediate", [True, False])
async def test_asr_cancellation_preserves_frames_and_allows_retry(
    speech_experiment, speech_context, asr_provider, immediate
):
    sessions, session = speech_experiment
    started = asyncio.Event()

    async def slow(path):
        started.set()
        await asyncio.Event().wait()

    asr_provider.get_text.side_effect = slow
    await sessions.models.start_asr(
        session, speech_context, session.revision, "test-asr", uuid.uuid4().hex
    )
    if not immediate:
        await asyncio.wait_for(started.wait(), 5)
    await sessions.cancel(session)
    assert session.model.asr_runs[-1].status == "cancelled"
    assert not session.task and not session.operation and session.error is None
    assert all(frame.path.is_file() for frame in session.pipeline.kept)
    assert not list(session.pipeline.media.cache_dir.glob("*.wav"))
    asr_provider.get_text.side_effect = None
    assert (await transcribe(sessions, session, speech_context)).status == "succeeded"


@pytest.mark.parametrize("operation", ["reset", "delete", "close"])
async def test_asr_work_is_cancelled_when_session_is_reset_or_removed(
    speech_experiment, speech_context, asr_provider, operation
):
    sessions, session = speech_experiment
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def slow(path):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    asr_provider.get_text.side_effect = slow
    await sessions.models.start_asr(
        session, speech_context, session.revision, "test-asr", uuid.uuid4().hex
    )
    await asyncio.wait_for(started.wait(), 5)
    if operation == "reset":
        await sessions.reset(session, {})
        assert not session.model.asr_runs
        assert not list(session.pipeline.media.cache_dir.iterdir())
    elif operation == "delete":
        await sessions.delete(session)
        assert not session.directory.exists()
    else:
        await sessions.close()
        assert not session.directory.exists()
    assert cancelled.is_set()
    assert not session.task


@pytest.mark.parametrize("outcome", ["no_audio", "failed", "timeout", "partial"])
async def test_asr_outcomes_remain_visible_and_do_not_prevent_visual_testing(
    speech_experiment, speech_context, asr_provider, outcome, monkeypatch
):
    sessions, session = speech_experiment
    payload = {
        "provider_id": "test-asr",
        "status": outcome,
        "segments": [
            {"segment_id": "segment", "start_sec": 0, "end_sec": 3, "text": "部分转写"}
        ]
        if outcome == "partial"
        else [],
        "complete": outcome == "no_audio",
        "message": "转写状态说明",
    }
    monkeypatch.setattr(
        "astrbot_plugin_vision.debug_models.VideoASR.enhance",
        AsyncMock(return_value=payload),
    )
    audio = await transcribe(sessions, session, speech_context)
    assert audio.result["status"] == outcome
    prepared = await prepare(
        sessions,
        session,
        speech_context,
        asr_enabled=True,
        asr_provider_id="test-asr",
        asr_run_id=audio.id,
    )
    assert json.loads(prepared["prompt"].split("\n", 1)[1])["asr"] == payload
    record = await run(sessions, session, speech_context, prepared)
    assert record.status == "succeeded"


async def test_invalid_asr_requests_never_call_provider(
    speech_experiment, speech_context, asr_provider, monkeypatch
):
    sessions, session = speech_experiment
    for provider_id, identifier, revision in [
        ("missing", uuid.uuid4().hex, session.revision),
        ("vision-test", uuid.uuid4().hex, session.revision),
        ("test-asr", "invalid", session.revision),
        ("test-asr", uuid.uuid4().hex, session.revision - 1),
    ]:
        with pytest.raises(ValueError):
            await sessions.models.start_asr(
                session, speech_context, revision, provider_id, identifier
            )
    with pytest.raises(ValueError, match="布尔"):
        await prepare(sessions, session, speech_context, asr_enabled="true")
    session.pipeline.completed -= 1
    with pytest.raises(ValueError, match="本地流水线"):
        await transcribe(sessions, session, speech_context)
    session.pipeline.completed += 1
    asr_provider.get_text.assert_not_awaited()
    asr_provider.provider_config["enable"] = False
    assert sessions.models.asr_providers(speech_context) == []
    with pytest.raises(ValueError, match="未启用"):
        await transcribe(sessions, session, speech_context)
    asr_provider.provider_config["enable"] = True
    monkeypatch.setattr("astrbot_plugin_vision.debug_models.MAX_ASR_RUNS", 1)
    await transcribe(sessions, session, speech_context)
    with pytest.raises(ValueError, match="20 次转写"):
        await transcribe(sessions, session, speech_context)


@pytest.fixture
async def experiment(tmp_path, sample_video):
    media = FFmpegMedia(
        Settings(overview_interval=0.5, overview_max_edge=128),
        tmp_path / "normal",
        [tmp_path],
    )
    sessions = DebugSessions(media, tmp_path / "debug", asyncio.Semaphore(2))
    upload = SimpleNamespace(
        filename="video.mp4",
        read=AsyncMock(side_effect=[sample_video.read_bytes(), b""]),
    )
    session = await sessions.upload("admin", upload)
    await session.pipeline.run()
    yield sessions, session
    await sessions.close()


async def prepare(sessions, session, context, mode="frames", **kwargs):
    return await sessions.models.prepare(
        session,
        context,
        session.revision,
        "vision-test",
        "请描述颜色变化",
        mode,
        **kwargs,
    )


async def run(sessions, session, context, prepared):
    identifier = uuid.uuid4().hex
    await sessions.models.start(
        session, context, session.revision, prepared["id"], identifier
    )
    await session.task
    return session.model.runs[-1]


async def test_preview_matches_chat_and_switching_modes_never_extracts(
    experiment, model_context, monkeypatch
):
    sessions, session = experiment
    original_ids = [frame.frame_id for frame in session.pipeline.kept]
    monkeypatch.setattr(
        session.pipeline.media,
        "extract",
        AsyncMock(side_effect=AssertionError("不得重新抽帧")),
    )
    for mode in ("frames", "grid"):
        prepared = await prepare(sessions, session, model_context, mode)
        shared = await prepare_overview_input(
            session.pipeline.media, session.pipeline.result, "请描述颜色变化", mode=mode
        )
        assert prepared["prompt"] == shared.prompt
        assert prepared["system_prompt"] == shared.system_prompt
        assert prepared["presentation"] == shared.presentation
        assert "private-key" not in json.dumps(prepared)
        assert '"requested_sec"' not in prepared["prompt"]
        payload = json.loads(prepared["prompt"].split("\n", 1)[1])
        assert [frame["frame_id"] for frame in payload["frames"]] == original_ids
        assert all(
            not {"x", "y", "width", "height"}.intersection(cell)
            for image in payload["presentation"]["images"]
            for cell in image["cells"]
        )
        if mode == "grid":
            assert all(
                (image["frame_width"], image["frame_height"])
                == (session.pipeline.kept[0].width, session.pipeline.kept[0].height)
                for image in payload["presentation"]["images"]
            )
        image = await sessions.image(session, "model", 0, "full", 0, prepared["id"])
        assert image["image"]["cells"] == payload["presentation"]["images"][0]["cells"]
        assert (
            base64.b64decode(image["data_url"].split(",")[1])
            == shared.images[0].path.read_bytes()
        )
        model_context.llm_generate.assert_not_awaited()
    assert len(sessions.snapshot(session)["stages"]) == 8


async def test_two_modes_keep_records_actual_inputs_and_usage(
    experiment, model_context
):
    sessions, session = experiment
    records = []
    for mode in ("frames", "grid"):
        prepared = await prepare(sessions, session, model_context, mode)
        record = await run(sessions, session, model_context, prepared)
        assert record.status == "succeeded"
        assert record.answer == "0 秒红色；1 秒蓝色；2 秒绿色。"
        assert record.usage == {
            "input": 150,
            "input_other": 120,
            "input_cached": 30,
            "output": 20,
            "total": 170,
        }
        call = model_context.llm_generate.await_args.kwargs
        assert call["prompt"] == prepared["prompt"]
        assert call["system_prompt"] == prepared["system_prompt"]
        assert len(call["image_urls"]) == prepared["image_count"]
        assert call["contexts"] == [] and call["tools"] is None
        assert 0 <= record.seconds < 2
        records.append(record)
    assert records[0].prepared.request.images[0].path.suffix == ".jpg"
    assert records[1].prepared.request.images[0].path.suffix == ".png"
    assert len(session.model.runs) == 2 and not session.task
    assert sessions.models.detail(session, records[0].id)["input"]["mode"] == "frames"


async def test_missing_usage_is_not_estimated(experiment, model_context):
    sessions, session = experiment
    model_context.llm_generate.return_value.usage = None
    record = await run(
        sessions,
        session,
        model_context,
        await prepare(sessions, session, model_context),
    )
    assert record.usage is None


async def test_stale_preview_and_duplicate_run_ids_do_not_call_twice(
    experiment, model_context
):
    sessions, session = experiment
    stale = await prepare(sessions, session, model_context)
    current = await prepare(sessions, session, model_context, "grid")
    with pytest.raises(ValueError, match="预览已变化"):
        await sessions.models.start(
            session, model_context, 0, stale["id"], uuid.uuid4().hex
        )
    identifier = uuid.uuid4().hex
    await asyncio.gather(
        *(
            sessions.models.start(session, model_context, 0, current["id"], identifier)
            for _ in range(2)
        )
    )
    if session.task is not None:
        await session.task
    await sessions.models.start(session, model_context, 0, current["id"], identifier)
    model_context.llm_generate.assert_awaited_once()
    assert len(session.model.runs) == 1


@pytest.mark.parametrize("immediate", [True, False])
async def test_model_cancel_preserves_frames_and_allows_retry(
    experiment, model_context, immediate
):
    sessions, session = experiment
    prepared = await prepare(sessions, session, model_context)
    started = asyncio.Event()

    async def slow(**kwargs):
        started.set()
        await asyncio.sleep(60)

    model_context.llm_generate.side_effect = slow
    await sessions.models.start(
        session, model_context, 0, prepared["id"], uuid.uuid4().hex
    )
    if not immediate:
        await asyncio.wait_for(started.wait(), 5)
    await sessions.cancel(session)
    assert session.model.runs[0].status == "cancelled"
    assert all(frame.path.is_file() for frame in session.pipeline.kept)
    assert session.pipeline.done and not session.task and not session.operation
    model_context.llm_generate.side_effect = None
    retried = await run(sessions, session, model_context, prepared)
    assert retried.status == "succeeded"


@pytest.mark.parametrize("failure", ["exception", "empty", "timeout"])
async def test_failure_and_timeout_are_recorded_without_losing_frames(
    experiment, model_context, failure
):
    sessions, session = experiment
    prepared = await prepare(sessions, session, model_context)
    if failure == "exception":
        model_context.llm_generate.side_effect = RuntimeError("secret-key")
    elif failure == "empty":
        model_context.llm_generate.return_value.completion_text = ""
    else:
        session.pipeline.settings = replace(
            session.pipeline.settings, tool_timeout=0.03
        )

        async def slow(**kwargs):
            await asyncio.sleep(60)

        model_context.llm_generate.side_effect = slow
    record = await run(sessions, session, model_context, prepared)
    assert record.status == "failed"
    assert record.error and "secret-key" not in record.error
    assert all(frame.path.is_file() for frame in session.pipeline.kept)


async def test_reset_and_delete_cancel_model_tasks(experiment, model_context):
    sessions, session = experiment
    prepared = await prepare(sessions, session, model_context)
    await sessions.models.start(
        session, model_context, 0, prepared["id"], uuid.uuid4().hex
    )
    await sessions.reset(session, {})
    assert not session.model.runs and not session.model.prepared
    assert not session.task and session.revision == 1
    with pytest.raises(ValueError, match="版本"):
        await sessions.models.start(
            session, model_context, 0, prepared["id"], uuid.uuid4().hex
        )


async def test_model_routes_require_auth_csrf_and_keep_records_private(
    speech_experiment, speech_context, asr_provider
):
    sessions, session = speech_experiment
    context = speech_context
    web = DebugWeb(context, sessions)
    web.register()
    app = FastAPI()
    app.state.core_lifecycle = SimpleNamespace(star_context=context)

    @app.api_route("/debug/{path:path}", methods=["GET", "POST"])
    async def endpoint(path: str, request: Request):
        return await _call_plugin_extension(
            "/astrbot_plugin_vision/debug/" + path, request, "admin"
        )

    token = jwt.encode(
        {"username": "admin", "exp": time.time() + 60},
        context.get_config()["dashboard"]["jwt_secret"],
        algorithm="HS256",
    )
    headers = {"Authorization": "Bearer " + token}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        assert (
            await client.get("/debug/model-records", params={"id": session.id})
        ).status_code == 403
        state = (await client.get("/debug/state", headers=headers)).json()["data"]
        assert [provider["id"] for provider in state["providers"]] == ["vision-test"]
        assert state["asr_providers"] == [
            {"id": "test-asr", "model": "test-speech-model"}
        ]
        assert state["asr_enabled"] is False
        body = {
            "id": session.id,
            "csrf": state["csrf"],
            "revision": session.revision,
            "provider_id": "vision-test",
            "question": "请描述颜色变化",
            "mode": "grid",
        }
        response = await client.post(
            "/debug/model-prepare", json={**body, "csrf": "wrong"}, headers=headers
        )
        assert response.status_code == 403
        response = await client.post("/debug/model-prepare", json=body, headers=headers)
        prepared = response.json()["data"]
        context.llm_generate.assert_not_awaited()
        asr_provider.get_text.assert_not_awaited()
        response = await client.post(
            "/debug/model-run",
            json={**body, "preview_id": prepared["id"], "run_id": uuid.uuid4().hex},
            headers=headers,
        )
        assert response.status_code == 200
        await session.task
        exported = (
            await client.get(
                "/debug/model-records", params={"id": session.id}, headers=headers
            )
        ).json()["data"]
        assert exported["runs"][0]["answer"]
        assert exported["runs"][0]["input"]["prompt"] == prepared["prompt"]
        assert "private-key" not in json.dumps(exported)
        asr_body = {**body, "provider_id": "test-asr", "run_id": uuid.uuid4().hex}
        assert (await client.post("/debug/model-asr", json=asr_body)).status_code == 403
        assert (
            await client.post(
                "/debug/model-asr", json={**asr_body, "csrf": "wrong"}, headers=headers
            )
        ).status_code == 403
        asr_provider.get_text.assert_not_awaited()
        assert (
            await client.post("/debug/model-asr", json=asr_body, headers=headers)
        ).status_code == 200
        assert (
            await client.post("/debug/model-asr", json=asr_body, headers=headers)
        ).status_code == 200
        if session.task is not None:
            await session.task
        asr_provider.get_text.assert_awaited_once()
        audio_state = (
            await client.get(
                "/debug/session", params={"id": session.id}, headers=headers
            )
        ).json()["data"]
        assert audio_state["model"]["asr"]["segment_count"] == 1
        assert asr_provider.get_text.return_value not in json.dumps(
            audio_state, ensure_ascii=False
        )
        prepared_audio = (
            await client.post(
                "/debug/model-prepare",
                json={
                    **body,
                    "asr_enabled": True,
                    "asr_provider_id": "test-asr",
                    "asr_run_id": asr_body["run_id"],
                },
                headers=headers,
            )
        ).json()["data"]
        assert (
            json.loads(prepared_audio["prompt"].split("\n", 1)[1])["asr"]["segments"][
                0
            ]["text"]
            == asr_provider.get_text.return_value
        )
        exported = (
            await client.get(
                "/debug/model-records", params={"id": session.id}, headers=headers
            )
        ).json()["data"]
        assert exported["asr_runs"][0]["result"]["segments"]
        assert "private-key" not in json.dumps(exported)
        session.owner = "another-admin"
        assert (
            await client.get(
                "/debug/model-records", params={"id": session.id}, headers=headers
            )
        ).status_code == 400
        assert (
            await client.get(
                "/debug/model-record",
                params={"id": session.id, "record": asr_body["run_id"]},
                headers=headers,
            )
        ).status_code == 400
    await web.close()


async def test_invalid_model_requests_never_call_provider(
    experiment, model_context, monkeypatch
):
    sessions, session = experiment
    for provider_id, question, mode in [
        ("text-only", "问题", "frames"),
        ("missing", "问题", "frames"),
        ("vision-test", " ", "frames"),
        ("vision-test", "a" * 4001, "frames"),
        ("vision-test", "问题", "unknown"),
    ]:
        with pytest.raises(ValueError):
            await sessions.models.prepare(
                session, model_context, 0, provider_id, question, mode
            )
    session.pipeline.completed -= 1
    with pytest.raises(ValueError, match="本地流水线"):
        await prepare(sessions, session, model_context)
    session.pipeline.completed += 1
    prepared = await prepare(sessions, session, model_context)
    with pytest.raises(ValueError, match="调用编号"):
        await sessions.models.start(
            session, model_context, 0, prepared["id"], "invalid"
        )
    monkeypatch.setattr("astrbot_plugin_vision.debug_models.MAX_MODEL_RUNS", 1)
    await run(sessions, session, model_context, prepared)
    with pytest.raises(ValueError, match="20 次"):
        await sessions.models.start(
            session, model_context, 0, prepared["id"], uuid.uuid4().hex
        )
    model_context.llm_generate.assert_awaited_once()


async def test_reset_cancels_model_input_preparation(
    experiment, model_context, monkeypatch
):
    sessions, session = experiment
    started = asyncio.Event()

    async def slow(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(
        "astrbot_plugin_vision.debug_models.prepare_overview_input", slow
    )
    preparing = asyncio.create_task(prepare(sessions, session, model_context))
    await asyncio.wait_for(started.wait(), 5)
    assert session.model.preparing
    await sessions.reset(session, {})
    with pytest.raises(asyncio.CancelledError):
        await preparing
    assert not session.assets and not session.model.preparing
    assert not session.model.prepared and not session.model.runs
    model_context.llm_generate.assert_not_awaited()


async def test_unload_cancels_active_model_and_cleans_test_files(
    experiment, model_context
):
    sessions, session = experiment
    prepared = await prepare(sessions, session, model_context)
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def slow(**kwargs):
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    model_context.llm_generate.side_effect = slow
    await sessions.models.start(
        session, model_context, 0, prepared["id"], uuid.uuid4().hex
    )
    await asyncio.wait_for(started.wait(), 5)
    await sessions.close()
    assert cancelled.is_set()
    assert not session.directory.exists()
    assert not sessions.sessions and not session.task
    assert session.model.runs[0].status == "cancelled"
