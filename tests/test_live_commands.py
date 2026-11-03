import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Plain
from astrbot.core.agent.message import ImageURLPart, Message
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.star.filter.command import CommandFilter
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.live import decode_frame
from astrbot_plugin_vision.live_reply import LiveReplyRoute
from astrbot_plugin_vision.main import VisionPlugin
from test_live import picture
from test_plugin import plugin as plugin  # 复用已有插件夹具。


class PlatformEvent(AstrMessageEvent):
    def __init__(self):
        message = AstrBotMessage()
        message.type = MessageType.FRIEND_MESSAGE
        message.session_id = "alice"
        message.sender = MessageMember(user_id="alice", nickname="Alice")
        message.self_id = "bot"
        message.message_id = "original-command"
        message.message_str = "/live 屏幕 变化 一起看"
        message.message = [Plain(text=message.message_str)]
        message.raw_message = {"command": "original"}
        super().__init__(
            message.message_str,
            message,
            PlatformMetadata(name="test", description="test", id="test"),
            "alice",
        )
        self.role = "admin"
        self.transport_client = object()
        self.sent = []

    async def send(self, message):
        self.sent.append(message)
        await super().send(message)


async def collect(generator):
    return [item async for item in generator]


@pytest.mark.parametrize(
    "mode,expected", [("提问", "demand"), ("定时", "interval"), ("变化", "change")]
)
async def test_command_starts_direct_capture_and_end_cancels(
    plugin, monkeypatch, mode, expected
):
    plugin.settings = replace(
        plugin.settings,
        live_enabled=True,
        live_presets=Settings.from_mapping(
            {
                "live": {
                    "screen": {
                        expected: (
                            {"sensitivity": "high", "min_analysis_interval": 20}
                            if expected == "change"
                            else {"observation_interval": 20}
                            if expected == "interval"
                            else {}
                        )
                    }
                }
            }
        ).live_presets,
    )
    plugin.live_sessions.settings = plugin.settings
    started = asyncio.Event()

    async def capture(sessions, session, ffmpeg):
        await sessions.add_frame(session, picture(), 0)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("astrbot_plugin_vision.main.read_capture", capture)
    event = PlatformEvent()
    results = await collect(
        plugin.live_command(event, f"屏幕 {mode} keep watching with me")
    )
    assert "已开始" in results[0].get_plain_text()
    assert f"响应模式：{mode}" in results[0].get_plain_text()
    assert started.is_set()
    session = plugin.live_sessions.for_chat(event.unified_msg_origin, "alice")
    assert session.options.focus == "keep watching with me"
    if mode == "变化":
        assert session.options.sensitivity == "high"
        assert session.options.min_analysis_interval == 20
    if mode == "定时":
        assert session.options.interval == 20
    assert session.options.mode == expected
    assert session.options.proactive is (mode != "提问")
    results = await collect(plugin.live_command(event, "end"))
    assert "已结束" in results[0].get_plain_text()
    assert session.stream_task.cancelled()
    assert session.reply_route is None


async def test_command_parser_keeps_entire_observation_instruction():
    handler = SimpleNamespace(handler=VisionPlugin.live_command)
    command = CommandFilter("live", handler_md=handler)
    params = command.validate_and_convert_params(
        ["屏幕", "提问", "keep", "watching", "with", "me"], command.handler_params
    )
    assert params["arguments"] == "屏幕 提问 keep watching with me"


@pytest.mark.parametrize("lookback,expected", [(None, [6, 9]), (0, [9]), (10, [0, 9])])
async def test_live_observe_uses_preset_window_and_allows_call_override(
    plugin, lookback, expected
):
    plugin.settings = replace(
        plugin.settings,
        overview_model_enabled=True,
        overview_provider_id="vision",
        live_presets=Settings.from_mapping(
            {"live": {"screen": {"demand": {"max_frames": 2, "lookback_seconds": 4}}}}
        ).live_presets,
    )
    manager = plugin.live_sessions
    manager.settings = plugin.settings
    manager.analyze = AsyncMock(return_value={"summary": "画面", "speak": False})
    event = PlatformEvent()
    session = await manager.create(event.unified_msg_origin, "alice")
    manager.configure(session, {"source": "screen", "mode": "demand"})
    try:
        for timestamp, color in zip(
            (0, 3, 6, 9), ((255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0))
        ):
            await manager.add_frame(session, picture(color), timestamp)
        await plugin.live_observe(event, "现在发生了什么", lookback_sec=lookback)
        assert session.observation["evidence_times"] == expected
    finally:
        await manager.close()


async def test_failed_capture_cleans_session_and_non_admin_cannot_start(
    plugin, monkeypatch
):
    plugin.settings = replace(plugin.settings, live_enabled=True)
    event = PlatformEvent()
    event.role = "member"
    results = await collect(plugin.live_command(event, "屏幕 变化 一起看"))
    assert "管理员" in results[0].get_plain_text()
    assert not plugin.live_sessions.sessions
    event.role = "admin"

    async def fail(sessions, session, ffmpeg):
        session.error = "设备无权限"
        session.ready.set()

    monkeypatch.setattr("astrbot_plugin_vision.main.read_capture", fail)
    results = await collect(plugin.live_command(event, "屏幕 变化 一起看"))
    assert "设备无权限" in results[0].get_plain_text()
    assert not plugin.live_sessions.sessions


async def test_reply_enters_normal_pipeline_with_current_conversation(plugin):
    original = PlatformEvent()
    session = await plugin.live_sessions.create(original.unified_msg_origin, "alice")
    plugin.live_sessions.configure(session, {})
    session.frames.append(decode_frame(picture(), 0, 160))
    session.observed_received = time.monotonic()
    session.observation = {"summary": "有人摔倒"}
    session.reply_route = LiveReplyRoute.from_event(original)
    queue = asyncio.Queue()
    plugin.context.get_event_queue = lambda: queue
    conversation = SimpleNamespace(history="[]")
    plugin.context.conversation_manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value="cid"),
        get_conversation=AsyncMock(return_value=conversation),
    )
    try:
        await plugin._speak_live(
            session,
            "值得吐槽",
            session.user_generation,
            image_urls=["data:image/jpeg;base64,AA=="],
        )
        payload = session.pending_reply
        event = queue.get_nowait()
        assert isinstance(event, PlatformEvent)
        assert event is not original
        assert event.transport_client is original.transport_client
        assert event.message_obj.raw_message is None
        assert event.get_extra("vision_agent_context") is None
        assert event.get_extra("enable_streaming") is False
        assert not original.get_extra("vision_live_reply")
        results = await collect(plugin.live_reply(event))
        assert results[0].conversation is conversation
        assert "值得吐槽" in results[0].prompt
        assert [
            part.image_url.url
            for part in results[0].extra_user_content_parts
            if isinstance(part, ImageURLPart)
        ] == ["data:image/jpeg;base64,AA=="]
        assert all(
            part._no_save
            for part in results[0].extra_user_content_parts
            if isinstance(part, ImageURLPart)
        )
        assert session.id not in results[0].prompt
        plugin.context.llm_generate.assert_not_awaited()
        await plugin.bind_agent_context(event, SimpleNamespace(messages=[]))
        assert session.user_busy == 0
        response = LLMResponse(role="assistant", completion_text="[SILENT]")
        await plugin.validate_live_reply_response(event, response)
        assert event.is_stopped()
        assert response.completion_text == ""
        assert session.pending_reply is None
        assert "image_urls" not in payload
        event = session.reply_route.create(event.get_extra("vision_live_reply"))
        plugin.live_sessions.user_turn(session)
        await plugin.validate_live_reply_request(event, None)
        assert event.is_stopped()
    finally:
        await plugin.live_sessions.close()


@pytest.mark.parametrize(
    "evidence_age,reply_age,valid",
    [(21, 0, True), (45, 24, True), (91, 1, False), (60, 46, False)],
)
async def test_reply_freshness_allows_model_latency(
    plugin, evidence_age, reply_age, valid
):
    original = PlatformEvent()
    session = await plugin.live_sessions.create(original.unified_msg_origin, "alice")
    plugin.live_sessions.configure(session, {})
    session.frames.append(decode_frame(picture(), 0, 160))
    now = time.monotonic()
    payload = {
        "session_id": session.id,
        "generation": session.generation,
        "user_generation": session.user_generation,
        "observed_received": now - evidence_age,
        "created_at": now - reply_age,
    }
    try:
        assert plugin._live_reply_valid(payload) is valid
        session.frames[-1] = replace(session.frames[-1], received=time.monotonic() - 11)
        assert not plugin._live_reply_valid(payload)
    finally:
        await plugin.live_sessions.close()


async def test_direct_proactive_queues_once_and_user_turn_cancels(plugin):
    original = PlatformEvent()
    session = await plugin.live_sessions.create(original.unified_msg_origin, "alice")
    plugin.live_sessions.configure(session, {"mode": "change"})
    session.frames.append(decode_frame(picture(), 0, 160))
    session.reply_route = LiveReplyRoute.from_event(original)
    queue = asyncio.Queue()
    plugin.context.get_event_queue = lambda: queue
    try:
        await plugin.live_sessions.observe(session, automatic=True)
        assert queue.qsize() == 1
        payload = session.pending_reply
        assert payload["image_urls"][0].startswith("data:image/jpeg;base64,")
        assert session.analysis_count == 1
        await plugin._speak_live(session, "不应重复入队", session.user_generation)
        assert queue.qsize() == 1
        plugin.context.llm_generate.assert_not_awaited()
        plugin.live_sessions.user_turn(session)
        assert payload["cancelled"]
        assert "image_urls" not in payload
        assert session.pending_reply is None
        event = queue.get_nowait()
        await plugin.validate_live_reply_request(event, None)
        assert event.is_stopped()
    finally:
        await plugin.live_sessions.close()


async def test_direct_proactive_main_loop_receives_history_and_frames(plugin):
    from test_plugin import PluginAgentHooks

    original = PlatformEvent()
    session = await plugin.live_sessions.create(original.unified_msg_origin, "alice")
    plugin.live_sessions.configure(session, {"mode": "change"})
    session.frames.append(decode_frame(picture(), 0, 160))
    session.reply_route = LiveReplyRoute.from_event(original)
    queue = asyncio.Queue()
    plugin.context.get_event_queue = lambda: queue
    conversation = SimpleNamespace(
        cid="current",
        history='[{"role": "user", "content": "我喜欢红色"}]',
        token_usage=0,
    )
    plugin.context.conversation_manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value="current"),
        get_conversation=AsyncMock(return_value=conversation),
    )
    provider = SimpleNamespace(
        provider_config={"id": "main", "modalities": ["image"]},
        text_chat=AsyncMock(
            return_value=LLMResponse(role="assistant", completion_text="[SILENT]")
        ),
    )
    try:
        await plugin.live_sessions.observe(session, automatic=True)
        event = queue.get_nowait()
        request = (await collect(plugin.live_reply(event)))[0]
        assert request.conversation is conversation
        # 核心准备请求时从绑定的当前会话加载历史和人设。
        request.contexts = json.loads(request.conversation.history)
        request.system_prompt = "你是陪看搭子"
        context = ContextWrapper(context=SimpleNamespace(event=event))
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider=provider,
            request=request,
            run_context=context,
            tool_executor=FunctionToolExecutor(),
            agent_hooks=PluginAgentHooks(plugin, event),
        )
        async for _ in runner.step_until_done(1):
            pass
        provider.text_chat.assert_awaited_once()
        plugin.context.llm_generate.assert_not_awaited()
        messages = [
            Message.model_validate(item)
            for item in provider.text_chat.await_args.kwargs["contexts"]
        ]
        serialized = json.dumps([m.model_dump() for m in messages], ensure_ascii=False)
        assert "我喜欢红色" in serialized
        assert "你是陪看搭子" in serialized
        assert any(
            isinstance(part, ImageURLPart)
            for message in messages
            if isinstance(message.content, list)
            for part in message.content
        )
        await plugin.validate_live_reply_response(event, runner.get_final_llm_resp())
        assert event.is_stopped()
        assert session.pending_reply is None
    finally:
        await plugin.live_sessions.close()


@pytest.mark.parametrize("evidence_age,reply_age", [(0, 0), (45, 24)])
async def test_live_reply_uses_core_decorate_and_respond_stages(
    plugin, monkeypatch, evidence_age, reply_age
):
    import copy

    from astrbot.core.config.default import DEFAULT_CONFIG
    from astrbot.core.message.message_event_result import ResultContentType
    from astrbot.core.pipeline.respond.stage import RespondStage
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
    from astrbot.core.star.star_handler import EventType

    monkeypatch.setattr(
        "astrbot.core.platform.astr_message_event.Metric.upload", AsyncMock()
    )

    original = PlatformEvent()
    session = await plugin.live_sessions.create(original.unified_msg_origin, "alice")
    plugin.live_sessions.configure(session, {})
    session.frames.append(decode_frame(picture(), 0, 160))
    session.observed_received = time.monotonic()
    session.observation = {"summary": "新的事件"}
    route = LiveReplyRoute.from_event(original)
    event = route.create(
        {
            "session_id": session.id,
            "generation": session.generation,
            "user_generation": session.user_generation,
            "observed_received": session.observed_received,
            "created_at": time.monotonic() - reply_age,
        }
    )
    event.get_extra("vision_live_reply")["observed_received"] -= evidence_age
    event.set_result(
        event.plain_result("正常评论").set_result_content_type(
            ResultContentType.LLM_RESULT
        )
    )
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["platform_settings"]["reply_prefix"] = "陪看："
    config["platform_settings"]["reply_with_mention"] = False
    config["platform_settings"]["reply_with_quote"] = False
    config["platform_settings"]["segmented_reply"]["enable"] = False
    config["provider_tts_settings"]["enable"] = False
    config["t2i"] = False
    plugin.context.get_using_tts_provider_async = AsyncMock(return_value=None)
    ctx = SimpleNamespace(
        astrbot_config=config, plugin_manager=SimpleNamespace(context=plugin.context)
    )
    monkeypatch.setattr(
        "astrbot.core.pipeline.result_decorate.stage.star_handlers_registry.get_handlers_by_event_type",
        lambda *args, **kwargs: [],
    )

    async def remember(sent_event, hook_type, *args, **kwargs):
        if hook_type == EventType.OnAfterMessageSentEvent:
            await plugin.remember_live_reply(sent_event)
        return False

    monkeypatch.setattr("astrbot.core.pipeline.respond.stage.call_event_hook", remember)
    try:
        decorate = ResultDecorateStage()
        await decorate.initialize(ctx)
        respond = RespondStage()
        await respond.initialize(ctx)
        await collect(decorate.process(event))
        await respond.process(event)
        assert event.sent[-1].get_plain_text() == "陪看：正常评论"
        assert session.recent_speech[-1] == "陪看：正常评论"
        assert event.get_result() is None
    finally:
        await plugin.live_sessions.close()


async def test_webchat_reply_uses_existing_adapter_delivery(monkeypatch):
    from astrbot.core.platform.sources.webchat.webchat_event import WebChatMessageEvent

    monkeypatch.setattr(
        "astrbot.core.platform.astr_message_event.Metric.upload", AsyncMock()
    )
    original = PlatformEvent()
    original.platform_meta.name = "webchat"
    event = WebChatMessageEvent(
        original.message_str,
        original.message_obj,
        original.platform_meta,
        original.session_id,
    )
    sender = AsyncMock()
    adapter = SimpleNamespace(
        meta=lambda: original.platform_meta, send_by_session=sender
    )
    context = SimpleNamespace(
        platform_manager=SimpleNamespace(platform_insts=[adapter])
    )
    route = LiveReplyRoute.from_event(event, context)
    reply = route.create({"session_id": "live"})
    await reply.send(reply.plain_result("通过平台原有发送方式"))
    sender.assert_awaited_once()
    assert sender.await_args.args[0] == reply.session
    assert sender.await_args.args[1].get_plain_text() == "通过平台原有发送方式"
    await reply.send(None)
    assert sender.await_count == 1
