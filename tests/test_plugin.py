import asyncio
import base64
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot.api import AstrBotConfig
from astrbot.api.message_components import Reply, Video
from astrbot.api.star import Context
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import (
    AssistantMessageSegment,
    ImageURLPart,
    Message,
    ToolCallMessageSegment,
)
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.provider.entities import LLMResponse, ProviderRequest, ToolCallsResult
from astrbot.core.star.register.star_handler import llm_tools
from astrbot.core.star.star import StarMetadata
from astrbot.core.star.star_manager import PluginManager
from astrbot_plugin_vision.agent_images import ToolImageBuffer
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.main import VisionPlugin
from astrbot_plugin_vision.media import VideoError
from astrbot_plugin_vision.tools import VideoInspectTool
from jsonschema import ValidationError, validate


class Event:
    def __init__(
        self, *, admin=False, user="alice", origin="chat", messages=None, agent=True
    ):
        self.admin = admin
        self.user = user
        self.unified_msg_origin = origin
        self.messages = messages or []
        self.extras = {}
        # 工具单测默认模拟已执行 on_agent_begin 的当前 Agent。
        self.agent_context = ContextWrapper(context=None)
        if agent:
            self.extras["vision_agent_context"] = self.agent_context

    def is_admin(self):
        return self.admin

    def get_sender_id(self):
        return self.user

    def get_messages(self):
        return self.messages

    def get_extra(self, name):
        return self.extras.get(name)

    def set_extra(self, name, value):
        self.extras[name] = value


class PluginAgentHooks(BaseAgentRunHooks):
    def __init__(self, plugin, event):
        self.plugin = plugin
        self.event = event

    async def on_agent_begin(self, run_context):
        await self.plugin.bind_agent_context(self.event, run_context)

    async def on_agent_done(self, run_context, llm_response):
        await self.plugin.release_agent_context(self.event, run_context, llm_response)


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "astrbot_plugin_vision.main.get_astrbot_data_path", lambda: str(tmp_path)
    )
    monkeypatch.setattr(
        "astrbot_plugin_vision.main.get_astrbot_temp_path",
        lambda: str(tmp_path / "temp"),
    )
    monkeypatch.setattr(
        "astrbot_plugin_vision.main.StarTools.get_data_dir",
        lambda name: tmp_path / "plugin",
    )
    context = SimpleNamespace(
        llm_generate=AsyncMock(), provider_manager=SimpleNamespace(llm_tools=llm_tools)
    )
    context.add_llm_tools = lambda *tools: Context.add_llm_tools(context, *tools)
    return VisionPlugin(context, {})


@pytest.fixture
def delegated_plugin(plugin):
    plugin.settings = replace(plugin.settings, overview_provider_id="configured-vision")
    plugin.context.llm_generate.return_value = SimpleNamespace(
        role="assistant", completion_text="0 秒（首帧）：红色画面。"
    )
    return plugin


def data(result):
    return json.loads(result.content[0].text)


def image_parts(event):
    return [
        part
        for message in event.agent_context.messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, ImageURLPart)
    ]


def record_tool_result(event, tool_name, result):
    """工具单测模拟核心批量写入消息；完整调用链另由真实 runner 测试覆盖。"""
    response = LLMResponse(
        role="assistant",
        tools_call_name=[tool_name],
        tools_call_args=[{}],
        tools_call_ids=["call_video"],
    )
    event.agent_context.messages.extend(
        ToolCallsResult(
            tool_calls_info=AssistantMessageSegment(
                tool_calls=response.to_openai_tool_calls_model()
            ),
            tool_calls_result=[
                ToolCallMessageSegment(
                    tool_call_id="call_video", content=result.content[0].text
                )
            ],
        ).to_openai_messages_model()
    )


def test_configuration_schema_matches_defaults(tmp_path):
    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    config = AstrBotConfig(str(tmp_path / "config.json"), schema=schema)
    assert Settings.from_mapping(config) == Settings()
    assert schema["overview_provider_id"]["_special"] == "select_provider"
    assert schema["overview_provider_id"]["default"] == ""


def test_registered_tools_have_provider_compatible_schema(plugin):
    overview = llm_tools.get_func("video_overview")
    inspect = llm_tools.get_func("video_inspect")
    assert isinstance(inspect, VideoInspectTool)
    assert inspect.handler is None
    assert overview.parameters["properties"]["video_path"]["type"] == "string"
    properties = inspect.parameters["properties"]
    assert set(properties) == {
        "video_path",
        "mode",
        "times_sec",
        "max_frames",
        "max_edge",
    }
    assert properties["mode"]["enum"] == ["range", "timestamps"]
    assert properties["times_sec"]["items"]["type"] == "number"
    assert properties["max_frames"]["type"] == ["integer", "null"]
    assert properties["max_edge"]["type"] == ["integer", "null"]
    assert set(inspect.parameters["required"]) == set(properties)
    assert inspect.parameters["additionalProperties"] is False
    assert (
        ToolSet([inspect]).openai_schema()[0]["function"]["parameters"]
        == inspect.parameters
    )
    for mode, times in (("range", [0, 11.5]), ("timestamps", [1, 5, 9, 10.5, 11.5])):
        validate(
            {
                "video_path": "video.mp4",
                "mode": mode,
                "times_sec": times,
                "max_frames": None,
                "max_edge": None,
            },
            inspect.parameters,
        )


def test_inspect_tool_belongs_to_plugin_for_loading_and_unloading(plugin, monkeypatch):
    module = "data.plugins.astrbot_plugin_vision"
    monkeypatch.setattr(VideoInspectTool, "__module__", f"{module}.tools")
    metadata = StarMetadata(module_path=f"{module}.main", star_cls=plugin)
    monkeypatch.setattr("astrbot.core.star.context.star_registry", [metadata])
    tool = VideoInspectTool(plugin)
    plugin.context.add_llm_tools(tool)
    assert tool.handler_module_path == metadata.module_path
    assert PluginManager._is_plugin_llm_tool(tool, metadata.module_path)
    assert tool.handler is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"start_sec": 0, "end_sec": 0, "timestamps_sec": [1, 5, 9, 10.5, 11.5]},
        {"start_sec": 0, "end_sec": 11.5, "timestamps_sec": []},
        {"start_sec": 0, "end_sec": 11.5, "timestamps_sec": [1, 5, 9, 10.5, 11.5]},
        {"mode": "range", "times_sec": []},
        {"mode": "auto", "times_sec": [0]},
        {"mode": "timestamps", "times_sec": [0], "start_sec": 0},
    ],
)
def test_schema_rejects_logged_mixed_arguments_and_unknown_fields(plugin, arguments):
    with pytest.raises(ValidationError):
        validate(
            {
                "video_path": "video.mp4",
                "max_frames": None,
                "max_edge": None,
                **arguments,
            },
            llm_tools.get_func("video_inspect").parameters,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"mode": "range"},
        {"mode": "range", "times_sec": []},
        {"mode": "auto", "times_sec": [0]},
        {"mode": "timestamps", "times_sec": [0], "max_frames": 0},
        {"mode": "timestamps", "times_sec": [0], "unexpected": True},
        {"start_sec": 0, "end_sec": 0, "timestamps_sec": [1, 5, 9, 10.5, 11.5]},
        {"start_sec": 0, "end_sec": 11.5, "timestamps_sec": []},
        {"start_sec": 0, "end_sec": 11.5, "timestamps_sec": [1, 5, 9, 10.5, 11.5]},
    ],
)
async def test_native_executor_rejects_invalid_arguments_before_processing(
    plugin, arguments
):
    event = Event(admin=True)
    plugin.service.inspect = AsyncMock()
    tool = llm_tools.get_func("video_inspect")
    tool_arguments = {
        "video_path": "video.mp4",
        "max_frames": None,
        "max_edge": None,
        **arguments,
    }
    results = [
        result
        async for result in FunctionToolExecutor.execute(
            tool,
            ContextWrapper(context=SimpleNamespace(event=event)),
            **tool_arguments,
        )
    ]
    assert len(results) == 1
    result = results[0]
    assert result.isError
    assert data(result)["error"]["code"] == "INVALID_ARGUMENT"
    assert data(result)["error"]["examples"]
    assert data(result)["remaining_inspect_calls"] == 3
    assert plugin._state(event)["inspect_calls"] == 0
    plugin.service.inspect.assert_not_awaited()
    assert not image_parts(event)


async def test_agent_context_is_bound_per_event_and_released(plugin):
    event = Event(agent=False)
    other_event = Event(agent=False)
    context = ContextWrapper(context=None)
    other_context = ContextWrapper(context=None)
    await plugin.bind_agent_context(event, context)
    await plugin.bind_agent_context(other_event, other_context)
    assert event.get_extra("vision_agent_context") is context
    assert other_event.get_extra("vision_agent_context") is other_context
    await plugin.release_agent_context(event, other_context, None)
    assert event.get_extra("vision_agent_context") is context
    await plugin.release_agent_context(event, context, None)
    assert event.get_extra("vision_agent_context") is None
    assert other_event.get_extra("vision_agent_context") is other_context


@pytest.mark.parametrize("tool_name", ["video_overview", "video_inspect"])
async def test_missing_agent_context_fails_before_extracting_frames(
    plugin, sample_video, tool_name
):
    plugin.service.overview = AsyncMock()
    plugin.service.inspect = AsyncMock()
    arguments = (
        {"mode": "timestamps", "times_sec": [0]} if tool_name == "video_inspect" else {}
    )
    event = Event(admin=True, agent=False)
    result = await getattr(plugin, tool_name)(event, str(sample_video), **arguments)
    assert result.isError
    assert "运行上下文" in data(result)["error"]["message"]
    plugin.service.overview.assert_not_awaited()
    plugin.service.inspect.assert_not_awaited()
    assert not event.agent_context.messages
    assert not plugin._state(event)["seen"]
    assert not plugin._state(event)["overviews"]


@pytest.mark.parametrize(
    "tool_name,mode,invalid_first",
    [
        ("video_overview", None, False),
        ("video_inspect", "range", False),
        ("video_inspect", "timestamps", False),
        ("video_inspect", "timestamps", True),
    ],
)
async def test_real_agent_loop_receives_images_without_generic_send_prompt(
    plugin, sample_video, tool_name, mode, invalid_first
):
    event = Event(admin=True, agent=False)
    arguments = {"video_path": str(sample_video)}
    if tool_name == "video_inspect":
        arguments.update(mode=mode, times_sec=[0, 1], max_frames=None, max_edge=None)
    tool = llm_tools.get_func(tool_name)
    if tool_name == "video_overview":
        tool = FunctionTool(
            name=tool_name,
            description=tool.description,
            parameters=tool.parameters,
            handler=plugin.video_overview,
        )
    responses = [
        LLMResponse(
            role="assistant",
            tools_call_name=[tool_name],
            tools_call_args=[arguments],
            tools_call_ids=["call_video"],
        ),
        LLMResponse(role="assistant", completion_text="已查看视频帧"),
    ]
    if invalid_first:
        responses.insert(
            0,
            LLMResponse(
                role="assistant",
                tools_call_name=[tool_name],
                tools_call_args=[{**arguments, "times_sec": []}],
                tools_call_ids=["call_invalid_video"],
            ),
        )
    provider = SimpleNamespace(
        provider_config={"id": "test", "modalities": ["image", "tool_use"]},
        text_chat=AsyncMock(side_effect=responses),
    )

    context = ContextWrapper(context=SimpleNamespace(event=event))
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=ProviderRequest(prompt="视频里发生了什么？", func_tool=ToolSet([tool])),
        run_context=context,
        tool_executor=FunctionToolExecutor(),
        agent_hooks=PluginAgentHooks(plugin, event),
    )
    async for _ in runner.step_until_done(len(responses)):
        pass
    assert provider.text_chat.await_count == len(responses)
    messages = [
        Message.model_validate(message)
        for message in provider.text_chat.await_args.kwargs["contexts"]
    ]
    results = [
        json.loads(message.content) for message in messages if message.role == "tool"
    ]
    if invalid_first:
        assert results[0]["error"]["code"] == "INVALID_ARGUMENT"
        assert results[0]["remaining_inspect_calls"] == 3
        assert results[0]["error"]["examples"]
    metadata = results[-1]
    assert metadata["success"]
    if tool_name == "video_inspect":
        assert metadata["remaining_inspect_calls"] == 2
    image_message = next(
        message
        for message in messages
        if message.role == "user"
        and isinstance(message.content, list)
        and any(part.type == "image_url" for part in message.content)
    )
    image_index = messages.index(image_message)
    assert messages[image_index - 1].role == "tool"
    assert messages[image_index - 1].tool_call_id == "call_video"
    assert messages[0].role == "user"
    assert all(part.type == "text" for part in messages[0].content)
    assert "".join(part.text for part in messages[0].content) == "视频里发生了什么？"
    assert type(context.messages) is list
    parts = image_message.content
    images = [part for part in parts if part.type == "image_url"]
    assert len(images) == metadata["new_frame_count"] > 0
    assert [json.loads(part.text) for part in parts if part.type == "text"] == metadata[
        "frames"
    ]
    for image in images:
        assert base64.b64decode(
            image.image_url.url.split(",", 1)[1]
        ) == await asyncio.to_thread(Path(image.image_url.id).read_bytes)
    serialized = json.dumps([message.model_dump() for message in messages])
    assert "Image returned and cached" not in serialized
    assert "Review the image below" not in serialized
    assert "send_message_to_user" not in serialized
    assert event.get_extra("vision_agent_context") is None

    conversation_manager = SimpleNamespace(update_conversation=AsyncMock())
    await InternalAgentSubStage._save_to_history(
        SimpleNamespace(conv_manager=conversation_manager),
        event,
        ProviderRequest(
            conversation=SimpleNamespace(cid="test-conversation"),
            tool_calls_result=runner.req.tool_calls_result,
        ),
        runner.get_final_llm_resp(),
        context.messages,
        None,
    )
    history = conversation_manager.update_conversation.await_args.kwargs["history"]
    assert [message["role"] for message in history] == [
        *[message.role for message in messages],
        "assistant",
    ]
    assert history[image_index - 1]["tool_call_id"] == "call_video"
    assert history[image_index]["content"] == image_message.model_dump()["content"]


@pytest.mark.parametrize("batch_size", [1, 2])
async def test_real_agent_loop_orders_images_across_batches_and_context_rebuilds(
    plugin, sample_video, batch_size
):
    plugin.settings = replace(plugin.settings, inspect_max_calls=4)
    event = Event(admin=True, agent=False)
    responses = [
        LLMResponse(
            role="assistant",
            tools_call_name=["video_inspect"] * batch_size,
            tools_call_args=[
                {
                    "video_path": str(sample_video),
                    "mode": "timestamps",
                    "times_sec": [(round_index * batch_size + index) * 0.8],
                    "max_frames": None,
                    "max_edge": 128,
                }
                for index in range(batch_size)
            ],
            tools_call_ids=[
                f"call_{round_index}_{index}" for index in range(batch_size)
            ],
        )
        for round_index in range(2)
    ]
    responses.append(LLMResponse(role="assistant", completion_text="已核实"))
    provider = SimpleNamespace(
        provider_config={"id": "test", "modalities": ["image", "tool_use"]},
        text_chat=AsyncMock(side_effect=responses),
    )
    context = ContextWrapper(context=SimpleNamespace(event=event))
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=ProviderRequest(
            prompt="请分两轮核实视频",
            func_tool=ToolSet([llm_tools.get_func("video_inspect")]),
        ),
        run_context=context,
        tool_executor=FunctionToolExecutor(),
        agent_hooks=PluginAgentHooks(plugin, event),
    )
    # 上下文处理可以重建列表；缓冲必须在每批抽帧时重新绑定到当前列表。
    runner.request_context_manager.process = AsyncMock(
        side_effect=lambda messages, **kwargs: list(messages)
    )
    async for _ in runner.step_until_done(3):
        pass
    assert provider.text_chat.await_count == 3
    messages = context.messages
    expected_roles = ["user"]
    offset = 1
    for round_index in range(2):
        expected_roles.extend(
            ["assistant", *["tool"] * batch_size, *["user"] * batch_size]
        )
        results = messages[offset + 1 : offset + 1 + batch_size]
        pictures = messages[offset + 1 + batch_size : offset + 1 + 2 * batch_size]
        assert [message.tool_call_id for message in results] == [
            f"call_{round_index}_{index}" for index in range(batch_size)
        ]
        for result, picture in zip(results, pictures):
            metadata = json.loads(result.content)
            assert metadata["success"]
            assert [
                json.loads(part.text) for part in picture.content if part.type == "text"
            ] == metadata["frames"]
            assert sum(part.type == "image_url" for part in picture.content) == 1
        offset += 1 + 2 * batch_size
    assert [message.role for message in messages] == [*expected_roles, "assistant"]
    provider_messages = provider.text_chat.await_args.kwargs["contexts"]
    assert [
        Message.model_validate(message).role for message in provider_messages
    ] == expected_roles
    assert type(context.messages) is list


async def test_aborted_tool_batch_discards_unrecorded_images(plugin, sample_video):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_tool(event):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    event = Event(admin=True, agent=False)
    provider = SimpleNamespace(
        provider_config={"id": "test", "modalities": ["image", "tool_use"]},
        text_chat=AsyncMock(
            return_value=LLMResponse(
                role="assistant",
                tools_call_name=["video_inspect", "slow_tool"],
                tools_call_args=[
                    {
                        "video_path": str(sample_video),
                        "mode": "timestamps",
                        "times_sec": [0],
                        "max_frames": None,
                        "max_edge": None,
                    },
                    {},
                ],
                tools_call_ids=["call_video", "call_slow"],
            )
        ),
    )
    context = ContextWrapper(context=SimpleNamespace(event=event))
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=ProviderRequest(
            prompt="看看视频",
            func_tool=ToolSet(
                [
                    llm_tools.get_func("video_inspect"),
                    FunctionTool(
                        name="slow_tool",
                        description="等待测试中断",
                        parameters={"type": "object", "properties": {}},
                        handler=slow_tool,
                    ),
                ]
            ),
        ),
        run_context=context,
        tool_executor=FunctionToolExecutor(),
        agent_hooks=PluginAgentHooks(plugin, event),
    )

    async def drive():
        async for _ in runner.step_until_done(3):
            pass

    task = asyncio.create_task(drive())
    try:
        await asyncio.wait_for(started.wait(), 10)
        assert isinstance(context.messages, ToolImageBuffer)
        assert len(context.messages) == 1
        runner.request_stop()
        await asyncio.wait_for(task, 10)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert runner.was_aborted()
    assert cancelled.is_set()
    assert type(context.messages) is list
    assert event.get_extra("vision_agent_context") is None
    assert not any(
        part.type == "image_url"
        for message in context.messages
        if isinstance(message.content, list)
        for part in message.content
    )


async def test_video_attachment_is_exposed_and_authorized(plugin, sample_video):
    event = Event(messages=[Video(file=str(sample_video))])
    request = SimpleNamespace(prompt="视频里发生了什么？")
    await plugin.prepare_video_resources(event, request)
    assert json.dumps(str(sample_video), ensure_ascii=False) in request.prompt
    assert plugin._authorize(event, str(sample_video)) == sample_video
    first = request.prompt
    await plugin.prepare_video_resources(event, request)
    assert request.prompt == first
    assert plugin._state(event)["question"] == "视频里发生了什么？"


async def test_cached_video_is_not_repeated_in_later_prompts(plugin, sample_video):
    attachment = Event(messages=[Video(file=str(sample_video))])
    await plugin.prepare_video_resources(attachment, SimpleNamespace(prompt="看看视频"))

    for prompt in (
        "两分钟后发下今日小猪",
        "抽到了什么",
        "我收到了，我只是想看看你看不看得到",
        "也有可能是定时任务的原因",
        "你再发一次",
        "好嘟",
        "那段视频的开头说了什么",
    ):
        event = Event()
        request = SimpleNamespace(prompt=prompt)
        await plugin.prepare_video_resources(event, request)
        assert request.prompt == prompt
        assert not plugin._state(event)["attachment_paths"]
        assert plugin._authorize(event, str(sample_video)) == sample_video

    result = await plugin.video_inspect(
        event, str(sample_video), mode="timestamps", times_sec=[0]
    )
    assert not result.isError


async def test_new_video_prompt_does_not_include_cached_old_video(plugin, sample_video):
    old_event = Event(messages=[Video(file=str(sample_video))])
    await plugin.prepare_video_resources(old_event, SimpleNamespace(prompt="旧视频"))
    new_video = sample_video.with_name("new video.mp4")
    await asyncio.to_thread(
        new_video.write_bytes, await asyncio.to_thread(sample_video.read_bytes)
    )

    event = Event(messages=[Video(file=str(new_video))])
    request = SimpleNamespace(prompt="新视频")
    await plugin.prepare_video_resources(event, request)
    resources = json.loads(request.prompt.split("\n")[3])
    assert resources["video_paths"] == [str(new_video)]
    assert plugin._authorize(event, str(sample_video)) == sample_video
    assert plugin._authorize(event, str(new_video)) == new_video


async def test_quoted_video_attachment_is_supported(plugin, sample_video):
    event = Event(messages=[Reply(id="1", chain=[Video(file=str(sample_video))])])
    request = SimpleNamespace(prompt="回看这个视频")
    await plugin.prepare_video_resources(event, request)
    assert json.dumps(str(sample_video), ensure_ascii=False) in request.prompt


async def test_early_hook_materializes_video_once(plugin, sample_video, monkeypatch):
    original = Video.convert_to_file_path
    downloads = []

    async def resolve(component):
        if component.file.startswith("https:"):
            downloads.append(component.file)
            return str(sample_video)
        return await original(component)

    monkeypatch.setattr(Video, "convert_to_file_path", resolve)
    video = Video(file="https://example.com/video.mp4")
    event = Event(messages=[video])
    await plugin.capture_video_attachments(event)
    assert await video.convert_to_file_path() == str(sample_video)
    await plugin.prepare_video_resources(event, SimpleNamespace(prompt=""))
    assert len(downloads) == 1


async def test_tools_return_real_images_and_do_not_duplicate_overview(
    plugin, sample_video
):
    event = Event(admin=True)
    result = await plugin.video_overview(event, str(sample_video))
    assert not result.isError
    assert all(content.type == "text" for content in result.content)
    assert not image_parts(event)
    record_tool_result(event, "video_overview", result)
    images = image_parts(event)
    assert len(images) == data(result)["new_frame_count"] > 0
    for image in images:
        assert image.image_url.url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(
            image.image_url.url.split(",", 1)[1]
        ) == await asyncio.to_thread(Path(image.image_url.id).read_bytes)
    assert [
        json.loads(part.text)
        for part in event.agent_context.messages[-1].content
        if part.type == "text"
    ] == data(result)["frames"]
    assert data(result)["sampling"]["requested_interval_sec"] == 4
    assert data(result)["sampling"]["budget_frames"] == 200
    assert data(result)["overview_mode"] == "direct"
    plugin.context.llm_generate.assert_not_awaited()
    again = await plugin.video_overview(event, str(sample_video))
    assert all(content.type == "text" for content in again.content)
    assert image_parts(event) == images
    inspect = await plugin.video_inspect(
        event, str(sample_video), mode="timestamps", times_sec=[0.5], max_edge=128
    )
    assert not inspect.isError
    assert data(inspect)["remaining_inspect_calls"] == 2
    assert all(content.type == "text" for content in inspect.content)
    record_tool_result(event, "video_inspect", inspect)
    assert len(image_parts(event)) == len(images) + data(inspect)["new_frame_count"]


async def test_selected_model_receives_frames_and_main_model_only_receives_text(
    delegated_plugin, sample_video
):
    plugin = delegated_plugin
    event = Event(messages=[Video(file=str(sample_video))])
    question = "请观察画面颜色的变化"
    await plugin.prepare_video_resources(event, SimpleNamespace(prompt=question))
    result = await plugin.video_overview(event, str(sample_video))
    assert not result.isError
    assert all(content.type == "text" for content in result.content)
    overview = data(result)
    assert overview["overview_mode"] == "delegated"
    assert overview["overview_provider_id"] == "configured-vision"
    assert overview["overview_text"] == "0 秒（首帧）：红色画面。"
    assert overview["new_frame_count"] == 0
    assert overview["submitted_frame_count"] == len(overview["frames"]) > 0
    assert overview["remaining_inspect_calls"] == 3
    call = plugin.context.llm_generate.await_args.kwargs
    assert call["chat_provider_id"] == "configured-vision"
    assert call["contexts"] == [] and call["tools"] is None
    assert len(call["image_urls"]) == len(overview["frames"])
    for path in call["image_urls"]:
        assert await asyncio.to_thread(Path(path).is_file)
    prompt_data = json.loads(call["prompt"].split("\n", 1)[1])
    assert prompt_data["user_question"] == question
    assert prompt_data["frames"] == overview["frames"]
    assert not plugin._state(event)["seen"]
    assert plugin._state(event)["overviews"]
    assert not image_parts(event)

    inspected = await plugin.video_inspect(
        event, str(sample_video), mode="timestamps", times_sec=[0], max_edge=640
    )
    assert not inspected.isError
    assert all(content.type == "text" for content in inspected.content)
    record_tool_result(event, "video_inspect", inspected)
    assert len(image_parts(event)) == data(inspected)["new_frame_count"] > 0
    assert data(inspected)["frames"][0]["frame_id"] == overview["frames"][0]["frame_id"]
    plugin.context.llm_generate.assert_awaited_once()


async def test_parallel_delegated_overview_calls_selected_model_once(
    delegated_plugin, sample_video
):
    plugin = delegated_plugin
    event = Event(admin=True)
    results = await asyncio.gather(
        *(plugin.video_overview(event, str(sample_video)) for _ in range(2))
    )
    assert all(not result.isError for result in results)
    assert all(
        content.type == "text" for result in results for content in result.content
    )
    plugin.context.llm_generate.assert_awaited_once()


@pytest.mark.parametrize(
    "response",
    [
        RuntimeError("private API detail"),
        None,
        SimpleNamespace(role="assistant", completion_text="  "),
        SimpleNamespace(role="err", completion_text="upstream error"),
        SimpleNamespace(role="assistant", completion_text=None),
    ],
)
async def test_delegated_failure_never_falls_back_and_allows_retry(
    delegated_plugin, sample_video, response
):
    plugin = delegated_plugin
    if isinstance(response, Exception):
        plugin.context.llm_generate.side_effect = response
    else:
        plugin.context.llm_generate.return_value = response
    event = Event(admin=True)
    result = await plugin.video_overview(event, str(sample_video))
    assert result.isError
    assert all(content.type == "text" for content in result.content)
    error = data(result)["error"]
    assert error["code"] == "VIDEO_ERROR"
    assert "不会自动改为主模型看图" in error["message"]
    assert "private API detail" not in error["message"]
    assert not plugin._state(event)["seen"]
    assert not plugin._state(event)["overviews"]
    plugin.context.llm_generate.side_effect = None
    plugin.context.llm_generate.return_value = SimpleNamespace(
        role="assistant", completion_text="重试成功"
    )
    retry = await plugin.video_overview(event, str(sample_video))
    assert not retry.isError
    assert data(retry)["overview_text"] == "重试成功"


async def test_delegated_model_shares_tool_timeout(
    delegated_plugin, sample_video, monkeypatch
):
    plugin = delegated_plugin
    overview = await plugin.service.overview(str(sample_video))
    cancelled = asyncio.Event()

    async def delayed_overview(*args):
        await asyncio.sleep(0.05)
        return overview

    async def delayed_model(**kwargs):
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    plugin.settings = replace(plugin.settings, tool_timeout=0.2)
    plugin.service.overview = delayed_overview
    plugin.context.llm_generate.side_effect = delayed_model
    wait_for = AsyncMock(wraps=asyncio.wait_for)
    monkeypatch.setattr("astrbot_plugin_vision.main.asyncio.wait_for", wait_for)
    event = Event(admin=True)
    result = await plugin.video_overview(event, str(sample_video))
    assert result.isError
    assert data(result)["error"]["code"] == "TIMEOUT"
    assert wait_for.await_count == 2
    assert wait_for.await_args_list[0].args[1] == 0.2
    assert 0 < wait_for.await_args_list[1].args[1] < 0.18
    assert cancelled.is_set()
    assert not plugin._state(event)["overviews"]
    assert not plugin._state(event)["seen"]


@pytest.mark.parametrize("unload", [False, True])
async def test_cancellation_stops_delegated_model(
    delegated_plugin, sample_video, unload
):
    plugin = delegated_plugin
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def delayed_model(**kwargs):
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    plugin.context.llm_generate.side_effect = delayed_model
    event = Event(admin=True)
    task = asyncio.create_task(plugin.video_overview(event, str(sample_video)))
    await asyncio.wait_for(started.wait(), 10)
    if unload:
        await plugin.terminate()
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert task.cancelled()
    assert cancelled.is_set()
    assert not plugin.tasks
    assert not plugin._state(event)["overviews"]
    assert not plugin._state(event)["seen"]


async def test_other_user_and_other_chat_cannot_access_attachment(plugin, sample_video):
    event = Event(messages=[Video(file=str(sample_video))])
    await plugin.prepare_video_resources(event, SimpleNamespace(prompt=""))
    for outsider in (Event(user="bob"), Event(origin="other-chat")):
        result = await plugin.video_overview(outsider, str(sample_video))
        assert result.isError
        assert data(result)["error"]["code"] == "PERMISSION_DENIED"


async def test_replaced_or_expired_attachment_loses_authorization(plugin, sample_video):
    owner = plugin._owner(Event())
    stat = sample_video.stat()
    plugin.resources[owner] = {
        str(sample_video): (stat.st_size + 1, stat.st_mtime_ns, time.monotonic() + 60)
    }
    result = await plugin.video_overview(Event(), str(sample_video))
    assert result.isError
    plugin.resources[owner][str(sample_video)] = (
        stat.st_size,
        stat.st_mtime_ns,
        time.monotonic() - 1,
    )
    result = await plugin.video_overview(Event(), str(sample_video))
    assert result.isError


async def test_invalid_inspect_arguments_preserve_budget_and_examples_work(
    plugin, sample_video
):
    event = Event(admin=True)
    for mode, times in (
        (None, None),
        ("range", [0]),
        ("range", [1, 0]),
        ("timestamps", []),
        ("timestamps", [-1]),
        ("timestamps", [99]),
    ):
        result = await plugin.video_inspect(
            event, str(sample_video), mode=mode, times_sec=times
        )
        assert result.isError
        assert data(result)["error"]["code"] == "INVALID_ARGUMENT"
        assert data(result)["remaining_inspect_calls"] == 3
        assert not image_parts(event)
    for example in data(result)["error"]["examples"]:
        validate(example, llm_tools.get_func("video_inspect").parameters)
        corrected = await plugin.video_inspect(event, **example)
        assert not corrected.isError
    assert plugin._state(event)["inspect_calls"] == 2


async def test_inspect_budget_is_per_question_and_counts_processing_failures(
    plugin, sample_video
):
    event = Event(admin=True)
    plugin.service.inspect = AsyncMock(side_effect=VideoError("抽帧失败"))
    for remaining in (2, 1, 0):
        result = await plugin.video_inspect(
            event, str(sample_video), mode="timestamps", times_sec=[0]
        )
        assert result.isError
        assert data(result)["remaining_inspect_calls"] == remaining
    plugin.service.inspect = AsyncMock()
    rejected = await plugin.video_inspect(
        event, str(sample_video), mode="timestamps", times_sec=[1]
    )
    assert rejected.isError
    assert "用尽" in data(rejected)["error"]["message"]
    plugin.service.inspect.assert_not_called()
    assert plugin._state(Event(admin=True))["inspect_calls"] == 0


async def test_missing_dependencies_do_not_prevent_plugin_loading(plugin):
    plugin.media.ffmpeg = ""
    plugin.media.ffprobe = ""
    await plugin.initialize()
    await plugin.terminate()
    assert not plugin.tasks and not plugin.media.processes


async def test_parallel_overview_does_not_send_duplicate_images(plugin, sample_video):
    event = Event(admin=True)
    results = await asyncio.gather(
        *(plugin.video_overview(event, str(sample_video)) for _ in range(2))
    )
    with_images = [
        result for result in results if result.content[0].text.startswith("{")
    ]
    assert len(with_images) == 1
    assert not event.agent_context.messages
    record_tool_result(event, "video_overview", with_images[0])
    assert [message.role for message in event.agent_context.messages] == [
        "assistant",
        "tool",
        "user",
    ]
    assert len(image_parts(event)) == data(with_images[0])["new_frame_count"]


async def test_unload_cancels_running_tools(plugin, sample_video):
    started = asyncio.Event()

    async def delayed(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)

    plugin.service.inspect = delayed
    task = asyncio.create_task(
        plugin.video_inspect(
            Event(admin=True), str(sample_video), mode="timestamps", times_sec=[1]
        )
    )
    await started.wait()
    await plugin.terminate()
    assert task.cancelled()
    assert not plugin.tasks and not plugin.media.processes
