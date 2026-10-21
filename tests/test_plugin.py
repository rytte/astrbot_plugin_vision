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
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import ImageURLPart, Message
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.star.register.star_handler import llm_tools
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.main import VisionPlugin


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
    return VisionPlugin(SimpleNamespace(llm_generate=AsyncMock()), {})


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


def test_configuration_schema_matches_defaults(tmp_path):
    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    config = AstrBotConfig(str(tmp_path / "config.json"), schema=schema)
    assert Settings.from_mapping(config) == Settings()
    assert schema["overview_provider_id"]["_special"] == "select_provider"
    assert schema["overview_provider_id"]["default"] == ""


def test_registered_tools_have_provider_compatible_schema():
    overview = llm_tools.get_func("video_overview")
    inspect = llm_tools.get_func("video_inspect")
    assert overview.parameters["properties"]["video_path"]["type"] == "string"
    assert inspect.parameters["properties"]["timestamps_sec"]["items"] == {
        "type": "number"
    }
    assert set(inspect.parameters) == {"type", "properties"}


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
    arguments = {"timestamps_sec": [0]} if tool_name == "video_inspect" else {}
    event = Event(admin=True, agent=False)
    result = await getattr(plugin, tool_name)(event, str(sample_video), **arguments)
    assert result.isError
    assert "运行上下文" in data(result)["error"]["message"]
    plugin.service.overview.assert_not_awaited()
    plugin.service.inspect.assert_not_awaited()
    assert not event.agent_context.messages
    assert not plugin._state(event)["seen"]
    assert not plugin._state(event)["overviews"]


@pytest.mark.parametrize("tool_name", ["video_overview", "video_inspect"])
async def test_real_agent_loop_receives_images_without_generic_send_prompt(
    plugin, sample_video, tool_name
):
    event = Event(admin=True, agent=False)
    arguments = {"video_path": str(sample_video)}
    if tool_name == "video_inspect":
        arguments["timestamps_sec"] = [0, 1]
    tool = FunctionTool(
        name=tool_name,
        description="查看视频帧",
        parameters=llm_tools.get_func(tool_name).parameters,
        handler=getattr(plugin, tool_name),
    )
    provider = SimpleNamespace(
        provider_config={"id": "test", "modalities": ["image", "tool_use"]},
        text_chat=AsyncMock(
            side_effect=[
                LLMResponse(
                    role="assistant",
                    tools_call_name=[tool_name],
                    tools_call_args=[arguments],
                    tools_call_ids=["call_video"],
                ),
                LLMResponse(role="assistant", completion_text="已查看视频帧"),
            ]
        ),
    )

    class Hooks(BaseAgentRunHooks):
        async def on_agent_begin(self, run_context):
            await plugin.bind_agent_context(event, run_context)

        async def on_agent_done(self, run_context, llm_response):
            await plugin.release_agent_context(event, run_context, llm_response)

    context = ContextWrapper(context=SimpleNamespace(event=event))
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=ProviderRequest(prompt="视频里发生了什么？", func_tool=ToolSet([tool])),
        run_context=context,
        tool_executor=FunctionToolExecutor(),
        agent_hooks=Hooks(),
    )
    async for _ in runner.step_until_done(2):
        pass
    assert provider.text_chat.await_count == 2
    messages = [
        Message.model_validate(message)
        for message in provider.text_chat.await_args.kwargs["contexts"]
    ]
    tool_result = next(message for message in messages if message.role == "tool")
    metadata = json.loads(tool_result.content)
    assert metadata["success"]
    image_message = next(
        message
        for message in messages
        if message.role == "user"
        and isinstance(message.content, list)
        and any(part.type == "image_url" for part in message.content)
    )
    parts = image_message.content
    images = [part for part in parts if part.type == "image_url"]
    assert len(images) == metadata["new_frame_count"] > 0
    assert [json.loads(part.text) for part in parts if part.type == "text"] == metadata[
        "frames"
    ]
    for image in images:
        assert (
            base64.b64decode(image.image_url.url.split(",", 1)[1])
            == Path(image.image_url.id).read_bytes()
        )
    serialized = json.dumps([message.model_dump() for message in messages])
    assert "Image returned and cached" not in serialized
    assert "Review the image below" not in serialized
    assert "send_message_to_user" not in serialized
    assert event.get_extra("vision_agent_context") is None


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
    images = image_parts(event)
    assert len(images) == data(result)["new_frame_count"] > 0
    for image in images:
        assert image.image_url.url.startswith("data:image/jpeg;base64,")
        assert (
            base64.b64decode(image.image_url.url.split(",", 1)[1])
            == Path(image.image_url.id).read_bytes()
        )
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
        event, str(sample_video), timestamps_sec=[0.5], max_edge=128
    )
    assert not inspect.isError
    assert data(inspect)["remaining_inspect_calls"] == 2
    assert all(content.type == "text" for content in inspect.content)
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
    assert all(Path(path).is_file() for path in call["image_urls"])
    prompt_data = json.loads(call["prompt"].split("\n", 1)[1])
    assert prompt_data["user_question"] == question
    assert prompt_data["frames"] == overview["frames"]
    assert not plugin._state(event)["seen"]
    assert plugin._state(event)["overviews"]
    assert not image_parts(event)

    inspected = await plugin.video_inspect(
        event, str(sample_video), timestamps_sec=[0], max_edge=640
    )
    assert not inspected.isError
    assert all(content.type == "text" for content in inspected.content)
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


async def test_inspect_budget_is_per_question_and_counts_failures(plugin, sample_video):
    event = Event(admin=True)
    for _ in range(3):
        result = await plugin.video_inspect(
            event, str(sample_video), timestamps_sec=[-1]
        )
        assert result.isError
    plugin.service.inspect = AsyncMock()
    rejected = await plugin.video_inspect(event, str(sample_video), timestamps_sec=[1])
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
    assert len(event.agent_context.messages) == 1
    assert len(image_parts(event)) == data(with_images[0])["new_frame_count"]


async def test_unload_cancels_running_tools(plugin, sample_video):
    started = asyncio.Event()

    async def delayed(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)

    plugin.service.inspect = delayed
    task = asyncio.create_task(
        plugin.video_inspect(Event(admin=True), str(sample_video), timestamps_sec=[1])
    )
    await started.wait()
    await plugin.terminate()
    assert task.cancelled()
    assert not plugin.tasks and not plugin.media.processes
