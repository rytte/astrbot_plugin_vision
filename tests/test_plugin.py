import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot.api import AstrBotConfig
from astrbot.api.message_components import Reply, Video
from astrbot.core.star.register.star_handler import llm_tools
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.main import VisionPlugin
from mcp.types import ImageContent


class Event:
    def __init__(self, *, admin=False, user="alice", origin="chat", messages=None):
        self.admin = admin
        self.user = user
        self.unified_msg_origin = origin
        self.messages = messages or []
        self.extras = {}

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
    assert any(isinstance(content, ImageContent) for content in result.content)
    assert data(result)["sampling"]["requested_interval_sec"] == 4
    assert data(result)["sampling"]["budget_frames"] == 200
    assert data(result)["overview_mode"] == "direct"
    plugin.context.llm_generate.assert_not_awaited()
    again = await plugin.video_overview(event, str(sample_video))
    assert not any(isinstance(content, ImageContent) for content in again.content)
    inspect = await plugin.video_inspect(
        event, str(sample_video), timestamps_sec=[0.5], max_edge=128
    )
    assert not inspect.isError
    assert data(inspect)["remaining_inspect_calls"] == 2
    assert any(isinstance(content, ImageContent) for content in inspect.content)


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

    inspected = await plugin.video_inspect(
        event, str(sample_video), timestamps_sec=[0], max_edge=640
    )
    assert not inspected.isError
    assert any(isinstance(content, ImageContent) for content in inspected.content)
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
        result
        for result in results
        if any(isinstance(content, ImageContent) for content in result.content)
    ]
    assert len(with_images) == 1


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
