import asyncio
import io
import json
import time
from types import SimpleNamespace

import httpx
import jwt
import pytest
from astrbot.core.star.star_manager import PluginManager
from astrbot.dashboard.api.plugins import _call_plugin_extension
from astrbot.dashboard.services.plugin_page_service import PluginPageService
from astrbot_plugin_vision.config import Settings
from astrbot_plugin_vision.debug import DebugSessions
from astrbot_plugin_vision.debug_web import DebugWeb
from astrbot_plugin_vision.media import FFmpegMedia
from fastapi import FastAPI, Request
from PIL import Image


class Upload:
    def __init__(self, content, filename="test.mp4"):
        self.filename = filename
        self.file = io.BytesIO(content)

    async def read(self, size):
        return self.file.read(size)


@pytest.fixture
async def sessions(tmp_path):
    media = FFmpegMedia(
        Settings(overview_interval=0.5, overview_max_edge=128),
        tmp_path / "normal_frames",
        [tmp_path],
    )
    manager = DebugSessions(media, tmp_path / "debug", asyncio.Semaphore(2))
    yield manager
    await manager.close()


async def test_real_debug_steps_preview_and_isolation(sessions, sample_video):
    source_bytes = sample_video.read_bytes()
    session = await sessions.upload("admin", Upload(source_bytes, "../../test.mp4"))
    assert session.source.parent == session.directory
    assert session.filename == "test.mp4"
    with pytest.raises(ValueError, match="不存在"):
        sessions.get("other", session.id)
    with pytest.raises(ValueError):
        await sessions.image(session, "frame", 0, "full", 0)
    for step in range(7):
        await sessions.next(session, step, 0)
        task = session.task
        with pytest.raises(ValueError):
            await sessions.next(session, step, 0)
        await task
        assert session.error is None
        assert session.pipeline.completed == step + 1
        if step == 2:
            preview = await sessions.image(session, "fixed", 0, "thumb", 0)
            assert preview["data_url"].startswith("data:image/jpeg;base64,")
            assert not session.pipeline.frames
        if step == 4:
            import base64

            fingerprint = await sessions.image(session, "frame", 0, "fingerprint", 0)
            image = Image.open(
                io.BytesIO(base64.b64decode(fingerprint["data_url"].split(",")[1]))
            )
            assert image.size == (32, 18)
            source = await sessions.image(session, "frame", 0, "source", 0)
            assert (source["frame"]["width"], source["frame"]["height"]) == (320, 180)
    snapshot = sessions.snapshot(session)
    assert snapshot["done"] and not snapshot["running"]
    assert [frame["requested_sec"] for frame in snapshot["frames"]] == [
        frame.requested_sec for frame in session.pipeline.frames
    ]
    assert [frame["frame_id"] for frame in snapshot["frames"]] == [
        frame.frame_id for frame in session.pipeline.frames
    ]
    assert snapshot["result"]["total_image_bytes"] > 0
    assert "video_path" not in snapshot["info"]
    assert "video_path" not in snapshot["result"]
    assert "normal_frames" not in json.dumps(snapshot)
    await sessions.reset(session, {"overview_interval": 1, "overview_max_edge": 64})
    assert session.source.read_bytes() == source_bytes
    assert session.revision == 1 and session.pipeline.completed == 0
    assert not list(session.pipeline.media.cache_dir.iterdir())
    with pytest.raises(ValueError, match="阶段已变化"):
        await sessions.next(session, 0, 0)
    with pytest.raises(ValueError, match="上一次"):
        await sessions.image(session, "fixed", 0, "thumb", 0)
    await sessions.delete(session)
    assert not session.directory.exists()
    assert sample_video.read_bytes() == source_bytes


async def test_grid_debug_output_preview_metadata_and_reset(sessions, sample_video):
    import base64

    session = await sessions.upload("admin", Upload(sample_video.read_bytes()))
    await sessions.reset(session, {"overview_output_mode": "grid"})
    with pytest.raises(ValueError, match="尚未生成"):
        await sessions.image(session, "grid", 0, "full", session.revision)
    while not session.pipeline.done:
        await sessions.next(session, session.pipeline.completed, session.revision)
        await session.task
        assert session.error is None
    snapshot = sessions.snapshot(session)
    presentation = snapshot["result"]["presentation"]
    assert snapshot["parameters"]["overview_output_mode"] == "grid"
    assert presentation["image_count"] < presentation["frame_count"]
    assert presentation["mime_type"] == "image/png"
    assert all("mime_type" not in image for image in presentation["images"])
    assert all(
        (image["frame_width"], image["frame_height"])
        == (session.pipeline.kept[0].width, session.pipeline.kept[0].height)
        for image in presentation["images"]
    )
    assert all(
        not {"x", "y", "width", "height"}.intersection(cell)
        for image in presentation["images"]
        for cell in image["cells"]
    )
    assert snapshot["result"]["output_image_bytes"] == presentation["total_bytes"]
    preview = await sessions.image(session, "grid", 0, "full", session.revision)
    assert preview["image"]["mime_type"] == presentation["mime_type"]
    assert "frame_id" not in json.dumps(presentation)
    assert all("frame_id" in cell for cell in preview["image"]["cells"])
    assert [
        {key: value for key, value in cell.items() if key != "frame_id"}
        for cell in preview["image"]["cells"]
    ] == presentation["images"][0]["cells"]
    assert preview["image"]["frame_width"] == presentation["images"][0]["frame_width"]
    assert preview["image"]["frame_height"] == presentation["images"][0]["frame_height"]
    assert preview["data_url"].startswith("data:image/png;base64,")
    with Image.open(
        io.BytesIO(base64.b64decode(preview["data_url"].split(",")[1]))
    ) as image:
        assert image.size == (preview["image"]["width"], preview["image"]["height"])
    thumbnail = await sessions.image(session, "grid", 0, "thumb", session.revision)
    with Image.open(
        io.BytesIO(base64.b64decode(thumbnail["data_url"].split(",")[1]))
    ) as image:
        assert image.width <= 480 and image.height <= 360
    with pytest.raises(ValueError):
        await sessions.image(session, "grid", 0, "source", session.revision)
    with pytest.raises(ValueError):
        await sessions.image(session, "grid", 999, "full", session.revision)
    paths = [image.path for image in session.pipeline.result.images]
    assert all(path.is_file() for path in paths)
    await sessions.reset(session, {"overview_output_mode": "frames"})
    assert not any(path.exists() for path in paths)
    assert not list(session.pipeline.media.cache_dir.iterdir())


async def test_cancellation_keeps_current_stage_retryable(sessions):
    session = await sessions.upload("admin", Upload(b"video"))
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(60)

    session.pipeline.advance = slow
    await sessions.next(session, 0, 0)
    await started.wait()
    await sessions.cancel(session)
    assert session.task is None and session.pipeline.completed == 0
    assert "取消" in session.error
    assert not session.pipeline.media.processes


async def test_immediate_cancel_releases_task_even_before_it_starts(sessions):
    session = await sessions.upload("admin", Upload(b"video"))
    await sessions.next(session, 0, 0)
    await sessions.cancel(session)
    assert session.task is None and session.pipeline.completed == 0
    assert "取消" in session.error
    await sessions.next(session, 0, 0)
    await sessions.cancel(session)
    assert session.task is None


async def test_debug_cache_limit_preserves_existing_frames(sessions, sample_video):
    from astrbot_plugin_vision.media import VideoError

    session = await sessions.upload("admin", Upload(sample_video.read_bytes()))
    for _ in range(5):
        await session.pipeline.advance()
    paths = [frame.path for frame in session.pipeline.frames]
    media = session.pipeline.media
    media.cache_limit_bytes = sum(
        path.stat().st_size for path in media.cache_dir.iterdir()
    )
    with pytest.raises(VideoError, match="容量上限"):
        await sessions.image(session, "frame", 0, "source", 0)
    assert all(path.is_file() for path in paths)
    assert not list(media.cache_dir.glob("*.tmp.jpg"))


async def test_reset_cancels_inflight_previews(sessions, sample_video):
    session = await sessions.upload("admin", Upload(sample_video.read_bytes()))
    for _ in range(3):
        await session.pipeline.advance()
    started = asyncio.Event()

    async def slow(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)

    session.pipeline.media.extract = slow
    image = asyncio.create_task(sessions.image(session, "fixed", 0, "thumb", 0))
    await started.wait()
    await sessions.reset(session, {})
    with pytest.raises(asyncio.CancelledError):
        await image
    assert not session.assets
    assert session.revision == 1


async def test_upload_limits_cleanup_and_invalid_settings(sessions, monkeypatch):
    monkeypatch.setattr("astrbot_plugin_vision.debug.MAX_UPLOAD", 16)
    with pytest.raises(ValueError, match="256"):
        await sessions.upload("admin", Upload(b"x" * 17))
    with pytest.raises(ValueError, match="为空"):
        await sessions.upload("admin", Upload(b""))
    with pytest.raises(ValueError, match="视频文件"):
        await sessions.upload("admin", Upload(b"video", "test.m3u8"))
    assert not list(sessions.directory.iterdir())
    monkeypatch.setattr("astrbot_plugin_vision.debug.MAX_SESSIONS", 1)
    session = await sessions.upload("admin", Upload(b"video"))
    with pytest.raises(ValueError, match="调试会话"):
        await sessions.upload("admin", Upload(b"video"))
    with pytest.raises(ValueError):
        await sessions.reset(session, {"ffmpeg_path": "other"})
    with pytest.raises(ValueError):
        await sessions.reset(session, {"overview_max_edge": 99999})
    session.touched -= sessions.ttl + 1
    await sessions.prune()
    assert not sessions.sessions and not session.directory.exists()
    with pytest.raises(ValueError, match="已删除"):
        await sessions.next(session, 0, 0)


async def test_native_routes_auth_upload_and_cleanup(sessions, sample_video):
    secret = "test-debug-secret-at-least-32-characters"
    context = SimpleNamespace(
        registered_web_apis=[], get_config=lambda: {"dashboard": {"jwt_secret": secret}}
    )
    context.register_web_api = lambda *entry: context.registered_web_apis.append(entry)
    web = DebugWeb(context, sessions)
    web.register()
    context.registered_web_apis.append(("/other", object(), ["GET"], "other"))
    app = FastAPI()
    app.state.core_lifecycle = SimpleNamespace(star_context=context)

    @app.api_route("/debug/{path:path}", methods=["GET", "POST"])
    async def endpoint(path: str, request: Request):
        return await _call_plugin_extension(
            "/astrbot_plugin_vision/debug/" + path, request, "admin"
        )

    def headers(**extra):
        token = jwt.encode(
            {"username": "admin", "exp": time.time() + 60, **extra},
            secret,
            algorithm="HS256",
        )
        return {"Authorization": "Bearer " + token}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        assert (await client.get("/debug/state")).status_code == 403
        assert (
            await client.get(
                "/debug/state", headers=headers(token_type="plugin_page_asset")
            )
        ).status_code == 403
        state = (await client.get("/debug/state", headers=headers())).json()["data"]
        response = await client.post(
            "/debug/upload/wrong",
            files={"file": ("test.mp4", b"video")},
            headers=headers(),
        )
        assert response.status_code == 403 and not sessions.sessions
        response = await client.post(
            "/debug/upload/" + state["csrf"],
            files={"file": ("test.mp4", sample_video.read_bytes())},
            headers=headers(),
        )
        assert response.status_code == 200
        session = response.json()["data"]
        assert (await client.get("/debug/state", headers=headers())).json()["data"][
            "active_session_id"
        ] == session["id"]
        page_body = {
            "id": session["id"],
            "csrf": state["csrf"],
            "revision": 0,
            "writer": "a" * 32,
            "sequence": 1,
            "state": {"view": -1, "parameters": {"overview_interval": "1.5"}},
        }
        assert (
            await client.post(
                "/debug/page-state",
                json={**page_body, "csrf": "wrong"},
                headers=headers(),
            )
        ).status_code == 403
        response = await client.post(
            "/debug/page-state", json=page_body, headers=headers()
        )
        assert response.status_code == 200
        restored = (
            await client.get(
                "/debug/session", params={"id": session["id"]}, headers=headers()
            )
        ).json()["data"]
        assert restored["page_state"] == page_body["state"]
        assert restored["completed"] == 0
        body = {
            "id": session["id"],
            "csrf": state["csrf"],
            "completed": 0,
            "revision": 0,
        }
        response = await client.post("/debug/next", json=body, headers=headers())
        assert response.status_code == 200
        await sessions.sessions[session["id"]].task
        response = await client.get(
            "/debug/session", params={"id": session["id"]}, headers=headers()
        )
        assert response.json()["data"]["completed"] == 1
        assert response.headers["cache-control"] == "no-store"
        response = await client.post("/debug/next", json=body, headers=headers())
        assert response.status_code == 400
    await web.close()
    assert len(context.registered_web_apis) == 1
    assert not sessions.sessions


async def test_page_state_restores_selection_and_rejects_stale_writes(sessions):
    first = await sessions.upload("admin", Upload(b"first", "first.mp4"))
    second = await sessions.upload("admin", Upload(b"second", "second.mp4"))
    other = await sessions.upload("other", Upload(b"other", "other.mp4"))
    assert sessions.active_session("admin") == second.id
    assert sessions.active_session("other") == other.id
    writer = "a" * 32
    state = {
        "view": 7,
        "parameters": {"overview_interval": "", "scene_threshold": "0.1"},
        "model_draft": {
            "provider_id": "vision",
            "mode": "grid",
            "question": "未提交的问题",
        },
        "pair_index": 4,
        "selected_runs": ["saved-run"],
        "pages": [[first.id + "/0/gallery/最终保留帧", 2]],
        "video_time": 12.5,
    }
    sessions.save_page_state(first, 0, writer, 2, state)
    # 较早的请求即使晚到，也不能覆盖最后一次编辑。
    sessions.save_page_state(first, 0, writer, 1, {"view": 0})
    assert sessions.snapshot(first)["page_state"] == state
    assert sessions.active_session("admin") == first.id
    assert sessions.active_session("other") == other.id
    assert not second.page_state and not other.page_state
    state["model_draft"]["question"] = "外部修改"
    assert first.page_state["model_draft"]["question"] == "未提交的问题"
    await sessions.reset(first, {})
    assert not first.page_state and not first.page_writers
    with pytest.raises(ValueError, match="版本"):
        sessions.save_page_state(first, 0, writer, 3, {"view": 7})
    sessions.save_page_state(first, 1, writer, 4, {"view": -1})
    await sessions.delete(first)
    assert sessions.active_session("admin") == second.id
    second.touched -= sessions.ttl + 1
    await sessions.prune()
    assert sessions.active_session("admin") is None
    assert sessions.active_session("other") == other.id


@pytest.mark.parametrize(
    ("writer", "sequence", "state"),
    [
        ("invalid", 1, {}),
        ("a" * 32, True, {}),
        ("a" * 32, 0, {}),
        ("a" * 32, 1, []),
        ("a" * 32, 1, {"unexpected": "value"}),
        ("a" * 32, 1, {"view": float("nan")}),
        ("a" * 32, 1, {"model_draft": {"question": "中" * 9000}}),
    ],
)
async def test_invalid_page_state_is_not_saved(sessions, writer, sequence, state):
    session = await sessions.upload("admin", Upload(b"video"))
    with pytest.raises(ValueError, match="页面状态"):
        sessions.save_page_state(session, 0, writer, sequence, state)
    assert not session.page_state and not session.page_writers


async def test_native_page_discovery_and_assets(tmp_path, monkeypatch):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    service = PluginPageService(SimpleNamespace(context=None), config={})
    monkeypatch.setattr(service, "get_plugin_root_dir", lambda plugin: root)
    metadata = SimpleNamespace(name="astrbot_plugin_vision")
    assert [page.name for page in await service.discover_plugin_pages(metadata)] == [
        "pipeline"
    ]
    assert (
        PluginManager._load_plugin_i18n(str(root))["zh-CN"]["pages"]["pipeline"][
            "title"
        ]
        == "视频流水线调试台"
    )
    for name in ["index.html", "app.js", "style.css"]:
        path = await service.resolve_plugin_page_file(metadata, "pipeline", name)
        assert path == root / "pages" / "pipeline" / name
