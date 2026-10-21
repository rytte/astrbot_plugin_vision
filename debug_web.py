"""通过 AstrBot 原生插件桥接开放调试 API。"""

from __future__ import annotations

import functools
import hmac
import secrets
import time

import jwt
from astrbot.api import logger
from astrbot.api.web import error_response, json_response, request
from astrbot.core.workspace import API_KEY_USERNAME_PREFIX

from .debug import MAX_SESSIONS, MAX_UPLOAD, PARAMETERS, DebugSessions
from .media import VideoError


class DebugWeb:
    def __init__(self, context, sessions: DebugSessions):
        self.context = context
        self.sessions = sessions
        self.handlers: list = []
        self.nonces: dict[str, tuple[str, float]] = {}

    def register(self) -> None:
        for route, method in {
            "state": "GET",
            "session": "GET",
            "image": "GET",
            "model-record": "GET",
            "model-records": "GET",
            "model-prepare": "POST",
            "model-run": "POST",
            "upload/<csrf_token>": "POST",
            "next": "POST",
            "reset": "POST",
            "cancel": "POST",
            "delete": "POST",
        }.items():
            handler = functools.partial(self.dispatch, route.split("/")[0])
            self.handlers.append(handler)
            self.context.register_web_api(
                f"/astrbot_plugin_vision/debug/{route}",
                handler,
                [method],
                "视频理解流水线调试（WebUI 管理员）",
            )

    def _authorize(self) -> str:
        username = request.username
        if (
            self.sessions.closed
            or not isinstance(username, str)
            or not username
            or username.startswith(API_KEY_USERNAME_PREFIX)
        ):
            raise PermissionError("请通过已登录的 AstrBot WebUI 插件页面访问")
        authorization = request.headers.get("Authorization", "")
        secret = self.context.get_config().get("dashboard", {}).get("jwt_secret")
        if (
            not authorization.startswith("Bearer ")
            or not isinstance(secret, str)
            or not secret
        ):
            raise PermissionError("缺少有效的 WebUI 登录凭据")
        try:
            payload = jwt.decode(
                authorization[7:].strip(),
                secret,
                algorithms=["HS256"],
                options={"require": ["exp", "username"]},
            )
        except jwt.InvalidTokenError as exc:
            raise PermissionError("WebUI 登录已过期，请重新登录") from exc
        if payload["username"] != username or payload.get("token_type") is not None:
            raise PermissionError("需要 WebUI 管理员登录凭据")
        return username

    def _nonce(self, owner: str) -> str:
        now = time.monotonic()
        self.nonces = {
            key: value for key, value in self.nonces.items() if value[1] > now
        }
        if owner not in self.nonces:
            self.nonces[owner] = (secrets.token_urlsafe(32), now + 3600)
        return self.nonces[owner][0]

    def _check_nonce(self, owner: str, token) -> None:
        entry = self.nonces.get(owner)
        if (
            not entry
            or entry[1] <= time.monotonic()
            or not isinstance(token, str)
            or not hmac.compare_digest(token, entry[0])
        ):
            raise PermissionError("页面操作凭据已过期，请刷新页面")

    async def dispatch(self, action: str, csrf_token: str | None = None):
        try:
            owner = self._authorize()
            if action == "state":
                await self.sessions.prune()
                data = {
                    "csrf": self._nonce(owner),
                    "defaults": {
                        key: getattr(self.sessions.media.settings, key)
                        for key in PARAMETERS
                    },
                    "max_edge_limit": self.sessions.media.settings.max_edge_limit,
                    "max_upload_bytes": MAX_UPLOAD,
                    "max_sessions": MAX_SESSIONS,
                    "ttl_seconds": self.sessions.ttl,
                    "providers": self.sessions.models.providers(self.context),
                    "default_provider_id": self.sessions.media.settings.overview_provider_id,
                    "sessions": [
                        {"id": session.id, "filename": session.filename}
                        for session in self.sessions.sessions.values()
                        if session.owner == owner
                    ],
                }
            elif action in {"session", "image", "model-record", "model-records"}:
                session = self.sessions.get(owner, request.query.get("id"))
                if action == "session":
                    data = self.sessions.snapshot(session)
                elif action == "model-record":
                    data = self.sessions.models.detail(
                        session, request.query.get("record")
                    )
                elif action == "model-records":
                    data = {
                        "prepared": session.model.prepared.detail()
                        if session.model.prepared
                        else None,
                        "runs": [run.detail() for run in session.model.runs],
                    }
                else:
                    try:
                        index = int(request.query.get("index", ""))
                        revision = int(request.query.get("revision", ""))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("图片索引和测试版本必须是整数") from exc
                    data = await self.sessions.image(
                        session,
                        request.query.get("kind"),
                        index,
                        request.query.get("variant", "thumb"),
                        revision,
                        request.query.get("record"),
                    )
            elif action == "upload":
                self._check_nonce(owner, csrf_token)
                files = await request.files()
                upload = files.get("file")
                if upload is None:
                    raise ValueError("缺少视频文件")
                data = self.sessions.snapshot(await self.sessions.upload(owner, upload))
            else:
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError("请求必须是 JSON 对象")
                self._check_nonce(owner, body.get("csrf"))
                session = self.sessions.get(owner, body.get("id"))
                if action == "next":
                    await self.sessions.next(
                        session, body.get("completed"), body.get("revision")
                    )
                elif action == "reset":
                    await self.sessions.reset(session, body.get("parameters"))
                elif action == "cancel":
                    await self.sessions.cancel(session)
                elif action == "delete":
                    await self.sessions.delete(session)
                    return json_response({"status": "ok", "data": {"deleted": True}})
                elif action == "model-prepare":
                    data = await self.sessions.models.prepare(
                        session,
                        self.context,
                        body.get("revision"),
                        body.get("provider_id"),
                        body.get("question"),
                        body.get("mode"),
                    )
                    return json_response(
                        {"status": "ok", "data": data},
                        headers={"Cache-Control": "no-store"},
                    )
                elif action == "model-run":
                    await self.sessions.models.start(
                        session,
                        self.context,
                        body.get("revision"),
                        body.get("preview_id"),
                        body.get("run_id"),
                    )
                else:
                    raise ValueError("未知调试操作")
                data = self.sessions.snapshot(session)
            return json_response(
                {"status": "ok", "data": data}, headers={"Cache-Control": "no-store"}
            )
        except PermissionError as exc:
            return error_response(str(exc), status_code=403)
        except (ValueError, TimeoutError, VideoError) as exc:
            return error_response(str(exc) or "处理超时", status_code=400)
        except Exception:
            logger.exception("视频流水线调试 API 失败")
            return error_response("调试请求失败，请查看 AstrBot 日志", status_code=500)

    async def close(self) -> None:
        await self.sessions.close()
        self.context.registered_web_apis[:] = [
            entry
            for entry in self.context.registered_web_apis
            if entry[1] not in self.handlers
        ]
        self.handlers.clear()
        self.nonces.clear()
