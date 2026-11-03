"""用新的平台消息事件进入 AstrBot 原对话流水线。"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.core.platform.sources.webchat.webchat_event import WebChatMessageEvent


class LiveWebChatEvent(WebChatMessageEvent):
    async def send(self, message):
        # 新事件没有浏览器请求队列，使用平台现有主动投递能力。
        if message is None:
            return
        await self._live_sender(self.session, message)
        await AstrMessageEvent.send(self, message)


class LiveReplyFilter(filter.CustomFilter):
    def filter(self, event, cfg) -> bool:
        return event.get_extra("vision_live_reply") is not None


class LiveNoticeFilter(filter.CustomFilter):
    def filter(self, event, cfg) -> bool:
        return event.get_extra("vision_live_notice") is not None


@dataclass
class LiveReplyRoute:
    event_type: type
    platform: object
    message: object
    transport: dict
    origin: str
    role: str
    webchat_sender: object | None = None

    @classmethod
    def from_event(cls, event: AstrMessageEvent, context=None):
        message = copy.copy(event.message_obj)
        message.message = []
        message.message_str = ""
        message.raw_message = None
        base = AstrMessageEvent("", message, event.platform_meta, event.session_id)
        transport = {
            key: value for key, value in vars(event).items() if key not in vars(base)
        }
        sender = None
        if isinstance(event, WebChatMessageEvent) and context is not None:
            sender = next(
                (
                    adapter.send_by_session
                    for adapter in context.platform_manager.platform_insts
                    if adapter.meta().id == event.platform_meta.id
                ),
                None,
            )
        return cls(
            type(event),
            event.platform_meta,
            message,
            transport,
            event.unified_msg_origin,
            event.role,
            sender,
        )

    def create(self, payload: dict) -> AstrMessageEvent:
        return self._create("[实时陪看观察]", "vision_live_reply", payload)

    def create_notice(self, text: str) -> AstrMessageEvent:
        return self._create("[实时观察结束]", "vision_live_notice", text)

    def _create(self, label: str, extra: str, payload: dict | str) -> AstrMessageEvent:
        message = copy.copy(self.message)
        message.message_id = "live_" + uuid.uuid4().hex
        message.timestamp = int(time.time())
        message.message_str = label
        message.message = [Plain(text=message.message_str)]
        event = object.__new__(
            LiveWebChatEvent if self.webchat_sender else self.event_type
        )
        event.__dict__.update(self.transport)
        if self.webchat_sender:
            event._live_sender = self.webchat_sender
        AstrMessageEvent.__init__(
            event, message.message_str, message, self.platform, message.session_id
        )
        event.unified_msg_origin = self.origin
        event.role = self.role
        event.is_at_or_wake_command = True
        event.is_wake = True
        event.set_extra(extra, payload)
        event.set_extra("enable_streaming", False)
        return event
