from collections.abc import Iterable

from astrbot.core.agent.message import Message
from astrbot.core.agent.run_context import ContextWrapper


class ToolImageBuffer(list[Message]):
    """暂存本批工具图片，等核心写入完整的调用和结果后再追加。"""

    def __init__(self, context: ContextWrapper):
        super().__init__(context.messages)
        self._context = context
        self._pending: list[tuple[str, Message]] = []

    def extend(self, messages: Iterable[Message]) -> None:
        batch = list(messages)
        super().extend(batch)
        if not self._pending:
            return
        calls = {}
        for message in batch:
            if message.role == "assistant":
                for call in message.tool_calls or []:
                    data = call if isinstance(call, dict) else call.model_dump()
                    calls[data["id"]] = data["function"]["name"]
        results = {message.tool_call_id for message in batch if message.role == "tool"}
        pending_tools = {name for name, _ in self._pending}
        if calls and calls.keys() <= results and pending_tools <= set(calls.values()):
            super().extend(message for _, message in self._pending)
            self.discard()

    def discard(self) -> None:
        """结束临时缓冲；未写入工具结果的图片不进入历史。"""
        self._pending.clear()
        if self._context.messages is self:
            self._context.messages = list(self)


def defer_tool_images(
    context: ContextWrapper, tool_name: str, message: Message
) -> None:
    if not isinstance(context.messages, ToolImageBuffer):
        context.messages = ToolImageBuffer(context)
    context.messages._pending.append((tool_name, message))
