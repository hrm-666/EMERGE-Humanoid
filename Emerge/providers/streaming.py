"""Request-scoped text delivery; tool calls execute only after a complete response."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class TextSink:
    callback: Callable[[str], Awaitable[None]]
    started: bool = False

    async def send(self, text: str) -> None:
        if text:
            self.started = True
            await self.callback(text)


text_sink: ContextVar[TextSink | None] = ContextVar("emerge_text_sink", default=None)
