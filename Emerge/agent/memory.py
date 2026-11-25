"""Read shared visual memory; compact conversations into their own session summaries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from Emerge.agent.reference_memory import ReferenceMemory
from Emerge.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from Emerge.providers.base import LLMProvider
    from Emerge.session.manager import Session, SessionManager


_SUMMARY_TOOL = [{
    "type": "function",
    "function": {
        "name": "save_context_summary",
        "description": "Return a compact summary for continuing this conversation.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string", "minLength": 1}},
            "required": ["summary"],
        },
    },
}]


class MemoryStore:
    """The monitor owns MEMORY.md; the Agent reads recent entries and follows image links."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"

    def read_long_term(self) -> str:
        return self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""

    def get_memory_context(self) -> str:
        """Only compact planning documents; raw actions/images are never implicit input."""
        reference = ReferenceMemory(self.memory_dir.parent)
        content = self.read_long_term()
        parts = []
        if content.startswith("# MEMORY — 当前任务"):
            parts.append(content)
        if reference.installed:
            parts.append(reference.context())
        else:
            parts.append("SPACE/EXPERIENCES 尚无总结；按当前观察执行。完整历史证据可按需查询。")
        return "\n\n".join(parts)

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(
                    part["text"] if part.get("type") == "text" else "[Image attached]"
                    for part in content
                )
            if message.get("tool_calls"):
                content += "\nTool calls: " + json.dumps(message["tool_calls"], ensure_ascii=False)
            if content:
                lines.append(f"{message['role'].upper()}: {content}")
        return "\n".join(lines)


class MemoryConsolidator:
    """Reduce old conversation context without changing the visual evidence archive."""

    def __init__(
        self, workspace: Path, provider: LLMProvider, model: str, sessions: SessionManager,
        context_window_tokens: int, build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
    ):
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._lock = asyncio.Lock()

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        channel, chat_id = session.key.split(":", 1) if ":" in session.key else (None, None)
        messages = self._build_messages(
            history=session.get_history(max_messages=0), current_message="[token-probe]",
            channel=channel, chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(self.provider, self.model, messages, self._get_tool_definitions())

    def pick_consolidation_boundary(self, session: Session, tokens_to_remove: int) -> tuple[int, int] | None:
        removed, boundary = 0, None
        for index in range(session.last_consolidated, len(session.messages)):
            message = session.messages[index]
            if index > session.last_consolidated and message["role"] == "user":
                boundary = (index, removed)
                if removed >= tokens_to_remove:
                    break
            removed += estimate_message_tokens(message)
        return boundary

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        if not session.messages or self.context_window_tokens <= 0:
            return
        async with self._lock:
            estimated, _ = self.estimate_session_prompt_tokens(session)
            if estimated < self.context_window_tokens:
                return
            target = self.context_window_tokens // 2
            for _ in range(5):
                boundary = self.pick_consolidation_boundary(session, estimated - target)
                if boundary is None:
                    return
                end = boundary[0]
                chunk = session.messages[session.last_consolidated:end]
                response = await self.provider.chat_with_retry(
                    messages=[
                        {"role": "system", "content": (
                            "Summarize this conversation for continuation. Preserve the task, decisions, "
                            "verified progress, failed attempts and unresolved questions. Keep evidence "
                            "image paths/action IDs verbatim. Distinguish historical observations from "
                            "current state. This is a session summary; do not rewrite visual MEMORY.md. "
                            "Call save_context_summary with the complete updated summary."
                        )},
                        {"role": "user", "content": (
                            f"Previous summary:\n{session.metadata.get('context_summary', '')}\n\n"
                            f"Conversation:\n{MemoryStore._format_messages(chunk)}"
                        )},
                    ],
                    tools=_SUMMARY_TOOL, model=self.model, tool_choice="required",
                )
                if (response.finish_reason == "error" or len(response.tool_calls) != 1
                        or response.tool_calls[0].name != "save_context_summary"):
                    logger.warning("Conversation compaction did not return a summary")
                    return
                summary = response.tool_calls[0].arguments.get("summary")
                if not isinstance(summary, str) or not summary.strip():
                    logger.warning("Conversation compaction returned an empty summary")
                    return
                session.metadata["context_summary"] = summary
                session.last_consolidated = end
                # Compaction invalidates old signed Claude thinking blocks.
                for message in session.messages[end:]:
                    message.pop("thinking_blocks", None)
                self.sessions.save(session)
                estimated, _ = self.estimate_session_prompt_tokens(session)
                if estimated <= target:
                    return
