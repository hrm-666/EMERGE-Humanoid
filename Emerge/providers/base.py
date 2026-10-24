"""Shared response contract and request-scoped retries for native SDK providers."""

from __future__ import annotations

import asyncio
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai
from loguru import logger


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_tool_call(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None
    thinking_blocks: list[dict] | None = None
    # Opaque items belong to this assistant turn, never to a client's mutable state.
    native_items: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float | None = None
    max_tokens: int = 8192
    reasoning_effort: str | None = None


def parse_arguments(raw: str) -> dict[str, Any]:
    arguments = json.loads(raw or "{}")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return arguments


def native_content(message: dict, *, provider: str, api_mode: str, model: str) -> list[dict]:
    return [
        item["data"] for item in message.get("native_items", [])
        if (item["provider"], item["api_mode"], item["model"]) == (provider, api_mode, model)
    ]


def content_parts(content: Any) -> list[dict]:
    """Normalize the framework's text and OpenAI-shaped image content."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, dict):
        return [content]
    return content or []


def discard_thinking(messages: list[dict]) -> None:
    """History compaction invalidates Claude's old thinking context."""
    for message in messages:
        message.pop("thinking_blocks", None)
        if "native_items" in message:
            message["native_items"] = [
                item for item in message["native_items"]
                if not (item["provider"] == "anthropic"
                        and item["data"].get("type") in {"thinking", "redacted_thinking"})
            ]


class LLMProvider(ABC):
    _SENTINEL = object()
    _MAX_RETRIES = 3

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base
        self.generation = GenerationSettings()

    @abstractmethod
    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        model: str | None = None, max_tokens: int = 8192, temperature: float | None = None,
        reasoning_effort: str | None = None, tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def get_default_model(self) -> str: ...

    async def aclose(self) -> None:
        """Release provider-owned resources after a run."""

    async def chat_with_retry(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        model: str | None = None, max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL, reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        from Emerge.providers.streaming import text_sink

        kwargs = dict(
            messages=messages, tools=tools, model=model, tool_choice=tool_choice,
            max_tokens=self.generation.max_tokens if max_tokens is self._SENTINEL else max_tokens,
            temperature=self.generation.temperature if temperature is self._SENTINEL else temperature,
            reasoning_effort=self.generation.reasoning_effort if reasoning_effort is self._SENTINEL else reasoning_effort,
        )
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                return await self.chat(**kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retryable = isinstance(exc, (TimeoutError, openai.APIConnectionError, anthropic.APIConnectionError))
                if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
                    retryable = exc.status_code in {408, 409, 429} or exc.status_code >= 500
                sink = text_sink.get()
                if not retryable or attempt == self._MAX_RETRIES or (sink and sink.started):
                    return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")
                delay = random.uniform(0.5, 1.0) * 2**attempt
                if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
                    retry_after = exc.response.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(0.0, min(float(retry_after), 30.0))
                        except ValueError:
                            pass
                logger.warning("LLM request failed ({}); retry {}/{} in {:.1f}s", type(exc).__name__, attempt + 1, self._MAX_RETRIES, delay)
                await asyncio.sleep(delay)

    async def chat_with_stream(self, *, on_delta, **kwargs) -> LLMResponse:
        from Emerge.providers.streaming import TextSink, text_sink

        token = text_sink.set(TextSink(on_delta))
        try:
            return await self.chat_with_retry(**kwargs)
        finally:
            text_sink.reset(token)
