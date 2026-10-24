"""Native Claude Messages adapter, including signed thinking and image inputs."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from Emerge.config.schema import AnthropicConfig
from Emerge.providers.base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
    content_parts,
    native_content,
    parse_arguments,
)
from Emerge.providers.streaming import text_sink


class AnthropicProvider(LLMProvider):
    def __init__(self, config: AnthropicConfig, default_model: str):
        super().__init__(config.api_key, config.api_base)
        self.default_model = default_model
        self.thinking = config.thinking
        self._client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.api_base.rstrip("/").removesuffix("/v1"),
            timeout=config.timeout, default_headers=config.extra_headers, max_retries=0,
        )

    def get_default_model(self) -> str:
        return self.default_model

    async def aclose(self) -> None:
        await self._client.close()

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        model: str | None = None, max_tokens: int = 8192, temperature: float | None = None,
        reasoning_effort: str | None = None, tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = model or self.default_model
        system, converted = self._messages(messages, model)
        params: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens, "messages": converted,
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            params["system"] = system
        # Claude's forced tool selection cannot be combined with extended thinking.
        forced_tool = tool_choice == "required" or isinstance(tool_choice, dict)
        if self.thinking == "adaptive" and not forced_tool:
            params["thinking"] = {"type": "adaptive"}
            if reasoning_effort:
                params["output_config"] = {"effort": reasoning_effort}
        elif temperature is not None:
            params["extra_body"] = {"temperature": temperature}
        if tools:
            params["tools"] = [
                {"name": t["function"]["name"], "description": t["function"].get("description", ""),
                 "input_schema": t["function"]["parameters"]} for t in tools
            ]
            params["tool_choice"] = (
                {"type": "tool", "name": tool_choice["function"]["name"]}
                if isinstance(tool_choice, dict) else {"type": "any" if tool_choice == "required" else tool_choice or "auto"}
            )
        sink = text_sink.get()
        if sink is None:
            result = await self._client.messages.create(**params)
        else:
            stopped = False
            async with self._client.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type in {"content_block_start", "content_block_delta"}:
                        sink.started = True
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        await sink.send(event.delta.text)
                    elif event.type == "message_stop":
                        stopped = True
                result = await stream.get_final_message()
            if not stopped:
                raise RuntimeError("Claude stream ended without message_stop")
        if result.stop_reason is None:
            raise RuntimeError("Claude response ended without a stop reason")
        blocks = [block.model_dump(mode="json", exclude_unset=True) for block in result.content]
        finish = {
            "end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls",
            "max_tokens": "length", "refusal": "content_filter", "pause_turn": "error",
        }.get(result.stop_reason, "error")
        complete = finish in {"stop", "tool_calls"}
        calls = [ToolCallRequest(b["id"], b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"] if complete else []
        if any(not isinstance(call.arguments, dict) for call in calls):
            raise ValueError("Claude tool input must be a JSON object")
        usage = result.usage
        cached = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        prompt = usage.input_tokens + cached + cache_write
        return LLMResponse(
            content="".join(b["text"] for b in blocks if b["type"] == "text") or None,
            tool_calls=calls, finish_reason=finish,
            native_items=[{"provider": "anthropic", "api_mode": "anthropic_messages", "model": model, "data": b} for b in blocks] if complete else [],
            usage={"prompt_tokens": prompt, "completion_tokens": usage.output_tokens,
                   "total_tokens": prompt + usage.output_tokens, "cached_tokens": cached,
                   "cache_creation_tokens": cache_write},
        )

    def _messages(self, messages: list[dict], model: str) -> tuple[str, list[dict]]:
        system, converted, tool_images = [], [], []
        for message in messages:
            role = message["role"]
            if role != "tool" and tool_images:
                converted[-1]["content"].extend(tool_images)
                tool_images = []
            parts = []
            for part in content_parts(message.get("content")):
                if part["type"] == "text":
                    if part["text"]:
                        parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        if not header.endswith(";base64"):
                            raise ValueError("Claude image data URLs must use base64")
                        source = {"type": "base64", "media_type": header[5:-7], "data": data}
                    else:
                        source = {"type": "url", "url": url}
                    parts.append({"type": "image", "source": source})
                else:
                    raise ValueError(f"Unsupported message content: {part['type']}")
            if role in {"system", "developer"}:
                system.extend(p["text"] for p in parts if p["type"] == "text")
                continue
            if role == "assistant":
                native = native_content(message, provider="anthropic", api_mode="anthropic_messages", model=model)
                if native:
                    parts = native
                else:
                    parts = [*message.get("thinking_blocks", []), *parts]
                    parts.extend({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                                  "input": parse_arguments(call["function"]["arguments"])}
                                 for call in message.get("tool_calls", []))
            elif role == "tool":
                tool_images.extend(p for p in parts if p["type"] == "image")
                parts = [{"type": "tool_result", "tool_use_id": message["tool_call_id"],
                          "content": "\n".join(p["text"] for p in parts if p["type"] == "text") or "(empty)"}]
                role = "user"
            if not parts:
                parts = [{"type": "text", "text": "(empty)"}]
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(parts)
            else:
                # Copy the list so merging consecutive roles never mutates saved native items.
                converted.append({"role": role, "content": list(parts)})
        if tool_images:
            converted[-1]["content"].extend(tool_images)
        return "\n\n".join(system), converted
