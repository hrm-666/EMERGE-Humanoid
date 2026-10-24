"""Native OpenAI Responses and Chat Completions adapters."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from Emerge.config.schema import OpenAIConfig
from Emerge.providers.base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
    content_parts,
    native_content,
    parse_arguments,
)
from Emerge.providers.streaming import text_sink


class OpenAIProvider(LLMProvider):
    def __init__(self, config: OpenAIConfig, default_model: str):
        super().__init__(config.api_key, config.api_base)
        self.default_model = default_model
        self.api_mode = config.api_mode
        self.reasoning_summary = config.reasoning_summary
        self._client = AsyncOpenAI(
            api_key=config.api_key, base_url=config.api_base, timeout=config.timeout,
            default_headers=config.extra_headers, max_retries=0,
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
        params: dict[str, Any] = {"model": model}
        if temperature is not None:
            params["temperature"] = temperature
        sink = text_sink.get()

        if self.api_mode == "responses":
            params.update(
                input=self._responses_input(messages, model), store=False,
                include=["reasoning.encrypted_content"], max_output_tokens=max_tokens,
            )
            if reasoning_effort:
                params["reasoning"] = {"effort": reasoning_effort, "summary": self.reasoning_summary}
            if tools:
                params["tools"] = [
                    {"type": "function", **tool["function"], "strict": False} for tool in tools
                ]
                params["tool_choice"] = (
                    {"type": "function", "name": tool_choice["function"]["name"]}
                    if isinstance(tool_choice, dict) else tool_choice or "auto"
                )
            if sink is None:
                result = await self._client.responses.create(**params)
            else:
                result = None
                stream = await self._client.responses.create(**params, stream=True)
                async with stream:
                    async for event in stream:
                        if event.type in {
                            "response.output_item.added", "response.output_text.delta",
                            "response.reasoning_summary_text.delta", "response.function_call_arguments.delta",
                            "response.refusal.delta",
                        }:
                            sink.started = True
                        if event.type in {"response.output_text.delta", "response.refusal.delta"}:
                            await sink.send(event.delta)
                        elif event.type in {"response.completed", "response.incomplete"}:
                            result = event.response
                        elif event.type == "response.failed":
                            raise RuntimeError(f"Responses request failed: {event.response.error}")
                        elif event.type == "error":
                            raise RuntimeError(f"Responses stream error: {event.message}")
                if result is None:
                    raise RuntimeError("Responses stream ended without a final response")
            raw = result.model_dump(mode="json", exclude_unset=True)
        else:
            params.update(messages=self._chat_messages(messages, model), max_completion_tokens=max_tokens)
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
            if tools:
                params.update(tools=tools, tool_choice=tool_choice or "auto")
            if sink is None:
                result = await self._client.chat.completions.create(**params)
                raw = result.model_dump(mode="json", exclude_unset=True)
            else:
                text, reasoning, refusal, calls = [], [], [], {}
                finish, usage = None, {}
                stream = await self._client.chat.completions.create(
                    **params, stream=True, stream_options={"include_usage": True},
                )
                async with stream:
                    async for chunk in stream:
                        if chunk.usage is not None:
                            usage = chunk.usage.model_dump(mode="json")
                        for choice in chunk.choices:
                            sink.started = True
                            delta = choice.delta
                            finish = choice.finish_reason or finish
                            if delta.content:
                                text.append(delta.content)
                                await sink.send(delta.content)
                            if delta.refusal:
                                refusal.append(delta.refusal)
                                await sink.send(delta.refusal)
                            if thought := (delta.model_extra or {}).get("reasoning_content"):
                                reasoning.append(thought)
                            for call in delta.tool_calls or []:
                                assembled = calls.setdefault(call.index, {
                                    "id": "", "type": "function", "function": {"name": "", "arguments": ""},
                                })
                                if call.id:
                                    assembled["id"] = call.id
                                if call.function:
                                    assembled["function"]["name"] += call.function.name or ""
                                    assembled["function"]["arguments"] += call.function.arguments or ""
                if finish is None:
                    raise RuntimeError("Chat Completions stream ended without a finish reason")
                message = {"role": "assistant", "content": "".join(text) or None}
                if refusal:
                    message["refusal"] = "".join(refusal)
                if reasoning:
                    message["reasoning_content"] = "".join(reasoning)
                if calls:
                    message["tool_calls"] = [calls[index] for index in sorted(calls)]
                raw = {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}
        return self._parse(raw, model)

    def _responses_input(self, messages: list[dict], model: str) -> list[dict]:
        items, tool_images = [], []
        for message in messages:
            role = message["role"]
            if role != "tool" and tool_images:
                items.append({"role": "user", "content": tool_images})
                tool_images = []
            native = native_content(message, provider="openai", api_mode=self.api_mode, model=model)
            if role == "assistant" and native:
                items.extend(native)
                continue
            parts = []
            for part in content_parts(message.get("content")):
                if part["type"] == "text":
                    if part["text"]:
                        parts.append({"type": "input_text", "text": part["text"]})
                elif part["type"] == "image_url":
                    image = part["image_url"]
                    parts.append({"type": "input_image", "image_url": image["url"], "detail": image.get("detail", "auto")})
                else:
                    raise ValueError(f"Unsupported message content: {part['type']}")
            text = "\n".join(p["text"] for p in parts if p["type"] == "input_text")
            if role == "tool":
                items.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": text})
                tool_images.extend(p for p in parts if p["type"] == "input_image")
            elif role == "assistant":
                if text:
                    items.append({"role": "assistant", "content": text})
                for call in message.get("tool_calls", []):
                    items.append({"type": "function_call", "call_id": call["id"], **call["function"]})
            else:
                items.append({"role": role, "content": parts or [{"type": "input_text", "text": "(empty)"}]})
        if tool_images:
            items.append({"role": "user", "content": tool_images})
        return items

    def _chat_messages(self, messages: list[dict], model: str) -> list[dict]:
        items, tool_images = [], []
        for message in messages:
            role = message["role"]
            if role != "tool" and tool_images:
                items.append({"role": "user", "content": tool_images})
                tool_images = []
            native = native_content(message, provider="openai", api_mode=self.api_mode, model=model)
            if role == "assistant" and native:
                items.extend(native)
                continue
            parts = content_parts(message.get("content"))
            text = "\n".join(p["text"] for p in parts if p["type"] == "text")
            if role == "tool":
                items.append({"role": "tool", "tool_call_id": message["tool_call_id"], "content": text or "(empty)"})
                tool_images.extend(p for p in parts if p["type"] == "image_url")
            elif role == "assistant":
                item = {"role": role, "content": text or None}
                if message.get("tool_calls"):
                    item["tool_calls"] = message["tool_calls"]
                items.append(item)
            else:
                items.append({"role": role, "content": parts if any(p["type"] == "image_url" for p in parts) else text or "(empty)"})
        if tool_images:
            items.append({"role": "user", "content": tool_images})
        return items

    def _parse(self, raw: dict, model: str) -> LLMResponse:
        calls, texts, reasoning = [], [], []
        usage = raw.get("usage") or {}
        if self.api_mode == "responses":
            if raw["status"] not in {"completed", "incomplete"}:
                raise RuntimeError(f"Responses request failed: {raw.get('error') or raw['status']}")
            finish = "stop"
            if raw["status"] == "incomplete":
                reason = (raw.get("incomplete_details") or {}).get("reason")
                finish = "length" if reason == "max_output_tokens" else "content_filter" if reason == "content_filter" else "error"
            native = raw["output"] if finish == "stop" else []
            for item in raw["output"]:
                if item["type"] == "message":
                    for block in item["content"]:
                        if block["type"] == "output_text":
                            texts.append(block["text"])
                        elif block["type"] == "refusal":
                            texts.append(block["refusal"])
                            finish = "content_filter"
                elif item["type"] == "reasoning":
                    reasoning.extend(part["text"] for part in item.get("summary", []))
                elif item["type"] == "function_call" and raw["status"] == "completed":
                    calls.append(ToolCallRequest(item["call_id"], item["name"], parse_arguments(item["arguments"])))
            prompt, completion = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            input_details, output_details = usage.get("input_tokens_details") or {}, usage.get("output_tokens_details") or {}
        else:
            choice = raw["choices"][0]
            message = choice["message"]
            finish = choice.get("finish_reason") or "stop"
            native = [message] if finish in {"stop", "tool_calls"} else []
            if message.get("content") or message.get("refusal"):
                texts.append(message.get("content") or message["refusal"])
            if message.get("reasoning_content"):
                reasoning.append(message["reasoning_content"])
            if finish in {"stop", "tool_calls"}:
                for call in message.get("tool_calls") or []:
                    if not call["id"] or not call["function"]["name"]:
                        raise ValueError("Incomplete tool call")
                    calls.append(ToolCallRequest(call["id"], call["function"]["name"], parse_arguments(call["function"]["arguments"])))
            prompt, completion = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            input_details, output_details = usage.get("prompt_tokens_details") or {}, usage.get("completion_tokens_details") or {}
        if finish not in {"stop", "tool_calls"}:
            calls, native = [], []
        return LLMResponse(
            content="".join(texts) or None, tool_calls=calls,
            finish_reason="tool_calls" if calls else finish,
            reasoning_content="".join(reasoning) or None,
            native_items=[{"provider": "openai", "api_mode": self.api_mode, "model": model, "data": item} for item in native],
            usage={"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
                   "cached_tokens": input_details.get("cached_tokens") or 0,
                   "reasoning_tokens": output_details.get("reasoning_tokens") or 0},
        )
