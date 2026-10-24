"""Create native providers while preserving the Agent's existing calling interface."""

from __future__ import annotations

import asyncio

from Emerge.config.schema import Config
from Emerge.providers.base import GenerationSettings, LLMProvider, LLMResponse
from Emerge.providers.registry import resolve_model


class ConfiguredProvider(LLMProvider):
    """Route each request using its model; SDK clients contain no conversation state."""

    def __init__(self, config: Config, model: str):
        super().__init__()
        self.config = config.model_copy(deep=True)
        self.default_model = model
        self._providers: dict[str, LLMProvider] = {}
        defaults = config.agents.defaults
        self.generation = GenerationSettings(defaults.temperature, defaults.max_tokens, defaults.reasoning_effort)
        provider = config.get_provider(model)
        if not provider.api_key:
            raise ValueError(f"providers.{config.get_provider_name(model)}.api_key is empty")

    def get_default_model(self) -> str:
        return self.default_model

    async def chat(self, messages, tools=None, model=None, max_tokens=8192,
                   temperature=None, reasoning_effort=None, tool_choice=None) -> LLMResponse:
        name, model_name = resolve_model(model or self.default_model, self.config.agents.defaults.provider)
        if name not in self._providers:
            settings = getattr(self.config.providers, name)
            if not settings.api_key:
                raise ValueError(f"providers.{name}.api_key is empty")
            if name == "openai":
                from Emerge.providers.openai_provider import OpenAIProvider

                self._providers[name] = OpenAIProvider(settings, model_name)
            else:
                from Emerge.providers.anthropic_provider import AnthropicProvider

                self._providers[name] = AnthropicProvider(settings, model_name)
        return await self._providers[name].chat(
            messages=messages, tools=tools, model=model_name, max_tokens=max_tokens,
            temperature=temperature, reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )

    async def aclose(self) -> None:
        await asyncio.gather(*(provider.aclose() for provider in self._providers.values()))
        self._providers.clear()


def create_provider(config: Config, *, model: str | None = None) -> LLMProvider:
    return ConfiguredProvider(config, model or config.agents.defaults.model)
