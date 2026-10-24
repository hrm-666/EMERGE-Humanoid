"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.Emerge/workspace"
    model: str = "openai/gpt-5.5"
    provider: Literal["auto", "openai", "anthropic"] = "auto"
    max_tokens: int = Field(default=8192, gt=0)
    context_window_tokens: int = 65_536
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tool_iterations: int = 40
    # Deprecated compatibility field: accepted from old configs but ignored at runtime.
    memory_window: int | None = Field(default=None, exclude=True)
    reasoning_effort: str | None = None  # Provider-specific reasoning effort, e.g. low / medium / high.

    @property
    def should_warn_deprecated_memory_window(self) -> bool:
        """Return True when old memoryWindow is present without contextWindowTokens."""
        return self.memory_window is not None and "context_window_tokens" not in self.model_fields_set


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ObjectLocationSubagentConfig(Base):
    """Runtime settings for the Object Location Subagent."""

    model: str | None = None
    geometry_backend: Literal["auto", "vggt", "rgbd"] = "auto"
    vggt_url: str = "ws://localhost:8001"
    sam3_url: str = "ws://localhost:8002"
    timeout: float = 120.0
    max_iterations: int = 8
    point_conf_threshold: float = 0.3
    min_points: int = 80
    bbox_padding_pixels: int = 4
    view_center_tolerance_m: float = Field(default=0.08, gt=0.0)
    ray_consensus_tolerance_m: float = Field(default=0.02, gt=0.0)
    min_consistent_views: int = Field(default=2, ge=1)


class TaskVerificationSubagentConfig(Base):
    """Runtime settings for the Task Verification Subagent."""

    model: str | None = None
    max_iterations: int = Field(default=2, ge=1)


class VisualMonitorConfig(Base):
    """Settings for the always-on visual interrupt monitor."""

    enabled: bool = True
    poll_interval_seconds: float = Field(default=0.5, gt=0.0)
    verification_timeout_seconds: float = Field(default=30.0, gt=0.0)
    confirmations: int = Field(default=1, ge=1)
    capture_interval_seconds: float = Field(default=0.1, gt=0.0)
    sample_every_n_frames: int = Field(default=3, ge=1)
    buffer_max_frames: int = Field(default=24, ge=4)


class SubagentsConfig(Base):
    """Configuration for specialist sub-agent instances."""

    object_location: ObjectLocationSubagentConfig = Field(
        default_factory=ObjectLocationSubagentConfig
    )
    task_verification: TaskVerificationSubagentConfig = Field(
        default_factory=TaskVerificationSubagentConfig
    )


class ProviderConfig(Base):
    """Connection settings. Timeout is in seconds; SDK retries are disabled."""

    model_config = ConfigDict(extra="forbid")
    api_key: str = ""
    api_base: str = Field(min_length=1)
    timeout: float = Field(default=600, gt=0)
    extra_headers: dict[str, str] | None = None


class OpenAIConfig(ProviderConfig):
    api_base: str = "https://api.openai.com/v1"
    api_mode: Literal["responses", "chat_completions"] = "responses"
    reasoning_summary: Literal["auto", "concise", "detailed"] = "auto"


class AnthropicConfig(ProviderConfig):
    api_base: str = "https://api.anthropic.com"
    api_mode: Literal["anthropic_messages"] = "anthropic_messages"
    thinking: Literal["adaptive", "disabled"] = "adaptive"


class ProvidersConfig(Base):
    """Configuration for LLM providers."""

    model_config = ConfigDict(extra="forbid")
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    timeout: int = 60
    path_append: str = ""


class ToolsConfig(Base):
    """Tools configuration."""

    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory


class Config(BaseSettings):
    """Root configuration for Emerge."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    visual_monitor: VisualMonitorConfig = Field(default_factory=VisualMonitorConfig)
    subagents: SubagentsConfig = Field(default_factory=SubagentsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded agent workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def get_provider_name(self, model: str | None = None) -> str:
        from Emerge.providers.registry import resolve_model

        name, _ = resolve_model(model or self.agents.defaults.model, self.agents.defaults.provider)
        return name

    def get_provider(self, model: str | None = None) -> ProviderConfig:
        return getattr(self.providers, self.get_provider_name(model))

    def get_api_key(self, model: str | None = None) -> str:
        return self.get_provider(model).api_key

    def get_api_base(self, model: str | None = None) -> str:
        return self.get_provider(model).api_base

    model_config = ConfigDict(env_prefix="Emerge_", env_nested_delimiter="__")
