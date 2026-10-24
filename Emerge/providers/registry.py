"""Explicit model routing for the two supported API providers."""

PROVIDERS = ("openai", "anthropic")


def resolve_model(model: str, provider: str = "auto") -> tuple[str, str]:
    """An explicit prefix wins; bare family names work with the legacy auto setting."""
    if "/" in model:
        provider, model = model.split("/", 1)
    elif provider == "auto":
        if model.startswith("claude-"):
            provider = "anthropic"
        elif model.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
            provider = "openai"
        else:
            raise ValueError("Specify openai/<model> or anthropic/<model>, or set agents.defaults.provider")
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider {provider!r}; supported providers: openai, anthropic")
    if not model.strip():
        raise ValueError("Model name cannot be empty")
    return provider, model
