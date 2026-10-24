"""Configuration loading utilities."""

import json
from pathlib import Path

from pydantic import ValidationError

from Emerge.config.schema import Config

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path(workspace: str | Path | None = None) -> Path:
    """Use the selected workspace's config unless an explicit config context is set."""
    if workspace is not None:
        return Path(workspace).expanduser().resolve() / "config.json"
    if _current_config_path:
        return _current_config_path
    return Path.home() / ".Emerge" / "workspace" / "config.json"


def load_config(config_path: str | Path | None = None, *, workspace: str | Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = Path(config_path).expanduser().resolve() if config_path else get_config_path(workspace)
    if config_path and not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        config = Config.model_validate(_migrate_config(data))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}") from None
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        raise ValueError(f"Invalid config {path}: {details}") from None
    if workspace is not None:
        config.agents.defaults.workspace = str(Path(workspace).expanduser().resolve())
    elif "workspace" not in config.agents.defaults.model_fields_set:
        config.agents.defaults.workspace = str(path.parent)
    return config


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = Path(config_path).expanduser() if config_path else get_config_path(config.workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(by_alias=False)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    # Old generated files included empty entries for every supported provider.
    providers = data.get("providers", {})
    if isinstance(providers, dict):
        for name, settings in list(providers.items()):
            if name not in {"openai", "anthropic"}:
                if isinstance(settings, dict) and not any(settings.values()):
                    del providers[name]
                else:
                    raise ValueError(f"Provider {name!r} was removed; configure providers.openai or providers.anthropic")
            elif isinstance(settings, dict):
                for key in ("api_base", "apiBase"):
                    if settings.get(key) is None:
                        settings.pop(key, None)
    # Drop the removed multi-workspace registry configuration so existing
    # user config files continue to load and are cleaned on the next save.
    data.pop("embodiments", None)

    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data
