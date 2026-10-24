"""Workspace configuration commands; never executes agent turns."""

from pathlib import Path

import typer
from rich.console import Console

from Emerge import __logo__
from Emerge.config.loader import get_config_path, load_config, save_config
from Emerge.config.schema import Config
from Emerge.providers.registry import PROVIDERS
from Emerge.utils.helpers import sync_workspace_templates

app = typer.Typer(help="Workspace and model configuration")
console = Console()


@app.command()
def onboard(
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    config: Path | None = typer.Option(None, "--config", "-c"),
):
    """Initialize config.json and templates in the selected workspace."""
    path = config.expanduser().resolve() if config else get_config_path(workspace)
    if path.exists():
        console.print(f"[yellow]Config already exists at {path}[/yellow]")
        if typer.confirm("Overwrite with defaults?", default=False):
            settings = Config()
            settings.agents.defaults.workspace = str(workspace.expanduser().resolve() if workspace else path.parent)
        else:
            settings = load_config(path, workspace=workspace)
    else:
        settings = Config()
        settings.agents.defaults.workspace = str(workspace.expanduser().resolve() if workspace else path.parent)
    save_config(settings, path)
    settings.workspace_path.mkdir(parents=True, exist_ok=True)
    sync_workspace_templates(settings.workspace_path)
    console.print(f"[green]✓[/green] Config saved at {path}")
    console.print(f"\n{__logo__} Emerge is ready!")
    console.print("Add your OpenAI or Anthropic API key to this config.")
    console.print("OpenAI api_mode defaults to responses; chat_completions is also supported.")
    console.print(f"Start: emerge --workspace {settings.workspace_path} --config {path}")


@app.command()
def wsinit(
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    config: Path | None = typer.Option(None, "--config", "-c"),
):
    """Show the selected workspace, model protocol and credential presence."""
    path = config.expanduser().resolve() if config else get_config_path(workspace)
    settings = load_config(config, workspace=workspace)
    console.print(f"{__logo__} Emerge Workspace\n")
    console.print(f"Config: {path} {'[green]✓[/green]' if path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {settings.workspace_path}")
    console.print(f"Model: {settings.agents.defaults.model}")
    for name in PROVIDERS:
        provider = getattr(settings.providers, name)
        state = "[green]key configured[/green]" if provider.api_key else "[dim]key not set[/dim]"
        console.print(f"{name}: {provider.api_mode} · {state}")
