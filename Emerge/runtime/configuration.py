"""Strict configuration loading for execution; no stdout or global config mutation."""

from Emerge.config.loader import load_config
from Emerge.config.schema import Config
from Emerge.runtime.protocol import RunRequest


def load_runtime_config(request: RunRequest) -> Config:
    config = load_config(request.config, workspace=request.workspace)
    if request.model:
        config.agents.defaults.model = request.model
    if request.max_iterations is not None:
        config.agents.defaults.max_tool_iterations = request.max_iterations
    if request.restrict_to_workspace is not None:
        config.tools.restrict_to_workspace = request.restrict_to_workspace
    if request.visual_monitor_enabled is not None:
        config.visual_monitor.enabled = request.visual_monitor_enabled
    return config
