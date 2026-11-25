"""Retrieve related controller attempts, their calls/results and evidence images."""
import asyncio
from pathlib import Path

from Emerge.agent.action_memory import retrieve
from Emerge.agent.reference_memory import ReferenceMemory
from Emerge.base import Tool


class SearchActionMemoryTool(Tool):
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.run_id = None

    @property
    def name(self):
        return "search_action_memory"

    @property
    def description(self):
        if ReferenceMemory(self.workspace).installed:
            return ("Compare current_run actions/results and images for this task. "
                    "Previous-task queries are redirected to text summaries; use search_reference_memory "
                    "tags to explicitly expand historical evidence.")
        return ("Search this controller's past tasks and current action history by target, skill, "
                "failure or action ID. Returns text first; include_images=true opens up to two archived images. "
                "Use previous_runs with sequence=true to recall earlier task routes and object "
                "locations; current_run to compare repeated views. task_run_id selects a specific task.")

    @property
    def parameters(self):
        return {"type": "object", "properties": {
            "query": {"type": "string", "minLength": 1},
            "scope": {"type": "string", "enum": ["controller", "current_run", "previous_runs"]},
            "task_run_id": {"type": "string", "description": "Restrict results to this archived run ID."},
            "sequence": {"type": "boolean", "description": "Return a chronological window around the best match in one task, with previous/next action IDs."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 8},
            "include_images": {"type": "boolean", "default": False}}, "required": ["query"], "additionalProperties": False}

    async def execute(self, query, scope="controller", limit=5, include_images=False,
                      task_run_id=None, sequence=False):
        reference = ReferenceMemory(self.workspace)
        if reference.installed and scope != "current_run":
            return await asyncio.to_thread(reference.search, query, limit=limit, run_id=self.run_id)
        return await asyncio.to_thread(retrieve, self.workspace, query, scope=scope, limit=limit,
                                       run_id=self.run_id, include_images=include_images,
                                       task_run_id=task_run_id, sequence=sequence)
