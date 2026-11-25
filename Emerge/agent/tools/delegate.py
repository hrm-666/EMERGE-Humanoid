"""Main-agent tool for invoking registered specialist sub-agents."""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from loguru import logger

from Emerge.agent.reference_memory import ReferenceMemory
from Emerge.agent.tools.view_image import ViewImageTool
from Emerge.base import Tool
from Emerge.providers.telemetry import usage_scope, workspace_sink
from Emerge.subagents import SubagentRegistry, SubagentTask, TextContent


class DelegateSubagentTool(Tool):
    def __init__(self, registry: SubagentRegistry, workspace: Path | None = None, on_event=None) -> None:
        self._registry = registry
        self.workspace, self.on_event = workspace, on_event
        self.run_id = None

    @property
    def name(self) -> str:
        return "delegate_subagent"

    @property
    def description(self) -> str:
        agents = "; ".join(
            f"{item.name}: {item.description}" for item in self._registry.list()
        )
        return (
            "Delegate a focused task to a registered specialist and wait for "
            f"its structured result. Available specialists: {agents}"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": list(self._registry.names),
                    "description": "Registered specialist name",
                },
                "task": {
                    "type": "string",
                    "description": "Self-contained task for the specialist",
                    "minLength": 1,
                },
                "evidence_images": {"type": "array", "maxItems": 2,
                    "items": {"type": "string"},
                    "description": "Selected historical frames for comparison with current observe_scene: E image label, memory/images/... path, or action:<current action id> for its final archived frame. Up to two."},
            },
            "required": ["agent_name", "task"],
        }

    def _images(self, references):
        if len(references) > 2 or not self.workspace:
            raise ValueError('At most two frames in the active workspace')
        parts = []
        for reference in references:
            if reference.startswith('E'):
                memory = ReferenceMemory(self.workspace)
                root, bundle = memory.load()
                item = bundle['evidence'][reference]
                if item['kind'] != 'image':
                    raise ValueError('Choose an E image label or action:<current action id>')
                memory._read(root, item['file'], item['sha256'])
                path = root / item['file']
            else:
                relative = reference
                if reference.startswith('action:'):
                    action_id = reference.split(':', 1)[1]
                    if not action_id or not all(c.isalnum() or c in '_-' for c in action_id):
                        raise ValueError('Invalid action id')
                    record = json.loads((self.workspace / 'memory/actions' / f'{action_id}.json').read_text())
                    if self.run_id and record.get('run_id') != self.run_id:
                        raise ValueError('Use reference E labels for previous tasks')
                    relative = record['images'][-1]
                path = (self.workspace / relative).resolve()
                if not path.is_relative_to((self.workspace / 'memory/images').resolve()):
                    raise ValueError('Choose an archived memory/images frame')
            loaded = ViewImageTool(workspace=self.workspace)._load_image(path)
            parts.extend((TextContent(f'Historical comparison frame {reference}; captured before the current observation.'),
                          *loaded.content))
        return parts

    async def execute(self, agent_name: str, task: str, evidence_images=None) -> str:
        try:
            parts = await asyncio.to_thread(self._images, evidence_images) if evidence_images else []
        except (OSError, ValueError, KeyError, IndexError) as exc:
            return f'Error: verification evidence unavailable: {exc}'
        context = (usage_scope(workspace_sink(self.workspace, self.on_event), scope='visual_subagent',
                               run_id=self.run_id, agent=agent_name)
                   if self.workspace else nullcontext())
        delegated = SubagentTask(content=(TextContent(task), *parts))
        if self.workspace:
            from Emerge.agent.action_memory import audit_access
            audit_access(self.workspace, 'verification_input', run_id=self.run_id,
                         agent=agent_name, evidence_images=evidence_images or [], task_id=delegated.task_id)
        with context:
            result = await self._registry.run(agent_name, delegated)
        payload = {
            "status": result.status.value,
            "agent_name": result.agent_name,
            "summary": result.summary,
            "output": result.output,
            "error": result.error,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
        logger.info(
            "Sub-agent result | agent={} status={} payload={}",
            result.agent_name,
            result.status.value,
            serialized,
        )
        return serialized
