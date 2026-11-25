"""Search summaries first; expand their explicitly selected evidence tags."""

import asyncio
from pathlib import Path

from Emerge.agent.reference_memory import ReferenceMemory
from Emerge.agent.tools.update_plan import UpdatePlanTool
from Emerge.base import Tool


class SearchReferenceMemoryTool(Tool):
    def __init__(self, workspace: Path):
        self.memory = ReferenceMemory(workspace)
        self.run_id = None
        self._task = None
        self._segment = None
        self._reuse = {}

    @property
    def name(self):
        return "search_reference_memory"

    @property
    def description(self):
        return ("Search SPACE/EXPERIENCES summaries by landmark, object or stage. Expand selected reference_ids "
                "(stage:table_to_cabinet, relation:partition/table or E423) with include_evidence=true. "
                "query prioritizes the matching route segment; explicit E tags come first. At most two images "
                "and four records. Previously viewed images reuse the PLAN comparison; reread_images explicitly compares them again.")

    @property
    def parameters(self):
        return {"type": "object", "properties": {
            "query": {"type": "string"},
            "reference_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "include_evidence": {"type": "boolean", "default": False},
            "include_images": {"type": "boolean", "default": True},
            "reread_images": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "minimum": 1, "maximum": 8},
        }, "additionalProperties": False}

    async def execute(self, query="", reference_ids=None, include_evidence=False,
                      include_images=True, limit=4, reread_images=False):
        plan, _ = UpdatePlanTool(self.memory.workspace)._load_previous()
        plan = plan or {}
        route = plan.get('exploration', {})
        task = (self.run_id,)
        if task != self._task:
            self._task, self._reuse = task, {'images': {}}
            self._segment = None
        segment = (task, plan.get('pointer'), route.get('route_revision', 0))
        if segment != self._segment:
            self._segment = segment
            self._reuse['records'] = set()
        self._reuse.update(route=route.get('next_route', ''), comparison=route.get('comparison'),
                           segment=segment, segment_event=route.get('segment_event'))
        return await asyncio.to_thread(self.memory.search, query, reference_ids=reference_ids,
                                       include_evidence=include_evidence, include_images=include_images,
                                       limit=limit, run_id=self.run_id, reuse=self._reuse,
                                       reread_images=reread_images)
