"""Search memory survives plan edits and renewed Agent context; no robot starts."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from Emerge.agent.context import ContextBuilder
from Emerge.agent.tools.update_plan import UpdatePlanTool


class ExplorationPlanTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.tool = UpdatePlanTool(self.workspace)
        self.plan = {
            "update_kind": "new_mission",
            "mission": "Find the cabinet",
            "main_line": [{"id": 1, "subgoal": "Search connected areas",
                           "done_criterion": "Cabinet visible", "status": "active", "retries": 0}],
            "pointer": 1, "branch_stack": [], "change_summary": "Start exploration",
        }
        self.search = {
            "target": "brown cabinet", "current_area": "table side of divider",
            "checked_routes": ["table corner: looked left; passage interior unseen"],
            "frontiers": ["opening past divider"],
            "next_route": "Walk through the opening past the divider",
        }

    async def start_search(self):
        reply = await self.tool.execute(**self.plan, exploration=self.search)
        self.assertTrue(reply.startswith("Successfully"), reply)
        return reply

    def snapshot(self):
        return self.tool._parse((self.workspace / "PLAN.md").read_text())

    async def test_progress_retains_memory_and_new_agent_context_reads_it(self):
        reply = await self.start_search()
        self.assertIn(self.search["next_route"], reply)
        self.plan["update_kind"] = "progress"
        self.tool = UpdatePlanTool(self.workspace)
        await self.tool.execute(**self.plan)
        self.assertEqual(self.snapshot()["exploration"], self.search)
        prompt = ContextBuilder(self.workspace).build_system_prompt()
        self.assertIn(self.search["next_route"], prompt)
        self.assertIn(self.search["checked_routes"][0], prompt)

    async def test_replan_retains_exploration_while_changing_strategy(self):
        await self.start_search()
        self.plan["update_kind"] = "replan"
        self.plan["main_line"][0]["subgoal"] = "Explore beyond the divider"
        result = await self.tool.execute(**self.plan)
        self.assertTrue(result.startswith("Successfully"), result)
        self.assertEqual(self.snapshot()["exploration"], self.search)

    async def test_updates_replace_search_snapshot_and_empty_object_clears(self):
        await self.start_search()
        self.plan["update_kind"] = "progress"
        revised = deepcopy(self.search)
        revised["current_area"] = "junction beyond divider"
        revised["checked_routes"].append("divider opening: traversed; new junction visible")
        revised["next_route"] = "Inspect passage beside refrigerator"
        await self.tool.execute(**self.plan, exploration=revised)
        self.assertEqual(self.snapshot()["exploration"], revised)
        await self.tool.execute(**self.plan, exploration={})
        self.assertNotIn("exploration", self.snapshot())

    async def test_new_mission_drops_previous_target_memory(self):
        await self.start_search()
        self.plan["mission"] = "Find the apple"
        await self.tool.execute(**self.plan)
        self.assertNotIn("exploration", self.snapshot())

    async def test_legacy_plan_without_memory_remains_compatible(self):
        await self.tool.execute(**self.plan)
        self.assertNotIn("exploration", self.snapshot())
        self.plan["update_kind"] = "progress"
        await self.tool.execute(**self.plan, exploration=self.search)
        self.assertEqual(self.snapshot()["exploration"], self.search)
        self.assertEqual(self.tool.validate_params({**self.plan, "exploration": self.search}), [])
        self.assertEqual(self.tool.validate_params({**self.plan, "exploration": {}}), [])

    async def test_invalid_memory_leaves_existing_plan_unchanged(self):
        await self.start_search()
        before = (self.workspace / "PLAN.md").read_text()
        self.plan["update_kind"] = "progress"
        for bad in (["left"], {"target": "cabinet"},
                    {**self.search, "frontiers": "left"},
                    {**self.search, "frontiers": [4]},
                    {**self.search, "target": "cabinet\nnew section"},
                    {**self.search, "x": 1.2}):
            with self.subTest(bad=bad):
                reply = await self.tool.execute(**self.plan, exploration=bad)
                self.assertTrue(reply.startswith("Error:"), reply)
                self.assertEqual((self.workspace / "PLAN.md").read_text(), before)

    async def test_json_roundtrip_preserves_fence_text_in_landmark(self):
        self.search["current_area"] = "area by the ``` labelled doorway"
        await self.start_search()
        self.assertEqual(self.snapshot()["exploration"], self.search)

    async def test_comparison_survives_full_progress_updates_and_clears_on_segment_events(self):
        await self.start_search()
        comparison = {'evidence_ids': ['E1', 'E2'], 'match': 'divider right, open floor left',
                      'exit': 'left passage', 'arrival_cue': 'junction beyond divider'}
        self.tool.advance_route('entry', 'Enter passage', segment_event='exit_selected', comparison=comparison)
        self.assertEqual(self.snapshot()['exploration']['route_revision'], 1)
        self.plan['update_kind'] = 'progress'
        result = await self.tool.execute(**self.plan, exploration={**self.search, 'current_area': 'Camera revision 103'})
        self.assertTrue(result.startswith('Successfully'), result)
        state = self.snapshot()['exploration']
        self.assertEqual(state['comparison'], comparison)
        self.assertEqual(state['route_revision'], 1)
        result = await self.tool.execute(**self.plan, exploration={**self.search, 'route_revision': 0})
        self.assertTrue(result.startswith('Successfully'), result)
        self.assertEqual(self.snapshot()['exploration']['route_revision'], 1)
        for event in ('landmark_reached', 'exit_selected', 'route_invalidated'):
            revision = self.snapshot()['exploration']['route_revision']
            result = await self.tool.execute(**self.plan, exploration={**self.search, 'segment_event': event})
            self.assertTrue(result.startswith('Successfully'), result)
            state = self.snapshot()['exploration']
            self.assertEqual(state['route_revision'], revision + 1)
            self.assertNotIn('comparison', state)
        self.tool.advance_route('entry', 'Follow passage', comparison=comparison)
        self.plan['main_line'][0]['status'] = 'done'
        self.plan['pointer'] = 2
        await self.tool.execute(**self.plan)
        self.assertNotIn('comparison', self.snapshot()['exploration'])
        self.plan['mission'] = 'New task'
        self.plan['update_kind'] = 'new_mission'
        await self.tool.execute(**self.plan, exploration=self.search)
        self.assertNotIn('comparison', self.snapshot()['exploration'])

    async def test_invalid_comparison_is_rejected_without_writing_plan(self):
        await self.start_search()
        before = self.tool.plan_file.read_bytes()
        for comparison in ({}, {'evidence_ids': ['E1'], 'match': 'x', 'exit': 'x', 'arrival_cue': 'x' * 121},
                           {'evidence_ids': ['E1'] * 7, 'match': 'x', 'exit': 'x', 'arrival_cue': 'x'}):
            with self.assertRaises(ValueError):
                self.tool.advance_route('area', 'route', comparison=comparison)
            self.assertEqual(self.tool.plan_file.read_bytes(), before)

    async def test_action_route_progress_saves_comparison_in_one_dispatch(self):
        from unittest.mock import patch

        from Emerge.agent.tools.embodied import EmbodiedActionTool
        await self.start_search()
        (self.workspace / 'EMBODIED.md').write_text('Offline fixture')
        action = EmbodiedActionTool(self.workspace)
        comparison = {'evidence_ids': ['E1'], 'match': 'passage left', 'exit': 'left', 'arrival_cue': 'junction'}
        args = {'action_type': 'navigate', 'parameters': {}, 'reasoning': 'Follow remembered exit',
                'route_progress': {'current_area': 'entry', 'next_route': 'Enter left passage',
                                   'segment_event': 'exit_selected', 'comparison': comparison}}
        self.assertEqual(action.validate_params(args), [])
        # End before queue creation; this only exercises route persistence through the real action tool.
        with patch.object(action, '_accept_action', return_value='offline dispatch') as dispatch:
            self.assertEqual(await action.execute(**args), 'offline dispatch')
            dispatch.assert_called_once()
        self.assertEqual(self.snapshot()['exploration']['comparison'], comparison)
        self.assertFalse((self.workspace / 'ACTION.md').exists())


if __name__ == "__main__":
    unittest.main()
