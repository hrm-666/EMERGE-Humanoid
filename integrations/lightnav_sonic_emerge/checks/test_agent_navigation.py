"""Agent tool -> real queue -> driver -> result, with an offline policy backend."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from Emerge.agent.tools.embodied import EmbodiedActionTool
from Emerge.utils.action_queue import infer_terminal_status
from Emerge.utils.action_queue import update_action_document
from integrations.lightnav_sonic_emerge import navigation_backend
from integrations.lightnav_sonic_emerge.checks import test_driver


class AgentNavigationTests(unittest.TestCase):

    def test_navigation_schema_and_runtime_timeout(self):
        tool = EmbodiedActionTool(Path("unused"))
        self.assertIn("navigate_vln", json.dumps(tool.parameters))
        self.assertEqual(tool._result_timeout("navigate_vln", {}), 180)
        self.assertEqual(
            tool._result_timeout("navigate_vln", {"max_runtime_s": 20}), 80)
        self.assertEqual(
            tool._result_timeout("navigate_vln", {"max_runtime_s": "nan"}), 180)

    def test_tool_queue_driver_result_round_trip(self):

        async def scenario(workspace):
            (workspace /
             "EMBODIED.md").write_text("Simulation driver: navigate_vln")
            tool = EmbodiedActionTool(workspace)
            driver = test_driver._TestDriver(
                navigation_backend.NavigationResult("budget_exhausted",
                                                    "max_steps", 1, 1, 0.1),
                workspace=workspace,
                elastic_band_release_file=None)
            task = asyncio.create_task(
                tool.execute(action_type="navigate_vln",
                             parameters={
                                 "instruction": "go to table",
                                 "max_steps": 1
                             },
                             reasoning="current navigation subgoal"))
            await asyncio.sleep(0)

            def controller(document):
                action = document["actions"][0]
                result = driver.execute_action(action["action_type"],
                                               action["parameters"])
                action.update(status=infer_terminal_status(result),
                              result=result)

            update_action_document(workspace / "ACTION.md", controller)
            reply = await asyncio.wait_for(task, timeout=5)
            self.assertIn("semantic goal is not automatically verified", reply)
            self.assertEqual(
                driver.get_runtime_state()["robots"][driver.ROBOT_ID]
                ["nav_state"]["ticks"], 1)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(scenario(Path(directory)))
