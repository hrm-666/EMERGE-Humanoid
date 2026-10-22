"""Native Skill/tool/queue docking contract, with a fake motion backend."""

from __future__ import annotations

import ast
import asyncio
import json
import re
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from Emerge.agent.skills import SkillsLoader
from Emerge.agent.tools.embodied import EmbodiedActionTool
from Emerge.utils import action_queue
from robot import controller
from robot.navigation.lightnav_sonic.station_docking import DockingResult
from scripts.G1_eval.checks import test_docking_driver as driver_checks
from scripts.G1_eval.checks import test_station_docking as docking_checks


class UnsupportedVisualCompletion:
    """A docking action must never use a visual success interrupt."""

    async def wait_for(self, _action_id):
        raise AssertionError("visual completion cannot certify station docking")


class NativeDockingTest(unittest.TestCase):

    def test_skill_example_validates(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            skill = SkillsLoader(workspace).load_skill("sim-station-docking")
            self.assertIsNotNone(skill)
            blocks = re.findall(r"```python\n(.*?)```", skill, re.DOTALL)
            self.assertEqual(len(blocks), 1)
            call = ast.parse(blocks[0]).body[0].value
            params = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
            self.assertEqual(
                EmbodiedActionTool(workspace).validate_params(params), [])

    def test_native_queue_returns_only_measured_result(self):
        for outcome in ["docked", "failed", "cancelled"]:
            with self.subTest(outcome=outcome):
                asyncio.run(self.round_trip(outcome))

    async def round_trip(self, outcome):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            config = workspace / "stations.json"
            station = asdict(docking_checks.make_station())
            station.pop("name")
            config.write_text(json.dumps({
                "schema_version": 1,
                "runtime": "sim",
                "docking_only": True,
                "stations": {
                    "apple_table_front": {
                        **station, "enabled": True,
                        "calibration_status": "trial"
                    }
                }
            }),
                              encoding="utf-8")
            driver = driver_checks.DockingDriver(workspace=workspace,
                                                 runtime="sim",
                                                 publish_observations=False,
                                                 docking_config_path=config)
            driver._client = docking_checks.FakeExecutive(
                docking_checks.FakePlant())
            driver.test_docking = driver_checks.FakeDocking(
                DockingResult(
                    outcome, "pose_and_rest_verified"
                    if outcome == "docked" else "test_failure",
                    "apple_table_front", 3.0, 30, 0.3, 0.02, 0.01))
            pending = cancellation = None
            try:
                controller._install_profile(driver, workspace)
                controller._poll_once(driver, workspace / "ACTION.md",
                                      workspace / "ROBOT_STATE.md")
                tool = EmbodiedActionTool(workspace,
                                          UnsupportedVisualCompletion())
                pending = asyncio.create_task(
                    tool.execute(action_type="dock_to_station",
                                 parameters={"station": "apple_table_front"},
                                 reasoning="Verify measured station arrival."))
                await asyncio.sleep(0)
                if outcome == "cancelled":
                    cancellation = asyncio.create_task(
                        tool.cancel_active("operator stop", 2.0))
                    await asyncio.sleep(0)
                controller._poll_once(driver, workspace / "ACTION.md",
                                      workspace / "ROBOT_STATE.md")
                reply = await asyncio.wait_for(pending, timeout=3)
                document = action_queue.parse_action_markdown(
                    (workspace / "ACTION.md").read_text())
                action = document["actions"][0]
                expected = "completed" if outcome == "docked" else outcome
                self.assertEqual(action["status"], expected)
                self.assertNotIn("visual monitor already verified", reply)
                if outcome == "docked":
                    self.assertIn('"result": "docked"', reply)
                elif outcome == "failed":
                    self.assertTrue(reply.startswith("Error:"))
                else:
                    self.assertEqual(driver.test_docking.calls, [])
                    self.assertTrue((await cancellation)["acknowledged"])
            finally:
                for task in (pending, cancellation):
                    if task is not None and not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                driver.close()

    def test_named_action_has_bounded_tool_wait(self):
        self.assertEqual(
            EmbodiedActionTool._result_timeout("dock_to_station", {
                "station": "apple_table_front",
                "max_runtime_s": 1e9
            }), 150.0)


if __name__ == "__main__":
    unittest.main()
