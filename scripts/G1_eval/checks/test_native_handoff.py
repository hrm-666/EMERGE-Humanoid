"""Native tool/queue/Controller handoff; policy and hardware are test doubles."""
from __future__ import annotations

import ast
import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path

from Emerge.agent.skills import SkillsLoader
from Emerge.agent.tools.delegate import DelegateSubagentTool
from Emerge.agent.tools.embodied import EmbodiedActionTool
from Emerge.agent.tools.update_plan import UpdatePlanTool
from Emerge.subagents.registry import SubagentRegistry
from Emerge.subagents.task_verification import build_task_verification_subagent
from Emerge.utils import action_queue
from robot import controller
from robot.drivers import load_driver
from robot.drivers.official01_g1_driver import Official01G1Driver
from robot.navigation.lightnav_sonic.executive_backend import (
    ExecutiveNavigationRunner,
)
from scripts.G1_eval.checks.test_driver import FakeClient
from scripts.G1_eval.checks.test_navigation import LightNav
from scripts.G1_eval.checks.test_navigation import Resource
from scripts.LightNav_eval.checks.test_native_skill import _VerificationProvider
from scripts.LightNav_eval.checks.test_observation_writer import _jpeg_bytes


class OfflineUnifiedDriver(Official01G1Driver):

    def __init__(self, workspace):
        super().__init__(workspace=workspace,
                         runtime="dry-run",
                         publish_observations=True)
        self._client = FakeClient()

    def _make_navigation(self, config, client):
        return ExecutiveNavigationRunner(
            config,
            client,
            self._observation_writer,
            camera_factory=lambda *_: Resource(_jpeg_bytes()),
            yaw_factory=lambda *_: Resource(0.0),
            lightnav_factory=lambda *_: LightNav(),
            sleep=lambda _: None)

    def _resource_exists(self, kind, name):
        return (kind, name) == ("prepare", "right2")


class NativeHandoffTest(unittest.TestCase):

    def test_both_skill_examples_validate_and_driver_is_native(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            tool = EmbodiedActionTool(workspace)
            loader = SkillsLoader(workspace)
            for name in ("lightnav-sonic-navigation", "gr00t-manipulation"):
                text = loader.load_skill(name)
                self.assertIsNotNone(text)
                for block in re.findall(r"```python\n(.*?)```", text,
                                        re.DOTALL):
                    call = ast.parse(block).body[0].value
                    if call.func.id != "execute_robot_action":
                        continue
                    params = {
                        kw.arg: ast.literal_eval(kw.value)
                        for kw in call.keywords
                    }
                    self.assertEqual(tool.validate_params(params), [])
            driver = load_driver("official01_g1",
                                 workspace=temp,
                                 publish_observations=False)
            self.assertIsInstance(driver, Official01G1Driver)
            self.assertTrue(driver.get_profile_path().is_file())
            driver.close()

    def test_chinese_and_english_missions_use_native_handoff(self):
        for mission in (
                "寻找放有苹果的桌子，走到桌子面前抓起苹果",
                "Find the table with an apple, walk up to it, and pick up the apple."
        ):
            with self.subTest(
                    mission=mission), tempfile.TemporaryDirectory() as temp:
                asyncio.run(self.round_trip(Path(temp), mission))

    async def round_trip(self, workspace, mission):
        driver = OfflineUnifiedDriver(workspace)
        try:
            controller._install_profile(driver, workspace)
            driver._observation_writer.write_jpeg(_jpeg_bytes())
            controller._poll_once(driver, workspace / "ACTION.md",
                                  workspace / "ROBOT_STATE.md")
            plan = UpdatePlanTool(workspace)
            reply = await plan.execute(
                update_kind="new_mission",
                mission=mission,
                main_line=[{
                    "id": 1,
                    "subgoal": "Approach and prepare to grasp",
                    "done_criterion": "Apple visible and grasp-ready",
                    "status": "active",
                    "retries": 0
                }],
                pointer=1,
                branch_stack=[],
                change_summary="Integration test plan")
            self.assertIn("Successfully", reply)
            before = (workspace / "PLAN.md").read_text()
            action = EmbodiedActionTool(workspace)
            # Scripted decisions test dispatch, not the real LLM's planning quality.
            for kind, params in (
                ("navigate_vln", {
                    "instruction": "approach the apple table",
                    "max_steps": 1
                }),
                ("move_micro", {
                    "direction": "turn_left",
                    "duration_s": 0.01
                }),
                ("run_fixed_action", {
                    "kind": "prepare",
                    "name": "right2"
                }),
                ("infer_gr00t", {
                    "instruction": "pick up the apple",
                    "duration_s": 0.01
                }),
            ):
                pending = asyncio.create_task(
                    action.execute(action_type=kind,
                                   parameters=params,
                                   reasoning="Current grounded phase"))
                try:
                    await asyncio.sleep(0)
                    controller._poll_once(driver, workspace / "ACTION.md",
                                          workspace / "ROBOT_STATE.md")
                    result = await asyncio.wait_for(pending, timeout=5)
                    self.assertNotIn("Error:", result)
                    self.assertEqual(driver._client.state, "READY")
                finally:
                    if not pending.done():
                        pending.cancel()
                        await asyncio.gather(pending, return_exceptions=True)
            self.assertIn(("key", "w"), driver._client.calls)
            self.assertIn(("key", "q"), driver._client.calls)
            queue = action_queue.parse_action_markdown(
                (workspace / "ACTION.md").read_text())
            self.assertEqual(len(queue["actions"]), 4)
            self.assertTrue(
                all(a["status"] == "completed" for a in queue["actions"]))
            # A completed grasp window must not advance the plan without evidence.
            self.assertEqual((workspace / "PLAN.md").read_text(), before)
            driver._observation_writer.write_jpeg(_jpeg_bytes())
            provider = _VerificationProvider("uncertain")
            agents = SubagentRegistry()
            agents.register(
                build_task_verification_subagent(provider, workspace))
            result = json.loads(await DelegateSubagentTool(agents).execute(
                agent_name="task_verification",
                task="Is the intended apple visibly held and lifted?"))
            self.assertTrue(provider.saw_image)
            self.assertEqual(result["output"]["outcome"], "uncertain")
            self.assertEqual((workspace / "PLAN.md").read_text(), before)
        finally:
            driver.close()
