"""Exercise native Skill dispatch and verification with offline policy doubles.

Only model outputs and robot transports are synthetic. Skill loading, tool
schema validation, Controller, navigation loop, and verification subagent run
through the repository implementations. These are not model quality tests.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Emerge.agent import context
from Emerge.agent import skills
from Emerge.agent import visual_monitor
from Emerge.agent.tools import delegate
from Emerge.agent.tools import embodied
from Emerge.agent.tools import filesystem
from Emerge.agent.tools import update_plan
from Emerge.base import registry
from Emerge.providers import base as providers
from Emerge.subagents import registry as subagent_registry
from Emerge.subagents import task_verification
from Emerge.subagents.task_verification.tools import observation
from Emerge.utils import action_queue
from integrations.lightnav_sonic_emerge import driver
from integrations.lightnav_sonic_emerge import navigation_backend
from integrations.lightnav_sonic_emerge.checks import test_navigation_backend
from integrations.lightnav_sonic_emerge.checks import test_observation_writer
from robot import controller

_SKILL_NAME = "lightnav-sonic-navigation"


class _VerificationProvider(providers.LLMProvider):
    """Script tool calls and require that the native agent supplies an image."""

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.calls = 0
        self.saw_image = False

    def get_default_model(self):
        return "offline-verification"

    async def chat(self, messages, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            name, arguments = "observe_scene", {}
        else:
            self.saw_image = "data:image/png;base64," in json.dumps(messages)
            if not self.saw_image:
                raise AssertionError("Verifier did not receive current RGB")
            name = "submit_task_verification"
            arguments = {
                "predicates": [{
                    "name":
                        "target_visible_ahead",
                    "state":
                        self.state,
                    "evidence":
                        "Synthetic fixture; not a visual-quality claim.",
                }],
                "scene_context": "Offline navigation integration fixture.",
            }
        return providers.LLMResponse(
            content=None,
            tool_calls=[
                providers.ToolCallRequest(str(self.calls), name, arguments)
            ],
        )


class _OfflineDriver(driver.SonicLightNavDriver):
    """Use real navigation lifecycle with in-memory camera, yaw, and policies."""

    def __init__(self, workspace, *, stop):
        self.camera_source = test_navigation_backend._SequenceSource(
            [test_observation_writer._jpeg_bytes()])
        self.publisher = test_navigation_backend._Publisher()
        self.stop = stop
        super().__init__(workspace=workspace, elastic_band_release_file=None)

    def _make_camera(self):
        return self.camera_source

    def _make_backend(self, config):
        return navigation_backend.NavigationBackend(
            config,
            self._observation_writer,
            camera_factory=lambda *_: test_navigation_backend._SequenceSource(
                [test_observation_writer._jpeg_bytes()]),
            yaw_factory=lambda *_: test_navigation_backend._SequenceSource(
                [0.3]),
            publisher_factory=lambda *_: self.publisher,
            client_factory=lambda *_: test_navigation_backend._Client(
                [test_navigation_backend._reply(stop=self.stop)]),
            adapter_factory=lambda *_: test_navigation_backend._Adapter(),
            sleep=lambda _: None,
        )


class NativeNavigationSkillTest(unittest.TestCase):

    def test_discovery_is_lazy_and_workspace_overrides_still_win(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            loader = skills.SkillsLoader(workspace)
            entry = next(item for item in loader.list_skills()
                         if item["name"] == _SKILL_NAME)
            self.assertEqual(entry["source"], "builtin")
            self.assertNotIn(_SKILL_NAME, loader.get_always_skills())
            prompt = context.ContextBuilder(workspace).build_system_prompt()
            self.assertIn(entry["path"], prompt)
            self.assertNotIn("## Execution Postconditions", prompt)
            text = asyncio.run(
                filesystem.ReadFileTool(workspace=workspace).execute(
                    path=entry["path"]))
            self.assertIn("## Execution Postconditions", text)
            override = workspace / "skills" / _SKILL_NAME / "SKILL.md"
            override.parent.mkdir(parents=True)
            override.write_text("custom navigation", encoding="utf-8")
            self.assertEqual(loader.load_skill(_SKILL_NAME),
                             "custom navigation")

    def test_skill_examples_use_native_tool_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            agents = subagent_registry.SubagentRegistry()
            agents.register(
                task_verification.build_task_verification_subagent(
                    _VerificationProvider("uncertain"), workspace))
            tools = registry.ToolRegistry()
            tools.register(embodied.EmbodiedActionTool(workspace))
            tools.register(delegate.DelegateSubagentTool(agents))
            text = skills.SkillsLoader(workspace).load_skill(_SKILL_NAME)
            examples = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
            self.assertEqual(len(examples), 2)
            for example in examples:
                call = ast.parse(example).body[0].value
                tool = tools.get(call.func.id)
                self.assertIsNotNone(tool, call.func.id)
                params = {
                    kw.arg: ast.literal_eval(kw.value) for kw in call.keywords
                }
                self.assertEqual(tool.validate_params(params), [])

    def test_controller_workspace_is_authoritative_for_navigation(self):
        config = {"workspace": "/wrong", "camera_transport": "official01_rgb"}
        result = controller._driver_kwargs_for_workspace(
            "lightnav_sonic_g1", Path("/selected"), config)
        self.assertEqual(result["workspace"], "/selected")
        self.assertEqual(config["workspace"], "/wrong")
        self.assertEqual(result["camera_transport"], "official01_rgb")
        self.assertEqual(
            controller._driver_kwargs_for_workspace("other_plugin",
                                                    Path("/selected"), {}), {})

    def test_native_controller_verifier_and_plan_transitions(self):
        cases = ((True, "satisfied", "achieved"),
                 (True, "not_satisfied", "not_achieved"), (False, "uncertain",
                                                           "uncertain"))
        for stop, state, outcome in cases:
            with self.subTest(stop=stop, outcome=outcome):
                with tempfile.TemporaryDirectory() as directory:
                    asyncio.run(
                        self._round_trip(Path(directory), stop, state, outcome))

    async def _round_trip(self, workspace, stop, state, outcome):
        with _OfflineDriver(workspace, stop=stop) as robot_driver:
            controller._install_profile(robot_driver, workspace)
            robot_driver.load_scene({})
            controller._poll_once(robot_driver, workspace / "ACTION.md",
                                  workspace / "ROBOT_STATE.md")
            # Before the first action, the native verifier already has RGB.
            self.assertEqual(
                observation.ObservationStore(workspace).load().reference_view,
                "ego_view")
            provider = _VerificationProvider(state)
            agents = subagent_registry.SubagentRegistry()
            agents.register(
                task_verification.build_task_verification_subagent(
                    provider, workspace))
            tools = registry.ToolRegistry()
            for tool in (update_plan.UpdatePlanTool(workspace),
                         embodied.EmbodiedActionTool(workspace),
                         delegate.DelegateSubagentTool(agents)):
                tools.register(tool)
            plan = {
                "update_kind": "new_mission",
                "mission": "Approach the visible table",
                "main_line": [{
                    "id": 1,
                    "subgoal": "Approach the table",
                    "done_criterion": "The intended table is visible ahead",
                    "status": "active",
                    "retries": 0,
                }],
                "pointer": 1,
                "branch_stack": [],
                "change_summary": "Ground the navigation phase",
            }
            result = await tools.execute("update_plan", plan)
            self.assertIn("Successfully", result)
            before = (workspace / "PLAN.md").read_text()
            pending = asyncio.create_task(
                tools.execute(
                    "execute_robot_action", {
                        "action_type": "navigate_vln",
                        "parameters": {
                            "instruction": "approach the table",
                            "max_steps": 1
                        },
                        "reasoning": "Current navigation phase",
                    }))
            try:
                await asyncio.sleep(0)
                controller._poll_once(robot_driver, workspace / "ACTION.md",
                                      workspace / "ROBOT_STATE.md")
                reply = await asyncio.wait_for(pending, timeout=5)
            finally:
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
            self.assertIn("semantic goal", reply)
            self.assertEqual((workspace / "PLAN.md").read_text(), before)
            queue = action_queue.parse_action_markdown(
                (workspace / "ACTION.md").read_text())
            self.assertEqual(queue["actions"][0]["status"], "completed")
            self.assertTrue(robot_driver.publisher.closed)
            self.assertTrue(robot_driver.publisher.idles)
            # Refresh as the real Controller does during idle polling.
            controller._poll_once(robot_driver, workspace / "ACTION.md",
                                  workspace / "ROBOT_STATE.md")
            result = json.loads(await tools.execute(
                "delegate_subagent", {
                    "agent_name": "task_verification",
                    "task": "Verify the active navigation Done Criterion.",
                }))
            self.assertTrue(provider.saw_image)
            self.assertEqual(result["output"]["outcome"], outcome)
            self.assertEqual((workspace / "PLAN.md").read_text(), before)
            # Script the Main Agent decision; the tools do not infer success.
            plan["update_kind"] = "progress"
            if outcome == "achieved":
                plan["main_line"][0]["status"] = "done"
                plan["pointer"] = 2
                plan["change_summary"] = "Verified navigation criterion"
            else:
                plan["main_line"][0]["retries"] = 1
                plan["branch_stack"] = [{
                    "id": "inspect-target",
                    "subgoal": "Reassess the target before another action",
                    "done_criterion": "The intended target is unambiguous",
                }]
                plan["change_summary"] = "Verification requires recovery"
            self.assertIn("Successfully", await
                          tools.execute("update_plan", plan))
            saved = update_plan.UpdatePlanTool._parse(
                (workspace / "PLAN.md").read_text())
            self.assertEqual(saved["pointer"], plan["pointer"])
            self.assertEqual(saved["branch_stack"], plan["branch_stack"])

    def test_stale_and_closed_camera_withdraw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            robot_driver = _OfflineDriver(workspace, stop=True)
            robot_driver.load_scene({})
            store = observation.ObservationStore(workspace)
            self.assertEqual(len(store.load().views), 1)
            robot_driver.camera_source.values = [None]
            state = robot_driver.get_runtime_state()["robots"][
                robot_driver.ROBOT_ID]
            self.assertEqual(state["observation_state"]["status"],
                             "unavailable")
            with self.assertRaises(FileNotFoundError):
                store.load()
            robot_driver.camera_source.values = [
                test_observation_writer._jpeg_bytes()
            ]
            robot_driver.get_runtime_state()
            robot_driver.close()
            self.assertTrue(robot_driver.camera_source.closed)
            with self.assertRaises(FileNotFoundError):
                store.load()

    def test_expiry_is_enforced_without_controller_poll(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with _OfflineDriver(workspace, stop=True) as robot_driver:
                robot_driver.load_scene({})
                path = workspace / "artifacts/observations/observation.json"
                payload = json.loads(path.read_text())
                with mock.patch.object(observation.time,
                                       "time",
                                       return_value=payload["expires_at"] + 1):
                    with self.assertRaisesRegex(ValueError, "expired"):
                        observation.ObservationStore(workspace).load()
                # Legacy LIBERO manifests have no expiry and remain supported.
                del payload["expires_at"]
                path.write_text(json.dumps(payload))
                self.assertEqual(
                    len(observation.ObservationStore(workspace).load().views),
                    1)

    def test_visual_monitor_ignores_expired_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for filename in ("PLAN.md", "ACTION.md"):
                (workspace / filename).touch()
            target = workspace / "artifacts/observations/observation.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({"expires_at": 0}))
            # No image or provider is needed if expiry is checked before inference.
            monitor = object.__new__(visual_monitor.VisualMonitor)
            monitor.workspace = workspace
            monitor._read_pending_action = lambda _: {"id": "offline"}
            with mock.patch.object(visual_monitor,
                                   "read_current_subgoal",
                                   return_value=object()):
                monitor._poll_once()
