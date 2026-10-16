from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from integrations.lightnav_sonic_emerge.navigation_backend import NavigationResult
from integrations.official01_g1_emerge.driver import Official01G1Driver


CONTROL_ROOT = Path("/home/embodied/visionpro_sonic_bridge")


class FakeClient:
    def __init__(self):
        self.state = "READY"
        self.running = True
        self.calls = []

    def begin_inference(self, instruction, *, timeout_s):
        self.calls.append(("infer", instruction, timeout_s))
        self.state = "INFERENCING"

    def pause_inference(self, *, timeout_s):
        self.calls.append(("pause", timeout_s))
        self.state = "READY"

    def run_fixed(self, kind, name, *, timeout_s):
        self.calls.append(("fixed", kind, name, timeout_s))

    def begin_moving(self, *, timeout_s):
        self.calls.append(("begin_moving", timeout_s))
        self.state = "MOVING"

    def send_nav_key(self, key):
        self.calls.append(("key", key))

    def stop_moving(self, *, timeout_s):
        self.calls.append(("stop", timeout_s))
        self.state = "READY"

    def close(self):
        self.running = False
        self.state = "CLOSED"


class FakeNavigation:
    def __init__(self, result):
        self.result = result

    def run(self, **_kwargs):
        return self.result


class TestDriver(Official01G1Driver):
    def __init__(self, *args, nav_result=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nav_result = nav_result

    def _make_navigation(self, config, client):
        del config, client
        return FakeNavigation(self.nav_result)


class Official01G1DriverTest(unittest.TestCase):
    def make_driver(self, workspace, nav_result=None):
        driver = TestDriver(
            workspace=workspace,
            control_root=CONTROL_ROOT,
            runtime="dry-run",
            command_watchdog_s=0.01,
            max_micro_duration_s=0.05,
            max_inference_duration_s=0.05,
            publish_observations=False,
            nav_result=nav_result,
        )
        driver._client = FakeClient()
        return driver

    @staticmethod
    def payload(result):
        return json.loads(result.split(": ", 1)[1])

    def test_fixed_inference_and_micro_return_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            driver = self.make_driver(temp)
            fixed = driver.execute_action(
                "run_fixed_action",
                {"kind": "pose", "name": "pick_ready_r", "timeout_s": 5},
            )
            infer = driver.execute_action(
                "infer_gr00t",
                {"instruction": "pick up the apple", "duration_s": 0.01, "timeout_s": 2},
            )
            move = driver.execute_action(
                "move_micro",
                {"direction": "forward", "duration_s": 0.02, "timeout_s": 2},
            )
            self.assertEqual(self.payload(fixed)["status"], "completed")
            self.assertFalse(self.payload(infer)["semantic_task_verified"])
            self.assertFalse(self.payload(move)["odometry_verified"])
            self.assertEqual(driver._client.state, "READY")
            self.assertIn(("key", "w"), driver._client.calls)

    def test_navigation_alias_and_semantic_caveat(self):
        nav_result = NavigationResult(
            status="completed", reason="lightnav_stop", ticks=2,
            command_steps=1, elapsed_s=0.5, lightnav_stop=True,
        )
        with tempfile.TemporaryDirectory() as temp:
            driver = self.make_driver(temp, nav_result=nav_result)
            result = driver.execute_action(
                "navigate_vln", {"instruction": "approach the table", "max_steps": 2}
            )
        payload = self.payload(result)
        self.assertEqual(payload["result"], "navigation_phase_ended")
        self.assertFalse(payload["semantic_arrival_verified"])

    def test_rejects_paths_unknown_actions_and_oversized_motion(self):
        with tempfile.TemporaryDirectory() as temp:
            driver = self.make_driver(temp)
            path_result = driver.execute_action(
                "run_fixed_action", {"kind": "pose", "name": "../unsafe"}
            )
            move_result = driver.execute_action(
                "move_micro", {"direction": "forward", "duration_s": 1.0}
            )
            unknown = driver.execute_action("shell", {"command": "whoami"})
        self.assertTrue(path_result.startswith("Failed:"))
        self.assertTrue(move_result.startswith("Failed:"))
        self.assertTrue(unknown.startswith("Failed:"))


if __name__ == "__main__":
    unittest.main()
