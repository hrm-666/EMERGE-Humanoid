"""Check native Driver dispatch and the docking-only manipulation barrier."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from robot.drivers.official01_g1_driver import Official01G1Driver
from robot.navigation.lightnav_sonic.station_docking import DockingResult
from scripts.G1_eval.checks import test_station_docking as docking_checks


class FakeDocking:

    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, station, **kwargs):
        self.calls.append((station.name, kwargs))
        return self.result


class DockingDriver(Official01G1Driver):

    def _make_docking(self):
        return self.test_docking


class DockingDriverTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        station = asdict(docking_checks.make_station())
        station.pop("name")
        path = Path(self.temp.name) / "stations.json"
        path.write_text(json.dumps({
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
        self.driver = DockingDriver(workspace=self.temp.name,
                                    runtime="sim",
                                    publish_observations=False,
                                    docking_config_path=path)
        self.client = docking_checks.FakeExecutive(docking_checks.FakePlant())
        self.client.running = True
        self.driver._client = self.client
        self.driver.test_docking = FakeDocking(
            DockingResult("docked", "pose_and_rest_verified",
                          "apple_table_front", 3.0, 30, 0.3, 0.02, 0.01))

    def payload(self, text):
        return json.loads(text.split(": ", 1)[1])

    def dock(self, **kwargs):
        return self.payload(
            self.driver.execute_action("dock_to_station", {
                "station": "apple_table_front",
                **kwargs
            }))

    def test_symbolic_dispatch_and_no_world_coordinates(self):
        result = self.dock()
        self.assertEqual(result["result"], "docked")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.driver.test_docking.calls[0][0],
                         "apple_table_front")
        self.assertFalse(result["grasp_ready"])
        state = self.driver.get_runtime_state()["robots"][self.driver.ROBOT_ID]
        self.assertNotIn("target_x", json.dumps(state))
        self.assertEqual(state["station_docking"]["status"],
                         "historical_result")

    def test_coordinate_and_budget_overrides_are_rejected(self):
        for kwargs in [{
                "target_x": 10
        }, {
                "max_runtime_s": 999
        }, {
                "station": "unknown"
        }]:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.dock(**kwargs)["status"], "failed")
        self.assertEqual(self.driver.test_docking.calls, [])

    def test_all_manipulation_stays_blocked_before_and_after_docking(self):
        for dock_first in [False, True]:
            if dock_first:
                self.dock()
            for action in ["infer_gr00t", "run_fixed_action", "move_micro"]:
                with self.subTest(docked=dock_first, action=action):
                    result = self.payload(self.driver.execute_action(
                        action, {}))
                    self.assertEqual(result["reason"], "docking_only_stage")
        self.assertEqual(self.client.calls, [])

    def test_failed_docking_does_not_receive_completed_prefix(self):
        self.driver.test_docking.result = DockingResult("failed",
                                                        "drifted_after_stop",
                                                        "apple_table_front",
                                                        3.0, 30, 0.3, 0.1, 0.01)
        reply = self.driver.execute_action("dock_to_station",
                                           {"station": "apple_table_front"})
        self.assertTrue(reply.startswith("Failed:"))
        self.assertEqual(self.payload(reply)["result"], "docking_not_verified")

    def test_navigation_invalidates_historical_evidence(self):
        self.dock()
        self.driver.execute_action("navigate_vln", {})
        self.assertEqual(self.driver._docking_state["status"],
                         "invalidated_by_navigation")

    def test_unconfigured_driver_cannot_dock(self):
        self.driver._docking_registry = None
        self.assertEqual(self.dock()["status"], "failed")
        self.assertEqual(self.driver.test_docking.calls, [])


if __name__ == "__main__":
    unittest.main()
