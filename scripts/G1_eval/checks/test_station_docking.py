"""Deterministic docking checks; fake kinematics are not SONIC validation."""

from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import msgpack

from robot.navigation.lightnav_sonic.sim_docking_pose import (
    SimDockingPoseSource,
)
from robot.navigation.lightnav_sonic.sim_docking_pose import SimRobotPose
from robot.navigation.lightnav_sonic.sim_docking_pose import (
    decode_sim_robot_pose,
)
from robot.navigation.lightnav_sonic.station_docking import Bounds
from robot.navigation.lightnav_sonic.station_docking import Station
from robot.navigation.lightnav_sonic.station_docking import StationDockingRunner
from robot.navigation.lightnav_sonic.station_docking import StationRegistry


def make_station(**kwargs):
    return replace(
        Station("apple_table_front",
                Bounds(0.0, 0.6, -0.2, 0.2),
                Bounds(-0.1, 0.75, -0.25, 0.25),
                0.5,
                0.0,
                0.0,
                stable_duration_s=0.5,
                max_runtime_s=15.0), **kwargs)


class FakePlant:
    """Integrate body velocity solely to exercise controller transitions."""

    def __init__(self):
        self.t = 0.0
        self.x, self.y, self.yaw = 0.1, 0.0, 0.0
        self.velocity = (0.0, 0.0, 0.0)
        self.after_stop_drift = 0.0

    def sleep(self, dt):
        forward, lateral, yaw_rate = self.velocity
        self.x += (math.cos(self.yaw) * forward - math.sin(self.yaw) * lateral +
                   self.after_stop_drift) * dt
        self.y += (math.sin(self.yaw) * forward +
                   math.cos(self.yaw) * lateral) * dt
        self.yaw += yaw_rate * dt
        self.t += dt

    def pose(self):
        return SimRobotPose(self.x, self.y, 0.75, self.yaw, 0.0, 100.0 + self.t,
                            10.0 + self.t, False)


class FakeExecutive:
    runtime = "sim"
    linear_speed_mps = 0.25
    yaw_rate_rad_s = 0.25
    command_watchdog_s = 0.4

    def __init__(self, plant):
        self.plant = plant
        self.state = "READY"
        self.running = True
        self.calls = []
        self.fail_stop = False
        self.fail_begin = False
        self.stop_drift = 0.0

    def begin_moving(self, *, timeout_s):
        self.calls.append(("begin", timeout_s))
        self.state = "MOVING"
        if self.fail_begin:
            raise RuntimeError("lost acknowledgment")

    def send_nav_velocity(self, forward, lateral, yaw_rate):
        if self.state != "MOVING":
            raise RuntimeError("not moving")
        self.calls.append(("velocity", forward, lateral, yaw_rate))
        self.plant.velocity = (forward, lateral, yaw_rate)

    def stop_moving(self, *, timeout_s):
        self.calls.append(("stop", timeout_s))
        if self.fail_stop:
            raise RuntimeError("stop timeout")
        self.plant.velocity = (0.0, 0.0, 0.0)
        self.plant.after_stop_drift = self.stop_drift
        self.state = "READY"

    def close(self):
        self.running = False
        self.state = "CLOSED"


class FakeSource:
    error = ""

    def __init__(self, plant):
        self.plant = plant
        self.closed = False
        self.transform = lambda pose: pose

    def latest(self, _max_age):
        return self.transform(self.plant.pose())

    def close(self):
        self.closed = True


class DockingRunnerTest(unittest.TestCase):

    def setUp(self):
        self.plant = FakePlant()
        self.client = FakeExecutive(self.plant)
        self.source = FakeSource(self.plant)
        self.runner = StationDockingRunner(
            self.client,
            "ipc:///tmp/not-used",
            source_factory=lambda _: self.source,
            monotonic=lambda: self.plant.t,
            wall_clock=lambda: 100.0 + self.plant.t,
            sleep=self.plant.sleep)

    def run_station(self, **kwargs):
        return self.runner.run(make_station(**kwargs))

    def assert_no_start(self, result, reason):
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, reason)
        self.assertEqual(self.client.calls, [])

    def test_walks_stops_and_measures_dwell(self):
        result = self.run_station()
        self.assertEqual(result.status, "docked")
        self.assertGreater(self.plant.x, 0.45)
        self.assertLessEqual(result.position_error_m, 0.04)
        self.assertGreater(result.samples, 20)
        self.assertEqual(self.client.state, "READY")
        self.assertTrue(self.source.closed)
        self.assertEqual(sum(c[0] == "begin" for c in self.client.calls), 1)
        self.assertEqual(sum(c[0] == "stop" for c in self.client.calls), 1)
        self.assertFalse(result.to_dict()["grasp_ready"])
        self.assertNotIn("target_x", json.dumps(result.to_dict()))

    def test_at_target_still_requires_new_stable_samples(self):
        self.plant.x = 0.5
        result = self.run_station()
        self.assertEqual(result.status, "docked")
        self.assertGreaterEqual(result.elapsed_s, 0.5)
        self.assertEqual(self.client.calls, [])

    def test_already_docked_outside_staging_region_only_verifies_rest(self):
        self.plant.x = 0.5
        result = self.run_station(entry=Bounds(0.0, 0.2, -0.2, 0.2))
        self.assertEqual(result.status, "docked")
        self.assertEqual(self.client.calls, [])

    def test_outside_entry_rejected_without_motion(self):
        self.plant.x = -0.05
        self.assert_no_start(self.run_station(), "outside_entry_region")

    def test_bad_heading_rejected_without_motion(self):
        self.plant.yaw = math.pi
        self.assert_no_start(self.run_station(), "outside_entry_heading")

    def test_support_rejected_without_motion(self):
        self.source.transform = lambda p: replace(p, support_enabled=True)
        self.assert_no_start(self.run_station(), "support_enabled")

    def test_tilt_and_height_rejected_without_motion(self):
        for field, value, reason in [("tilt", 0.4, "excessive_tilt"),
                                     ("z", 0.2, "invalid_base_height")]:
            with self.subTest(field=field):
                self.source.transform = lambda p: replace(p, **{field: value})
                self.assert_no_start(self.run_station(), reason)

    def test_real_runtime_rejected_without_source_or_motion(self):
        self.client.runtime = "real"
        self.assert_no_start(self.run_station(), "execution_error:ValueError")
        self.assertFalse(self.source.closed)

    def test_pending_cancellation_never_starts(self):
        result = self.runner.run(make_station(), cancel_check=lambda: "cancel")
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(self.client.calls, [])

    def test_cancellation_stops_motion(self):
        result = self.runner.run(make_station(),
                                 cancel_check=lambda: "cancel"
                                 if self.plant.t > 0.2 else None)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(self.client.state, "READY")
        self.assertEqual(self.client.calls[-1][0], "stop")

    def test_stale_mid_motion_stops(self):
        self.source.transform = lambda p: None if self.plant.t > 0.3 else p
        result = self.run_station()
        self.assertEqual(result.reason, "stale_or_invalid_pose")
        self.assertEqual(self.client.state, "READY")

    def test_duplicate_frames_cannot_establish_dwell(self):
        self.plant.x = 0.5
        frozen = self.plant.pose()
        self.source.transform = lambda _: frozen
        result = self.run_station()
        self.assertEqual(result.reason, "stale_or_invalid_pose")
        self.assertNotEqual(result.status, "docked")

    def test_simulation_reset_stops_motion(self):
        self.source.transform = lambda p: replace(p, sim_time_s=0.0
                                                 ) if self.plant.t > 0.3 else p
        result = self.run_station()
        self.assertEqual(result.reason, "simulation_reset_or_reordered_pose")
        self.assertEqual(self.client.state, "READY")

    def test_jumping_pose_cannot_claim_arrival(self):
        self.source.transform = lambda p: replace(p, x=0.5
                                                 ) if self.plant.t > 0.3 else p
        result = self.run_station()
        self.assertEqual(result.reason, "pose_discontinuity")
        self.assertEqual(self.client.state, "READY")

    def test_leave_corridor_stops_motion(self):
        self.source.transform = lambda p: replace(p, y=0.26
                                                 ) if self.plant.t > 0.3 else p
        result = self.run_station()
        self.assertEqual(result.reason, "outside_corridor")
        self.assertEqual(self.client.state, "READY")

    def test_path_budget_stops_without_success(self):
        result = self.run_station(max_path_m=0.03)
        self.assertEqual(result.reason, "path_budget_exhausted")
        self.assertEqual(self.client.state, "READY")

    def test_timeout_stops_without_success(self):
        result = self.run_station(max_runtime_s=0.8)
        self.assertEqual(result.reason, "max_runtime_s")
        self.assertEqual(self.client.state, "READY")

    def test_failed_stop_never_reports_docked(self):
        self.client.fail_stop = True
        result = self.run_station()
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "stop_failed")

    def test_lost_begin_ack_attempts_cleanup(self):
        self.client.fail_begin = True
        result = self.run_station()
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.client.calls[-1][0], "stop")
        self.assertEqual(self.client.state, "READY")

    def test_drift_after_ready_is_not_success(self):
        self.client.stop_drift = 0.1
        result = self.run_station()
        self.assertEqual(result.reason, "drifted_after_stop")
        self.assertNotEqual(result.status, "docked")

    def test_executive_limits_and_watchdog_are_respected(self):
        result = self.run_station(control_hz=2.0)
        self.assert_no_start(result, "execution_error:ValueError")

    def test_rejected_zero_does_not_skip_stop(self):
        original = self.client.send_nav_velocity

        def reject_zero(*velocity):
            if velocity == (0.0, 0.0, 0.0):
                raise RuntimeError("rejected zero")
            original(*velocity)

        self.client.send_nav_velocity = reject_zero
        result = self.run_station()
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.client.calls[-1][0], "stop")
        self.assertEqual(self.client.state, "READY")

    def test_pose_cleanup_failure_is_reported(self):

        def fail_close():
            raise RuntimeError("close failed")

        self.source.close = fail_close
        result = self.run_station()
        self.assertEqual(result.reason, "pose_source_close_failed")
        self.assertEqual(result.status, "failed")


class DockingGeometryTest(unittest.TestCase):

    def test_world_delta_rotates_into_body_frame(self):
        station = make_station(target_yaw=math.pi / 2)
        pose = replace(FakePlant().pose(), yaw=math.pi / 2)
        forward, lateral, yaw = station.body_velocity(pose)
        self.assertAlmostEqual(forward, 0.0)
        self.assertLess(lateral, 0.0)
        self.assertAlmostEqual(yaw, 0.0)

    def test_combined_translation_caps_vector_norm(self):
        station = make_station(target_y=0.15)
        forward, lateral, yaw = station.body_velocity(FakePlant().pose())
        self.assertLessEqual(math.hypot(forward, lateral),
                             station.max_speed_mps + 1e-12)
        self.assertLessEqual(abs(yaw), station.max_yaw_rate_rad_s)

    def test_large_yaw_error_turns_without_translation(self):
        command = make_station().body_velocity(
            replace(FakePlant().pose(), yaw=0.4))
        self.assertEqual(command[:2], (0.0, 0.0))
        self.assertLess(command[2], 0.0)

    def test_angle_wrap_near_pi(self):
        station = make_station(target_yaw=-math.pi + 0.01)
        pose = replace(FakePlant().pose(), yaw=math.pi - 0.01)
        self.assertAlmostEqual(station.errors(pose)[1], 0.02)

    def test_invalid_geometry_and_nonfinite_limits(self):
        for kwargs in [{
                "max_speed_mps": float("nan")
        }, {
                "target_x": True
        }, {
                "target_x": 0.74
        }, {
                "entry": Bounds(-0.2, 0.2, -0.1, 0.1)
        }, {
                "max_runtime_s": 0.1
        }, {
                "max_runtime_s": 121.0
        }]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_station(**kwargs)

    def test_registry_sim_modes_and_disabled_station(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stations.json"
            config = {
                "schema_version": 1,
                "runtime": "sim",
                "docking_only": True,
                "stations": {
                    "apple_table_front": {
                        "enabled": False
                    }
                }
            }
            path.write_text(json.dumps(config), encoding="utf-8")
            registry = StationRegistry(path, "sim")
            with self.assertRaises(ValueError):
                registry.get("apple_table_front")
            with self.assertRaises(ValueError):
                registry.get("unknown")
            with self.assertRaises(ValueError):
                StationRegistry(path, "real")
            config["docking_only"] = False
            path.write_text(json.dumps(config), encoding="utf-8")
            registry = StationRegistry(path, "sim")
            self.assertFalse(registry.docking_only)
            self.assertEqual(registry.public_status()["mode"],
                             "simulation_docking_and_grasp")
            self.assertTrue(registry.public_status()["grasp_enabled"])

    def test_valid_registry_does_not_publish_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stations.json"
            data = asdict(make_station())
            data.pop("name")
            config = {
                "schema_version": 1,
                "runtime": "sim",
                "docking_only": True,
                "stations": {
                    "apple_table_front": {
                        **data, "enabled": True,
                        "calibration_status": "trial"
                    }
                }
            }
            path.write_text(json.dumps(config), encoding="utf-8")
            registry = StationRegistry(path, "sim")
            self.assertEqual(registry.get("apple_table_front").target_x, 0.5)
            self.assertNotIn("target_x", json.dumps(registry.public_status()))


class PoseProtocolTest(unittest.TestCase):

    def payload(self):
        return {
            "wall_time_s": 100.0,
            "sim_time_s": 1.0,
            "base_quat": [1.0, 0.0, 0.0, 0.0],
            "diagnostics": {
                "robot_base_xyz": [0.1, 0.0, 0.75],
                "support_enabled": False
            }
        }

    def decode(self, payload):
        return decode_sim_robot_pose(msgpack.packb(payload, use_bin_type=True))

    def test_wxyz_quaternion_and_no_object_dependency(self):
        payload = self.payload()
        payload["base_quat"] = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
        pose = self.decode(payload)
        self.assertAlmostEqual(pose.yaw, math.pi / 2)
        self.assertAlmostEqual(pose.tilt, 0.0)
        self.assertTrue(pose.fresh(100.2, 0.3))
        self.assertFalse(pose.fresh(101.0, 0.3))
        self.assertFalse(pose.fresh(99.0, 0.3))

    def test_reject_missing_support_and_nonfinite_positions(self):
        payload = self.payload()
        del payload["diagnostics"]["support_enabled"]
        with self.assertRaises(KeyError):
            self.decode(payload)
        for value in [True, float("nan"), "0.1"]:
            payload = self.payload()
            payload["diagnostics"]["robot_base_xyz"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.decode(payload)

    def test_reject_invalid_quaternion_and_support_type(self):
        for quat in [[0, 0, 0, 0], [1, 0, 0], [float("inf"), 0, 0, 0]]:
            payload = self.payload()
            payload["base_quat"] = quat
            with self.subTest(quat=quat), self.assertRaises(ValueError):
                self.decode(payload)
        payload = self.payload()
        payload["diagnostics"]["support_enabled"] = "false"
        with self.assertRaises(ValueError):
            self.decode(payload)


class PoseSubscriberTest(unittest.TestCase):

    def setUp(self):
        import zmq

        self.temp = tempfile.TemporaryDirectory()
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        endpoint = "ipc://" + str(Path(self.temp.name) / "pose.ipc")
        self.publisher.bind(endpoint)
        self.source = SimDockingPoseSource(endpoint)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.context.term)
        self.addCleanup(self.publisher.close)
        self.addCleanup(self.source.close)

    def send_until(self, raw, predicate):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.publisher.send(raw)
            time.sleep(0.01)
            result = self.source.latest(1.0)
            if predicate(result):
                return result
        self.fail("local pose subscriber did not receive expected frame")

    def payload(self, sim_time=1.0):
        return msgpack.packb(
            {
                "wall_time_s": time.time(),
                "sim_time_s": sim_time,
                "base_quat": [1.0, 0.0, 0.0, 0.0],
                "diagnostics": {
                    "robot_base_xyz": [0.1, 0.0, 0.75],
                    "support_enabled": False
                }
            },
            use_bin_type=True)

    def test_actual_ipc_frame_and_invalid_data_latch(self):
        pose = self.send_until(self.payload(), lambda p: p is not None)
        self.assertAlmostEqual(pose.x, 0.1)
        self.send_until(b"invalid",
                        lambda _: self.source.error.startswith("invalid_pose"))
        self.publisher.send(self.payload(2.0))
        time.sleep(0.02)
        self.assertIsNone(self.source.latest(1.0))

    def test_simulation_reset_latches_failure(self):
        self.send_until(self.payload(10.0), lambda p: p is not None)
        self.send_until(self.payload(1.0),
                        lambda _: self.source.error.startswith("invalid_pose"))
        self.assertIsNone(self.source.latest(1.0))


if __name__ == "__main__":
    unittest.main()
