"""Simulation launch must not need privileged networking or a password."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robot.drivers.official01_executive import ExecutiveProcessClient
from robot.drivers.official01_runtime import NavigationHoldMixin


class ExecutiveLaunchTest(unittest.TestCase):

    def test_simulation_uses_direct_loopback_process(self):
        with tempfile.TemporaryDirectory() as temp:
            script = Path(
                temp
            ) / "integrations/official01_real_deployment/official01_executive.py"
            script.parent.mkdir(parents=True)
            script.touch()
            client = ExecutiveProcessClient(temp, runtime="sim")
            argv = client._build_argv()
            self.assertNotIn("sudo", argv)
            self.assertNotIn("ip", argv)
            self.assertIn("--confirm-sim", argv)
            self.assertEqual(argv[argv.index("--network-interface") + 1], "lo")
            for flag in ("--policy-host", "--action-host", "--state-host",
                         "--observation-host"):
                self.assertEqual(argv[argv.index(flag) + 1], "127.0.0.1")

    def test_invalid_watchdog_is_rejected(self):
        for value in (float("nan"), float("inf"), -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExecutiveProcessClient("/tmp/control", command_watchdog_s=value)


class LegacyCapture:
    """Model the legacy measurement capture, preserving its transition guard."""

    def _handle_command(self, command):
        if command.name == "move_keyboard":
            if self.state != "READY":
                raise ValueError("not ready")
            self.arm_hold = self.measured.copy()
            self.state = "MOVING"
        elif command.name == "pause":
            self.arm_hold = self.measured.copy()
            self.state = "READY"


class StableHold(NavigationHoldMixin, LegacyCapture):
    pass


class NavigationHoldTest(unittest.TestCase):

    def test_repeated_navigation_does_not_accumulate_tracking_error(self):
        runtime = StableHold()
        target = np.linspace(-0.2, 0.6, 14)
        runtime.arm_hold = target.copy()
        for _ in range(5):
            runtime.state = "READY"
            runtime.measured = runtime.arm_hold + 0.15
            runtime._handle_command(SimpleNamespace(name="move_keyboard"))
            self.assertEqual(runtime.state, "MOVING")
            np.testing.assert_array_equal(runtime.arm_hold, target)

    def test_grasp_pause_still_captures_current_pose(self):
        runtime = StableHold()
        runtime.state = "INFERENCING"
        runtime.arm_hold = np.zeros(14)
        runtime.measured = np.ones(14)
        runtime._handle_command(SimpleNamespace(name="pause"))
        np.testing.assert_array_equal(runtime.arm_hold, runtime.measured)

    def test_transition_guard_is_preserved(self):
        runtime = StableHold()
        runtime.state = "INFERENCING"
        runtime.arm_hold = np.zeros(14)
        runtime.measured = np.ones(14)
        with self.assertRaises(ValueError):
            runtime._handle_command(SimpleNamespace(name="move_keyboard"))
        self.assertEqual(runtime.state, "INFERENCING")


class LegacyPoseGate:

    def _stable(self, debug):
        self.legacy_debug = debug
        return False


class NavigationRestRuntime(NavigationHoldMixin, LegacyPoseGate):

    @staticmethod
    def _roll_pitch(quaternion):
        return 0.0, 0.0


class NavigationRestTest(unittest.TestCase):

    def setUp(self):
        self.runtime = NavigationRestRuntime()
        self.runtime.state = "MOVING"
        self.runtime.stop_requested = True
        self.runtime.nav = np.zeros(3)
        self.runtime.arm_hold = np.zeros(14)
        self.runtime.args = SimpleNamespace(ready_error_rad=0.30,
                                            ready_tilt_rad=0.25,
                                            ready_gyro_rad_s=0.8,
                                            stable_hold_s=0.5)
        self.runtime._stable_since = None
        self.measured = np.r_[np.zeros(15), np.full(14, 0.34)]
        self.runtime.body = mock.Mock()
        self.runtime.body.read.return_value = {
            "last_action": self.measured.copy()
        }
        self.debug = {
            "measured": self.measured.copy(),
            "target": self.measured.copy(),
            "dq": np.zeros(29),
            "base_quat": np.array([1., 0., 0., 0.]),
            "base_ang_vel": np.zeros(3)
        }

    def test_navigation_rest_is_not_a_reference_pose_success(self):
        with mock.patch("robot.drivers.official01_runtime.time.monotonic",
                        side_effect=[1., 1.6]):
            self.assertFalse(self.runtime._stable(self.debug))
            self.assertTrue(self.runtime._stable(self.debug))
        np.testing.assert_array_equal(self.runtime.arm_hold, np.zeros(14))

    def test_pose_preparation_retains_legacy_gate(self):
        self.runtime.state = "ADJUSTING"
        self.assertFalse(self.runtime._stable(self.debug))
        self.assertIs(self.runtime.legacy_debug, self.debug)
        self.runtime.body.read.assert_not_called()

    def test_rejects_missing_or_invalid_motor_feedback(self):
        for value in (None, {}, {
                "last_action": [0.] * 28
        }, {
                "last_action": [float("nan")] * 29
        }):
            with self.subTest(value=value):
                self.runtime._stable_since = 0.0
                self.runtime.body.read.return_value = value
                self.assertFalse(self.runtime._stable(self.debug))
                self.assertIsNone(self.runtime._stable_since)

    def test_rejects_unfinished_motion(self):
        self.runtime.nav[0] = 0.25
        self.assertFalse(self.runtime._stable(self.debug))
        self.runtime.nav[:] = 0.
        self.runtime.stop_requested = False
        self.assertFalse(self.runtime._stable(self.debug))

    def test_tracking_velocity_and_body_gates_reset_hold_timer(self):
        for field, index, value in [("measured", 15, 1.), ("dq", 20, .2),
                                    ("target", 0, .4), ("base_ang_vel", 0, 1.)]:
            with self.subTest(field=field):
                debug = {k: v.copy() for k, v in self.debug.items()}
                debug[field][index] = value
                self.runtime._stable_since = 0.
                self.assertFalse(self.runtime._stable(debug))
                self.assertIsNone(self.runtime._stable_since)
        self.runtime._roll_pitch = lambda _q: (.3, 0.)
        self.assertFalse(self.runtime._stable(self.debug))

    def test_reached_reference_allows_static_motor_torque_offset(self):
        self.runtime.arm_hold = self.measured[15:].copy()
        self.runtime.body.read.return_value = {
            "last_action": self.measured + 0.45
        }
        with mock.patch("robot.drivers.official01_runtime.time.monotonic",
                        side_effect=[1., 1.6]):
            self.assertFalse(self.runtime._stable(self.debug))
            self.assertTrue(self.runtime._stable(self.debug))
