from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

from robot.navigation.lightnav_sonic.adapter import AdapterLimits
from robot.navigation.lightnav_sonic.adapter import LightNav2SonicAdapter
from robot.navigation.lightnav_sonic.executive_backend import (
    ExecutiveNavigationRunner,
)
from robot.navigation.lightnav_sonic.navigation_backend import NavigationConfig
from robot.navigation.lightnav_sonic.observation_writer import (
    EgoObservationWriter,
)


class Resource:

    def __init__(self, value=None):
        self.value = value
        self.closed = False

    def latest(self, _timeout):
        return self.value

    def close(self):
        self.closed = True


class Reply:
    first_waypoint = (0.4, 0.0, 0.0)
    stop = False


class LightNav(Resource):

    def connect(self):
        pass

    def predict(self, _jpeg, _instruction):
        return Reply()


class Executive:

    linear_speed_mps = 0.25
    yaw_rate_rad_s = 0.25

    def __init__(self):
        self.keys = []
        self.velocities = []
        self.stopped = False

    def begin_moving(self, *, timeout_s):
        self.begin_timeout = timeout_s

    def send_nav_key(self, key):
        self.keys.append(key)

    def send_nav_velocity(self, *values):
        self.velocities.append(values)

    def stop_moving(self, *, timeout_s):
        self.stop_timeout = timeout_s
        self.stopped = True


class ExecutiveNavigationTest(unittest.TestCase):

    def test_preserves_velocity_and_always_stops(self):
        camera = Resource(b"jpeg")
        yaw = Resource(0.0)
        lightnav = LightNav()
        executive = Executive()
        config = NavigationConfig(
            lightnav_server="ws://test",
            camera_transport="official01_rgb",
            max_runtime_s=2.0,
            control_hz=100.0,
        )
        with tempfile.TemporaryDirectory() as temp:
            writer = EgoObservationWriter(temp, enabled=False)
            runner = ExecutiveNavigationRunner(
                config,
                executive,
                writer,
                camera_factory=lambda *_args: camera,
                yaw_factory=lambda *_args: yaw,
                lightnav_factory=lambda *_args: lightnav,
                sleep=lambda _seconds: None,
            )
            result = runner.run(instruction="approach table", max_steps=1)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(executive.velocities, [(0.25, 0.0, 0.0)])
        self.assertTrue(executive.stopped)
        self.assertTrue(camera.closed and yaw.closed and lightnav.closed)

    def test_missing_yaw_fails_before_entering_moving(self):
        camera = Resource(b"jpeg")
        yaw = Resource(None)
        lightnav = LightNav()
        executive = Executive()
        config = NavigationConfig(
            lightnav_server="ws://test",
            camera_transport="official01_rgb",
            max_runtime_s=2.0,
            control_hz=100.0,
            max_stale_ticks=2,
        )
        with tempfile.TemporaryDirectory() as temp:
            writer = EgoObservationWriter(temp, enabled=False)
            runner = ExecutiveNavigationRunner(
                config,
                executive,
                writer,
                camera_factory=lambda *_args: camera,
                yaw_factory=lambda *_args: yaw,
                lightnav_factory=lambda *_args: lightnav,
                sleep=lambda _seconds: None,
            )
            result = runner.run(instruction="approach table", max_steps=1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "stale_input")
        self.assertIn("yaw", result.error)
        self.assertFalse(hasattr(executive, "begin_timeout"))
        self.assertFalse(executive.stopped)
        self.assertTrue(camera.closed and yaw.closed and lightnav.closed)

    def test_small_forward_waypoint_is_not_a_stop(self):
        adapter = LightNav2SonicAdapter(
            AdapterLimits(position_tolerance_m=0.02))
        # Actual step 19 from the failed UI run: old 12 cm threshold erased it.
        command = adapter.convert((0.1079648, -0.0016599, -0.0503904), 0.0)
        forward, lateral, yaw_rate = ExecutiveNavigationRunner._body_velocity(
            command, 0.0, 0.25)
        self.assertGreater(forward, 0.24)
        self.assertLess(lateral, 0.0)
        self.assertLess(yaw_rate, -0.19)

    def test_turning_does_not_discard_translation(self):
        adapter = LightNav2SonicAdapter()
        command = adapter.convert((0.15, 0.03, 0.3), 1.0)
        forward, lateral, yaw_rate = ExecutiveNavigationRunner._body_velocity(
            command, 1.0, 0.25)
        self.assertGreater(forward, 0.0)
        self.assertGreater(lateral, 0.0)
        self.assertAlmostEqual(yaw_rate, 0.35)

    def test_yaw_only_has_no_translation(self):
        adapter = LightNav2SonicAdapter()
        command = adapter.convert((0, 0, -0.3), 2.0)
        velocity = ExecutiveNavigationRunner._body_velocity(command, 2.0, 0.25)
        self.assertAlmostEqual(velocity[0], 0.0)
        self.assertAlmostEqual(velocity[1], 0.0)
        self.assertAlmostEqual(velocity[2], -0.35)


class NavigationInterruptionTest(unittest.TestCase):

    def run_case(self,
                 predict,
                 *,
                 cancelled=lambda: None,
                 clock=lambda: 0.0,
                 camera_values=None):

        class Camera(Resource):

            def __init__(self):
                super().__init__(b"jpeg")
                self.values = iter(camera_values or [b"jpeg"] * 10)

            def latest(self, _timeout):
                return next(self.values, b"jpeg")

        camera = Camera()
        policy = LightNav()
        policy.predict = predict
        executive = Executive()
        with tempfile.TemporaryDirectory() as temp:
            runner = ExecutiveNavigationRunner(
                NavigationConfig(lightnav_server="ws://test",
                                 max_runtime_s=1,
                                 control_hz=4,
                                 max_stale_ticks=1),
                executive,
                EgoObservationWriter(temp, enabled=False),
                camera_factory=lambda *_: camera,
                yaw_factory=lambda *_: Resource(0.0),
                lightnav_factory=lambda *_: policy,
                sleep=lambda _: None,
                monotonic=clock,
            )
            result = runner.run(instruction="approach apple table",
                                max_steps=2,
                                cancel_check=cancelled)
        self.assertTrue(camera.closed and policy.closed)
        return result, executive

    def test_cancel_during_prediction_sends_no_motion(self):
        state = {"cancel": None}

        def predict(*_):
            state["cancel"] = "user_stop"
            return Reply()

        result, executive = self.run_case(predict,
                                          cancelled=lambda: state["cancel"])
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(executive.velocities, [])
        self.assertTrue(executive.stopped)

    def test_expired_prediction_sends_no_motion(self):
        state = {"time": 0.0}

        def predict(*_):
            state["time"] = 2.0
            return Reply()

        result, executive = self.run_case(predict, clock=lambda: state["time"])
        self.assertEqual(result.reason, "max_runtime_s")
        self.assertEqual(executive.velocities, [])
        self.assertTrue(executive.stopped)

    def test_camera_loss_stops_previous_motion(self):
        result, executive = self.run_case(
            lambda *_: Reply(), camera_values=[b"jpeg", b"jpeg", None])
        self.assertEqual(result.reason, "stale_input")
        self.assertEqual(executive.velocities, [(0.25, 0.0, 0.0)])
        self.assertTrue(executive.stopped)

    def test_model_stop_sends_no_velocity(self):
        result, executive = self.run_case(
            lambda *_: SimpleNamespace(first_waypoint=(0.4, 0, 0), stop=True))
        self.assertEqual(result.reason, "lightnav_stop")
        self.assertTrue(result.lightnav_stop)
        self.assertEqual(executive.velocities, [])
        self.assertTrue(executive.stopped)

    def test_zero_waypoint_keeps_moving_until_action_cleanup(self):
        from unittest import mock
        with mock.patch.object(Executive,
                               "begin_moving") as begin, mock.patch.object(
                                   Executive, "stop_moving") as stop:
            result, executive = self.run_case(lambda *_: SimpleNamespace(
                first_waypoint=(0, 0, 0), stop=False))
        self.assertEqual(result.reason, "max_steps")
        self.assertEqual(executive.velocities, [(0., 0., 0.)] * 2)
        self.assertEqual(begin.call_count, 1)
        self.assertEqual(stop.call_count, 1)

    def test_stop_failure_is_not_success(self):
        from unittest import mock
        with mock.patch.object(Executive,
                               "stop_moving",
                               side_effect=RuntimeError("lost owner")):
            result, _ = self.run_case(lambda *_: Reply())
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "stop_or_stability_failed")


if __name__ == "__main__":
    unittest.main()
