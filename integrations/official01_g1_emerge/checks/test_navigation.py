from __future__ import annotations

import tempfile
import unittest

from integrations.lightnav_sonic_emerge.bridge.adapter import LightNav2SonicAdapter
from integrations.lightnav_sonic_emerge.navigation_backend import NavigationConfig
from integrations.lightnav_sonic_emerge.observation_writer import EgoObservationWriter
from integrations.official01_g1_emerge.navigation import ExecutiveNavigationRunner


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
    def __init__(self):
        self.keys = []
        self.stopped = False

    def begin_moving(self, *, timeout_s):
        self.begin_timeout = timeout_s

    def send_nav_key(self, key):
        self.keys.append(key)

    def stop_moving(self, *, timeout_s):
        self.stop_timeout = timeout_s
        self.stopped = True


class ExecutiveNavigationTest(unittest.TestCase):
    def test_quantizes_waypoint_and_always_stops(self):
        camera = Resource(b"jpeg")
        yaw = Resource(0.0)
        lightnav = LightNav()
        executive = Executive()
        config = NavigationConfig(
            lightnav_server="ws://test", camera_transport="official01_rgb",
            max_runtime_s=2.0, control_hz=100.0,
        )
        with tempfile.TemporaryDirectory() as temp:
            writer = EgoObservationWriter(temp, enabled=False)
            runner = ExecutiveNavigationRunner(
                config, executive, writer,
                camera_factory=lambda *_args: camera,
                yaw_factory=lambda *_args: yaw,
                lightnav_factory=lambda *_args: lightnav,
                sleep=lambda _seconds: None,
            )
            result = runner.run(instruction="approach table", max_steps=1)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(executive.keys, ["w"])
        self.assertTrue(executive.stopped)
        self.assertTrue(camera.closed and yaw.closed and lightnav.closed)

    def test_yaw_command_mapping(self):
        adapter = LightNav2SonicAdapter()
        left = adapter.convert((0.0, 0.0, 0.5), 0.0)
        right = adapter.convert((0.0, 0.0, -0.5), 0.0)
        self.assertEqual(ExecutiveNavigationRunner._command_key(left, 0.0), "q")
        self.assertEqual(ExecutiveNavigationRunner._command_key(right, 0.0), "e")


if __name__ == "__main__":
    unittest.main()
