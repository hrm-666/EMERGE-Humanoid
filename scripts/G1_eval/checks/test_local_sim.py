"""Check ordinary-user simulation endpoints without launching services."""
import argparse
import tempfile
import unittest
from unittest import mock

from scripts.G1_eval.local_sim import build_command


class LocalSimulationLaunchTest(unittest.TestCase):

    def test_services_never_use_privileged_networking(self):
        with tempfile.TemporaryDirectory() as temp:
            for service in ("sim", "sonic", "gr00t", "lightnav"):
                with self.subTest(service=service), mock.patch.dict(
                        "os.environ", {"DISPLAY": ":0"}):
                    args = argparse.Namespace(service=service,
                                              output_dir=temp,
                                              headless=False,
                                              duration=0,
                                              checkpoint=temp)
                    command, env = build_command(args)
                    self.assertNotIn("sudo", command)
                    self.assertNotIn("runuser", command)
                    self.assertNotIn("netns", command)
                    self.assertIn("127.0.0.1", " ".join(command))
                    if service in ("sim", "sonic"):
                        self.assertIn("lo", command)
                    self.assertNotIn("OPENAI_API_KEY", " ".join(command))


class ViewerTimingTest(unittest.TestCase):

    def test_physics_ticks_do_not_force_gui_refresh_or_catchup_bursts(self):
        from scripts.G1_eval.sim_runtime import RateLimitedViewer
        viewer = mock.Mock()
        now = [0.0]
        wrapper = RateLimitedViewer(viewer, clock=lambda: now[0])
        for i in range(500):
            now[0] = i * 0.002
            wrapper.sync()
        self.assertGreaterEqual(viewer.sync.call_count, 29)
        self.assertLessEqual(viewer.sync.call_count, 30)
        before = viewer.sync.call_count
        now[0] = 20.0
        for _ in range(10):
            wrapper.sync()
        self.assertEqual(viewer.sync.call_count, before + 1)
        wrapper.close()
        viewer.close.assert_called_once()
