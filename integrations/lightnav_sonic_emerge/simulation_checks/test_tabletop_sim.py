"""Viewer pacing must not prevent status/close or change the physics clock."""

import unittest
from unittest.mock import Mock
from unittest.mock import patch

from integrations.lightnav_sonic_emerge.bridge import tabletop_sim


class SimulationDisplayTests(unittest.TestCase):

    def test_display_is_throttled_without_blocking_viewer_status_or_close(self):
        original = Mock()
        viewer = tabletop_sim.RateLimitedViewer(original, 60)
        with patch(
                "integrations.lightnav_sonic_emerge.bridge.tabletop_sim.time.monotonic",
                return_value=1.0):
            viewer.sync()
        with patch(
                "integrations.lightnav_sonic_emerge.bridge.tabletop_sim.time.monotonic",
                return_value=1.002):
            viewer.sync()
        self.assertEqual(original.sync.call_count, 1)
        viewer.is_running()
        viewer.close()
        original.is_running.assert_called_once()
        original.close.assert_called_once()
        with patch(
                "integrations.lightnav_sonic_emerge.bridge.tabletop_sim.time.monotonic",
                return_value=1.02):
            viewer.sync()
        self.assertEqual(original.sync.call_count, 2)
        self.assertEqual(viewer.sync_count, 2)

    def test_clock_report_measures_slow_physics_without_scaling_time(self):
        timing = tabletop_sim.SimulationTiming(10, 100)
        self.assertIsNone(timing.record(10.5, 0.002, 104))
        report = timing.record(10.7, 0.002, 105)
        self.assertIn("realtime_ratio=0.140", report)
        self.assertIn("BELOW_REALTIME", report)
        report = timing.record(15.7, 0.002, 110)
        self.assertIn("realtime_ratio=1.000", report)
        self.assertNotIn("BELOW_REALTIME", report)


if __name__ == "__main__":
    unittest.main()
