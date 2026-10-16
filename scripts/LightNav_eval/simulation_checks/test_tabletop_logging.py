"""Task boundaries and delayed-reply ownership in tabletop logs."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.LightNav_eval import tabletop_logging


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class TabletopLoggingTests(unittest.TestCase):

    def test_late_prediction_stays_in_cancelled_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = tabletop_logging.TabletopRunLog(root)
            try:
                first = log.start(1, "Go to the apple table.", {"yaw_rad": 0})
                log.finish("simulation reset requested", {"yaw_rad": 0.1})
                second = log.start(4,
                                   "Go straight to the apple table and stop.",
                                   None)
                image = log.save_frame("frame-000001.jpg",
                                       b"jpeg",
                                       generation=1)
                log.record("prediction",
                           generation=1,
                           frame=image,
                           reply={"seq": 0},
                           stop=False,
                           raw_text="old reply",
                           pointing=None,
                           first_waypoint=[1, 0, 0],
                           robot_pose=None,
                           reason="discarded: cancelled navigation")
                log.record("control_telemetry", mode="navigation")
                log.finish("LightNav stop=true", None)
                first_events = read_events(first.directory / "events.jsonl")
                second_events = read_events(second.directory / "events.jsonl")
                self.assertEqual(
                    [e["event"] for e in first_events],
                    ["navigation_start", "navigation_end", "prediction"])
                self.assertTrue(first_events[-1]["after_task_end"])
                self.assertEqual(first_events[-1]["frame"], "frame-000001.jpg")
                self.assertEqual(
                    (first.directory / first_events[-1]["frame"]).read_bytes(),
                    b"jpeg")
                self.assertFalse(
                    any(e["event"] == "prediction" for e in second_events))
                summary = json.loads(
                    (first.directory / "summary.json").read_text())
                self.assertEqual(summary["end_reason"],
                                 "simulation reset requested")
                self.assertEqual(summary["late_prediction_count"], 1)
                self.assertEqual(summary["discarded_prediction_count"], 1)
                text = (first.directory / "inference.txt").read_text()
                self.assertIn("TASK END task-0001", text)
                self.assertIn("late=True", text)
                self.assertEqual((root / "latest_task.txt").read_text(),
                                 "tasks/task-0002\n")
            finally:
                log.close()

    def test_idle_and_reset_events_stay_outside_completed_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = tabletop_logging.TabletopRunLog(root)
            try:
                task = log.start(2, "Go forward and stop.", None)
                log.record("prediction",
                           generation=2,
                           reply={"seq": 0},
                           stop=True,
                           raw_text="stop",
                           first_waypoint=[0, 0, 0],
                           reason="LightNav stop=true")
                log.finish("LightNav stop=true", None)
                log.record("control_telemetry", mode="idle")
                log.record("simulation_reset_start", generation=3)
                task_events = read_events(task.directory / "events.jsonl")
                self.assertEqual(task_events[-1]["event"], "navigation_end")
                self.assertEqual(len(task_events), 3)
                session_events = read_events(root / "events.jsonl")
                self.assertIsNone(session_events[-1]["task_id"])
                self.assertIsNone(session_events[-2]["task_id"])
                summary = json.loads(
                    (task.directory / "summary.json").read_text())
                self.assertEqual(summary["status"], "ended")
                self.assertTrue(summary["last_model_stop"])
                self.assertGreaterEqual(summary["duration_s"], 0)
                self.assertIn("+", summary["started_at"])
                self.assertIsNone(log.finish("shutdown", None))
            finally:
                log.close()


if __name__ == "__main__":
    unittest.main()
