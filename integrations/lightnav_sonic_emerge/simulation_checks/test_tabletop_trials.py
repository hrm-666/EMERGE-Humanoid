"""Reproducible trial starts, capture isolation, and offline video/evaluation."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import mujoco
import numpy as np

from integrations.lightnav_sonic_emerge.bridge import official01_observation
from integrations.lightnav_sonic_emerge.bridge import tabletop
from integrations.lightnav_sonic_emerge.bridge import tabletop_evaluate
from integrations.lightnav_sonic_emerge.bridge import tabletop_logging
from integrations.lightnav_sonic_emerge.bridge import tabletop_pose
from integrations.lightnav_sonic_emerge.bridge import tabletop_recording
from integrations.lightnav_sonic_emerge.bridge import tabletop_start
from integrations.lightnav_sonic_emerge.bridge import tabletop_video


class RandomTrialTests(unittest.TestCase):

    def test_reproducible_varied_disk_samples_with_clearance(self):
        table = tabletop_start.TableFootprint(0.5, 0, 0.6, 0.4, math.pi / 2)
        first, second = tabletop_start.RandomStartSampler(
            table, 7), tabletop_start.RandomStartSampler(table, 7)
        samples = [first.sample(3) for _ in range(500)]
        self.assertEqual(samples, [second.sample(3) for _ in range(500)])
        for sample in samples:
            x, y = sample["position_xy_m"]
            self.assertLessEqual(math.hypot(x - 0.5, y), 3)
            self.assertFalse(table.contains(x, y, 0.6))
        self.assertLess(min(s["yaw_rad"] for s in samples), -3)
        self.assertGreater(max(s["yaw_rad"] for s in samples), 3)
        # Conditional uniform disk: cells with equal area outside the rectangle
        # have comparable counts. Test radial second moment with negligible exclusion.
        large = tabletop_start.RandomStartSampler(table, 19)
        radii = [
            sum(v * v
                for v in (s["position_xy_m"][0] - 0.5, s["position_xy_m"][1]))
            for s in (large.sample(100) for _ in range(4000))
        ]
        self.assertAlmostEqual(sum(radii) / len(radii) / 10000,
                               0.5,
                               delta=0.025)
        replay = second.sample(3, samples[0]["sample_seed"])
        self.assertEqual(replay["position_xy_m"], samples[0]["position_xy_m"])
        self.assertEqual(replay["yaw_rad"], samples[0]["yaw_rad"])
        for radius in (0, 0.9, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                first.sample(radius)

    def test_only_xy_yaw_changed_roll_pitch_height_joints_preserved(self):
        qpos = np.array(
            [0, 0, 0.8, math.cos(0.1),
             math.sin(0.1), 0, 0, 0.3, 0.4])
        before = qpos.copy()
        tabletop_start.apply_start(qpos, 0, {
            "position_xy_m": [2, 3],
            "yaw_rad": 1.4
        })
        np.testing.assert_array_equal(qpos[7:], before[7:])
        self.assertEqual(qpos[2], before[2])
        pose = tabletop_pose.RobotPose(tuple(qpos[:3]), tuple(qpos[3:7]), 1, 1)
        self.assertAlmostEqual(pose.yaw_rad, 1.4)
        self.assertAlmostEqual(qpos[4]**2 + qpos[5]**2, before[4]**2)

    def test_reset_syntax_and_bad_values(self):
        self.assertEqual(tabletop.parse_reset_command("reset"), {})
        self.assertEqual(tabletop.parse_reset_command("reset random 3 42"), {
            "start_mode": "random",
            "radius_m": 3,
            "sample_seed": 42
        })
        self.assertEqual(tabletop.parse_reset_command("reset fixed"),
                         {"start_mode": "fixed"})
        for text in ("reset random", "reset random nan", "reset random -1",
                     "reset fixed 1"):
            with self.assertRaises(ValueError):
                tabletop.parse_reset_command(text)

    def test_any_side_goal_and_model_stop_are_separate(self):
        table = tabletop_start.TableFootprint(0, 0, 0.5, 0.3)
        trial = {
            "end_reason": "LightNav stop=true",
            "last_model_stop": True,
            "end_wall_time_s": 100.1,
            "end_robot_pose": {
                "position_m": [1.5, 0, 0.8],
                "yaw_rad": math.pi,
                "quaternion_wxyz": [0, 0, 0, 1],
                "wall_time_s": 100
            }
        }
        self.assertFalse(
            tabletop_evaluate.evaluate_summary(trial, -0.5,
                                               0)["geometry_arrival"])
        self.assertTrue(
            tabletop_evaluate.evaluate_any_side(trial,
                                                table)["model_stop_at_goal"])
        trial["end_reason"] = "maximum navigation runtime reached"
        self.assertTrue(
            tabletop_evaluate.evaluate_any_side(trial,
                                                table)["geometry_arrival"])
        self.assertFalse(
            tabletop_evaluate.evaluate_any_side(trial,
                                                table)["model_stop_at_goal"])
        trial["end_robot_pose"]["position_m"] = [5, 0, 0.8]
        self.assertFalse(
            tabletop_evaluate.evaluate_any_side(trial,
                                                table)["geometry_arrival"])

    def test_late_prediction_cannot_change_terminal_model_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = tabletop_logging.TabletopRunLog(Path(temporary))
            task = log.start(1, "go", None)
            log.record("prediction", generation=1, stop=True)
            log.finish("LightNav stop=true", None)
            log.record("prediction",
                       generation=1,
                       stop=False,
                       reason="discarded: old")
            self.assertTrue(task.summary["model_stop_at_end"])
            log.close()


class RecordingTests(unittest.TestCase):

    def test_capture_isolated_capped_and_mp4_preserves_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log = tabletop_logging.TabletopRunLog(directory)
            task = log.start(1, "go", None)
            start = task.summary["start_wall_time_s"]
            recording = tabletop_recording.TaskRecording(task, 10)
            ok, jpeg = cv2.imencode(".jpg",
                                    np.full((48, 64, 3), 120, dtype=np.uint8))
            self.assertTrue(ok)
            for offset in (0, 0.02, 0.2, 0.9):
                pose = tabletop_pose.RobotPose((0, 0, 0.8), (1, 0, 0, 0),
                                               start + offset, offset, (1, 2))
                recording.capture(
                    official01_observation.ObservationFrame(
                        jpeg.tobytes(), start + offset, offset), pose)
            self.assertEqual(recording.frames, 3)
            self.assertEqual(recording.poses, 3)
            with patch(
                    "integrations.lightnav_sonic_emerge.bridge.tabletop_logging.time.time",
                    return_value=start + 1):
                log.finish("operator stop", None)
            recording.finish(task)
            report = tabletop_video.render_ego(
                task.directory,
                json.loads((task.directory / "recording.json").read_text()))
            self.assertEqual(report["frames"], 10)
            capture = cv2.VideoCapture(str(task.directory / "ego.mp4"))
            self.assertTrue(capture.read()[0])
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 10)
            capture.release()
            other = log.start(2, "next", None)
            self.assertFalse((other.directory / "ego.mp4").exists())
            self.assertEqual(
                len((task.directory /
                     "scene_poses.jsonl").read_text().splitlines()), 3)
            log.close()

    def test_postprocess_removes_jpegs_after_mp4_generation(self):

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "video_frames").mkdir()
            for name in ("000000.jpg", "000001.jpg"):
                (directory / "video_frames" / name).write_bytes(b"jpeg")
            for name in ("frame-000001.jpg", "state-000001.jpg", "end_ego.jpg"):
                (directory / name).write_bytes(b"jpeg")
            removed = tabletop_video._cleanup_intermediate_images(directory)
            self.assertEqual(len(removed), 5)
            self.assertFalse((directory / "video_frames").exists())
            self.assertEqual(list(directory.glob("*.jpg")), [])

    def test_timeline_repeats_missing_frames(self):
        samples = [{"wall_time_s": 0, "n": 0}, {"wall_time_s": 0.4, "n": 1}]
        self.assertEqual([
            s["n"]
            for s in tabletop_recording.timeline_samples(samples, 0, 1, 5)
        ], [0, 0, 1, 1, 1])
        with self.assertRaises(ValueError):
            list(tabletop_recording.timeline_samples([], 0, 1, 5))

    def test_artifact_submission_does_not_block_control(self):
        import threading

        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as temporary:
            task = SimpleNamespace(directory=Path(temporary))
            worker = tabletop_recording.TaskArtifactWorker()

            def blocked(directory):
                entered.set()
                release.wait(2)
                (directory /
                 "artifacts.json").write_text('{"status":"complete"}')

            try:
                with patch.object(worker, "_process", side_effect=blocked):
                    worker.submit(task)
                    self.assertTrue(entered.wait(1))
                    self.assertEqual(
                        json.loads((task.directory /
                                    "artifacts.json").read_text())["status"],
                        "pending")
            finally:
                release.set()
                worker.close()

    def test_third_person_replay_never_steps_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model_path = directory / "scene.xml"
            model_path.write_text('''<mujoco><worldbody>
            <light pos="0 0 4"/><geom type="plane" size="10 10 .1"/>
            <body name="pelvis" pos="-2 0 .8"><freejoint/><geom size=".1"/></body>
            <body name="tabletop_lab_table" pos=".5 0 .7"><geom type="box" size=".4 .6 .03"/></body>
            </worldbody></mujoco>''')
            model = mujoco.MjModel.from_xml_path(str(model_path))
            rows = [{
                "wall_time_s": t,
                "sim_time_s": t,
                "qpos": model.qpos0.tolist()
            } for t in (0, 0.2)]
            (directory / "scene_poses.jsonl").write_text("\n".join(
                json.dumps(row) for row in rows))
            with patch.object(mujoco,
                              "mj_step",
                              side_effect=AssertionError("must not step")):
                report = tabletop_video.render_third_person(
                    directory, {
                        "start_wall_time_s": 0,
                        "end_wall_time_s": 0.4,
                        "fps": 5
                    }, model_path)
            self.assertEqual(report["frames"], 2)
            cap = cv2.VideoCapture(str(directory / "third_person.mp4"))
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
