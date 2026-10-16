"""Keep model termination separate from physical arrival in simulation reports."""

import unittest

from integrations.lightnav_sonic_emerge.bridge import tabletop_evaluate


def summary(reason="LightNav stop=true", model_stop=True, x=-1.0, yaw=0.0):
    return {
        "end_reason": reason,
        "last_model_stop": model_stop,
        "end_wall_time_s": 100.1,
        "end_robot_pose": {
            "position_m": [x, 0.0, 0.78],
            "yaw_rad": yaw,
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "wall_time_s": 100.0,
        },
    }


class TabletopEvaluationTests(unittest.TestCase):

    def test_guard_stop_at_goal_is_not_model_success(self):
        report = tabletop_evaluate.evaluate_summary(
            summary("turn guard", False), 0.0, 0.0)
        self.assertTrue(report["geometry_arrival"])
        self.assertFalse(report["model_stop_at_goal"])

    def test_model_stop_far_from_table_is_not_success(self):
        report = tabletop_evaluate.evaluate_summary(summary(x=-3.0), 0.0, 0.0)
        self.assertTrue(report["model_stop"])
        self.assertFalse(report["geometry_arrival"])
        self.assertFalse(report["model_stop_at_goal"])

    def test_valid_model_stop_in_declared_region_passes(self):
        self.assertTrue(
            tabletop_evaluate.evaluate_summary(summary(), 0.0,
                                               0.0)["model_stop_at_goal"])

    def test_wrong_heading_or_fallen_pose_fails(self):
        for trial in (summary(yaw=1.5), summary()):
            if trial["end_robot_pose"]["yaw_rad"] == 0.0:
                trial["end_robot_pose"]["quaternion_wxyz"] = [
                    0.0, 1.0, 0.0, 0.0
                ]
            self.assertFalse(
                tabletop_evaluate.evaluate_summary(trial, 0.0,
                                                   0.0)["geometry_arrival"])

    def test_stale_pose_is_unknown_not_success(self):
        trial = summary()
        trial["end_robot_pose"]["wall_time_s"] = 90.0
        report = tabletop_evaluate.evaluate_summary(trial, 0.0, 0.0)
        self.assertIsNone(report["geometry_arrival"])
        self.assertIsNone(report["model_stop_at_goal"])


if __name__ == "__main__":
    unittest.main()
