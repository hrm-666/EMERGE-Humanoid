import math
import unittest

from robot.navigation.lightnav_sonic import adapter
from scripts.LightNav_eval import teleop


class TeleopCommandTests(unittest.TestCase):

    def test_forward_at_zero_yaw_is_world_x(self) -> None:
        state = teleop.TeleopState(forward=1.0, speed_mps=0.18)
        command = teleop.command_from_state(state,
                                            0.0,
                                            heading_tolerance_rad=0.05)
        self.assertEqual(command.mode, adapter.SONIC_SLOW_WALK_MODE)
        self.assertEqual(command.movement_direction, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(command.movement_speed, 0.18)

    def test_local_left_rotates_into_world_frame(self) -> None:
        state = teleop.TeleopState(lateral=1.0, speed_mps=0.18)
        command = teleop.command_from_state(state,
                                            math.pi / 2.0,
                                            heading_tolerance_rad=0.05)
        self.assertAlmostEqual(command.movement_direction[0], -1.0, places=6)
        self.assertAlmostEqual(command.movement_direction[1], 0.0, places=6)

    def test_turn_in_place_uses_default_planner_speed(self) -> None:
        state = teleop.TeleopState(target_yaw=math.pi / 2.0)
        command = teleop.command_from_state(state,
                                            0.0,
                                            heading_tolerance_rad=0.05)
        self.assertEqual(command.mode, adapter.SONIC_SLOW_WALK_MODE)
        self.assertEqual(command.movement_direction, (0.0, 0.0, 0.0))
        self.assertEqual(command.movement_speed, -1.0)
        self.assertAlmostEqual(command.facing_direction[1], 1.0)

    def test_reached_heading_is_idle(self) -> None:
        state = teleop.TeleopState(target_yaw=0.02)
        command = teleop.command_from_state(state,
                                            0.0,
                                            heading_tolerance_rad=0.05)
        self.assertEqual(command.mode, adapter.SONIC_IDLE_MODE)


if __name__ == "__main__":
    unittest.main()
