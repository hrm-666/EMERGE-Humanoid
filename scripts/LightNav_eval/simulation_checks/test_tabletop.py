import io
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgpack
import numpy as np
from PIL import Image

from robot.vla.lightnav_sonic import lightnav_ws
from robot.vla.lightnav_sonic import official01_observation
from scripts.LightNav_eval import tabletop


def observation_bytes(*,
                      wall_time=None,
                      dtype="|u1",
                      shape=(1, 1, 480, 640, 3)):
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # RGB red, not OpenCV BGR.
    return msgpack.packb({
        "wall_time_s": time.time() if wall_time is None else wall_time,
        "sim_time_s": 1.25,
        "observation": {
            "video": {
                "ego_view": {
                    b"nd": True,
                    b"type": dtype,
                    b"kind": b"",
                    b"shape": shape,
                    b"data": rgb.tobytes(),
                }
            }
        },
        "apple_xyz": [999, 999, 999],  # Must not influence the image.
    })


class ObservationTests(unittest.TestCase):

    def test_original_msgpack_numpy_layout_and_rgb_order(self):
        frame = official01_observation.decode_observation(observation_bytes())
        image = np.array(Image.open(io.BytesIO(frame.jpeg)))
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertGreater(int(image[0, 0, 0]), 250)
        self.assertLess(int(image[0, 0, 2]), 5)
        self.assertEqual(frame.sim_time_s, 1.25)

    def test_queued_old_frame_does_not_become_fresh_on_receipt(self):
        frame = official01_observation.decode_observation(
            observation_bytes(wall_time=time.time() - 5))
        self.assertFalse(frame.fresh(1.0))

    def test_wrong_dtype_shape_and_pickle_envelopes_are_rejected(self):
        for options in ({
                "dtype": "|O"
        }, {
                "shape": (480, 640, 3)
        }, {
                "dtype": "<f4"
        }):
            with self.subTest(options=options), self.assertRaises(ValueError):
                official01_observation.decode_observation(
                    observation_bytes(**options))

    def test_nonfinite_and_future_timestamps_cannot_drive_motion(self):
        with self.assertRaises(ValueError):
            official01_observation.decode_observation(
                observation_bytes(wall_time=float("nan")))
        frame = official01_observation.decode_observation(
            observation_bytes(wall_time=time.time() + 10))
        self.assertFalse(frame.fresh(1.0))


class TabletopControlTests(unittest.TestCase):

    def setUp(self):
        self.args = tabletop.build_parser().parse_args([])
        tabletop.validate_args(self.args)
        self.state = tabletop.TabletopControl(self.args)
        self.frame = official01_observation.ObservationFrame(
            b"jpeg", time.time(), 1.0)
        self.reply = lightnav_ws.LightNavReply(0, ((0.5, 0.0, 0.0),), False)

    def test_default_is_idle_and_motion_requires_opt_in(self):
        self.assertFalse(self.args.enable_motion)
        self.assertTrue(self.state.tick(0, True).is_idle)

    def test_idle_heading_does_not_follow_imu_jitter_or_drift(self):
        first = self.state.tick(0.01, True)
        for yaw in (0.011, 0.008, 0.1, 0.6):
            command = self.state.tick(yaw, True)
            self.assertEqual(command.facing_direction, first.facing_direction)
            self.assertEqual(command.movement_speed, 0)
            self.assertEqual(command.movement_direction, (0, 0, 0))
            self.assertTrue(command.is_idle)

    def test_idle_keeps_heading_on_feedback_loss(self):
        first = self.state.tick(0.4, True)
        self.assertEqual(
            self.state.tick(None, False).facing_direction,
            first.facing_direction)

    def test_stop_after_manual_move_latches_new_heading(self):
        self.state.tick(0, True)
        self.state.key("w", 0.5)
        self.assertFalse(self.state.tick(0.5, True).is_idle)
        self.state.stop("operator stop")
        first = self.state.tick(0.52, True)
        self.assertAlmostEqual(
            math.atan2(first.facing_direction[1], first.facing_direction[0]),
            0.52)
        self.assertEqual(
            self.state.tick(0.6, True).facing_direction, first.facing_direction)

    def test_reset_can_replace_the_previous_idle_heading(self):
        self.state.tick(0.6, True)
        self.state.hold_heading(0)
        self.assertEqual(
            self.state.tick(0.01, True).facing_direction, (1, 0, 0))

    def test_keyboard_is_body_relative_and_expires(self):
        self.state.key("s", math.pi / 2)
        command = self.state.tick(math.pi / 2, True)
        self.assertAlmostEqual(command.movement_direction[1], -1)
        self.assertAlmostEqual(command.movement_speed, 0.50)
        self.state.manual_until = time.monotonic() - 0.01
        self.assertTrue(self.state.tick(0, True).is_idle)

    def test_repeat_key_keeps_heading_and_direction_stable(self):
        self.state.key("s", 0)
        first = self.state.tick(0, True)
        self.state.key("s", 0.1)
        self.assertEqual(self.state.tick(0.2, True), first)

    def test_repeated_turn_keys_accumulate_instead_of_resetting_from_feedback(
            self):
        for key, sign in (("q", 1), ("e", -1)):
            state = tabletop.TabletopControl(self.args)
            for _ in range(3):
                state.key(key, 0)
            self.assertAlmostEqual(state.manual.target_yaw,
                                   sign * math.radians(45))
            command = state.tick(0, True, root_yaw_rad=0)
            self.assertFalse(command.is_idle)
            self.assertEqual(command.movement_direction, (0, 0, 0))
            self.assertEqual(command.movement_speed, -1)
            target = math.atan2(command.facing_direction[1],
                                command.facing_direction[0])
            self.assertAlmostEqual(target, sign * math.radians(45))

    def test_auto_repeat_has_bounded_turn_lead_and_can_reverse(self):
        for _ in range(20):
            self.state.key("q", 0)
        self.assertAlmostEqual(self.state.manual.target_yaw, math.radians(60))
        self.state.key("e", 0)
        self.assertAlmostEqual(self.state.manual.target_yaw, math.radians(45))
        self.state.key("q", math.radians(30))
        self.assertAlmostEqual(self.state.manual.target_yaw, math.radians(60))
        self.state.key("q", math.radians(30))
        self.assertAlmostEqual(self.state.manual.target_yaw, math.radians(75))

    def test_upper_body_heading_alone_cannot_complete_pelvis_turn(self):
        self.state.key("q", 0)
        target = self.state.manual.target_yaw
        command = self.state.tick(target, True, root_yaw_rad=0)
        self.assertFalse(command.is_idle)
        self.assertEqual(self.state.mode, "manual")
        command = self.state.tick(target,
                                  True,
                                  root_yaw_rad=target - math.radians(1))
        self.assertTrue(command.is_idle)
        self.assertEqual(self.state.reason, "keyboard turn complete")
        self.assertEqual(self.state.mode, "idle")
        self.assertAlmostEqual(self.state.idle_yaw_rad, target)

    def test_single_turn_key_has_time_to_finish_then_times_out(self):
        with patch("scripts.LightNav_eval.tabletop.time.monotonic",
                   return_value=10):
            self.state.key("q", 0)
        with patch("scripts.LightNav_eval.tabletop.time.monotonic",
                   return_value=12):
            self.assertFalse(self.state.tick(0, True, root_yaw_rad=0).is_idle)
        with patch("scripts.LightNav_eval.tabletop.time.monotonic",
                   return_value=16.01):
            self.assertTrue(self.state.tick(0, True, root_yaw_rad=0).is_idle)
            self.assertEqual(self.state.reason, "keyboard turn timeout")

    def test_turn_stops_on_missing_pelvis_feedback_and_keeps_last_root_heading(
            self):
        self.state.key("q", 0)
        self.state.tick(0.2, True, root_yaw_rad=0.01)
        command = self.state.tick(0.2, True)
        self.assertTrue(command.is_idle)
        self.assertEqual(self.state.reason,
                         "pelvis yaw feedback lost; turn cancelled")
        self.assertAlmostEqual(self.state.idle_yaw_rad, 0.01)

    def test_space_translation_and_new_trial_cancel_pending_turn(self):
        self.state.key("q", 0)
        self.state.key(" ", 0)
        self.assertTrue(self.state.tick(0, True, root_yaw_rad=0).is_idle)
        self.state.key("q", 0)
        self.state.key("s", 0)
        command = self.state.tick(0, True, root_yaw_rad=0)
        self.assertEqual(command.movement_direction, (-1, 0, 0))
        self.assertAlmostEqual(self.state.manual.target_yaw, 0)
        self.state.key("q", 0)
        self.state.navigate()
        self.assertTrue(self.state.tick(0, True, root_yaw_rad=0).is_idle)

    def test_turn_target_wraps_and_new_key_after_expiry_uses_current_root_heading(
            self):
        self.state.key("q", math.radians(175))
        self.assertAlmostEqual(self.state.manual.target_yaw, math.radians(-170))
        self.state.manual_until = time.monotonic() - 1
        self.state.key("e", math.radians(100))
        self.assertAlmostEqual(self.state.manual.target_yaw, math.radians(85))

    def test_invalid_keyboard_turn_settings_are_rejected(self):
        for name, values in (
            ("keyboard_turn_step_deg", (0, -1, 30.1, float("nan"))),
            ("keyboard_turn_timeout_s", (0, -1, 10.1, float("inf"))),
        ):
            for value in values:
                args = tabletop.build_parser().parse_args([])
                setattr(args, name, value)
                with self.subTest(name=name,
                                  value=value), self.assertRaises(ValueError):
                    tabletop.validate_args(args)

    def test_small_navigation_waypoint_is_sent_to_sonic(self):
        self.state.navigate()
        reply = lightnav_ws.LightNavReply(0, ((0.07, 0.0, 0.0),), False)
        self.state.accept(self.state.generation, reply, self.frame, 0)
        command = self.state.tick(0, True)
        self.assertFalse(command.is_idle)
        self.assertAlmostEqual(command.movement_speed, 0.20)

    def test_simulation_navigation_uses_supported_walking_speed(self):
        self.state.navigate()
        self.state.accept(self.state.generation, self.reply, self.frame, 0)
        self.assertAlmostEqual(self.state.tick(0, True).movement_speed, 0.20)

    def test_simulation_executes_left_and_right_facing_waypoints(self):
        # The logged regression was a -18 degree waypoint becoming -1.15
        # degrees. Check actual world targets, including angle wrapping.
        for capture_yaw in (0.0, 0.47, math.pi - 0.1):
            for delta in (math.radians(18), math.radians(-18)):
                with self.subTest(capture_yaw=capture_yaw, delta=delta):
                    self.state.navigate()
                    reply = lightnav_ws.LightNavReply(0,
                                                      ((0.00004, 0.0, delta),),
                                                      False)
                    self.state.accept(self.state.generation, reply, self.frame,
                                      capture_yaw)
                    command = self.state.tick(capture_yaw, True)
                    target = capture_yaw + delta
                    self.assertAlmostEqual(command.facing_direction[0],
                                           math.cos(target))
                    self.assertAlmostEqual(command.facing_direction[1],
                                           math.sin(target))
                    self.assertEqual(command.movement_direction, (0, 0, 0))
                    self.assertEqual(command.movement_speed, -1)
                    self.assertFalse(command.is_idle)

    def test_simulation_turn_bound_is_independent_of_control_period(self):
        for control_hz in (2, 4, 8):
            self.args.control_hz = control_hz
            state = tabletop.TabletopControl(self.args)
            state.navigate()
            reply = lightnav_ws.LightNavReply(0, ((0, 0, -math.pi / 2),), False)
            state.accept(state.generation, reply, self.frame, 0)
            command = state.tick(0, True)
            target = math.atan2(command.facing_direction[1],
                                command.facing_direction[0])
            self.assertAlmostEqual(target, math.radians(-18))
            state.command_until = time.monotonic() - 1
            self.assertTrue(state.tick(0, True).is_idle)

    def test_simulation_preserves_combined_translation_and_facing(self):
        self.state.navigate()
        reply = lightnav_ws.LightNavReply(0, ((0.15, 0.0, -0.2),), False)
        self.state.accept(self.state.generation, reply, self.frame, 0.4)
        command = self.state.tick(0.4, True)
        self.assertAlmostEqual(
            math.atan2(command.facing_direction[1],
                       command.facing_direction[0]), 0.2)
        self.assertAlmostEqual(
            math.atan2(command.movement_direction[1],
                       command.movement_direction[0]), 0.4)
        self.assertAlmostEqual(command.movement_speed, 0.2)

    def test_invalid_simulation_yaw_bounds_are_rejected(self):
        for value in (0, -1, 30.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.args.navigation_yaw_step_deg = value
                with self.assertRaises(ValueError):
                    tabletop.validate_args(self.args)

    def test_invalid_waypoint_cannot_become_a_simulation_turn(self):
        self.state.navigate()
        reply = lightnav_ws.LightNavReply(0, ((0, 0, float("nan")),), False)
        self.state.accept(self.state.generation, reply, self.frame, 0.4)
        self.assertTrue(self.state.tick(0.4, True).is_idle)

    def test_rejects_speeds_outside_simulation_profile(self):
        for value in (0.05, 0.81, float("nan")):
            with self.subTest(value=value):
                self.args.keyboard_speed_mps = value
                with self.assertRaises(ValueError):
                    tabletop.validate_args(self.args)

    def test_keyboard_speed_does_not_raise_navigation_speed_cap(self):
        self.args.keyboard_speed_mps = 0.8
        tabletop.validate_args(self.args)
        self.args.navigation_speed_mps = 0.5
        with self.assertRaises(ValueError):
            tabletop.validate_args(self.args)

    def test_simulation_preserves_shallow_waypoint_direction(self):
        self.state.navigate()
        reply = lightnav_ws.LightNavReply(0, ((0.15, 0.027, 0.22),), False)
        self.state.accept(self.state.generation, reply, self.frame, math.pi / 2)
        command = self.state.tick(0, True)
        norm = math.hypot(0.15, 0.027)
        self.assertAlmostEqual(command.movement_direction[0], -0.027 / norm)
        self.assertAlmostEqual(command.movement_direction[1], 0.15 / norm)

    def test_short_inference_gap_keeps_motion_but_deadline_still_expires(self):
        self.state.navigate()
        with patch("scripts.LightNav_eval.tabletop.time.monotonic",
                   return_value=10.0):
            self.state.accept(self.state.generation, self.reply, self.frame, 0)
        with patch("scripts.LightNav_eval.tabletop.time.monotonic",
                   return_value=10.30):
            self.assertFalse(self.state.tick(0, True).is_idle)
        with patch("scripts.LightNav_eval.tabletop.time.monotonic",
                   return_value=10.51):
            self.assertTrue(self.state.tick(0, True).is_idle)

    def test_long_command_hold_is_rejected(self):
        self.args.command_hold_s = 0.76
        with self.assertRaises(ValueError):
            tabletop.validate_args(self.args)

    def test_motion_requires_existing_simulation_namespace(self):
        self.args.enable_motion = True
        with patch(
                "scripts.LightNav_eval.tabletop.os.stat",
                side_effect=[
                    SimpleNamespace(st_ino=1),
                    SimpleNamespace(st_ino=2)
                ],
        ):
            with self.assertRaisesRegex(RuntimeError, "sonic_deploy_dds"):
                tabletop.require_simulation_namespace(self.args)
        with patch("scripts.LightNav_eval.tabletop.os.stat",
                   return_value=SimpleNamespace(st_ino=2)):
            tabletop.require_simulation_namespace(self.args)

    def test_dry_run_does_not_require_namespace_or_open_motion(self):
        with patch("scripts.LightNav_eval.tabletop.os.stat") as stat:
            tabletop.require_simulation_namespace(self.args)
        stat.assert_not_called()

    def test_model_result_after_operator_stop_cannot_restart_motion(self):
        self.state.navigate()
        generation = self.state.generation
        self.state.stop("operator stop")
        reason = self.state.accept(generation, self.reply, self.frame, 0)
        self.assertIn("cancelled", reason)
        self.assertTrue(self.state.tick(0, True).is_idle)

    def test_result_from_previous_trial_cannot_override_new_trial(self):
        self.state.navigate()
        previous = self.state.generation
        self.state.navigate()
        self.state.accept(previous, self.reply, self.frame, 0)
        self.assertTrue(self.state.tick(0, True).is_idle)

    def test_waypoint_uses_capture_yaw_and_has_bounded_command_lifetime(self):
        self.state.navigate()
        self.state.accept(self.state.generation, self.reply, self.frame,
                          math.pi / 2)
        command = self.state.tick(0, True)
        self.assertAlmostEqual(command.movement_direction[1], 1)
        self.state.command_until = time.monotonic() - 0.01
        self.assertTrue(self.state.tick(0, True).is_idle)

    def test_command_cannot_outlive_its_source_frame(self):
        self.state.navigate()
        frame = official01_observation.ObservationFrame(b"jpeg", 98.1, 1)
        with (
                patch("scripts.LightNav_eval.tabletop.time.time",
                      return_value=100.0),
                patch("scripts.LightNav_eval.tabletop.time.monotonic",
                      return_value=10.0),
        ):
            self.state.accept(self.state.generation, self.reply, frame, 0)
        self.assertAlmostEqual(self.state.command_until, 10.1)
        with patch("scripts.LightNav_eval.tabletop.time.monotonic",
                   return_value=10.11):
            self.assertTrue(self.state.tick(0, True).is_idle)

    def test_stop_token_is_a_latched_stop(self):
        self.state.navigate()
        stopped = lightnav_ws.LightNavReply(0, ((0.5, 0, 0),), True)
        self.state.accept(self.state.generation, stopped, self.frame, 0)
        self.assertEqual(self.state.mode, "idle")
        self.assertEqual(self.state.reason, "LightNav stop=true")
        self.assertTrue(self.state.tick(0, True).is_idle)

    def test_lost_feedback_cancels_manual_and_autonomous_motion(self):
        for mode in ("manual", "navigation"):
            with self.subTest(mode=mode):
                self.state.key("w", 0)
                if mode == "navigation":
                    self.state.navigate()
                self.assertTrue(self.state.tick(None, True).is_idle)
                self.assertEqual(self.state.mode, "idle")
                self.assertTrue(self.state.tick(0, True).is_idle)

    def test_old_prediction_and_trial_deadline_stop_motion(self):
        self.state.navigate()
        old = official01_observation.ObservationFrame(b"jpeg",
                                                      time.time() - 10, 1)
        self.state.accept(self.state.generation, self.reply, old, 0)
        self.assertEqual(self.state.mode, "idle")
        self.state.navigate()
        self.state.navigation_until = time.monotonic() - 0.01
        self.state.accept(self.state.generation, self.reply, self.frame, 0)
        self.assertTrue(self.state.tick(0, True).is_idle)

    def test_nan_timeouts_are_rejected(self):
        self.args.prediction_max_age_s = float("nan")
        with self.assertRaises(ValueError):
            tabletop.validate_args(self.args)

    def test_turn_guard_can_be_disabled_without_disabling_model_stop(self):
        for disabled in (False, True):
            self.args.disable_turn_guard = disabled
            state = tabletop.TabletopControl(self.args)
            state.navigate()
            generation = state.generation
            state.accept(generation, self.reply, self.frame, 0)
            turn = lightnav_ws.LightNavReply(1, ((0.0, 0.0, -0.3),), False)
            for _ in range(self.args.turn_guard_consecutive):
                state.accept(generation, turn, self.frame, 0)
            self.assertEqual(state.mode, "navigation" if disabled else "idle")
            if disabled:
                stop = lightnav_ws.LightNavReply(2, ((0.0, 0.0, 0.0),), True)
                state.accept(generation, stop, self.frame, 0)
                self.assertEqual(state.reason, "LightNav stop=true")
                self.assertTrue(state.tick(0, True).is_idle)

    def test_arrival_guard_can_be_disabled_independently(self):
        reply = SimpleNamespace(
            stop=False,
            first_waypoint=(0.04, 0.0, 0.0),
            pointing={
                "opos_px": [320, 240],
                "frame_size": [640, 480]
            },
        )
        for disabled in (False, True):
            self.args.disable_arrival_guard = disabled
            state = tabletop.TabletopControl(self.args)
            state.navigate()
            state.navigation_started_at -= 3
            generation = state.generation
            for _ in range(self.args.arrival_consecutive):
                state.accept(generation, reply, self.frame, 0)
            self.assertEqual(state.mode, "navigation" if disabled else "idle")

    def test_disabled_heuristics_keep_runtime_and_feedback_gates(self):
        self.args.disable_arrival_guard = True
        self.args.disable_turn_guard = True
        state = tabletop.TabletopControl(self.args)
        state.navigate()
        state.navigation_until = time.monotonic() - 1
        state.accept(state.generation, self.reply, self.frame, 0)
        self.assertEqual(state.reason, "maximum navigation runtime reached")
        state.navigate()
        self.assertTrue(state.tick(None, False).is_idle)
        self.assertEqual(state.mode, "idle")

    def test_worker_wait_does_not_extend_a_command(self):
        self.state.navigate()
        self.state.accept(self.state.generation, self.reply, self.frame, 0)
        with patch(
                "scripts.LightNav_eval.tabletop.time.monotonic",
                return_value=self.state.command_until + 0.01,
        ):
            self.assertTrue(self.state.tick(0, True).is_idle)


if __name__ == "__main__":
    unittest.main()
