"""Keyboard positioning and LightNav navigation in the unchanged apple-table scene.

One process owns the planner socket across manual/autonomous handoff. Inference
runs in a worker; the main thread keeps feedback, keyboard stop and IDLE alive.
"""

from __future__ import annotations

import argparse
import codecs
import math
import os
import select
import sys
import termios
import time
import tty
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
from datetime import timezone
from pathlib import Path

from robot.vla.lightnav_sonic.adapter import AdapterLimits
from robot.vla.lightnav_sonic.adapter import LightNav2SonicAdapter
from robot.vla.lightnav_sonic.adapter import SonicPlannerCommand
from robot.vla.lightnav_sonic.feedback import SonicYawSubscriber
from robot.vla.lightnav_sonic.lightnav_ws import LightNavReply
from robot.vla.lightnav_sonic.lightnav_ws import LightNavWsClient
from robot.vla.lightnav_sonic.official01_observation import ObservationFrame
from robot.vla.lightnav_sonic.official01_observation import Official01Subscriber
from robot.vla.lightnav_sonic.sonic_zmq import SonicZmqPublisher

from .tabletop_logging import TabletopRunLog
from .tabletop_pose import RobotPose
from .tabletop_pose import TabletopPoseSubscriber
from .tabletop_recording import TaskArtifactWorker
from .tabletop_recording import TaskRecording
from .tabletop_reset import DEFAULT_RESET_ENDPOINT
from .tabletop_reset import TabletopResetClient
from .teleop import TeleopState
from .teleop import command_from_state
from .teleop import wrap_angle

DEFAULT_INSTRUCTION = "Go straight to the table directly ahead with the apple on top and stop."
SIMULATION_MAX_KEYBOARD_SPEED_MPS = 0.80
SIMULATION_MAX_NAVIGATION_SPEED_MPS = 0.32
SIMULATION_MAX_KEYBOARD_TURN_LEAD_DEG = 60.0


class ConsoleInput:
    """Line commands normally; unbuffered WASD keys in keyboard mode."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.settings = termios.tcgetattr(self.fd) if os.isatty(
            self.fd) else None
        self.keyboard = False
        self.buffer = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")()

    def set_keyboard(self, enabled: bool) -> None:
        if enabled and self.settings is None:
            raise RuntimeError(
                "move keyboard requires an interactive terminal (SSH: ssh -t)")
        if self.settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)
            if enabled:
                tty.setcbreak(self.fd)
                settings = termios.tcgetattr(self.fd)
                settings[3] &= ~termios.ECHO
                termios.tcsetattr(self.fd, termios.TCSADRAIN, settings)
        self.keyboard = enabled
        self.buffer = ""

    def poll(self) -> list[str]:
        if not select.select([self.fd], [], [], 0)[0]:
            return []
        raw = os.read(self.fd, 4096)
        if not raw:
            return ["\x04"]
        text = self.decoder.decode(raw)
        if self.keyboard:
            return list(text)
        self.buffer += text
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()
        return [line.strip() for line in lines]

    def close(self) -> None:
        self.set_keyboard(False)


class TabletopControl:
    """Transport-free mode/expiry logic, including late-result rejection."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.adapter = LightNav2SonicAdapter(
            AdapterLimits(
                max_forward_mps=0.08,
                max_lateral_mps=0.08,
                max_planar_mps=0.08,
                max_yaw_rate_rps=0.08,
                control_period_s=1 / args.control_hz,
                position_tolerance_m=args.waypoint_idle_m,
            ))
        self.manual = TeleopState(speed_mps=args.keyboard_speed_mps)
        self.manual_command: SonicPlannerCommand | None = None
        self.manual_key: str | None = None
        self.mode = "idle"
        self.generation = 0
        self.manual_until = 0.0
        self.navigation_until = 0.0
        self.command_until = 0.0
        self.command: SonicPlannerCommand | None = None
        self.reason = "operator idle"
        self.latched_stop_reason: str | None = None
        self.navigation_started_at = 0.0
        self.arrival_hits = 0
        self.yaw_only_hits = 0
        self.progress_seen = False
        self.idle_yaw_rad: float | None = None
        self.last_yaw_rad: float | None = None

    def hold_heading(self, yaw: float) -> None:
        """Latch a standing heading; IMU noise must not rewrite the target."""
        if not math.isfinite(yaw):
            raise ValueError("standing heading must be finite")
        self.idle_yaw_rad = yaw

    def idle_command(self, *, reason: str) -> SonicPlannerCommand:
        if self.idle_yaw_rad is None and self.last_yaw_rad is not None:
            self.hold_heading(self.last_yaw_rad)
        # In this simulation client, stationary commands request zero speed,
        # rather
        # than the planner's -1 (choose a default speed) sentinel.
        return replace(self.adapter.idle(self.idle_yaw_rad, reason=reason),
                       movement_speed=0.0)

    def stop(self, reason: str, *, latch: bool = False) -> None:
        self.mode = "idle"
        self.generation += 1
        self.command = None
        self.manual_command = None
        self.manual_key = None
        self.reason = reason
        if latch:
            self.latched_stop_reason = reason

    def navigate(self) -> None:
        self.stop("waiting for LightNav")
        self.latched_stop_reason = None
        self.navigation_started_at = time.monotonic()
        self.arrival_hits = 0
        self.yaw_only_hits = 0
        self.progress_seen = False
        self.mode = "navigation"
        self.navigation_until = time.monotonic() + self.args.max_runtime_s

    def key(self, key: str, yaw: float) -> None:
        if key.lower() not in "wasdqe" or len(key) != 1:
            self.stop("keyboard stop")
            return
        key = key.lower()
        if key in "qe":
            # Repeated keys accumulate a world-heading target. Resetting it
            # from feedback on every press used to discard the previous turn.
            continuing_turn = (self.mode == "manual" and
                               self.manual_key in ("q", "e") and
                               time.monotonic() < self.manual_until)
            if not continuing_turn:
                self.stop("keyboard positioning")
                self.mode = "manual"
                self.manual.stop(yaw)
            delta = math.radians(
                self.args.keyboard_turn_step_deg) * (1 if key == "q" else -1)
            self.manual.turn(delta, yaw)
            # Auto-repeat cannot queue an unlimited spin. Further presses
            # extend the target only as the pelvis follows it.
            lead = math.radians(SIMULATION_MAX_KEYBOARD_TURN_LEAD_DEG)
            error = wrap_angle(self.manual.target_yaw - yaw)
            self.manual.target_yaw = wrap_angle(yaw +
                                                max(-lead, min(lead, error)))
            self.manual_key = key
            self.idle_yaw_rad = None
            self.manual_until = time.monotonic(
            ) + self.args.keyboard_turn_timeout_s
            return
        if self.mode == "manual" and key in "wasd" and key == self.manual_key:
            # Auto-repeat renews the command without resetting world heading
            # to IMU noise and forcing the planner to replan each keypress.
            self.manual_until = time.monotonic() + self.args.keyboard_watchdog_s
            return
        self.stop("keyboard positioning")
        self.mode = "manual"
        self.manual.stop(yaw)
        self.manual_key = key
        directions = {"w": (1, 0), "s": (-1, 0), "a": (0, 1), "d": (0, -1)}
        if key in directions:
            self.manual.set_translation(*directions[key], yaw)
        self.manual_command = command_from_state(
            self.manual, yaw, heading_tolerance_rad=math.radians(3))
        self.manual_until = time.monotonic() + self.args.keyboard_watchdog_s

    def accept(self, generation: int, reply: LightNavReply,
               frame: ObservationFrame, capture_yaw: float) -> str:
        if generation != self.generation or self.mode != "navigation":
            return "discarded: cancelled navigation"
        if time.monotonic() >= self.navigation_until:
            self.stop("maximum navigation runtime reached")
        elif not frame.fresh(self.args.prediction_max_age_s):
            self.stop("LightNav result too old; start a new trial")
        elif reply.stop:
            self.stop("LightNav stop=true", latch=True)
        else:
            # Waypoints belong to the camera capture frame, not the later yaw.
            waypoint = reply.first_waypoint
            translation_m = math.hypot(waypoint[0], waypoint[1])
            yaw_magnitude = abs(waypoint[2])
            if translation_m >= self.args.turn_guard_min_progress_m:
                self.progress_seen = True
            if (self.progress_seen and
                    translation_m <= self.args.turn_guard_waypoint_m and
                    yaw_magnitude >= self.args.turn_guard_yaw_rad):
                self.yaw_only_hits += 1
            else:
                self.yaw_only_hits = 0
            if self._target_is_centered(reply) and (
                    translation_m <= self.args.arrival_waypoint_m and
                    yaw_magnitude <= self.args.arrival_yaw_rad and
                    time.monotonic() - self.navigation_started_at
                    >= self.args.arrival_min_runtime_s):
                self.arrival_hits += 1
            else:
                self.arrival_hits = 0
            if not self.args.disable_arrival_guard and self.arrival_hits >= self.args.arrival_consecutive:
                self.stop(
                    "arrival guard: centered target and settled waypoint (latched)",
                    latch=True)
                return self.latched_stop_reason or self.reason
            if not self.args.disable_turn_guard and self.yaw_only_hits >= self.args.turn_guard_consecutive:
                self.stop("turn guard: repeated yaw-only waypoints (latched)",
                          latch=True)
                return self.latched_stop_reason or self.reason
            self.command = self.adapter.convert_or_idle(waypoint, capture_yaw)
            if not self.command.is_idle:
                # SONIC consumes an absolute facing target, not a yaw rate.
                # The generic adapter's conservative rate * period gives only
                # 0.02 rad here, too small to track an 18-degree model turn.
                # Preserve the waypoint angle within this simulation profile;
                # this bound is an angle, not a guaranteed physical turn rate.
                yaw_limit = math.radians(self.args.navigation_yaw_step_deg)
                delta_yaw = max(-yaw_limit, min(yaw_limit, waypoint[2]))
                target_yaw = capture_yaw + delta_yaw
                self.command = replace(
                    self.command,
                    facing_direction=(math.cos(target_yaw),
                                      math.sin(target_yaw), 0.0),
                )
            if self.command.movement_speed > 0:
                # SONIC documents SLOW_WALK as 0.1--0.8 m/s. The generic
                # real-robot bridge's 0.08 m/s cap is outside that range.
                # Use a separate simulation speed; never alter that adapter.
                # Preserve the model's planar direction. Independently clipping
                # forward/lateral velocities can turn a shallow curve into a
                # diagonal command before the simulation speed is applied.
                forward, lateral, _ = reply.first_waypoint
                movement_yaw = capture_yaw + math.atan2(lateral, forward)
                self.command = replace(
                    self.command,
                    movement_direction=(math.cos(movement_yaw),
                                        math.sin(movement_yaw), 0.0),
                    movement_speed=self.args.navigation_speed_mps,
                )
            remaining_age_s = self.args.prediction_max_age_s - (
                time.time() - frame.wall_time_s)
            # A short, bounded overlap absorbs inference jitter. Waiting for
            # another prediction never extends this deadline or source age.
            self.command_until = time.monotonic() + min(
                self.args.command_hold_s, remaining_age_s)
            return self.command.reason
        return self.reason

    @staticmethod
    def _target_is_centered(reply: LightNavReply) -> bool:
        pointing = reply.pointing
        if not isinstance(pointing, dict):
            return False
        point = pointing.get("opos_px")
        frame_size = pointing.get("frame_size")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False
        if not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2:
            return False
        try:
            x, y = float(point[0]), float(point[1])
            width, height = float(frame_size[0]), float(frame_size[1])
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            return False
        if width <= 0.0 or height <= 0.0:
            return False
        return abs(x / width - 0.5) <= 0.18 and 0.16 <= y / height <= 0.84

    def tick(self,
             yaw: float | None,
             fresh_observation: bool,
             *,
             root_yaw_rad: float | None = None) -> SonicPlannerCommand:
        if yaw is not None:
            if root_yaw_rad is not None:
                self.last_yaw_rad = root_yaw_rad
            elif not (self.mode == "manual" and self.manual_key in ("q", "e")):
                self.last_yaw_rad = yaw
        if yaw is None or not fresh_observation:
            if self.mode != "idle":
                self.stop("observation or yaw feedback lost; motion cancelled")
            return self.idle_command(
                reason="waiting for fresh observation and yaw")
        now = time.monotonic()
        if self.mode == "manual":
            turning = self.manual_key in ("q", "e")
            if turning and root_yaw_rad is None:
                self.stop("pelvis yaw feedback lost; turn cancelled")
                return self.idle_command(reason=self.reason)
            if now >= self.manual_until:
                self.stop(
                    "keyboard turn timeout" if turning else "keyboard watchdog")
            elif self.manual.has_translation:
                assert self.manual_command is not None
                self.idle_yaw_rad = None
                return self.manual_command
            else:
                # An upper-body/IMU heading alone does not prove that the legs
                # have turned. Completion uses the floating pelvis pose.
                command = command_from_state(
                    self.manual,
                    root_yaw_rad,
                    heading_tolerance_rad=math.radians(3))
                if not command.is_idle:
                    self.idle_yaw_rad = None
                    return command
                target_yaw = self.manual.target_yaw
                self.stop("keyboard turn complete")
                self.hold_heading(root_yaw_rad if target_yaw is
                                  None else target_yaw)
                return self.idle_command(reason=self.reason)
        if self.mode == "navigation":
            if now >= self.navigation_until:
                self.stop("maximum navigation runtime reached")
            elif self.command is not None and now < self.command_until:
                if self.command.is_idle:
                    return self.idle_command(reason=self.command.reason)
                self.idle_yaw_rad = None
                return self.command
        return self.idle_command(reason=self.reason)


class NavigationWorker:
    """Serialize model requests; a new trial always creates a fresh session."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client: LightNavWsClient | None = None
        self.generation = -1
        self.executor = ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix="lightnav")

    def predict(self, generation: int, jpeg: bytes,
                instruction: str) -> LightNavReply:
        if generation != self.generation:
            self.close_client()
            self.client = LightNavWsClient(
                self.args.lightnav_server,
                client_id=f"tabletop-{os.getpid()}-{generation}",
                request_timeout_s=self.args.inference_timeout_s,
            )
            self.generation = generation
        assert self.client is not None
        return self.client.predict(jpeg, instruction)

    def close_client(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def close(self) -> None:
        # Motion has already stopped; at most the bounded WS timeout remains.
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.close_client()


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone simulation control argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-endpoint",
                        default="tcp://10.254.240.2:5560")
    parser.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--reset-endpoint", default=DEFAULT_RESET_ENDPOINT)
    parser.add_argument("--sonic-feedback-endpoint",
                        default="tcp://127.0.0.1:5557")
    parser.add_argument("--sonic-bind", default="tcp://127.0.0.1:5556")
    parser.add_argument("--lightnav-server", default="ws://127.0.0.1:8051")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("outputs/tabletop_navigation"))
    parser.add_argument(
        "--video-fps",
        type=float,
        default=10.0,
        help="task video capture/output wall-clock rate, 1 to 10 fps")
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="keep task logs/automatic evaluation but skip both MP4s")
    parser.add_argument("--control-hz", type=float, default=4.0)
    parser.add_argument("--input-timeout-s", type=float, default=1.0)
    parser.add_argument("--keyboard-watchdog-s", type=float, default=1.5)
    parser.add_argument("--keyboard-speed-mps", type=float, default=0.50)
    parser.add_argument(
        "--keyboard-turn-step-deg",
        type=float,
        default=15.0,
        help=
        "simulation Q/E heading increment per keypress (0 < value <= 30 degrees)",
    )
    parser.add_argument(
        "--keyboard-turn-timeout-s",
        type=float,
        default=6.0,
        help=
        "maximum time to finish a Q/E pelvis turn after the last keypress (at most 10 s)",
    )
    parser.add_argument("--navigation-speed-mps", type=float, default=0.20)
    parser.add_argument(
        "--navigation-yaw-step-deg",
        type=float,
        default=18.0,
        help=
        "simulation facing-target offset bound in degrees (0 < value <= 30); not a yaw rate",
    )
    parser.add_argument(
        "--waypoint-idle-m",
        type=float,
        default=0.02,
        help=
        "idle below this planar waypoint distance; larger steps are sent to SONIC",
    )
    parser.add_argument("--prediction-max-age-s", type=float, default=2.0)
    parser.add_argument("--command-hold-s", type=float, default=0.50)
    parser.add_argument("--inference-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-runtime-s", type=float, default=120.0)
    parser.add_argument("--arrival-consecutive", type=int, default=3)
    parser.add_argument("--arrival-waypoint-m", type=float, default=0.08)
    parser.add_argument("--arrival-yaw-rad", type=float, default=0.12)
    parser.add_argument("--arrival-min-runtime-s", type=float, default=2.0)
    parser.add_argument(
        "--disable-turn-guard",
        action="store_true",
        help=
        "simulation ablation: keep executing repeated yaw-only model predictions",
    )
    parser.add_argument(
        "--disable-arrival-guard",
        action="store_true",
        help=
        "simulation ablation: require model stop instead of the arrival heuristic",
    )
    parser.add_argument("--turn-guard-consecutive", type=int, default=4)
    parser.add_argument("--turn-guard-waypoint-m", type=float, default=0.03)
    parser.add_argument("--turn-guard-yaw-rad", type=float, default=0.15)
    parser.add_argument("--turn-guard-min-progress-m", type=float, default=0.10)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--simulation-loopback", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate finite control limits and the simulation-only motion boundary."""
    if not math.isfinite(args.video_fps) or not 1 <= args.video_fps <= 10:
        raise ValueError("--video-fps must be between 1 and 10")
    for name in (
            "control_hz",
            "waypoint_idle_m",
            "input_timeout_s",
            "keyboard_watchdog_s",
            "keyboard_turn_timeout_s",
            "prediction_max_age_s",
            "command_hold_s",
            "inference_timeout_s",
            "max_runtime_s",
            "arrival_waypoint_m",
            "arrival_yaw_rad",
            "arrival_min_runtime_s",
            "turn_guard_waypoint_m",
            "turn_guard_yaw_rad",
            "turn_guard_min_progress_m",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and positive")
    if args.waypoint_idle_m >= args.arrival_waypoint_m:
        raise ValueError(
            "--waypoint-idle-m must be smaller than --arrival-waypoint-m")
    if args.arrival_consecutive < 1 or args.turn_guard_consecutive < 1:
        raise ValueError("guard consecutive counts must be positive")
    if not args.instruction.strip():
        raise ValueError("--instruction must not be empty")
    if args.command_hold_s > 0.75:
        raise ValueError("--command-hold-s must not exceed 0.75 seconds")
    if not math.isfinite(args.navigation_yaw_step_deg
                        ) or not 0 < args.navigation_yaw_step_deg <= 30:
        raise ValueError(
            "--navigation-yaw-step-deg must be finite, greater than 0 and at most 30"
        )
    if not math.isfinite(args.keyboard_turn_step_deg
                        ) or not 0 < args.keyboard_turn_step_deg <= 30:
        raise ValueError(
            "--keyboard-turn-step-deg must be finite, greater than 0 and at most 30"
        )
    if args.keyboard_turn_timeout_s > 10:
        raise ValueError("--keyboard-turn-timeout-s must not exceed 10 seconds")
    for name, maximum in (
        ("keyboard_speed_mps", SIMULATION_MAX_KEYBOARD_SPEED_MPS),
        ("navigation_speed_mps", SIMULATION_MAX_NAVIGATION_SPEED_MPS),
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.1 <= value <= maximum:
            raise ValueError(
                f"--{name.replace('_', '-')} must be between 0.1 and {maximum:g} m/s"
            )


def require_simulation_namespace(args: argparse.Namespace) -> None:
    """Keep the simulation speed profile inside the existing DDS sandbox."""
    if not args.enable_motion:
        return
    if args.simulation_loopback:
        from .local_tabletop import require_loopback_simulation

        require_loopback_simulation(args)
        return
    try:
        current = os.stat("/proc/self/ns/net").st_ino
        expected = os.stat("/run/netns/sonic_deploy_dds").st_ino
    except OSError as exc:
        raise RuntimeError(
            "motion requires the existing sonic_deploy_dds namespace") from exc
    if current != expected:
        raise RuntimeError(
            "use run_tabletop.sh control: motion requires sonic_deploy_dds")


def parse_reset_command(text: str) -> dict:
    """Validate trial start changes before cancelling or resetting a task."""
    words = text.split()
    if words == ["reset"]:
        return {}
    if words == ["reset", "fixed"]:
        return {"start_mode": "fixed"}
    if len(words) in (3, 4) and words[:2] == ["reset", "random"]:
        try:
            radius = float(words[2])
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("radius must be finite and positive")
            result = {"start_mode": "random", "radius_m": radius}
            if len(words) == 4:
                result["sample_seed"] = int(words[3])
            return result
        except ValueError as exc:
            raise ValueError(
                "use: reset random R [integer sample seed]; R in metres"
            ) from exc
    raise ValueError("use: reset | reset random R [seed] | reset fixed")


def run(args: argparse.Namespace) -> int:
    """Run keyboard and model control with a single simulation planner owner."""
    validate_args(args)
    require_simulation_namespace(args)
    output = args.output_dir / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    state = TabletopControl(args)
    worker = NavigationWorker(args)
    observation = Official01Subscriber(args.observation_endpoint)
    feedback = SonicYawSubscriber(args.sonic_feedback_endpoint, "g1_debug")
    pose_feedback = TabletopPoseSubscriber(args.pose_endpoint)
    reset_client = TabletopResetClient(args.reset_endpoint)
    reset_waiting = False
    reset_wall_time = None
    reset_feedback_deadline = 0.0
    reset_support_enabled = False
    publisher = None
    console = None
    pending: Future | None = None
    request: tuple[int, ObservationFrame, float, RobotPose | None,
                   str] | None = None
    instruction = args.instruction
    last_pose: RobotPose | None = None
    ready = False
    last_state = None
    decision = 0
    state_image_count = 0
    next_request_at = 0.0
    next_activation_at = 0.0
    next_telemetry_at = 0.0
    drift_warned = False
    log = TabletopRunLog(output)
    record = log.record
    artifacts = TaskArtifactWorker()
    recording = None
    frame = None

    def finish_task(reason: str | None = None) -> None:
        nonlocal recording
        if log.active is not None and (
                reason is not None or state.mode != "navigation" or
                state.generation != log.active.generation):
            if frame is not None and frame.fresh(args.input_timeout_s):
                log.save_frame("end_ego.jpg", frame.jpeg, log.active.generation)
            task = log.finish(
                reason or state.reason,
                None if last_pose is None else last_pose.as_dict())
            if recording is not None:
                recording.capture(frame, last_pose)
                recording.finish(task)
                recording = None
            artifacts.submit(task)
            print(
                f"TASK END {task.task_id}: {task.summary['end_reason']}\n"
                f"Task summary: {task.directory / 'summary.json'}",
                flush=True)

    try:
        console = ConsoleInput()
        if args.enable_motion:
            publisher = SonicZmqPublisher(args.sonic_bind)
        print(f"SIMULATION_ONLY logs={output}", flush=True)
        print(
            f"Simulation speeds: keyboard={args.keyboard_speed_mps:g} m/s; "
            f"navigation={args.navigation_speed_mps:g} m/s; "
            f"yaw_step_limit={args.navigation_yaw_step_deg:g} deg",
            flush=True,
        )
        print(
            f"Simulation guards: turn={'OFF' if args.disable_turn_guard else 'ON'}; "
            f"arrival={'OFF' if args.disable_arrival_guard else 'ON'}; "
            f"max_runtime={args.max_runtime_s:g}s",
            flush=True,
        )
        print(
            f"Keyboard turns: {args.keyboard_turn_step_deg:g} deg/key; "
            f"pelvis feedback; timeout={args.keyboard_turn_timeout_s:g}s",
            flush=True,
        )
        print(
            "Commands: move keyboard | nav [instruction] | reset [random R [seed] | fixed] | stop | status | exit",
            flush=True)
        print(f"Default instruction: {instruction}", flush=True)
        record(
            "startup",
            motion=args.enable_motion,
            instruction=instruction,
            args={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        )
        while True:
            artifacts.poll()
            frame = observation.latest(args.input_timeout_s)
            yaw = feedback.latest(args.input_timeout_s)
            pose = pose_feedback.latest(args.input_timeout_s)
            if pose is not None:
                last_pose = pose
            if recording is not None:
                recording.capture(frame, pose)
            healthy = frame is not None and yaw is not None
            reset_reply = reset_client.poll()
            if reset_reply is not None:
                record("simulation_reset_reply", reply=reset_reply)
                if reset_reply.get("ok"):
                    reset_wall_time = reset_reply["wall_time_s"]
                    reset_feedback_deadline = time.monotonic() + 5.0
                    reset_support_enabled = bool(
                        reset_reply.get("support_enabled", False))
                    print(
                        "Reset applied; waiting for fresh RGB, pose and SONIC yaw.",
                        flush=True)
                else:
                    reset_waiting = False
                    state.stop("simulation reset failed")
                    print(
                        f"Reset failed: {reset_reply.get('error')}; motion remains idle.",
                        flush=True)
            if reset_waiting and reset_wall_time is not None and healthy and pose is not None:
                yaw_error = abs(
                    math.atan2(math.sin(yaw - pose.yaw_rad),
                               math.cos(yaw - pose.yaw_rad)))
                if (frame.wall_time_s > reset_wall_time and
                        pose.wall_time_s > reset_wall_time and
                        yaw_error < 0.15):
                    reset_waiting = False
                    reset_wall_time = None
                    state.stop("simulation reset complete; operator idle")
                    state.hold_heading(pose.yaw_rad)
                    record("simulation_reset_complete",
                           robot_pose=pose.as_dict())
                    if reset_support_enabled:
                        print(
                            "RESET COMPLETE; support is enabled. Release it with viewer key 9 before nav.",
                            flush=True)
                    else:
                        print(
                            "RESET COMPLETE; enter nav for a new trial. No need to press 9 again.",
                            flush=True)
            if reset_waiting and reset_wall_time is not None and time.monotonic(
            ) >= reset_feedback_deadline:
                reset_waiting = False
                reset_wall_time = None
                state.stop("reset feedback timed out; motion remains idle")
                record("simulation_reset_error", reason=state.reason)
                print(
                    "Reset feedback timed out; check RGB/pose/SONIC in status before retrying.",
                    flush=True)
            if (publisher is not None and frame is not None and not ready and
                    time.monotonic() >= next_activation_at):
                # g1_debug is emitted only in SONIC's CONTROL state. Waiting
                # for yaw before sending start would deadlock startup.
                publisher.activate_planner(warmup_s=0.0)
                publisher.send_idle(state.idle_command(reason="startup IDLE"))
                next_activation_at = time.monotonic() + 1.0
            if healthy and not ready:
                ready = True
                state.hold_heading(yaw)
                print(
                    "SIMULATION_ONLY STATE = READY; now release support with viewer key 9 once.",
                    flush=True,
                )
                record("ready")
            for text in console.poll():
                if text == "\x04":
                    return 0
                if reset_waiting and text not in ("status", "stop", "exit",
                                                  "quit"):
                    print("Reset in progress; wait for RESET COMPLETE.",
                          flush=True)
                    continue
                if console.keyboard:
                    if text.lower() == "x":
                        console.set_keyboard(False)
                        state.stop("left keyboard mode")
                        print("\nKeyboard stopped. Enter: nav [instruction]",
                              flush=True)
                    elif healthy:
                        if text.lower() in ("q", "e") and pose is None:
                            print("Turn unavailable: need fresh pelvis pose.",
                                  flush=True)
                            continue
                        keyboard_yaw = yaw if pose is None else pose.yaw_rad
                        state.key(text, keyboard_yaw)
                        record(
                            "keyboard_input",
                            key=text,
                            yaw=keyboard_yaw,
                            sonic_yaw_rad=yaw,
                            pelvis_yaw_rad=None
                            if pose is None else pose.yaw_rad,
                            target_yaw_rad=state.manual.target_yaw,
                            command_until_monotonic=state.manual_until,
                        )
                        if text.lower() in ("q", "e"):
                            print(
                                f"KEYBOARD TURN: pelvis={math.degrees(keyboard_yaw):+.1f} deg; "
                                f"target={math.degrees(state.manual.target_yaw):+.1f} deg",
                                flush=True,
                            )
                elif text in ("exit", "quit"):
                    return 0
                elif text == "reset" or text.startswith("reset "):
                    if not args.enable_motion or not args.simulation_loopback:
                        print(
                            "Reset requires local_tabletop control --enable-motion.",
                            flush=True)
                        continue
                    if not ready or not publisher.activated:
                        print(
                            "Reset is available after SIMULATION_ONLY STATE = READY.",
                            flush=True)
                        continue
                    try:
                        start_options = parse_reset_command(text)
                    except ValueError as exc:
                        print(f"Reset not requested: {exc}", flush=True)
                        continue
                    state.stop("simulation reset requested")
                    publisher.send_idle(state.idle_command(reason=state.reason))
                    request_id = reset_client.request(**start_options)
                    reset_waiting = True
                    reset_wall_time = None
                    record("simulation_reset_start",
                           request_id=request_id,
                           generation=state.generation)
                    print(
                        "Resetting G1 to the configured start; motion is idle.",
                        flush=True)
                elif text in ("move keyboard", "keyboard"):
                    state.stop("entered keyboard mode")
                    console.set_keyboard(True)
                    print(
                        f"W/S/A/D move; Q/E add {args.keyboard_turn_step_deg:g} deg left/right; "
                        "SPACE stop; X return to command input. "
                        f"Movement watchdog={args.keyboard_watchdog_s:g}s; "
                        f"turn timeout={args.keyboard_turn_timeout_s:g}s after the last keypress.",
                        flush=True,
                    )
                elif text == "stop":
                    state.stop("operator stop")
                elif text == "status":
                    task = log.active or next(reversed(log.tasks.values()),
                                              None)
                    print(
                        f"mode={state.mode} observation={frame is not None} yaw={yaw} "
                        f"idle_heading={state.idle_yaw_rad} "
                        f"pelvis_yaw={None if pose is None else pose.yaw_rad} "
                        f"model_busy={pending is not None} reason={state.reason} "
                        f"task={None if task is None else task.task_id} "
                        f"task_logs={None if task is None else task.directory}",
                        flush=True,
                    )
                elif text == "nav" or text.startswith("nav "):
                    if not healthy:
                        print(
                            "Not ready: need fresh observation and SONIC yaw.",
                            flush=True)
                        continue
                    instruction = text[4:].strip() if text.startswith(
                        "nav ") else args.instruction
                    finish_task("replaced by new navigation instruction")
                    state.navigate()
                    task = log.start(
                        state.generation,
                        instruction,
                        None if last_pose is None else last_pose.as_dict(),
                        control_config={
                            "turn_guard_enabled":
                                not args.disable_turn_guard,
                            "arrival_guard_enabled":
                                not args.disable_arrival_guard,
                            "waypoint_idle_m":
                                args.waypoint_idle_m,
                            "navigation_speed_mps":
                                args.navigation_speed_mps,
                            "navigation_yaw_step_deg":
                                args.navigation_yaw_step_deg,
                            "max_runtime_s":
                                args.max_runtime_s,
                        })
                    recording = TaskRecording(task, args.video_fps,
                                              not args.no_record_video)
                    recording.capture(frame, pose)
                    print(
                        f"\nTASK START {task.task_id}\nNavigating: {instruction}\n"
                        f"Task logs: {task.directory}\n"
                        f"Readable inference: {task.directory / 'inference.txt'}",
                        flush=True)
                elif text:
                    print(
                        "Commands: move keyboard | nav [instruction] | reset [random R [seed] | fixed] | stop | status | exit",
                        flush=True,
                    )
                finish_task()

            # Apply loss/timeout gates before accepting or scheduling results.
            state.tick(yaw,
                       frame is not None,
                       root_yaw_rad=None if pose is None else pose.yaw_rad)
            finish_task()
            if pending is not None and pending.done():
                assert request is not None
                generation, capture_frame, capture_yaw, capture_pose, image_name = request
                try:
                    reply = pending.result()
                    reason = state.accept(generation, reply, capture_frame,
                                          capture_yaw)
                    planner_command = (state.command
                                       if state.generation == generation and
                                       state.mode == "navigation" else None)
                    command_yaw_step_deg = None
                    if planner_command is not None and not planner_command.is_idle:
                        target_yaw = math.atan2(
                            planner_command.facing_direction[1],
                            planner_command.facing_direction[0])
                        command_yaw_step_deg = math.degrees(
                            math.atan2(math.sin(target_yaw - capture_yaw),
                                       math.cos(target_yaw - capture_yaw)))
                    record(
                        "prediction",
                        generation=generation,
                        reply=asdict(reply),
                        capture_yaw=capture_yaw,
                        frame=image_name,
                        frame_wall_time_s=capture_frame.wall_time_s,
                        frame_sim_time_s=capture_frame.sim_time_s,
                        result_age_s=time.time() - capture_frame.wall_time_s,
                        raw_text=reply.raw_text,
                        stop=reply.stop,
                        pointing=reply.pointing,
                        first_waypoint=list(reply.first_waypoint),
                        planner_command=None
                        if planner_command is None else asdict(planner_command),
                        command_yaw_step_deg=command_yaw_step_deg,
                        robot_pose=None
                        if capture_pose is None else capture_pose.as_dict(),
                        reason=reason,
                    )
                    print(
                        f"[{log.task_for(generation).task_id}] LightNav seq={reply.seq} stop={reply.stop} "
                        f"latency_ms={reply.latency_ms} waypoint={reply.first_waypoint} "
                        f"command_yaw_step_deg={command_yaw_step_deg} "
                        f"reason={reason}",
                        flush=True,
                    )
                except Exception as exc:
                    if generation == state.generation:
                        state.stop(f"model failure: {exc}")
                    record("model_error", generation=generation, error=str(exc))
                    print(f"LightNav error: {exc}; motion stopped.", flush=True)
                pending = None
                request = None
                finish_task()
            if (state.mode == "navigation" and pending is None and healthy and
                    time.monotonic() >= next_request_at):
                decision += 1
                image_name = f"frame-{decision:06d}.jpg"
                image_name = log.save_frame(image_name, frame.jpeg,
                                            state.generation)
                request = (state.generation, frame, yaw, last_pose, image_name)
                pending = worker.executor.submit(worker.predict,
                                                 state.generation, frame.jpeg,
                                                 instruction)
                next_request_at = time.monotonic() + 1 / args.control_hz
            command = state.tick(
                yaw,
                frame is not None,
                root_yaw_rad=None if pose is None else pose.yaw_rad)
            finish_task()
            if reset_waiting and pose is not None:
                # Reacquire heading only across a teleport, not on every small
                # standing oscillation while waiting for the reset reply.
                if state.idle_yaw_rad is None or abs(
                        math.atan2(
                            math.sin(pose.yaw_rad - state.idle_yaw_rad),
                            math.cos(pose.yaw_rad - state.idle_yaw_rad),
                        )) > 0.15:
                    state.hold_heading(pose.yaw_rad)
                command = state.idle_command(
                    reason="simulation reset in progress")
            if publisher is not None and publisher.activated:
                publisher.send_planner(command)
            if time.monotonic() >= next_telemetry_at:
                target_yaw = math.atan2(command.facing_direction[1],
                                        command.facing_direction[0])
                heading_error = None if yaw is None else math.atan2(
                    math.sin(yaw - target_yaw), math.cos(yaw - target_yaw))
                record(
                    "control_telemetry",
                    mode=state.mode,
                    command=asdict(command),
                    measured_yaw_rad=yaw,
                    target_yaw_rad=target_yaw,
                    heading_error_rad=heading_error,
                    pelvis_yaw_rad=None if pose is None else pose.yaw_rad,
                    robot_pose=None if pose is None else pose.as_dict(),
                )
                if command.is_idle and not reset_waiting and heading_error is not None:
                    drifting = abs(heading_error) > math.radians(5)
                    if drifting and not drift_warned:
                        print(
                            f"IDLE HEADING DRIFT: {math.degrees(heading_error):+.1f} deg; "
                            "target held fixed, no navigation command sent.",
                            flush=True,
                        )
                    drift_warned = drifting
                else:
                    drift_warned = False
                next_telemetry_at = time.monotonic() + 1.0
            state_label = (state.mode, command.reason)
            if state_label != last_state:
                state_image = None
                if frame is not None:
                    state_image_count += 1
                    state_image = f"state-{state_image_count:06d}.jpg"
                    state_image = log.save_frame(state_image, frame.jpeg)
                record("control_state",
                       mode=state.mode,
                       command=asdict(command),
                       frame=state_image,
                       robot_pose=None
                       if last_pose is None else last_pose.as_dict())
                print(f"state={state.mode}: {command.reason}", flush=True)
                last_state = state_label
            time.sleep(0.02)
    except KeyboardInterrupt:
        return 130
    finally:
        # Stop before waiting for the inference thread or restoring the terminal.
        try:
            if publisher is not None and publisher.activated:
                publisher.send_idle(
                    state.idle_command(reason="tabletop client shutdown"))
        finally:
            if publisher is not None:
                publisher.close()
            if console is not None:
                console.close()
            observation.close()
            feedback.close()
            pose_feedback.close()
            reset_client.close()
            finish_task("control client shutdown")
            record("shutdown", reason=state.reason)
            log.close()
            worker.close()
            artifacts.close()


def main() -> int:
    """Parse CLI arguments and run the standalone simulation controller."""
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
