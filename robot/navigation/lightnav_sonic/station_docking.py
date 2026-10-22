"""Bounded simulation station docking through the existing Executive owner.

The native Skill selects a station. Only this execution layer sees world pose.
Success requires fresh, measured rest after the Executive acknowledges READY.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

from robot.navigation.lightnav_sonic.sim_docking_pose import (
    SimDockingPoseSource,
)
from robot.navigation.lightnav_sonic.sim_docking_pose import SimRobotPose
from robot.navigation.lightnav_sonic.sim_docking_pose import finite_number


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class Bounds:
    """Allowed root-center rectangle, already reduced for body clearance."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            finite_number(value, key)
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("bounds must have positive area")

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.xmin + margin <= x <= self.xmax - margin and
                self.ymin + margin <= y <= self.ymax - margin)


@dataclass(frozen=True)
class Station:
    """Operator-owned station geometry and bounded controller settings."""

    name: str
    entry: Bounds
    corridor: Bounds
    target_x: float
    target_y: float
    target_yaw: float
    position_tolerance_m: float = 0.04
    yaw_tolerance_rad: float = 0.06
    entry_yaw_tolerance_rad: float = 0.5
    settled_position_tolerance_m: float | None = None
    settled_yaw_tolerance_rad: float | None = None
    max_speed_mps: float = 0.08
    max_yaw_rate_rad_s: float = 0.12
    position_gain: float = 0.5
    yaw_gain: float = 0.6
    stable_duration_s: float = 1.0
    stable_speed_mps: float = 0.025
    stable_yaw_rate_rad_s: float = 0.05
    max_pose_age_s: float = 0.3
    max_runtime_s: float = 35.0
    max_path_m: float = 1.0
    max_step_m: float = 0.10
    max_step_yaw_rad: float = 0.12
    max_tilt_rad: float = 0.25
    min_base_height_m: float = 0.55
    max_base_height_m: float = 1.0
    control_hz: float = 10.0

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.name):
            raise ValueError("invalid station name")
        for key, value in asdict(self).items():
            if key in {"name", "entry", "corridor"}:
                continue
            if value is None and key in {
                    "settled_position_tolerance_m", "settled_yaw_tolerance_rad"
            }:
                continue
            finite_number(value, key)
            if not key.startswith("target_") and value <= 0:
                raise ValueError(f"{key} must be positive")
        if not -math.pi <= self.target_yaw <= math.pi:
            raise ValueError("target_yaw must be in [-pi, pi]")
        if self.entry_yaw_tolerance_rad > math.pi / 2:
            raise ValueError("station must be approached facing forward")
        if self.min_base_height_m >= self.max_base_height_m:
            raise ValueError("invalid base height interval")
        if self.stable_duration_s >= self.max_runtime_s:
            raise ValueError("stable dwell must fit runtime budget")
        if self.max_runtime_s > 120.0:
            raise ValueError("station runtime cannot exceed 120 seconds")
        if not self.corridor.contains(self.target_x, self.target_y,
                                      self.position_tolerance_m):
            raise ValueError("target tolerance region must fit corridor")
        if not (self.corridor.contains(self.entry.xmin, self.entry.ymin) and
                self.corridor.contains(self.entry.xmax, self.entry.ymax)):
            raise ValueError("entry region must fit corridor")

    def errors(self, pose: SimRobotPose) -> tuple[float, float]:
        return (math.hypot(self.target_x - pose.x, self.target_y - pose.y),
                abs(wrap_angle(self.target_yaw - pose.yaw)))

    def arrived(self, pose: SimRobotPose) -> bool:
        position, yaw = self.errors(pose)
        return (position <= self.position_tolerance_m and
                yaw <= self.yaw_tolerance_rad)

    def settled_arrived(self, pose: SimRobotPose) -> bool:
        """Accept the configured resting pose without widening the corridor."""
        position, yaw = self.errors(pose)
        return (position <= (self.settled_position_tolerance_m or
                             self.position_tolerance_m) and
                yaw <= (self.settled_yaw_tolerance_rad or
                        self.yaw_tolerance_rad))

    def pose_fault(self, pose: SimRobotPose) -> str | None:
        if pose.support_enabled:
            return "support_enabled"
        if not self.corridor.contains(pose.x, pose.y):
            return "outside_corridor"
        if pose.tilt > self.max_tilt_rad:
            return "excessive_tilt"
        if not self.min_base_height_m <= pose.z <= self.max_base_height_m:
            return "invalid_base_height"
        return None

    def body_velocity(self, pose: SimRobotPose) -> tuple[float, float, float]:
        """Bound planar speed as a vector; face station before translating."""
        error = wrap_angle(self.target_yaw - pose.yaw)
        yaw_rate = max(-self.max_yaw_rate_rad_s,
                       min(self.max_yaw_rate_rad_s, self.yaw_gain * error))
        if abs(error) > min(self.entry_yaw_tolerance_rad / 2, 0.175):
            return (0.0, 0.0, yaw_rate)
        wx = self.position_gain * (self.target_x - pose.x)
        wy = self.position_gain * (self.target_y - pose.y)
        scale = min(1.0, self.max_speed_mps / max(math.hypot(wx, wy), 1e-12))
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return (scale * (c * wx + s * wy), scale * (-s * wx + c * wy), yaw_rate)


class StationRegistry:
    """Explicit simulation opt-in for docking-only or docking-and-grasp mode."""

    def __init__(self, path: str | Path, runtime: str):
        if runtime != "sim":
            raise ValueError("station docking is simulation-only")
        self.path = Path(path).expanduser().resolve()
        config = json.loads(self.path.read_text(encoding="utf-8"))
        if (config.get("schema_version") != 1 or
                config.get("runtime") != "sim" or
                type(config.get("docking_only")) is not bool):
            raise ValueError("require schema 1, runtime sim, boolean docking_only")
        self.docking_only = config["docking_only"]
        if not isinstance(config.get("stations"), dict):
            raise ValueError("stations must be an object")
        self._stations = config["stations"]

    def get(self, name: str) -> Station:
        if not isinstance(name, str) or name not in self._stations:
            raise ValueError("unknown station")
        data = dict(self._stations[name])
        if data.pop("enabled", False) is not True:
            raise ValueError("station is disabled pending calibration")
        if data.pop("calibration_status", None) not in {"trial", "validated"}:
            raise ValueError("station requires explicit calibration status")
        data.pop("notes", None)
        data["name"] = name
        try:
            data["entry"] = Bounds(**data["entry"])
            data["corridor"] = Bounds(**data["corridor"])
            return Station(**data)
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid station configuration") from exc

    def public_status(self) -> dict[str, Any]:
        return {
            "mode": ("simulation_docking_only" if self.docking_only else
                     "simulation_docking_and_grasp"),
            "grasp_enabled": not self.docking_only,
            "stations": sorted(self._stations)
        }


@dataclass(frozen=True)
class DockingResult:
    """Symbolic result and error magnitudes, without raw world coordinates."""

    status: str
    reason: str
    station: str
    elapsed_s: float
    samples: int
    path_m: float
    position_error_m: float | None
    yaw_error_rad: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self), "privileged_sim_pose": True,
            "grasp_ready": False
        }


class StationDockingRunner:
    """Own no publisher; delegate every command to the current Executive."""

    def __init__(self,
                 executive: Any,
                 endpoint: str,
                 *,
                 source_factory: Callable = SimDockingPoseSource,
                 monotonic: Callable = time.monotonic,
                 wall_clock: Callable = time.time,
                 sleep: Callable = time.sleep):
        self.executive = executive
        self.endpoint = endpoint
        self._source_factory = source_factory
        self._clock = monotonic
        self._wall_clock = wall_clock
        self._sleep = sleep

    def run(self,
            station: Station,
            *,
            cancel_check: Callable[[], str | None] | None = None,
            state_callback: Callable[[dict], None] | None = None
           ) -> DockingResult:
        """Approach once, stop, then verify measured rest; never auto-retry."""
        started = self._clock()
        deadline = started + station.max_runtime_s
        period = 1.0 / station.control_hz
        source = None
        attempted_motion = False
        samples = 0
        path = 0.0
        last: SimRobotPose | None = None
        final: SimRobotPose | None = None
        stable_start: tuple[float, float] | None = None
        phase = "preflight"
        status, reason = "failed", "max_runtime_s"

        def timeout() -> float:
            return max(0.1, min(10.0, deadline - self._clock()))

        try:
            client = self.executive
            if client.runtime != "sim" or client.state != "READY":
                raise ValueError("simulation Executive must be READY")
            if (station.max_speed_mps > client.linear_speed_mps or
                    station.max_yaw_rate_rad_s > client.yaw_rate_rad_s or
                    period > client.command_watchdog_s / 2):
                raise ValueError("station limits exceed Executive contract")
            source = self._source_factory(self.endpoint)
            while self._clock() < deadline:
                cancellation = cancel_check() if cancel_check else None
                if cancellation:
                    status, reason = "cancelled", "cancel_requested"
                    break
                pose = source.latest(station.max_pose_age_s)
                if pose is None:
                    if phase == "preflight" and not source.error.startswith(
                            "invalid_pose") and self._clock() - started < 2.0:
                        self._sleep(period)
                        continue
                    reason = "stale_or_invalid_pose"
                    break
                if not pose.fresh(self._wall_clock(), station.max_pose_age_s):
                    reason = "stale_or_invalid_pose"
                    break
                fault = station.pose_fault(pose)
                if fault == "excessive_tilt" and phase == "approaching":
                    client.stop_moving(timeout_s=timeout())
                    attempted_motion = False
                    if client.state != "READY":
                        reason = "stop_not_acknowledged"
                        break
                    phase = "settling"
                    stable_start = None
                    last = None
                    continue
                if fault:
                    reason = fault
                    break
                final = pose
                if last is not None:
                    if (pose.wall_time_s, pose.sim_time_s) == (last.wall_time_s,
                                                               last.sim_time_s):
                        self._sleep(period)
                        continue
                    dt = pose.sim_time_s - last.sim_time_s
                    dw = pose.wall_time_s - last.wall_time_s
                    if dt <= 0 or dw <= 0:
                        reason = "simulation_reset_or_reordered_pose"
                        break
                    if dw > station.max_pose_age_s:
                        reason = "observation_gap"
                        break
                    distance = math.hypot(pose.x - last.x, pose.y - last.y)
                    angle = abs(wrap_angle(pose.yaw - last.yaw))
                    if (distance > station.max_step_m or
                            angle > station.max_step_yaw_rad):
                        reason = "pose_discontinuity"
                        break
                    path += distance
                    if path > station.max_path_m:
                        reason = "path_budget_exhausted"
                        break
                    slow = (distance / dt <= station.stable_speed_mps and
                            angle / dt <= station.stable_yaw_rate_rad_s)
                else:
                    slow = False
                last = pose
                samples += 1
                if phase == "preflight":
                    if (not station.settled_arrived(pose) and
                            not station.entry.contains(pose.x, pose.y)):
                        reason = "outside_entry_region"
                        break
                    if abs(wrap_angle(station.target_yaw - pose.yaw)) > (
                            station.entry_yaw_tolerance_rad):
                        reason = "outside_entry_heading"
                        break
                    if station.settled_arrived(pose):
                        phase = "settling"
                    else:
                        attempted_motion = True
                        client.begin_moving(timeout_s=timeout())
                        phase = "approaching"
                        # Reacquire pose after the potentially blocking handshake.
                        last = None
                        continue
                if phase == "approaching":
                    if client.state != "MOVING":
                        reason = "executive_left_moving"
                        break
                    if station.arrived(pose):
                        client.send_nav_velocity(0.0, 0.0, 0.0)
                        client.stop_moving(timeout_s=timeout())
                        attempted_motion = False
                        if client.state != "READY":
                            reason = "stop_not_acknowledged"
                            break
                        phase = "settling"
                        stable_start = None
                        # Only post-stop observations may establish stable dwell.
                        last = None
                        continue
                    velocity = station.body_velocity(pose)
                    forward, lateral, _ = velocity
                    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
                    horizon = client.command_watchdog_s + station.max_pose_age_s
                    predicted_x = pose.x + (c * forward - s * lateral) * horizon
                    predicted_y = pose.y + (s * forward + c * lateral) * horizon
                    if not station.corridor.contains(predicted_x, predicted_y):
                        reason = "command_would_leave_corridor"
                        break
                    client.send_nav_velocity(*velocity)
                else:
                    if client.state != "READY":
                        reason = "executive_not_ready_after_stop"
                        break
                    if not station.settled_arrived(pose):
                        reason = "drifted_after_stop"
                        break
                    if not slow:
                        stable_start = None
                    elif stable_start is None:
                        stable_start = (pose.sim_time_s, self._clock())
                    elif (pose.sim_time_s - stable_start[0]
                          >= station.stable_duration_s and
                          self._clock() - stable_start[1]
                          >= station.stable_duration_s):
                        status, reason = "docked", "pose_and_rest_verified"
                        break
                if state_callback:
                    state_callback({
                        "station": station.name,
                        "phase": phase,
                        "status": "running",
                        "samples": samples,
                        "privileged_sim_pose": True
                    })
                self._sleep(period)
        except Exception as exc:
            status, reason = "failed", f"execution_error:{type(exc).__name__}"
        finally:
            if attempted_motion:
                try:
                    if self.executive.state == "MOVING":
                        try:
                            self.executive.send_nav_velocity(0.0, 0.0, 0.0)
                        except Exception:
                            # A rejected zero command must not skip native stop.
                            pass
                    self.executive.stop_moving(timeout_s=10.0)
                    if self.executive.state != "READY":
                        raise RuntimeError("Executive did not return READY")
                except Exception:
                    status, reason = "failed", "stop_failed"
            if source is not None:
                try:
                    source.close()
                except Exception:
                    if reason != "stop_failed":
                        status, reason = "failed", "pose_source_close_failed"
        errors = station.errors(final) if final else (None, None)
        return DockingResult(status, reason, station.name,
                             self._clock() - started, samples, path, *errors)
