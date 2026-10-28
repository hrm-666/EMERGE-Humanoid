"""World-frame geometric actions through RoboCasa's native action12 interface.

This layer servos the grip site; it does not plan collision-free paths or move
the base/torso intentionally. It borrows the environment and never resets it.
Only NumPy and the existing pure pose helpers are needed at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from robot.mujoco_simulation.pose_utils import PoseUtils

if TYPE_CHECKING:
    from robot.robocasa.env import RoboCasaEnvManager


@dataclass(slots=True)
class RoboCasaActionResult:
    """One geometric request, separate from the environment's task success.

    total_steps includes steps whose later status I/O raised. Error metrics are
    measured at the last observed state, or None after an unobserved step.
    gripper_command_applied confirms command execution, not grasp or width.
    """

    completed: bool
    task_success: bool
    total_steps: int
    reason: str
    position_error_m: float | None = None
    rotation_error_rad: float | None = None
    error_message: str | None = None
    error_stage: str | None = None
    observation_error: str | None = None


@dataclass
class _Progress:
    steps: int = 0
    dirty: bool = False
    stage: str = "request_validation"
    terminated: bool = False
    truncated: bool = False
    position_error: float | None = None
    rotation_error: float | None = None


class _Stopped(Exception):
    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason
        self.message = message


class RoboCasaActionController:
    """Execute move_to_pose, move_linear and binary set_gripper requests.

    Targets and deltas use world metres, and orientations use XYZW quaternions
    or XYZ roll/pitch/yaw radians. move_to_pose requires an orientation;
    move_linear preserves the initial orientation when omitted. Its steps are
    interpolation steps, followed by at most settle_steps of endpoint servoing.

    Motion retains env.last_gripper_close. When that is unknown, initialize with
    set_gripper or explicitly pass gripper_command='open'/'close' on the motion.
    set_gripper accepts command only; intermediate opening_m is unsupported.

    Cancellation is checked before every step. All steps use arm mode, zero
    base velocity and zero torso increment. Observations publish periodically
    and on return when dirty. No applied action is retried after an error.
    """

    _DEFAULTS = {
        "position_gain": .5,
        "rotation_gain": .5,
        "position_tolerance_m": .005,
        "rotation_tolerance_rad": .05,
        "max_steps": 100,
        "move_linear_steps": 40,
        "settle_steps": 30,
        "gripper_steps": 30,
        "publish_every_n_steps": 5,
    }

    def __init__(
        self, environment: RoboCasaEnvManager, config: dict[str, Any] | None = None
    ) -> None:
        if config is not None and not isinstance(config, dict):
            raise ValueError("RoboCasa motion config must be a mapping")
        config = dict(config or {})
        unknown = set(config) - self._DEFAULTS.keys()
        if unknown:
            raise ValueError(f"unsupported RoboCasa motion settings: {', '.join(sorted(unknown))}")
        self._environment = environment
        self.config = self._DEFAULTS | config
        for name in self._DEFAULTS:
            if name in {"position_gain", "rotation_gain", "position_tolerance_m", "rotation_tolerance_rad"}:
                self.config[name] = self._positive_number(self.config[name], name)
            else:
                self.config[name] = self._integer(self.config[name], name, allow_zero=name == "settle_steps")

    def execute(
        self, action_type: str, params: dict[str, Any], *,
        cancel_check: Callable[[], str | None] | None = None,
    ) -> RoboCasaActionResult:
        """Return operational/validation failures; propagate BaseExceptions.

        Request completion means pose convergence or gripper command execution.
        Native/evaluation success can end any request early as goal_reached.
        """
        progress = _Progress()
        completed, reason, message, error_stage = False, "error", None, None
        observation_error = None
        try:
            request = self._request(action_type, params)
            progress.stage = "environment_state"
            self._check_stop(progress, cancel_check)
            state = self._environment.get_robot_state()
            progress.terminated = bool(state["terminated"])
            progress.truncated = bool(state["truncated"])
            self._check_stop(progress, cancel_check)
            progress.stage = "control_spec"
            spec = self._environment.get_arm_control_spec()
            for prefix in ("input", "output"):
                for suffix in ("min", "max"):
                    key = f"{prefix}_{suffix}"
                    spec[key] = self._vector(spec[key], key, length=6)
            if np.any(spec["output_min"] >= 0) or np.any(spec["output_max"] <= 0):
                raise ValueError("OSC output ranges must contain zero and both motion directions")
            progress.stage = "geometry"
            start, start_quaternion = self._environment.get_eef_pose_world()
            if action_type == "set_gripper":
                target, quaternion = start, start_quaternion
            else:
                target = request["position"]
                if request["relative"]:
                    target = start + target
                quaternion = request["orientation"]
                if quaternion is None:
                    quaternion = start_quaternion
            gripper = request["gripper"]
            if action_type == "set_gripper":
                for _ in range(request["steps"]):
                    self._advance(target, quaternion, gripper, spec, progress, cancel_check)
                self._measure(target, quaternion, progress)
                completed, reason = True, "gripper_command_applied"
            elif self._converged(target, quaternion, progress):
                completed, reason = True, "pose_reached"
            else:
                if action_type == "move_linear":
                    for index in range(request["steps"]):
                        waypoint = start + (target - start) * ((index + 1) / request["steps"])
                        self._advance(waypoint, quaternion, gripper, spec, progress, cancel_check)
                    servo_steps = request["settle_steps"]
                else:
                    servo_steps = request["steps"]
                completed = self._servo(target, quaternion, gripper, spec, servo_steps, progress, cancel_check)
                reason = "pose_reached" if completed else "not_converged"
        except _Stopped as exc:
            completed, reason, message = exc.reason == "goal_reached", exc.reason, exc.message
            if reason in {"error", "environment_unavailable", "gripper_uninitialized"}:
                error_stage = progress.stage
        except Exception as exc:
            reason = "invalid_request" if progress.stage == "request_validation" else "error"
            message, error_stage = f"{type(exc).__name__}: {exc}", progress.stage
        finally:
            if progress.dirty and self._environment.is_connected():
                try:
                    self._publish(progress)
                except Exception as exc:
                    observation_error = f"{type(exc).__name__}: {exc}"
                    if completed:
                        completed, reason = False, "error"
                        message, error_stage = observation_error, "observation_publish"
        return RoboCasaActionResult(
            completed=completed, task_success=bool(self._environment.check_success()),
            total_steps=progress.steps, reason=reason,
            position_error_m=progress.position_error, rotation_error_rad=progress.rotation_error,
            error_message=message, error_stage=error_stage, observation_error=observation_error,
        )

    def _request(self, action_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if action_type not in {"move_to_pose", "move_linear", "set_gripper"}:
            raise ValueError(f"unsupported RoboCasa geometric action {action_type!r}")
        if not isinstance(params, dict):
            raise ValueError("action parameters must be a mapping")
        allowed = {"steps", "frame"}
        if action_type == "set_gripper":
            allowed.add("command")
        else:
            allowed |= {"position_m", "orientation_quat", "orientation_euler", "gripper_command"}
            if action_type == "move_linear":
                allowed |= {"delta_m", "settle_steps"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported {action_type} parameters: {', '.join(sorted(unknown))}")
        if params.get("frame", "world") != "world":
            raise ValueError("geometric targets and deltas must use frame='world'")
        default_steps = {"move_to_pose": "max_steps", "move_linear": "move_linear_steps", "set_gripper": "gripper_steps"}
        request = {
            "steps": self._integer(params.get("steps", self.config[default_steps[action_type]]), "steps"),
            "settle_steps": 0, "relative": False, "position": None, "orientation": None,
            "gripper": None,
        }
        if action_type == "set_gripper":
            request["gripper"] = self._gripper_command(params.get("command"))
            return request
        if "gripper_command" in params:
            request["gripper"] = self._gripper_command(params["gripper_command"])
        if ("position_m" in params) == ("delta_m" in params):
            raise ValueError("supply exactly one position_m or delta_m target")
        request["relative"] = "delta_m" in params
        key = "delta_m" if request["relative"] else "position_m"
        request["position"] = self._vector(params[key], key)
        if "orientation_quat" in params and "orientation_euler" in params:
            raise ValueError("supply only one orientation representation")
        if "orientation_quat" in params:
            quaternion = self._vector(params["orientation_quat"], "orientation_quat", length=4)
            request["orientation"] = PoseUtils.normalize_quaternion(quaternion)
        elif "orientation_euler" in params:
            euler = self._vector(params["orientation_euler"], "orientation_euler", keys=("roll", "pitch", "yaw"))
            request["orientation"] = PoseUtils.euler_to_quaternion(euler)
        elif action_type == "move_to_pose":
            raise ValueError("move_to_pose requires orientation_quat or orientation_euler")
        if action_type == "move_linear":
            request["settle_steps"] = self._integer(
                params.get("settle_steps", self.config["settle_steps"]), "settle_steps", allow_zero=True
            )
        return request

    def _check_stop(self, progress: _Progress, cancel_check: Callable[[], str | None] | None) -> None:
        progress.stage = "environment_state"
        environment = self._environment
        if getattr(environment, "episode_done", False):
            if environment.check_success():
                raise _Stopped("goal_reached")
            reason = environment.termination_reason or "environment_done"
            message = environment.get_episode_status().get("error_message") if reason == "error" else None
            raise _Stopped(reason, message)
        if not environment.is_connected():
            raise _Stopped("environment_unavailable", "create/reset the RoboCasa environment first")
        if environment.check_success():
            raise _Stopped("goal_reached")
        if progress.terminated or progress.truncated:
            raise _Stopped("environment_terminated" if progress.terminated else "environment_truncated")
        if getattr(environment, "remaining_steps", None) == 0:
            raise _Stopped("max_action_steps")
        progress.stage = "cancel_check"
        message = cancel_check() if cancel_check else None
        if message:
            raise _Stopped("interrupted", str(message))

    def _measure(self, target: np.ndarray, quaternion: np.ndarray, progress: _Progress) -> tuple[np.ndarray, np.ndarray]:
        progress.stage = "geometry"
        position, current_quaternion = self._environment.get_eef_pose_world()
        position_error = target - position
        rotation_error = PoseUtils.quaternion_error(current_quaternion, quaternion)
        progress.position_error = float(np.linalg.norm(position_error))
        progress.rotation_error = float(np.linalg.norm(rotation_error))
        return position_error, rotation_error

    def _converged(self, target: np.ndarray, quaternion: np.ndarray, progress: _Progress) -> bool:
        self._measure(target, quaternion, progress)
        return (
            progress.position_error <= self.config["position_tolerance_m"]
            and progress.rotation_error <= self.config["rotation_tolerance_rad"]
        )

    def _servo(
        self, target: np.ndarray, quaternion: np.ndarray, gripper: bool | None,
        spec: dict[str, Any], steps: int, progress: _Progress,
        cancel_check: Callable[[], str | None] | None,
    ) -> bool:
        for _ in range(steps):
            self._check_stop(progress, cancel_check)
            if self._converged(target, quaternion, progress):
                return True
            self._advance(target, quaternion, gripper, spec, progress, cancel_check)
        # Inspect the actual state after the final allowed action as well.
        return self._converged(target, quaternion, progress)

    def _advance(
        self, target: np.ndarray, quaternion: np.ndarray, gripper: bool | None,
        spec: dict[str, Any], progress: _Progress,
        cancel_check: Callable[[], str | None] | None,
    ) -> None:
        self._check_stop(progress, cancel_check)
        position_error, rotation_error = self._measure(target, quaternion, progress)
        if gripper is None:
            gripper = self._environment.last_gripper_close
        if gripper is None:
            progress.stage = "gripper_state"
            raise _Stopped("gripper_uninitialized", "use set_gripper or an explicit gripper_command before motion")
        delta = np.r_[position_error * self.config["position_gain"], rotation_error * self.config["rotation_gain"]]
        if spec["input_ref_frame"] == "base":
            progress.stage = "control_frame"
            rotation = self._environment.get_arm_control_frame()[:3, :3]
            delta = np.r_[rotation.T @ delta[:3], rotation.T @ delta[3:]]
        delta = np.clip(delta, spec["output_min"], spec["output_max"])
        arm = spec["input_min"] + (delta - spec["output_min"]) * (
            (spec["input_max"] - spec["input_min"]) / (spec["output_max"] - spec["output_min"])
        )
        action = np.zeros(12, dtype=np.float64)
        action[:6] = np.clip(arm, spec["input_min"], spec["input_max"])
        action[6] = float(gripper)
        # action[7:10]: base velocities; [10]: torso delta; [11]=0: arm mode.
        self._check_stop(progress, cancel_check)
        progress.stage = "environment_step"
        before = self._environment.action_steps
        # Even a step that raises before its counter increments may have changed
        # physics. Do not expose pre-step errors as measurements of that state.
        progress.position_error = progress.rotation_error = None
        try:
            _, _, progress.terminated, progress.truncated, _ = self._environment.step(
                action, publish_observation=False
            )
        finally:
            count = self._environment.action_steps - before
            if count > 0:
                progress.steps += count
                progress.dirty = True
        if count != 1:
            raise RuntimeError("RoboCasa step did not increment its action counter exactly once")
        self._check_stop(progress, cancel_check)
        if progress.steps % self.config["publish_every_n_steps"] == 0:
            self._publish(progress)

    def _publish(self, progress: _Progress) -> None:
        progress.stage = "observation_publish"
        self._environment.refresh_cameras()
        progress.dirty = False

    @staticmethod
    def _gripper_command(value: Any) -> bool:
        if not isinstance(value, str) or value.strip().lower() not in {"open", "close"}:
            raise ValueError("gripper command must be 'open' or 'close'; intermediate widths are unsupported")
        return value.strip().lower() == "close"

    @staticmethod
    def _integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
        minimum = 0 if allow_zero else 1
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return int(value)

    @staticmethod
    def _positive_number(value: Any, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a positive finite number")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    @staticmethod
    def _vector(value: Any, name: str, *, length: int = 3, keys: tuple[str, ...] | None = None) -> np.ndarray:
        if isinstance(value, dict):
            keys = keys or tuple("xyzw"[:length])
            if set(value) != set(keys):
                raise ValueError(f"{name} must have exactly the keys {keys}")
            value = [value[key] for key in keys]
        array = np.asarray(value)
        if (
            array.shape != (length,) or array.dtype.kind not in "fiu"
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(f"{name} must contain exactly {length} finite real numbers")
        return array.astype(np.float64, copy=True)
