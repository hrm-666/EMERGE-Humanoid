"""Dispatch geometric and VLA actions through RoboCasa's action12 interface.

Geometric actions servo the grip site without planning collision-free paths or
moving the base/torso intentionally. The environment and VLA executor are borrowed.
Only NumPy and the existing pure pose helpers are needed at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from robot.mujoco_simulation.pose_utils import PoseUtils

if TYPE_CHECKING:
    from robot.robocasa.env import RoboCasaEnvManager
    from robot.vla.robocasa_policy_executor import RoboCasaPolicyExecutor, RoboCasaVLAResult


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
    """Dispatch VLA requests and execute world-frame geometric requests."""

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
        self, environment: RoboCasaEnvManager, config: dict[str, Any] | None = None,
        *, vla_executor: RoboCasaPolicyExecutor | None = None,
    ) -> None:
        config = dict(config or {})
        self._environment = environment
        self._vla = vla_executor
        self.config = self._DEFAULTS | config

    def execute(
        self, action_type: str, params: dict[str, Any], *,
        cancel_check: Callable[[], str | None] | None = None,
    ) -> RoboCasaActionResult | RoboCasaVLAResult:
        """Dispatch VLA to the borrowed executor, or run a geometric request.

        Completion means pose convergence, gripper execution or requested VLA steps.
        Native/evaluation success can end any request early as goal_reached.
        """
        if action_type == "vla_execute":
            return self._vla_execute(params, cancel_check)
        progress = _Progress()
        completed, reason, message, error_stage = False, "error", None, None
        observation_error = None
        try:
            request = self._request(action_type, params)
            progress.stage = "environment_state"
            state = self._environment.get_robot_state()
            progress.terminated = bool(state["terminated"])
            progress.truncated = bool(state["truncated"])
            self._check_stop(progress, cancel_check)
            progress.stage = "control_spec"
            spec = self._environment.get_arm_control_spec()
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

    def _vla_execute(
        self, params: dict[str, Any], cancel_check: Callable[[], str | None] | None,
    ) -> RoboCasaVLAResult:
        if self._vla is None:
            raise RuntimeError("RoboCasa VLA executor is not initialized")
        return self._vla.execute(
            params.get("instruction", params.get("prompt")),
            step=params.get("step"), cancel_check=cancel_check,
        )

    def _request(self, action_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if action_type not in {"move_to_pose", "move_linear", "set_gripper"}:
            raise ValueError(f"unsupported RoboCasa geometric action {action_type!r}")
        if params.get("frame", "world") != "world":
            raise ValueError("geometric targets and deltas must use frame='world'")
        default_steps = {"move_to_pose": "max_steps", "move_linear": "move_linear_steps", "set_gripper": "gripper_steps"}
        request = {
            "steps": max(1, int(params.get("steps", self.config[default_steps[action_type]]))),
            "settle_steps": 0, "relative": False, "position": None, "orientation": None,
            "gripper": None,
        }
        if action_type == "set_gripper":
            request["gripper"] = self._gripper_command(params.get("command"))
            return request
        if "gripper_command" in params:
            request["gripper"] = self._gripper_command(params["gripper_command"])
        request["relative"] = action_type == "move_linear" and "position_m" not in params
        key = "delta_m" if request["relative"] else "position_m"
        request["position"] = self._vector(params[key], key)
        orientation = {key: self._vector(params[key], key, length=len(keys), keys=keys)
                       for key, keys in (("orientation_quat", "xyzw"), ("orientation_euler", ("roll", "pitch", "yaw")))
                       if key in params}
        request["orientation"] = PoseUtils.resolve_orientation(orientation)
        if action_type == "move_to_pose" and request["orientation"] is None:
            raise ValueError("move_to_pose requires orientation_quat or orientation_euler")
        if action_type == "move_linear":
            request["settle_steps"] = max(0, int(params.get("settle_steps", self.config["settle_steps"])))
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
    def _vector(
        value: Any, name: str, *, length: int = 3, keys: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        if isinstance(value, dict):
            value = [value[key] for key in (keys or "xyzw"[:length])]
        return PoseUtils.vector(value, name=name, length=length)
