"""Execute OpenPI action12 chunks against the RoboCasa environment adapters.

Policy inputs follow local openpi/examples/robocasa/main.py: three native RGB
images, resize-with-pad, unchanged XYZW quaternions and a 16-D state. Perception
camera artifacts are independent of these inputs. OpenPI is imported lazily.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from robot.robocasa.env import RoboCasaEnvManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RoboCasaVLAResult:
    """One executor call, with task success separate from request completion.

    completed means the requested steps finished or the goal was reached.
    task_success is the environment's success signal, including its evaluation
    latch when applicable. Exhausting a call's steps alone is not task success.
    total_steps counts completed Gym steps, also when later status I/O raises.
    last_gripper_command retains raw action12[6], before RoboCasa's threshold.
    observation_error reports a final publication failure alongside any primary
    execution error or cancellation. No action is automatically retried.
    """

    completed: bool
    task_success: bool
    total_steps: int
    reason: str
    error_message: str | None = None
    error_stage: str | None = None
    observation_error: str | None = None
    last_gripper_command: float | None = None


@dataclass
class _Progress:
    steps: int = 0
    stage: str = "environment_state"
    dirty: bool = False
    last_gripper: float | None = None
    inference_count: int = 0
    inference_seconds: float = 0.0


class RoboCasaPolicyExecutor:
    """Observe, infer and execute bounded action chunks without resetting the env.

    Config supports server_url, timeout (passed to Pi05Client), resize_size
    (224), replan_steps (5), publish_every_n_steps (defaults to replan_steps),
    stop_on_success (True), and base/wrist/right_image_key overrides. Publication
    runs periodically and once on return if there are unpublished actions.

    Calls are synchronous. Cancellation is checked before inference and before
    every action, including immediately after inference returns; it cannot abort
    an in-flight client.infer(). Pi05Client's timeout does not impose an RPC
    deadline. An injected client is borrowed and is never closed here.
    """

    _IMAGES = (
        ("base_image_key", "video.robot0_agentview_left", "observation/image"),
        ("wrist_image_key", "video.robot0_eye_in_hand", "observation/wrist_image"),
        ("right_image_key", "video.robot0_agentview_right", "observation/right_image"),
    )
    _STATE = (
        ("state.end_effector_position_relative", 3),
        ("state.end_effector_rotation_relative", 4),
        ("state.base_position", 3),
        ("state.base_rotation", 4),
        ("state.gripper_qpos", 2),
    )

    def __init__(
        self,
        environment: RoboCasaEnvManager,
        *,
        config: dict[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self._environment = environment
        if config is not None and not isinstance(config, dict):
            raise ValueError("RoboCasa VLA config must be a mapping")
        self.config = deepcopy(config or {})
        allowed = {
            "server_url", "timeout", "resize_size", "replan_steps",
            "publish_every_n_steps", "stop_on_success",
            *(item[0] for item in self._IMAGES),
        }
        unknown = set(self.config) - allowed
        if unknown:
            raise ValueError(f"unsupported RoboCasa VLA settings: {', '.join(sorted(unknown))}")
        self.resize_size = self._positive_int(self.config.get("resize_size", 224), "resize_size")
        self.replan_steps = self._positive_int(self.config.get("replan_steps", 5), "replan_steps")
        self.publish_every_n_steps = self._positive_int(
            self.config.get("publish_every_n_steps", self.replan_steps), "publish_every_n_steps"
        )
        self.stop_on_success = self.config.get("stop_on_success", True)
        if not isinstance(self.stop_on_success, bool):
            raise ValueError("stop_on_success must be a boolean")
        for config_key, default, _ in self._IMAGES:
            value = self.config.get(config_key, default)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{config_key} must be a non-empty observation key")
        self._server_url = self.config.get("server_url", "ws://localhost:8000")
        if not isinstance(self._server_url, str) or not self._server_url.strip():
            raise ValueError("server_url must be non-empty")
        raw_timeout = self.config.get("timeout", 120.0)
        if isinstance(raw_timeout, (bool, np.bool_)) or not np.isscalar(raw_timeout):
            raise ValueError("timeout must be a positive finite number")
        self._timeout = float(raw_timeout)
        if not np.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self._client = client
        self._owns_client = client is None
        self._closed = False

    def execute(
        self,
        instruction: str | None = None,
        *,
        step: int,
        cancel_check: Callable[[], str | None] | None = None,
    ) -> RoboCasaVLAResult:
        """Execute at most step actions; None instruction uses the native task text.

        Invalid step requests raise ValueError before inference or physics.
        Operational errors return a structured result. KeyboardInterrupt and
        other BaseExceptions propagate after best-effort final publication.
        """
        step = self._positive_int(step, "step")
        progress = _Progress()
        started = time.perf_counter()
        completed, reason, error_message = False, "error", None
        error_stage = observation_error = None
        try:
            if self._closed:
                reason, error_message = "closed", "RoboCasa policy executor is closed"
            else:
                completed, reason, error_message = self._run(instruction, step, cancel_check, progress)
                if reason in {"error", "server_unavailable", "environment_unavailable"}:
                    error_stage = progress.stage
        except Exception as exc:
            completed, reason = False, "error"
            error_message = f"{type(exc).__name__}: {exc}"
            error_stage = progress.stage
        finally:
            if progress.dirty and self._environment.is_connected():
                try:
                    self._publish(progress)
                except Exception as exc:
                    observation_error = f"{type(exc).__name__}: {exc}"
                    if reason not in {"error", "interrupted", "server_unavailable", "closed"}:
                        completed, reason = False, "error"
                        error_message, error_stage = observation_error, "observation_publish"
            logger.info(
                "RoboCasa VLA | total=%.3fs actions=%d inference_calls=%d inference=%.3fs reason=%s",
                time.perf_counter() - started, progress.steps,
                progress.inference_count, progress.inference_seconds, reason,
            )
        return RoboCasaVLAResult(
            completed=completed, task_success=bool(self._environment.check_success()),
            total_steps=progress.steps, reason=reason, error_message=error_message,
            error_stage=error_stage, observation_error=observation_error,
            last_gripper_command=progress.last_gripper,
        )

    def _run(
        self, instruction: str | None, step: int,
        cancel_check: Callable[[], str | None] | None, progress: _Progress,
    ) -> tuple[bool, str, str | None]:
        stopped = self._stop_status()
        if stopped is not None:
            return stopped
        state = self._environment.get_robot_state()
        stopped = self._stop_status(bool(state["terminated"]), bool(state["truncated"]))
        if stopped is not None:
            return stopped
        progress.stage = "preprocess"
        if instruction is None:
            instruction = self._environment.instruction
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("RoboCasa policy instruction must be a non-empty string")
        health_checked = False
        while True:
            progress.stage = "environment_state"
            stopped = self._stop_status()
            if stopped is not None:
                return stopped
            progress.stage = "cancel_check"
            cancellation = cancel_check() if cancel_check else None
            if cancellation:
                return False, "interrupted", cancellation
            progress.stage = "preprocess"
            element = self._build_element(instruction)
            progress.stage = "health_check"
            client = self._get_client()
            if not health_checked:
                health_check = getattr(client, "health_check", None)
                if callable(health_check) and not health_check():
                    return False, "server_unavailable", "policy server health check failed"
                health_checked = True
            # Preprocessing and the health probe may take time. Recheck directly
            # before RPC so cancellation during them does not start inference.
            progress.stage = "environment_state"
            stopped = self._stop_status()
            if stopped is not None:
                return stopped
            progress.stage = "cancel_check"
            cancellation = cancel_check() if cancel_check else None
            if cancellation:
                return False, "interrupted", cancellation
            progress.stage = "inference"
            inference_started = time.perf_counter()
            progress.inference_count += 1
            try:
                result = client.infer(element)
            finally:
                progress.inference_seconds += time.perf_counter() - inference_started
            progress.stage = "action_validation"
            chunk = self._action_chunk(result)
            execute_count = min(len(chunk), self.replan_steps, step - progress.steps)
            remaining = getattr(self._environment, "remaining_steps", None)
            if remaining is not None:
                execute_count = min(execute_count, remaining)
            for action in chunk[:execute_count]:
                progress.stage = "environment_state"
                stopped = self._stop_status()
                if stopped is not None:
                    return stopped
                progress.stage = "cancel_check"
                cancellation = cancel_check() if cancel_check else None
                if cancellation:
                    return False, "interrupted", cancellation
                progress.stage = "environment_step"
                before = self._environment.action_steps
                try:
                    _, _, terminated, truncated, _ = self._environment.step(
                        action, publish_observation=False
                    )
                finally:
                    # Evaluation status I/O may raise after the Gym step counted.
                    delta = self._environment.action_steps - before
                    if delta > 0:
                        progress.steps += delta
                        progress.last_gripper = float(action[6])
                        progress.dirty = True
                if delta != 1:
                    raise RuntimeError("RoboCasa step did not increment its action counter exactly once")
                progress.stage = "environment_state"
                stopped = self._stop_status(terminated, truncated)
                if stopped is not None:
                    return stopped
                if progress.steps >= step:
                    return True, "step_completed", None
                if progress.steps % self.publish_every_n_steps == 0:
                    self._publish(progress)

    def _stop_status(
        self, terminated: bool = False, truncated: bool = False
    ) -> tuple[bool, str, str | None] | None:
        if getattr(self._environment, "episode_done", False):
            if self._environment.check_success():
                return True, "goal_reached", None
            if self._environment.termination_reason == "error":
                return False, "error", self._environment.get_episode_status().get("error_message")
            return False, self._environment.termination_reason or "environment_done", None
        if not self._environment.is_connected():
            return False, "environment_unavailable", "RoboCasa observation is unavailable; create/reset first"
        if self.stop_on_success and self._environment.check_success():
            return True, "goal_reached", None
        if terminated:
            return False, "environment_terminated", None
        if truncated:
            return False, "environment_truncated", None
        if getattr(self._environment, "remaining_steps", None) == 0:
            return False, "max_action_steps", None
        return None

    def _publish(self, progress: _Progress) -> None:
        progress.stage = "observation_publish"
        self._environment.refresh_cameras()
        progress.dirty = False

    def _build_element(self, instruction: str) -> dict[str, Any]:
        obs = self._environment.latest_obs
        vectors = []
        for key, length in self._STATE:
            value = np.asarray(obs[key])
            if (
                value.shape != (length,) or not np.issubdtype(value.dtype, np.number)
                or np.iscomplexobj(value) or not np.all(np.isfinite(value))
            ):
                raise ValueError(f"RoboCasa state {key!r} must contain {length} finite real values")
            vectors.append(value)
        # Preserve the baseline ordering and native state precision/quaternions.
        element = {"observation/state": np.concatenate(vectors), "prompt": instruction}
        for config_key, default, output_key in self._IMAGES:
            element[output_key] = self._prepare_image(obs[self.config.get(config_key, default)])
        return element

    def _prepare_image(self, value: Any) -> np.ndarray:
        from openpi_client import image_tools

        image = np.asarray(value)
        if (
            image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or min(image.shape[:2]) < 1
        ):
            raise ValueError("RoboCasa native policy images must be non-empty HWC uint8 RGB")
        # Gym already flipped RGB upright; the RoboCasa baseline adds no rotation.
        image = image_tools.resize_with_pad(
            np.ascontiguousarray(image), self.resize_size, self.resize_size
        )
        return image_tools.convert_to_uint8(image).copy(order="C")

    @staticmethod
    def _action_chunk(result: Any) -> np.ndarray:
        if not isinstance(result, Mapping) or "actions" not in result:
            raise ValueError("OpenPI result must contain actions")
        chunk = np.asarray(result["actions"])
        if (
            chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != 12
            or not np.issubdtype(chunk.dtype, np.number) or np.iscomplexobj(chunk)
            or not np.all(np.isfinite(chunk))
        ):
            raise ValueError(f"OpenPI actions must have non-empty shape (N, 12) and finite real values; got {chunk.shape}")
        return chunk

    def _get_client(self) -> Any:
        if self._client is None:
            from robot.vla.openpi_bridge import Pi05Client

            self._client = Pi05Client(self._server_url, timeout=self._timeout)
        return self._client

    def close(self) -> None:
        """Close an owned client once; keep the environment owned by its caller."""
        if self._closed:
            return
        self._closed = True
        client, self._client = self._client, None
        if client is not None and self._owns_client:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)
