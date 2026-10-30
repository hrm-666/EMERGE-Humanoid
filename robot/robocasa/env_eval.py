"""Episode budgets and evaluation status on top of the RoboCasa Gym adapter."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from Emerge.runtime.storage import atomic_json
from robot.robocasa.env import RoboCasaEnvManager, StepResult

logger = logging.getLogger(__name__)


class RoboCasaEvalEnvManager(RoboCasaEnvManager):
    """Enforce one episode's budget without changing native Gym return values.

    Success and budget exhaustion set episode_done; check it alongside Gym flags.
    """

    def __init__(
        self,
        *,
        max_action_steps: int | None = None,
        status_path: str | Path | None = None,
        video_path: str | Path | None = None,
        **env_kwargs: Any,
    ) -> None:
        super().__init__(**env_kwargs)
        if max_action_steps is not None:
            self._validate_budget(max_action_steps)
        self._budget_override = (
            None if max_action_steps is None else int(max_action_steps)
        )
        self._max_action_steps = self._budget_override
        self.status_path = (
            Path(status_path).expanduser().resolve() if status_path is not None else None
        )
        self.video_path = Path(video_path).expanduser().resolve() if video_path is not None else None
        self._video_writer = None
        self._reset_evaluation_state()

    @property
    def max_action_steps(self) -> int | None:
        """Resolved budget, or None before the first registry-based reset."""
        return self._max_action_steps

    @property
    def action_steps(self) -> int:
        """Keep the last episode's count available after resource cleanup."""
        if self._closed_status is not None:
            return int(self._closed_status["action_steps"])
        return super().action_steps

    @property
    def remaining_steps(self) -> int | None:
        if self._max_action_steps is None:
            return None
        return max(0, self._max_action_steps - self.action_steps)

    @property
    def episode_done(self) -> bool:
        """Evaluation completion, including success, budget, errors or close."""
        return self._episode_done

    @property
    def episode_success(self) -> bool:
        return self._episode_success

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    def create(self) -> None:
        """Use the parent's initialization sequence, also recording setup errors."""
        self.close()
        self._reset_evaluation_state()
        try:
            self._invalidate_published_observation()
            # Replace any previous episode's ready status before construction.
            self._write_evaluation_status()
            super().create()
        except Exception as exc:
            self._record_error(exc)
            self._write_status_safely()
            raise

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset evaluation state around exactly one call to the base reset."""
        self._require_environment()
        self._validate_seed(seed)
        self._reset_evaluation_state()
        # Invalidate the preceding episode even if horizon lookup or reset fails.
        self._clear_observation()
        self._action_steps = 0
        try:
            self._close_video()
            self._invalidate_published_observation()
            self._write_evaluation_status()
            if self._budget_override is None:
                registry = importlib.import_module("robocasa.utils.dataset_registry_utils")
                budget = registry.get_task_horizon(self.env_name)
                self._validate_budget(budget)
                self._max_action_steps = int(budget)
            result = super().reset(seed=seed)
            if super().check_success():
                self._finish("success")
            self._record_video()
            self._write_evaluation_status()
            return result
        except Exception as exc:
            self._record_error(exc)
            self._write_status_safely()
            raise

    def step(self, action12: Any, *, publish_observation: bool = True) -> StepResult:
        """Count only base step calls and stop before any action beyond the limit."""
        if self._episode_done:
            raise RuntimeError(
                f"RoboCasa evaluation ended ({self._termination_reason}); reset before stepping again"
            )
        self._require_ready()
        before = self.action_steps
        try:
            try:
                result = super().step(action12, publish_observation=publish_observation)
            finally:
                if self.action_steps > before and self.is_connected():
                    self._record_video()
            # Success wins when the final allowed action also reaches the goal.
            if super().check_success():
                self._finish("success")
            elif result[2]:
                self._finish("environment_terminated")
            elif result[3]:
                self._finish("environment_truncated")
            elif self.remaining_steps == 0:
                self._finish("max_action_steps")
            self._write_evaluation_status()
            return result
        except Exception as exc:
            # The Gym action can succeed before camera publication raises.
            # Preserve that native goal signal even without a returned result.
            if super().check_success():
                self._finish("success")
            self._record_error(exc)
            self._write_status_safely()
            raise

    def check_success(self) -> bool:
        """Latched episode success, also available after close()."""
        return self._episode_success

    def get_robot_state(self) -> dict[str, Any]:
        """Add evaluation completion and budget to the live robot summary."""
        state = super().get_robot_state()
        state.update(
            done=self._episode_done,
            success=self._episode_success,
            max_action_steps=self._max_action_steps,
            remaining_steps=self.remaining_steps,
            termination_reason=self._termination_reason,
        )
        return state

    def get_episode_status(self) -> dict[str, Any]:
        """Live internal status, retained after close for the driver and evaluator."""
        if self._closed_status is not None:
            return dict(self._closed_status)
        return {
            "ready": self.is_connected(),
            "success": self._episode_success,
            "done": self._episode_done,
            "reward": self._latest_reward,
            "instruction": self.instruction,
            "action_steps": self.action_steps,
            "max_action_steps": self._max_action_steps,
            "termination_reason": self._termination_reason,
            "error_message": self._error_message,
        }

    def close(self) -> None:
        """Save final status before the base class clears its counters and cache."""
        if self._environment is None:
            return
        if not self._episode_done:
            self._finish("closed")
        snapshot = self.get_episode_status()
        try:
            try:
                self._close_video()
            finally:
                super().close()
        except Exception as exc:
            self._record_error(exc)
            snapshot.update(
                done=self._episode_done,
                success=self._episode_success,
                termination_reason=self._termination_reason,
                error_message=self._error_message,
            )
            raise
        finally:
            snapshot["ready"] = False
            self._closed_status = snapshot
            # Reporting failures during cleanup must not mask a simulator error.
            self._write_status_safely()

    def _record_video(self) -> None:
        """Encode native upright RGB; do not render, step, or publish PNG files."""
        if self.video_path is None:
            return
        if self._video_writer is None:
            import imageio.v2 as imageio

            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = self._environment.unwrapped.env.control_freq
            self._video_writer = imageio.get_writer(
                self.video_path, format="FFMPEG", fps=fps, codec="libx264",
                macro_block_size=None, output_params=["-movflags", "+faststart"],
            )
        self._video_writer.append_data(self.latest_obs["video.robot0_agentview_left"])

    def _close_video(self) -> None:
        writer, self._video_writer = self._video_writer, None
        if writer is not None:
            writer.close()

    def _reset_evaluation_state(self) -> None:
        self._episode_done = False
        self._episode_success = False
        self._termination_reason: str | None = None
        self._error_message: str | None = None
        self._closed_status: dict[str, Any] | None = None

    def _finish(self, reason: str) -> None:
        self._episode_done = True
        self._episode_success = self._episode_success or reason == "success"
        self._termination_reason = reason

    def _record_error(self, exc: Exception) -> None:
        # An infrastructure error after task success must not erase the goal.
        if not self._episode_success:
            self._finish("error")
        self._error_message = f"{type(exc).__name__}: {exc}"
        if self._closed_status is not None:
            self._closed_status.update(
                done=self._episode_done,
                success=self._episode_success,
                termination_reason=self._termination_reason,
                error_message=self._error_message,
            )

    def _write_evaluation_status(self) -> None:
        path = self.status_path
        if path is None:
            return
        payload = self.get_episode_status()
        # Match LIBERO's file format without changing internal status consumers.
        payload.update(
            schema_version="Emerge.robocasa_evaluation_status.v1",
            environment_steps=payload["action_steps"], settling_steps=0,
            metadata={
                "env_name": self.env_name, "split": self.split, "seed": self.seed,
                "instruction": payload.pop("instruction"),
                "error_message": payload.pop("error_message"),
            },
        )
        atomic_json(path, payload)

    def _write_status_safely(self) -> None:
        try:
            self._write_evaluation_status()
        except Exception:
            logger.exception("Could not publish RoboCasa evaluation status to %s", self.status_path)

    @staticmethod
    def _validate_budget(budget: Any) -> None:
        if (
            isinstance(budget, (bool, np.bool_))
            or not isinstance(budget, (int, np.integer))
            or budget <= 0
        ):
            raise ValueError("max_action_steps must be a positive integer")
