"""Episode budgets and evaluation status on top of the RoboCasa Gym adapter."""

from __future__ import annotations

import importlib
import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from robot.robocasa.env import RoboCasaEnvManager, StepResult

logger = logging.getLogger(__name__)


class RoboCasaEvalEnvManager(RoboCasaEnvManager):
    """Enforce one episode's budget without changing native Gym return values.

    Check ``episode_done`` in addition to the returned terminated/truncated
    flags: success and evaluation budget exhaustion need not terminate the
    underlying simulator. Completed episodes reject further actions until reset.

    The default horizon is resolved lazily from RoboCasa's task registry during
    reset. An explicit override is recorded as such in the evaluation status.
    No extra simulator resets, settling actions or automatic reseeding are added.
    """

    def __init__(
        self,
        *,
        env_name: str,
        split: str | None = "target",
        seed: int | None = None,
        env_kwargs: dict[str, Any] | None = None,
        mujoco_gl: str = "egl",
        max_action_steps: int | None = None,
        status_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            env_name=env_name,
            split=split,
            seed=seed,
            env_kwargs=env_kwargs,
            mujoco_gl=mujoco_gl,
        )
        if max_action_steps is not None:
            self._validate_budget(max_action_steps)
        self._budget_override = (
            None if max_action_steps is None else int(max_action_steps)
        )
        self._max_action_steps = self._budget_override
        self.status_path = (
            Path(status_path).expanduser().resolve() if status_path is not None else None
        )
        self._metadata = deepcopy(metadata or {})
        # Fail before allocating a simulator if status cannot represent metadata.
        json.dumps(self._metadata, allow_nan=False)
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
        self._reset_seed = None if seed is None else int(seed)
        # Invalidate the preceding episode even if horizon lookup or reset fails.
        self._clear_observation()
        self._action_steps = 0
        try:
            self._write_evaluation_status()
            if self._budget_override is None:
                registry = importlib.import_module("robocasa.utils.dataset_registry_utils")
                budget = registry.get_task_horizon(self.env_name)
                self._validate_budget(budget)
                self._max_action_steps = int(budget)
            result = super().reset(seed=seed)
            if super().check_success():
                self._finish("success")
            self._write_evaluation_status()
            return result
        except Exception as exc:
            self._record_error(exc)
            self._write_status_safely()
            raise

    def step(self, action12: Any) -> StepResult:
        """Count only base step calls and stop before any action beyond the limit."""
        if self._episode_done:
            raise RuntimeError(
                f"RoboCasa evaluation ended ({self._termination_reason}); reset before stepping again"
            )
        self._require_ready()
        if self.remaining_steps == 0:
            self._finish("max_action_steps")
            self._write_evaluation_status()
            raise RuntimeError("RoboCasa evaluation budget exhausted; reset before stepping again")
        try:
            result = super().step(action12)
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
            self._record_error(exc)
            self._write_status_safely()
            raise

    def check_success(self) -> bool:
        """Latched episode success, also available after close()."""
        return self._episode_success

    def get_robot_state(self) -> dict[str, Any]:
        """Add evaluation completion and budget to the live robot summary.

        done follows episode_done, success is latched, and terminated/truncated
        remain native Gym flags. After close(), use get_episode_status() for
        the retained final snapshot; live robot observations are unavailable.
        """
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
        """Return a detached status snapshot, including the last closed episode.

        ``seed`` is the constructor seed; ``reset_seed`` records an explicit
        reset seed, or None when continuing the environment's RNG stream.
        """
        if self._closed_status is not None:
            return deepcopy(self._closed_status)
        return {
            "schema_version": "Emerge.robocasa_evaluation_status.v1",
            "ready": self.is_connected(),
            "env_name": self.env_name,
            "split": self.split,
            "seed": self.seed,
            "reset_seed": self._reset_seed,
            "instruction": self.instruction,
            "success": self._episode_success,
            "done": self._episode_done,
            "terminated": self._latest_terminated,
            "truncated": self._latest_truncated,
            "reward": self._latest_reward,
            "action_steps": self.action_steps,
            "max_action_steps": self._max_action_steps,
            "remaining_steps": self.remaining_steps,
            "budget_source": "task_registry" if self._budget_override is None else "override",
            "termination_reason": self._termination_reason,
            "error_message": self._error_message,
            "metadata": deepcopy(self._metadata),
        }

    def close(self) -> None:
        """Save final status before the base class clears its counters and cache."""
        if self._environment is None:
            return
        if not self._episode_done:
            self._finish("closed")
        snapshot = self.get_episode_status()
        try:
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

    def _reset_evaluation_state(self) -> None:
        self._episode_done = False
        self._episode_success = False
        self._termination_reason: str | None = None
        self._error_message: str | None = None
        self._reset_seed: int | None = None
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
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

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
