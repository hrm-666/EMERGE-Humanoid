"""Adapt native RoboCasa episodes and VLA execution to EMERGE's driver API."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robot.drivers.base_driver import BaseDriver, CancelCheck
from robot.robocasa.actions import RoboCasaActionController
from robot.robocasa.env import RoboCasaEnvManager
from robot.robocasa.env_eval import RoboCasaEvalEnvManager
from robot.vla.robocasa_policy_executor import RoboCasaPolicyExecutor

logger = logging.getLogger(__name__)


class RoboCasaDriver(BaseDriver):
    """Expose VLA and geometric actions without creating a simulator eagerly.

    Task generation is controlled by the robocasa configuration. load_scene()
    accepts workspace object annotations, but does not spawn them in RoboCasa.
    Evaluation is opt-in via evaluation.enabled; relative status paths are
    anchored to the same workspace used by Controller and visual readers.
    """

    def __init__(
        self,
        gui: bool = False,
        workspace: str | Path | None = None,
        robocasa: dict[str, Any] | None = None,
        cameras: dict[str, dict[str, Any]] | None = None,
        observation: dict[str, Any] | None = None,
        motion: dict[str, Any] | None = None,
        vla: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        profile_path: str | Path | None = None,
    ) -> None:
        if gui:
            raise ValueError("robocasa uses offscreen rendering; gui must be false")
        self._workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        repository = Path(__file__).resolve().parents[2]
        profile = (
            Path(profile_path).expanduser()
            if profile_path is not None else Path("robot/profiles/robocasa.md")
        )
        self._profile_path = (profile if profile.is_absolute() else repository / profile).resolve()
        native = dict(robocasa or {})
        eval_config = dict(evaluation or {})
        enabled = eval_config.pop("enabled", False)
        # Per-action diagnostics are written by this driver, not by the
        # environment manager, so keep the path out of eval_config.
        metrics = eval_config.pop("action_metrics_path", None)
        self._action_metrics_path: Path | None = None
        if metrics:
            path = Path(metrics).expanduser()
            self._action_metrics_path = path if path.is_absolute() else self._workspace / path
        common = dict(
            **native,
            camera_config=dict(cameras or {}),
            workspace=self._workspace,
            observation_config=dict(observation or {}),
        )
        if enabled:
            eval_config.setdefault("status_path", "artifacts/evaluation/status.json")
            for key in ("status_path", "video_path"):
                if eval_config.get(key) is not None:
                    path = Path(eval_config[key]).expanduser()
                    eval_config[key] = path if path.is_absolute() else self._workspace / path
            self._environment = RoboCasaEvalEnvManager(**common, **eval_config)
        else:
            self._environment = RoboCasaEnvManager(**common)
        self._vla = RoboCasaPolicyExecutor(self._environment, config=vla)
        self._actions = RoboCasaActionController(
            self._environment, config=dict(motion or {}), vla_executor=self._vla
        )
        self._last_execution: dict[str, Any] | None = None
        self._load_error: str | None = None
        self._closed = False

    def get_profile_path(self) -> Path:
        return self._profile_path

    def load_scene(self, scene: dict[str, dict]) -> None:
        """Create the configured task; workspace annotations stay in the workspace."""
        if self._closed:
            raise RuntimeError("RoboCasa driver is closed")
        self._last_execution = None
        self._load_error = None
        try:
            self._environment.create()
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise

    def execute_action(
        self,
        action_type: str,
        params: dict,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        self._last_execution = None
        if self._closed:
            return "Failed: RoboCasa driver is closed"
        if not self._environment.is_connected():
            return "Failed: RoboCasa environment is unavailable; load the configured task first"
        try:
            result = self._actions.execute(action_type, params, cancel_check=cancel_check)
        except Exception as exc:
            self._record_action_metrics(
                action_type, params, error=f"{type(exc).__name__}: {exc}"
            )
            return f"Failed: {type(exc).__name__}: {exc}"
        self._last_execution = {"action_type": action_type, **asdict(result)}
        self._record_action_metrics(action_type, params)
        label = "VLA execution" if action_type == "vla_execute" else action_type
        summary = (
            f"steps={result.total_steps}, task_success={str(result.task_success).lower()}, "
            f"reason={result.reason}"
        )
        if result.reason == "interrupted":
            # Controller detects cancellation from this text. Keep arbitrary
            # cancellation text in last_execution so words such as 'failed'
            # in the user's reason cannot change its queue classification.
            return f"Interrupted: RoboCasa {label}; {summary}"
        if result.completed:
            return f"{label} finished: {summary}"
        detail = f"; {result.error_message}" if result.error_message else ""
        return f"Failed: RoboCasa {label}; {summary}{detail}"

    def _record_action_metrics(
        self, action_type: str, params: dict, *, error: str | None = None
    ) -> None:
        """Append one structured action record for offline accuracy analysis.

        This is diagnostics only: it writes after the action has already
        completed and never changes the controller's result or success signal.
        """
        path = self._action_metrics_path
        if path is None:
            return
        try:
            check_success = bool(self._environment.check_success())
        except Exception:
            check_success = None
        try:
            action_steps = int(self._environment.action_steps)
        except Exception:
            action_steps = None
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_type": action_type,
            "parameters": deepcopy(params),
            "action_steps": action_steps,
            "environment_success": check_success,
        }
        if error is not None:
            record["error"] = error
        elif self._last_execution is not None:
            record.update(self._last_execution)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.warning("failed to append action metrics to %s: %s", path, exc)

    def get_runtime_state(self) -> dict[str, Any]:
        """Publish native frame names under the EMERGE robots.robocasa key.

        An unavailable simulator has no live pose. Evaluation summaries remain
        available after errors or close; do not recycle a stale pose as current.
        """
        environment = self._environment
        if environment.is_connected():
            state = environment.get_robot_state()
        else:
            state = {
                "type": "panda_omron",
                "task": {
                    "env_name": environment.env_name,
                    "split": environment.split,
                    "instruction": environment.instruction,
                },
                "action_steps": environment.action_steps,
                "success": environment.check_success(),
                "done": self._closed or self._load_error is not None,
            }
        state["ready"] = environment.is_connected()
        if isinstance(environment, RoboCasaEvalEnvManager):
            status = environment.get_episode_status()
            state["evaluation"] = status
            state.update(done=self._closed or status["done"], success=status["success"])
            state["task"]["instruction"] = status["instruction"]
        if self._load_error is not None:
            state["setup_error"] = self._load_error
        state["last_execution"] = deepcopy(self._last_execution)
        manifest = environment.observation_manifest_path
        state["observation_manifest_path"] = str(manifest) if manifest is not None else None
        return {"robots": {"robocasa": state}}

    def close(self) -> None:
        """Attempt both cleanups, preserving the first error if both fail."""
        if self._closed:
            return
        self._closed = True
        try:
            self._vla.close()
        finally:
            self._environment.close()
