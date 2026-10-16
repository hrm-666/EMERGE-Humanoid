"""Composite EMERGE driver for LightNav, GR00T, poses, and micro motion."""

from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
import time
from typing import Any

from robot.drivers.base_driver import BaseDriver, CancelCheck
from integrations.lightnav_sonic_emerge.navigation_backend import NavigationConfig
from integrations.lightnav_sonic_emerge.navigation_backend import NavigationResult
from integrations.lightnav_sonic_emerge.observation_writer import EgoObservationWriter

try:
    from .executive_client import ExecutiveClientError, ExecutiveProcessClient
    from .navigation import ExecutiveNavigationRunner
except ImportError:  # External plugin loading imports driver as a top-level module.
    from executive_client import ExecutiveClientError, ExecutiveProcessClient
    from navigation import ExecutiveNavigationRunner


class Official01G1Driver(BaseDriver):
    """Serialize all G1 capabilities through one persistent executive."""

    ROBOT_ID = "g1_official01_001"

    def __init__(
        self,
        gui: bool = False,
        workspace: str | Path = "~/.Emerge/workspace",
        control_root: str | Path = "~/visionpro_sonic_bridge",
        runtime: str = "sim",
        startup_timeout_s: float = 90.0,
        command_watchdog_s: float = 0.40,
        linear_speed_mps: float = 0.12,
        yaw_rate_rad_s: float = 0.25,
        max_micro_duration_s: float = 2.0,
        max_inference_duration_s: float = 60.0,
        max_fixed_timeout_s: float = 120.0,
        stop_timeout_s: float = 15.0,
        lightnav_server: str = "ws://127.0.0.1:8051",
        camera_endpoint: str = "tcp://127.0.0.1:5560",
        camera_name: str = "ego_view",
        camera_transport: str = "official01_rgb",
        sonic_feedback_endpoint: str = "tcp://127.0.0.1:5557",
        sonic_feedback_topic: str = "g1_debug",
        client_id: str = "emerge-official01-g1",
        control_hz: float = 4.0,
        input_timeout_s: float = 0.75,
        max_stale_ticks: int = 8,
        inference_timeout_s: float = 10.0,
        max_action_runtime_s: float = 120.0,
        default_max_steps: int = 120,
        max_action_steps: int = 120,
        publish_observations: bool = True,
        lock_file: str | Path = "/tmp/emerge_official01_g1.lock",
        profile_path: str | Path | None = None,
        **_kwargs: Any,
    ) -> None:
        del gui
        self.workspace = Path(workspace).expanduser().resolve()
        self.control_root = Path(control_root).expanduser().resolve()
        self.runtime = runtime
        own_dir = Path(__file__).resolve().parent
        selected_profile = Path(profile_path).expanduser() if profile_path else own_dir / "profile_official01_g1.md"
        if not selected_profile.is_absolute():
            selected_profile = own_dir / selected_profile
        self._profile_path = selected_profile.resolve()
        self.max_micro_duration_s = self._bounded_float(max_micro_duration_s, "max_micro_duration_s")
        self.max_inference_duration_s = self._bounded_float(
            max_inference_duration_s, "max_inference_duration_s"
        )
        self.max_fixed_timeout_s = self._bounded_float(max_fixed_timeout_s, "max_fixed_timeout_s")
        self.stop_timeout_s = self._bounded_float(stop_timeout_s, "stop_timeout_s")
        self.default_max_steps = self._positive_int(default_max_steps, "default_max_steps")
        self.max_action_steps = self._positive_int(max_action_steps, "max_action_steps")
        self._lock_path = Path(lock_file).expanduser().resolve()
        self._lock_handle = None
        self._client_settings = {
            "control_root": self.control_root,
            "runtime": runtime,
            "startup_timeout_s": self._bounded_float(startup_timeout_s, "startup_timeout_s"),
            "command_watchdog_s": self._bounded_float(command_watchdog_s, "command_watchdog_s"),
            "linear_speed_mps": self._bounded_float(linear_speed_mps, "linear_speed_mps"),
            "yaw_rate_rad_s": self._bounded_float(yaw_rate_rad_s, "yaw_rate_rad_s"),
        }
        self.linear_speed_mps = self._client_settings["linear_speed_mps"]
        self.yaw_rate_rad_s = self._client_settings["yaw_rate_rad_s"]
        self.command_watchdog_s = self._client_settings["command_watchdog_s"]
        self._navigation_base = {
            "lightnav_server": lightnav_server,
            "camera_endpoint": camera_endpoint,
            "camera_name": camera_name,
            "camera_transport": camera_transport,
            "sonic_feedback_endpoint": sonic_feedback_endpoint,
            "sonic_feedback_topic": sonic_feedback_topic,
            "sonic_bind": "executive-owned",
            "client_id": client_id,
            "control_hz": self._bounded_float(control_hz, "control_hz"),
            "input_timeout_s": self._bounded_float(input_timeout_s, "input_timeout_s"),
            "max_stale_ticks": self._positive_int(max_stale_ticks, "max_stale_ticks"),
            "inference_timeout_s": self._bounded_float(inference_timeout_s, "inference_timeout_s"),
        }
        self.max_action_runtime_s = self._bounded_float(
            max_action_runtime_s, "max_action_runtime_s"
        )
        self._observation_writer = EgoObservationWriter(
            self.workspace,
            view_name=camera_name,
            enabled=publish_observations,
            max_age_s=self._navigation_base["input_timeout_s"],
        )
        self._client: ExecutiveProcessClient | None = None
        self._last_action: dict[str, Any] = {
            "type": None, "status": "idle", "result": None, "reason": None
        }

    @staticmethod
    def _bounded_float(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite number")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return result

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        result = int(value)
        if result < 1 or float(result) != float(value):
            raise ValueError(f"{name} must be a positive integer")
        return result

    def _make_client(self) -> ExecutiveProcessClient:
        return ExecutiveProcessClient(**self._client_settings)

    def _make_navigation(
        self, config: NavigationConfig, client: ExecutiveProcessClient
    ) -> ExecutiveNavigationRunner:
        return ExecutiveNavigationRunner(config, client, self._observation_writer)

    def _acquire_control_lock(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                f"another EMERGE G1 controller owns {self._lock_path}; do not start competing motion publishers"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={__import__('os').getpid()} runtime={self.runtime}\n")
        handle.flush()
        self._lock_handle = handle

    def get_profile_path(self) -> Path:
        return self._profile_path

    def load_scene(self, scene: dict[str, dict]) -> None:
        del scene
        if not self.control_root.is_dir():
            raise FileNotFoundError(f"G1 control repository not found: {self.control_root}")
        self._acquire_control_lock()
        try:
            self._client = self._make_client()
            self._client.start()
        except Exception:
            self.close()
            raise

    def _require_client(self) -> ExecutiveProcessClient:
        if self._client is None or not self._client.running:
            raise ExecutiveClientError("Official01 executive is not ready")
        return self._client

    @staticmethod
    def _json_result(status: str, result: str, reason: str, **evidence: Any) -> str:
        payload = {"status": status, "result": result, "reason": reason, **evidence}
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if status == "failed":
            return f"Failed: {text}"
        if status == "cancelled":
            return f"Cancelled: {text}"
        return f"Completed: {text}"

    def execute_action(
        self,
        action_type: str,
        params: dict,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        if not isinstance(params, dict):
            return self._json_result("failed", "rejected", "parameters_must_be_object")
        action_type = "navigate" if action_type == "navigate_vln" else action_type
        self._last_action = {
            "type": action_type, "status": "running", "result": None, "reason": None
        }
        try:
            if action_type == "navigate":
                result = self._navigate(params, cancel_check)
            elif action_type == "infer_gr00t":
                result = self._infer(params, cancel_check)
            elif action_type == "run_fixed_action":
                result = self._run_fixed(params)
            elif action_type == "move_micro":
                result = self._move_micro(params, cancel_check)
            else:
                result = self._json_result("failed", "rejected", f"unknown_action:{action_type}")
        except (ExecutiveClientError, TypeError, ValueError, OSError) as exc:
            result = self._json_result(
                "failed", "exception", f"{type(exc).__name__}: {exc}",
                executive_state=self._client.state if self._client else "unavailable",
            )
        status = "failed" if result.startswith("Failed:") else (
            "cancelled" if result.startswith("Cancelled:") else "completed"
        )
        self._last_action.update(status=status, result=result)
        return result

    def _navigate(self, params: dict[str, Any], cancel_check: CancelCheck | None) -> str:
        instruction = params.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("navigate requires a non-empty instruction")
        max_steps = self._positive_int(
            params.get("max_steps", self.default_max_steps), "max_steps"
        )
        if max_steps > self.max_action_steps:
            raise ValueError(f"max_steps cannot exceed {self.max_action_steps}")
        runtime = self._bounded_float(
            params.get("max_runtime_s", min(120.0, self.max_action_runtime_s)),
            "max_runtime_s",
        )
        if runtime > self.max_action_runtime_s:
            raise ValueError(f"max_runtime_s cannot exceed {self.max_action_runtime_s}")
        control_hz = self._bounded_float(
            params.get("control_hz", self._navigation_base["control_hz"]), "control_hz"
        )
        config_values = dict(self._navigation_base)
        config_values.update(
            control_hz=control_hz,
            max_runtime_s=runtime,
            elastic_band_release_file=None,
        )
        config = NavigationConfig(**config_values)
        nav_result = self._make_navigation(config, self._require_client()).run(
            instruction=instruction.strip(),
            max_steps=max_steps,
            cancel_check=cancel_check,
            state_callback=self._receive_navigation_state,
        )
        evidence = nav_result.to_dict()
        evidence.pop("status", None)
        evidence.pop("reason", None)
        if nav_result.status == "failed":
            return self._json_result("failed", "navigation_failed", nav_result.reason, **evidence)
        if nav_result.status == "cancelled":
            return self._json_result("cancelled", "navigation_stopped", nav_result.reason, **evidence)
        return self._json_result(
            "completed", "navigation_phase_ended", nav_result.reason,
            semantic_arrival_verified=False, **evidence,
        )

    def _receive_navigation_state(self, state: dict[str, Any]) -> None:
        self._last_action.update(state)

    def _infer(self, params: dict[str, Any], cancel_check: CancelCheck | None) -> str:
        instruction = params.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("infer_gr00t requires a non-empty instruction")
        duration = self._bounded_float(params.get("duration_s", 10.0), "duration_s")
        if duration > self.max_inference_duration_s:
            raise ValueError(f"duration_s cannot exceed {self.max_inference_duration_s}")
        timeout = self._bounded_float(params.get("timeout_s", duration + 15.0), "timeout_s")
        if timeout < duration:
            raise ValueError("timeout_s must be at least duration_s")
        client = self._require_client()
        started = time.monotonic()
        deadline = started + timeout
        client.begin_inference(instruction.strip(), timeout_s=min(10.0, timeout))
        cancelled = None
        try:
            while time.monotonic() - started < duration:
                cancelled = cancel_check() if cancel_check is not None else None
                if cancelled:
                    break
                if client.state != "INFERENCING":
                    raise ExecutiveClientError(
                        f"GR00T inference left INFERENCING unexpectedly: {client.state}"
                    )
                time.sleep(max(0.0, min(0.1, duration - (time.monotonic() - started))))
        finally:
            if client.state == "INFERENCING":
                client.pause_inference(timeout_s=max(0.1, deadline - time.monotonic()))
        elapsed = time.monotonic() - started
        if cancelled:
            return self._json_result(
                "cancelled", "inference_stopped", str(cancelled),
                instruction=instruction.strip(), elapsed_s=round(elapsed, 3),
                executive_state=client.state,
            )
        return self._json_result(
            "completed", "inference_window_ended", "duration_elapsed",
            instruction=instruction.strip(), elapsed_s=round(elapsed, 3),
            semantic_task_verified=False, executive_state=client.state,
        )

    def _resource_exists(self, kind: str, name: str) -> bool:
        if kind == "prepare":
            return name == "right2" and (
                self.control_root /
                "integrations/official01_real_deployment/body_prepare/right2_to_sonic_ready_v1.yaml"
            ).is_file()
        subdir = "poses" if kind == "pose" else "trajectories"
        root = (
            self.control_root /
            f"integrations/official01_real_deployment/executive_poses/{subdir}"
        )
        suffixes = ("", ".yaml", ".yml") if kind == "pose" else ("", ".npz")
        return any((root / f"{name}{suffix}").is_file() for suffix in suffixes)

    def _run_fixed(self, params: dict[str, Any]) -> str:
        kind = params.get("kind")
        name = params.get("name")
        if kind not in {"prepare", "pose", "trajectory"}:
            raise ValueError("kind must be prepare, pose, or trajectory")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("run_fixed_action requires a resource name")
        timeout = self._bounded_float(params.get("timeout_s", 30.0), "timeout_s")
        if timeout > self.max_fixed_timeout_s:
            raise ValueError(f"timeout_s cannot exceed {self.max_fixed_timeout_s}")
        if not self._resource_exists(kind, name):
            raise ValueError(f"fixed action is not in the configured inventory: {kind}/{name}")
        client = self._require_client()
        started = time.monotonic()
        client.run_fixed(kind, name, timeout_s=timeout)
        return self._json_result(
            "completed", "fixed_action_finished", "ready_and_stable",
            kind=kind, name=name, elapsed_s=round(time.monotonic() - started, 3),
            executive_state=client.state,
        )

    def _move_micro(self, params: dict[str, Any], cancel_check: CancelCheck | None) -> str:
        direction = params.get("direction")
        keys = {
            "forward": "w", "backward": "s", "left": "a", "right": "d",
            "turn_left": "q", "turn_right": "e",
        }
        if direction not in keys:
            raise ValueError(f"direction must be one of {sorted(keys)}")
        duration = self._bounded_float(params.get("duration_s", 0.5), "duration_s")
        if duration > self.max_micro_duration_s:
            raise ValueError(f"duration_s cannot exceed {self.max_micro_duration_s}")
        timeout = self._bounded_float(params.get("timeout_s", duration + 15.0), "timeout_s")
        if timeout < duration:
            raise ValueError("timeout_s must be at least duration_s")
        client = self._require_client()
        started = time.monotonic()
        deadline = started + timeout
        cancelled = None
        client.begin_moving(timeout_s=min(10.0, timeout))
        try:
            next_send = started
            while time.monotonic() - started < duration:
                cancelled = cancel_check() if cancel_check is not None else None
                if cancelled:
                    break
                now = time.monotonic()
                if now >= next_send:
                    client.send_nav_key(keys[direction])
                    next_send = now + self.command_watchdog_s * 0.5
                time.sleep(max(0.0, min(0.05, duration - (now - started))))
        finally:
            client.stop_moving(timeout_s=max(0.1, deadline - time.monotonic()))
        elapsed = min(duration, time.monotonic() - started)
        if direction.startswith("turn_"):
            bound = {"max_angle_rad": round(self.yaw_rate_rad_s * elapsed, 4)}
        else:
            bound = {"max_distance_m": round(self.linear_speed_mps * elapsed, 4)}
        if cancelled:
            return self._json_result(
                "cancelled", "micro_motion_stopped", str(cancelled),
                direction=direction, commanded_duration_s=duration,
                executive_state=client.state, **bound,
            )
        return self._json_result(
            "completed", "micro_motion_finished", "ready_and_stable",
            direction=direction, commanded_duration_s=duration,
            executive_state=client.state, odometry_verified=False, **bound,
        )

    def get_runtime_state(self) -> dict[str, Any]:
        client_state = self._client.state if self._client is not None else "unavailable"
        return {
            "robots": {
                self.ROBOT_ID: {
                    "name": "Official01 G1 + Inspire",
                    "type": "humanoid",
                    "runtime": self.runtime,
                    "executive_state": client_state,
                    "control_owner": "official01_g1",
                    "last_action": deepcopy(self._last_action),
                    "safety": {
                        "single_owner_lock": str(self._lock_path),
                        "max_micro_duration_s": self.max_micro_duration_s,
                        "max_inference_duration_s": self.max_inference_duration_s,
                        "requires_ready_between_actions": True,
                    },
                }
            }
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._lock_handle is not None:
            try:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_handle.close()
                self._lock_handle = None
