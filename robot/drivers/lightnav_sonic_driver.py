"""EMERGE BaseDriver for bounded LightNav-0 + SONIC navigation actions."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from robot.drivers.base_driver import BaseDriver
from robot.drivers.base_driver import CancelCheck
from robot.vla.lightnav_sonic import camera
from robot.vla.lightnav_sonic.navigation_backend import NavigationBackend
from robot.vla.lightnav_sonic.navigation_backend import NavigationConfig
from robot.vla.lightnav_sonic.navigation_backend import NavigationResult
from robot.vla.lightnav_sonic.observation_writer import EgoObservationWriter


class SonicLightNavDriver(BaseDriver):
    """Expose LightNav-guided SONIC locomotion as an EMERGE action."""

    ROBOT_ID = "g1_sonic_001"

    def __init__(
        self,
        gui: bool = False,
        workspace: str | Path = "~/.Emerge/workspace",
        lightnav_bridge_root: str | Path | None = None,
        lightnav_server: str = "ws://127.0.0.1:8051",
        camera_endpoint: str = "tcp://127.0.0.1:5555",
        camera_name: str = "ego_view",
        camera_transport: str = "sonic_jpeg",
        sonic_feedback_endpoint: str = "tcp://127.0.0.1:5557",
        sonic_feedback_topic: str = "g1_debug",
        sonic_bind: str = "tcp://*:5556",
        client_id: str = "emerge-lightnav-sonic-g1",
        control_hz: float = 4.0,
        input_timeout_s: float = 0.75,
        max_stale_ticks: int = 8,
        inference_timeout_s: float = 10.0,
        max_action_runtime_s: float = 120.0,
        default_max_steps: int = 120,
        max_action_steps: int = 120,
        elastic_band_release_file: str | Path |
        None = "/tmp/lightnav_sonic_release",
        publish_observations: bool = True,
        profile_path: str | Path | None = None,
        **_kwargs: Any,
    ) -> None:
        del gui  # The separately launched SONIC MuJoCo process owns its GUI.
        del lightnav_bridge_root  # Deprecated: the bridge now ships with this plugin.
        self.workspace = Path(workspace).expanduser().resolve()
        if camera_transport not in ("sonic_jpeg", "official01_rgb"):
            raise ValueError("unsupported camera_transport")
        if camera_transport == "official01_rgb" and camera_name != "ego_view":
            raise ValueError("official01_rgb only publishes ego_view")
        self._camera: camera.CameraSubscriber | None = None
        self._observation_state: dict[str, Any] = {"status": "unavailable"}

        own_dir = Path(__file__).resolve().parents[1] / "profiles"
        configured_profile = Path(profile_path).expanduser(
        ) if profile_path else (own_dir / "lightnav_sonic_g1.md")
        if not configured_profile.is_absolute():
            configured_profile = own_dir / configured_profile
        self._profile_path = configured_profile.resolve()
        self._base_config = {
            "lightnav_server":
                lightnav_server,
            "camera_endpoint":
                camera_endpoint,
            "camera_name":
                camera_name,
            "camera_transport":
                camera_transport,
            "sonic_feedback_endpoint":
                sonic_feedback_endpoint,
            "sonic_feedback_topic":
                sonic_feedback_topic,
            "sonic_bind":
                sonic_bind,
            "client_id":
                client_id,
            "control_hz":
                self._positive_float(control_hz, "control_hz"),
            "input_timeout_s":
                self._positive_float(input_timeout_s, "input_timeout_s"),
            "max_stale_ticks":
                self._positive_int(max_stale_ticks, "max_stale_ticks"),
            "inference_timeout_s":
                self._positive_float(inference_timeout_s,
                                     "inference_timeout_s"),
            "elastic_band_release_file":
                (Path(elastic_band_release_file).expanduser().resolve()
                 if elastic_band_release_file else None),
        }
        self.max_action_runtime_s = self._positive_float(
            max_action_runtime_s, "max_action_runtime_s")
        self.default_max_steps = self._positive_int(default_max_steps,
                                                    "default_max_steps")
        self.max_action_steps = self._positive_int(max_action_steps,
                                                   "max_action_steps")
        self._observation_writer = EgoObservationWriter(
            self.workspace,
            view_name=camera_name,
            enabled=publish_observations,
            max_age_s=self._base_config["input_timeout_s"],
        )
        self._nav_state: dict[str, Any] = {
            "mode": "idle",
            "status": "idle",
            "instruction": None,
            "ticks": 0,
            "command_steps": 0,
            "last_waypoint": None,
            "last_error": None,
        }
        self._connection_state: dict[str, Any] = {
            "status": "configured",
            "lightnav_server": lightnav_server,
            "camera_endpoint": camera_endpoint,
            "sonic_feedback_endpoint": sonic_feedback_endpoint,
            "sonic_bind": sonic_bind,
        }

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive number")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be a positive number")
        return result

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        result = int(value)
        if result < 1 or float(result) != float(value):
            raise ValueError(f"{name} must be a positive integer")
        return result

    def get_profile_path(self) -> Path:
        return self._profile_path

    def load_scene(self, scene: dict[str, dict]) -> None:
        del scene
        self._connection_state["status"] = "ready"
        self._refresh_observation()

    def execute_action(
        self,
        action_type: str,
        params: dict,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        if action_type != "navigate_vln":
            return f"Unknown action type: {action_type!r}"
        if not isinstance(params, dict):
            return "Failed: navigate_vln parameters must be an object"

        instruction = params.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            return "Failed: navigate_vln requires a non-empty instruction"
        try:
            max_steps = self._positive_int(
                params.get("max_steps", self.default_max_steps), "max_steps")
            if max_steps > self.max_action_steps:
                raise ValueError(
                    f"max_steps cannot exceed configured limit {self.max_action_steps}"
                )
            max_runtime_s = self._positive_float(
                params.get("max_runtime_s", min(120.0,
                                                self.max_action_runtime_s)),
                "max_runtime_s")
            if max_runtime_s > self.max_action_runtime_s:
                raise ValueError(
                    f"max_runtime_s cannot exceed configured limit {self.max_action_runtime_s}"
                )
            control_hz = self._positive_float(
                params.get("control_hz", self._base_config["control_hz"]),
                "control_hz",
            )
        except (TypeError, ValueError) as exc:
            return f"Failed: invalid navigate_vln parameters: {exc}"

        config_values = dict(self._base_config)
        config_values["control_hz"] = control_hz
        config_values["max_runtime_s"] = max_runtime_s
        config = NavigationConfig(**config_values)
        self._connection_state["status"] = "running"
        self._nav_state.update(
            mode="navigating",
            status="starting",
            instruction=instruction.strip(),
            ticks=0,
            command_steps=0,
            max_steps=max_steps,
            last_error=None,
        )
        backend = self._make_backend(config)
        result = backend.run(
            instruction=instruction.strip(),
            max_steps=max_steps,
            cancel_check=cancel_check,
            state_callback=self._receive_backend_state,
        )
        self._connection_state["status"] = ("ready" if result.status != "failed"
                                            else "degraded")
        self._nav_state.update(
            mode="idle",
            status=result.status,
            reason=result.reason,
            ticks=result.ticks,
            command_steps=result.command_steps,
            last_waypoint=(list(result.last_waypoint)
                           if result.last_waypoint is not None else None),
            last_yaw_rad=result.last_yaw_rad,
            last_error=result.error,
        )
        return self._format_result(result)

    def _make_backend(self, config: NavigationConfig) -> NavigationBackend:
        return NavigationBackend(config, self._observation_writer)

    def _receive_backend_state(self, state: dict[str, Any]) -> None:
        self._nav_state.update(state)
        self._nav_state["mode"] = ("navigating" if state.get("status")
                                   == "running" else "idle")

    @staticmethod
    def _format_result(result: NavigationResult) -> str:
        payload = json.dumps(result.to_dict(),
                             ensure_ascii=False,
                             sort_keys=True)
        if result.status == "failed":
            return f"Failed: navigation backend failed. {payload}"
        if result.status == "cancelled":
            return f"Cancelled: navigation action stopped. {payload}"
        if result.status == "budget_exhausted":
            budget = "runtime budget" if result.reason == "max_runtime_s" else "step budget"
            return (
                f"Navigation action completed its {budget}; semantic goal is not "
                f"automatically verified. {payload}")
        return (
            "Navigation action completed; semantic goal still requires observation "
            f"verification. {payload}")

    def _make_camera(self) -> camera.CameraSubscriber:
        return camera.create_camera(
            self._base_config["camera_endpoint"],
            self._base_config["camera_name"],
            self._base_config["camera_transport"],
        )

    def _refresh_observation(self) -> None:
        """Publish passive evidence on the Controller thread, including idle polls."""
        if not self._observation_writer.enabled:
            self._observation_state = {"status": "disabled"}
            return
        if self._connection_state["status"] == "closed":
            return
        try:
            if self._camera is None:
                self._camera = self._make_camera()
            jpeg = self._camera.latest(self._base_config["input_timeout_s"])
            if jpeg is None:
                self._observation_writer.invalidate()
                self._observation_state = {"status": "unavailable"}
                return
            self._observation_writer.write_jpeg(jpeg)
            self._observation_state = {
                "status": "available",
                "revision": self._observation_writer.revision,
            }
        except Exception as exc:
            # Camera failures must not prevent Controller cancellation/results.
            self._observation_writer.invalidate()
            self._observation_state = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def get_runtime_state(self) -> dict[str, Any]:
        self._refresh_observation()
        yaw = self._nav_state.get("last_yaw_rad")
        robot: dict[str, Any] = {
            "connection_state": deepcopy(self._connection_state),
            "nav_state": deepcopy(self._nav_state),
            "observation_state": deepcopy(self._observation_state),
        }
        if isinstance(yaw, (int, float)) and math.isfinite(float(yaw)):
            robot["robot_pose"] = {
                "frame": "sonic_world",
                "yaw": float(yaw),
            }
        return {"robots": {self.ROBOT_ID: robot}}

    def close(self) -> None:
        self._nav_state.update(mode="idle", status="closed")
        self._connection_state["status"] = "closed"
        self._observation_state = {"status": "unavailable"}
        try:
            if self._camera is not None:
                self._camera.close()
        finally:
            self._camera = None
            self._observation_writer.invalidate()
