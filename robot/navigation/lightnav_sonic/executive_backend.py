"""LightNav execution routed through the single Official01 executive owner."""

from __future__ import annotations

import math
import time
from typing import Any
from typing import Callable

from robot.drivers.official01_executive import ExecutiveProcessClient
from robot.navigation.lightnav_sonic import camera
from robot.navigation.lightnav_sonic.adapter import AdapterLimits
from robot.navigation.lightnav_sonic.adapter import LightNav2SonicAdapter
from robot.navigation.lightnav_sonic.feedback import SonicYawSubscriber
from robot.navigation.lightnav_sonic.lightnav_ws import LightNavWsClient
from robot.navigation.lightnav_sonic.navigation_backend import NavigationConfig
from robot.navigation.lightnav_sonic.navigation_backend import NavigationResult
from robot.navigation.lightnav_sonic.official01_observation import (
    Official01YawSubscriber,
)

CancelCheck = Callable[[], str | None]
StateCallback = Callable[[dict[str, Any]], None]


class ExecutiveNavigationRunner:
    """Run LightNav while the executive remains the only motion publisher."""

    def __init__(
        self,
        config: NavigationConfig,
        executive: ExecutiveProcessClient,
        observation_writer: Any,
        *,
        camera_factory: Callable[[str, str], Any] | None = None,
        yaw_factory: Callable[[str, str], Any] | None = None,
        lightnav_factory: Callable[[str, str], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.executive = executive
        self.observation_writer = observation_writer
        self._camera_factory = camera_factory or (
            lambda endpoint, name: camera.create_camera(
                endpoint, name, self.config.camera_transport))
        if yaw_factory is not None:
            self._yaw_factory = yaw_factory
        elif self.config.camera_transport == "official01_rgb":
            # Read camera and root yaw from the same simulation IPC stream.
            # This also supports the original isolated-network publisher.
            self._yaw_factory = (
                lambda _endpoint, topic: Official01YawSubscriber(
                    self.config.camera_endpoint, topic))
        else:
            self._yaw_factory = SonicYawSubscriber
        self._lightnav_factory = lightnav_factory or (
            lambda endpoint, client_id: LightNavWsClient(
                endpoint,
                client_id=client_id,
                request_timeout_s=self.config.inference_timeout_s,
            ))
        self._monotonic = monotonic
        self._sleep = sleep

    @staticmethod
    def _command_key(command: Any, yaw: float) -> str | None:
        """Quantize a validated SONIC command to the executive's safe key set."""
        if command.is_idle:
            return None
        target_yaw = math.atan2(command.facing_direction[1],
                                command.facing_direction[0])
        yaw_error = math.atan2(math.sin(target_yaw - yaw),
                               math.cos(target_yaw - yaw))
        if abs(yaw_error) >= 0.04:
            return "q" if yaw_error > 0 else "e"
        wx, wy, _ = command.movement_direction
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        forward = cos_yaw * wx + sin_yaw * wy
        lateral = -sin_yaw * wx + cos_yaw * wy
        if max(abs(forward), abs(lateral)) < 1e-6:
            return None
        if abs(forward) >= abs(lateral):
            return "w" if forward >= 0 else "s"
        return "a" if lateral >= 0 else "d"

    def run(
        self,
        *,
        instruction: str,
        max_steps: int,
        cancel_check: CancelCheck | None = None,
        state_callback: StateCallback | None = None,
    ) -> NavigationResult:
        if not instruction.strip():
            raise ValueError("navigation instruction must be non-empty")
        if max_steps < 1:
            raise ValueError("max_steps must be at least one")

        started = self._monotonic()
        deadline = started + self.config.max_runtime_s
        period_s = 1.0 / self.config.control_hz
        adapter = LightNav2SonicAdapter(
            AdapterLimits(control_period_s=period_s))
        camera_source = yaw_source = lightnav = None
        ticks = command_steps = stale_ticks = 0
        last_waypoint: tuple[float, float, float] | None = None
        yaw: float | None = None
        lightnav_stop = False
        moving = False
        cleanup_error: str | None = None

        def finish(status: str,
                   reason: str,
                   error: str | None = None) -> NavigationResult:
            result = NavigationResult(
                status=status,
                reason=reason,
                ticks=ticks,
                command_steps=command_steps,
                elapsed_s=max(0.0,
                              self._monotonic() - started),
                last_waypoint=last_waypoint,
                last_yaw_rad=yaw,
                lightnav_stop=lightnav_stop,
                error=error or cleanup_error,
            )
            if state_callback is not None:
                state_callback({"status": status, **result.to_dict()})
            return result

        result: NavigationResult | None = None
        try:
            camera_source = self._camera_factory(self.config.camera_endpoint,
                                                 self.config.camera_name)
            yaw_source = self._yaw_factory(
                self.config.sonic_feedback_endpoint,
                self.config.sonic_feedback_topic,
            )
            lightnav = self._lightnav_factory(self.config.lightnav_server,
                                              self.config.client_id)
            lightnav.connect()

            # Establish fresh inputs before taking the executive out of READY.
            # This avoids entering MOVING when the model is reachable but the
            # camera/yaw publishers are absent or still joining their PUB/SUB
            # transports.
            preflight_jpeg = None
            preflight_yaw = None
            for attempt in range(1, self.config.max_stale_ticks + 1):
                cancellation = cancel_check(
                ) if cancel_check is not None else None
                if cancellation:
                    result = finish("cancelled", str(cancellation))
                    break
                if self._monotonic() >= deadline:
                    result = finish("budget_exhausted", "max_runtime_s")
                    break
                preflight_jpeg = camera_source.latest(
                    self.config.input_timeout_s)
                preflight_yaw = yaw_source.latest(self.config.input_timeout_s)
                if preflight_jpeg is not None and preflight_yaw is not None:
                    break
                missing = ("camera"
                           if preflight_jpeg is None else "robot yaw feedback")
                if state_callback is not None:
                    state_callback({
                        "status": "running",
                        "instruction": instruction,
                        "ticks": 0,
                        "command_steps": 0,
                        "max_steps": max_steps,
                        "last_yaw_rad": preflight_yaw,
                        "last_waypoint": None,
                        "last_error": f"waiting for fresh {missing}",
                    })
                if attempt < self.config.max_stale_ticks:
                    self._sleep(min(period_s, self.config.input_timeout_s))
            else:
                missing = ("camera"
                           if preflight_jpeg is None else "robot yaw feedback")
                result = finish("failed", "stale_input",
                                f"stale or missing {missing}")
            if result is not None:
                return result

            self.executive.begin_moving(timeout_s=10.0)
            moving = True
            while ticks < max_steps:
                tick_started = self._monotonic()
                cancellation = cancel_check(
                ) if cancel_check is not None else None
                if cancellation:
                    result = finish("cancelled", str(cancellation))
                    break
                if self._monotonic() >= deadline:
                    result = finish("budget_exhausted", "max_runtime_s")
                    break
                ticks += 1
                jpeg = camera_source.latest(self.config.input_timeout_s)
                yaw = yaw_source.latest(self.config.input_timeout_s)
                if jpeg is None or yaw is None:
                    if moving:
                        self.executive.stop_moving(timeout_s=15.0)
                        moving = False
                    if jpeg is None:
                        self.observation_writer.invalidate()
                    stale_ticks += 1
                    missing = "camera" if jpeg is None else "robot yaw feedback"
                    if state_callback is not None:
                        state_callback({
                            "status":
                                "running",
                            "instruction":
                                instruction,
                            "ticks":
                                ticks,
                            "command_steps":
                                command_steps,
                            "max_steps":
                                max_steps,
                            "last_yaw_rad":
                                yaw,
                            "last_waypoint":
                                list(last_waypoint) if last_waypoint else None,
                            "last_error":
                                f"stale or missing {missing}",
                        })
                    if stale_ticks >= self.config.max_stale_ticks:
                        result = finish("failed", "stale_input",
                                        f"stale or missing {missing}")
                        break
                else:
                    stale_ticks = 0
                    self.observation_writer.write_jpeg(jpeg)
                    reply = lightnav.predict(jpeg, instruction)
                    cancellation = cancel_check(
                    ) if cancel_check is not None else None
                    if cancellation:
                        result = finish("cancelled", str(cancellation))
                        break
                    if self._monotonic() >= deadline:
                        result = finish("budget_exhausted", "max_runtime_s")
                        break
                    last_waypoint = tuple(reply.first_waypoint)
                    lightnav_stop = bool(reply.stop)
                    if lightnav_stop:
                        result = finish("completed", "lightnav_stop")
                        break
                    command = adapter.convert_or_idle(last_waypoint, yaw)
                    key = self._command_key(command, yaw)
                    if key is not None:
                        if not moving:
                            self.executive.begin_moving(timeout_s=10.0)
                            moving = True
                        self.executive.send_nav_key(key)
                        command_steps += 1
                    elif moving:
                        self.executive.stop_moving(timeout_s=15.0)
                        moving = False
                    if state_callback is not None:
                        state_callback({
                            "status": "running",
                            "instruction": instruction,
                            "ticks": ticks,
                            "command_steps": command_steps,
                            "max_steps": max_steps,
                            "last_yaw_rad": yaw,
                            "last_waypoint": list(last_waypoint),
                            "last_error": None,
                        })
                remaining = period_s - (self._monotonic() - tick_started)
                if remaining > 0:
                    self._sleep(remaining)
            if result is None:
                result = finish("budget_exhausted", "max_steps")
        except Exception as exc:
            result = finish("failed", "exception",
                            f"{type(exc).__name__}: {exc}")
        finally:
            if moving:
                try:
                    self.executive.stop_moving(timeout_s=15.0)
                except Exception as exc:
                    cleanup_error = f"failed to stop and stabilize: {type(exc).__name__}: {exc}"
            for resource in (lightnav, camera_source, yaw_source):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        pass
        if cleanup_error:
            return finish("failed", "stop_or_stability_failed", cleanup_error)
        assert result is not None
        return result
