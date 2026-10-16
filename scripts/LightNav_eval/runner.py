"""Run the LightNav → SONIC navigation bridge against SONIC Sim2Sim.

The process is intentionally a thin, standalone robot client:

* subscribes to the upstream MuJoCo image publisher (default port 5555),
* subscribes to the upstream SONIC ``g1_debug`` feedback topic for base yaw,
* calls a remote ``lightnav-serve`` WebSocket server, and
* owns the sole SONIC planner PUB socket (default port 5556).

It never imports or edits SONIC.  ``--enable-motion`` is deliberately required
before planner activation or packet transmission.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import Sequence

from robot.navigation.lightnav_sonic.adapter import AdapterLimits
from robot.navigation.lightnav_sonic.adapter import LightNav2SonicAdapter
from robot.navigation.lightnav_sonic.feedback import SonicCameraSubscriber
from robot.navigation.lightnav_sonic.feedback import SonicYawSubscriber
from robot.navigation.lightnav_sonic.lightnav_ws import LightNavWsClient
from robot.navigation.lightnav_sonic.sonic_zmq import SonicZmqPublisher


def _command_log(command: Any, *, waypoint: Sequence[float] | None,
                 robot_yaw: float | None) -> str:
    payload = {
        "waypoint": None if waypoint is None else list(waypoint),
        "robot_yaw_rad": robot_yaw,
        "sonic_command": asdict(command),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _safe_idle(publisher: SonicZmqPublisher | None,
               adapter: LightNav2SonicAdapter, yaw: float | None,
               reason: str) -> None:
    if publisher is None or not publisher.activated:
        return
    command = adapter.idle(yaw, reason=reason)
    try:
        publisher.send_idle(command)
    except Exception as exc:  # pragma: no cover - physical transport failure
        print(f"failed to send SONIC IDLE after {reason}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--lightnav-server",
        help="LightNav WebSocket endpoint, e.g. ws://gpu-host:8051")
    source.add_argument("--mock-waypoint",
                        nargs=3,
                        type=float,
                        metavar=("FORWARD_M", "LATERAL_M", "YAW_RAD"))
    parser.add_argument("--instruction",
                        default="walk forward",
                        help="language instruction for LightNav")
    parser.add_argument("--client-id", default="lightnav-sonic-g1")
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--camera-name", default="ego_view")
    parser.add_argument("--sonic-feedback-endpoint",
                        default="tcp://127.0.0.1:5557")
    parser.add_argument("--sonic-feedback-topic", default="g1_debug")
    parser.add_argument("--sonic-bind", default="tcp://*:5556")
    parser.add_argument("--control-hz", type=float, default=4.0)
    parser.add_argument("--input-timeout-s", type=float, default=0.75)
    parser.add_argument("--max-steps",
                        type=int,
                        default=0,
                        help="0 runs until LightNav stop or Ctrl-C")
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="explicitly activate SONIC planner and publish commands")
    parser.add_argument(
        "--elastic-band-release-file",
        type=Path,
        help=(
            "create this bridge-local signal after SONIC yaw feedback is live; "
            "use it with room_sim --elastic-band-release-file"),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if not math.isfinite(args.control_hz) or args.control_hz <= 0.0:
        raise ValueError("--control-hz must be positive")
    if not math.isfinite(args.input_timeout_s) or args.input_timeout_s <= 0.0:
        raise ValueError("--input-timeout-s must be positive")
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative")
    if args.mock_waypoint is not None and args.enable_motion and args.max_steps == 0:
        raise ValueError(
            "mock motion requires a finite --max-steps safety limit")

    adapter = LightNav2SonicAdapter(
        AdapterLimits(control_period_s=1.0 / args.control_hz))
    camera = SonicCameraSubscriber(args.camera_endpoint, args.camera_name)
    yaw_source = SonicYawSubscriber(args.sonic_feedback_endpoint,
                                    args.sonic_feedback_topic)
    publisher: SonicZmqPublisher | None = None
    client: LightNavWsClient | None = None
    elastic_band_release_file = args.elastic_band_release_file
    if elastic_band_release_file is not None:
        elastic_band_release_file = elastic_band_release_file.resolve()
        if elastic_band_release_file.exists():
            raise FileExistsError(
                f"elastic-band release file already exists: {elastic_band_release_file}; "
                "remove this stale signal before starting the runner")
    period_s = 1.0 / args.control_hz
    step = 0
    try:
        if args.enable_motion:
            publisher = SonicZmqPublisher(args.sonic_bind)
            print(
                f"SONIC planner publisher bound at {args.sonic_bind}; waiting for subscriber"
            )
            publisher.activate_planner()
            print(
                "SONIC planner mode activated; normal stop sends locomotion IDLE only"
            )
        if args.lightnav_server:
            client = LightNavWsClient(args.lightnav_server,
                                      client_id=args.client_id)
            client.connect()
            print(f"connected to LightNav server {args.lightnav_server}")

        while args.max_steps == 0 or step < args.max_steps:
            tick_started = time.monotonic()
            jpeg = camera.latest(args.input_timeout_s)
            yaw = yaw_source.latest(args.input_timeout_s)
            if jpeg is None or yaw is None:
                missing = "camera" if jpeg is None else "robot yaw feedback"
                _safe_idle(publisher, adapter, yaw,
                           f"stale or missing {missing}")
                print(f"idle: stale or missing {missing}")
            else:
                try:
                    if elastic_band_release_file is not None:
                        elastic_band_release_file.touch(exist_ok=False)
                        print(
                            f"released room ElasticBand via {elastic_band_release_file}"
                        )
                        elastic_band_release_file = None
                    if args.mock_waypoint is not None:
                        waypoint = tuple(args.mock_waypoint)
                        stop = False
                    else:
                        assert client is not None
                        reply = client.predict(jpeg, args.instruction)
                        waypoint = reply.first_waypoint
                        stop = reply.stop
                    if stop:
                        command = adapter.idle(yaw, reason="LightNav stop=true")
                        _safe_idle(publisher, adapter, yaw, command.reason)
                        print(
                            _command_log(command,
                                         waypoint=waypoint,
                                         robot_yaw=yaw))
                        return 0
                    command = adapter.convert_or_idle(waypoint, yaw)
                    if publisher is not None and publisher.activated:
                        publisher.send_planner(command)
                    print(
                        _command_log(command, waypoint=waypoint, robot_yaw=yaw))
                    step += 1
                # Any recoverable bridge failure must degrade to a planner-level
                # IDLE packet. KeyboardInterrupt is deliberately not caught, so
                # normal shutdown still reaches the finally block below.
                except Exception as exc:
                    _safe_idle(publisher, adapter, yaw,
                               f"bridge failure: {exc}")
                    print(f"idle: bridge failure: {exc}")

            remaining = period_s - (time.monotonic() - tick_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("interrupted; sending SONIC locomotion IDLE")
        return 130
    finally:
        _safe_idle(publisher, adapter, yaw_source.latest(args.input_timeout_s),
                   "runner shutdown")
        if client is not None:
            client.close()
        if publisher is not None:
            publisher.close()
        camera.close()
        yaw_source.close()
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
