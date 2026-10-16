"""Exercise the console subprocess over real ZMQ and WebSocket transports."""

import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import msgpack
import zmq
from websockets.sync.server import serve

from integrations.lightnav_sonic_emerge.simulation_checks import test_tabletop


class TabletopTransportTests(unittest.TestCase):

    def exercise(self,
                 *,
                 cancel_during_inference,
                 reset_during_inference=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = zmq.Context()
            camera = context.socket(zmq.PUB)
            debug = context.socket(zmq.PUB)
            planner = context.socket(zmq.SUB)
            pose = context.socket(zmq.PUB)
            reset = context.socket(zmq.REP)
            for socket in (camera, debug, planner, pose, reset):
                socket.setsockopt(zmq.LINGER, 0)
            camera_port = camera.bind_to_random_port("tcp://127.0.0.1")
            debug_port = debug.bind_to_random_port("tcp://127.0.0.1")
            pose_port = pose.bind_to_random_port("tcp://127.0.0.1")
            reset_port = reset.bind_to_random_port("tcp://127.0.0.1")
            reset_requests = []
            planner.setsockopt(zmq.SUBSCRIBE, b"")
            # Reserve a free local port for the client-owned PUB socket.
            port_probe = context.socket(zmq.PUB)
            port = port_probe.bind_to_random_port("tcp://127.0.0.1")
            port_probe.close()
            planner.connect(f"tcp://127.0.0.1:{port}")
            request_received = threading.Event()
            allow_reply = threading.Event()
            packets = []

            def handler(socket):
                for raw in socket:
                    payload = json.loads(raw)
                    action = payload["action"]
                    data = {"rc": 0}
                    if action == "next":
                        seq = payload["data"]["seq"]
                        request_received.set()
                        if cancel_during_inference or (
                                reset_during_inference and
                                not allow_reply.is_set()):
                            allow_reply.wait(5)
                        data.update(seq=seq,
                                    stop=seq > 0,
                                    actions={"actions": [[0.5, 0, 0]]},
                                    latency_ms=20)
                    socket.send(json.dumps({"action": action, "data": data}))

            server = serve(handler, "127.0.0.1", 0)
            server_port = server.socket.getsockname()[1]
            server_thread = threading.Thread(target=server.serve_forever,
                                             daemon=True)
            server_thread.start()
            output = (root / "console.log").open("w")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    # The fake transports live on the test host's loopback;
                    # only this test replaces the namespace-entry boundary.
                    "from unittest.mock import patch\n"
                    "from integrations.lightnav_sonic_emerge.bridge.tabletop import main\n"
                    "with (patch('integrations.lightnav_sonic_emerge.bridge.tabletop.require_simulation_namespace'),\n"
                    "     patch('integrations.lightnav_sonic_emerge.bridge.tabletop_recording.TaskArtifactWorker._process', "
                    "side_effect=lambda directory: (directory / 'artifacts.json').write_text("
                    "'{\"status\":\"complete\"}'))):\n"
                    "    raise SystemExit(main())\n",
                    "--enable-motion",
                    "--simulation-loopback",
                    "--pose-endpoint",
                    f"tcp://127.0.0.1:{pose_port}",
                    "--reset-endpoint",
                    f"tcp://127.0.0.1:{reset_port}",
                    "--observation-endpoint",
                    f"tcp://127.0.0.1:{camera_port}",
                    "--sonic-feedback-endpoint",
                    f"tcp://127.0.0.1:{debug_port}",
                    "--sonic-bind",
                    f"tcp://127.0.0.1:{port}",
                    "--lightnav-server",
                    f"ws://127.0.0.1:{server_port}",
                    "--output-dir",
                    str(root / "run"),
                ],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ, "PYTHONUNBUFFERED": "1"
                },
                text=True,
            )

            def pump():
                camera.send(test_tabletop.observation_bytes())
                debug.send(b"g1_debug" +
                           msgpack.packb({"base_quat": [1, 0, 0, 0]}))
                pose.send(
                    msgpack.packb({
                        "wall_time_s": time.time(),
                        "sim_time_s": 1,
                        "position_m": [-3.371, 0, 0.793],
                        "quaternion_wxyz": [1, 0, 0, 0],
                    }))
                if reset.poll(0):
                    request = reset.recv_json()
                    reset_requests.append(request)
                    reset.send_json({
                        "ok": True,
                        "request_id": request["request_id"],
                        "wall_time_s": time.time(),
                        "support_enabled": False
                    })
                while planner.poll(0):
                    raw = planner.recv()
                    if raw.startswith(b"planner"):
                        mode = struct.unpack_from("<i", raw,
                                                  len(b"planner") + 1280)[0]
                        packets.append((time.monotonic(), mode))

            def events():
                files = list((root / "run").glob("*/events.jsonl"))
                if not files:
                    return []
                return [
                    json.loads(line)
                    for line in files[0].read_text().splitlines()
                ]

            def wait_for(predicate, timeout=5):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    pump()
                    if predicate():
                        return
                    if process.poll() is not None:
                        self.fail((root / "console.log").read_text())
                    time.sleep(0.02)
                self.fail("Timed out:\n" + (root / "console.log").read_text())

            try:
                wait_for(lambda: any(e["event"] == "ready" for e in events()))
                process.stdin.write(
                    "nav Go straight to the table with the apple and stop.\n")
                process.stdin.flush()
                wait_for(request_received.is_set)
                if reset_during_inference:
                    stopped_at = time.monotonic()
                    process.stdin.write("reset\n")
                    process.stdin.flush()
                    wait_for(
                        lambda: any(e["event"] == "simulation_reset_complete"
                                    for e in events()))
                    self.assertEqual(len(reset_requests), 1)
                    self.assertFalse(allow_reply.is_set())
                    wait_for(lambda: any(
                        t >= stopped_at and mode == 0 for t, mode in packets))
                    allow_reply.set()
                    wait_for(lambda: any(
                        e["event"] == "prediction" for e in events()))
                    self.assertTrue(
                        all(mode == 0
                            for t, mode in packets
                            if t >= stopped_at))
                    self.assertTrue(
                        any(
                            e.get("reason") == "discarded: cancelled navigation"
                            for e in events()))
                    process.stdin.write(
                        "nav Go straight to the table with the apple and stop.\n"
                    )
                    process.stdin.flush()
                    wait_for(lambda: any(
                        e.get("reason") == "LightNav stop=true"
                        for e in events()))
                    starts = [
                        e for e in events() if e["event"] == "navigation_start"
                    ]
                    self.assertEqual(len(starts), 2)
                    self.assertNotEqual(starts[0]["generation"],
                                        starts[1]["generation"])
                elif cancel_during_inference:
                    stopped_at = time.monotonic()
                    process.stdin.write("stop\n")
                    process.stdin.flush()
                    wait_for(
                        lambda: any(
                            e.get("command", {}).get("reason") ==
                            "operator stop" for e in events()),
                        timeout=0.5,
                    )
                    wait_for(
                        lambda: any(t >= stopped_at and mode == 0
                                    for t, mode in packets),
                        timeout=0.5,
                    )
                    self.assertFalse(allow_reply.is_set())
                    allow_reply.set()
                    wait_for(lambda: any(
                        e["event"] == "prediction" for e in events()))
                    self.assertTrue(
                        all(mode == 0
                            for t, mode in packets
                            if t >= stopped_at))
                    self.assertTrue(
                        any(
                            e.get("reason") == "discarded: cancelled navigation"
                            for e in events()))
                else:
                    wait_for(lambda: any(
                        e.get("reason") == "LightNav stop=true"
                        for e in events()))
                    self.assertTrue(any(mode != 0 for _, mode in packets))
                    # A flushed log event can precede PUB delivery. Wait for
                    # actual IDLE evidence instead of assuming both transports
                    # become visible in the same polling iteration.
                    wait_for(lambda: bool(packets) and packets[-1][1] == 0,
                             timeout=0.5)
                    self.assertEqual(packets[-1][1], 0)
                wait_for(lambda: any(
                    e["event"] == "navigation_end" for e in events()))
                run = next((root / "run").iterdir())
                tasks = sorted((run / "tasks").iterdir())
                self.assertEqual(len(tasks), 2 if reset_during_inference else 1)
                for task in tasks:
                    summary = json.loads((task / "summary.json").read_text())
                    task_events = [
                        json.loads(line)
                        for line in (task /
                                     "events.jsonl").read_text().splitlines()
                    ]
                    self.assertEqual(summary["status"], "ended")
                    self.assertEqual(task_events[0]["event"],
                                     "navigation_start")
                    self.assertEqual(
                        sum(e["event"] == "navigation_end"
                            for e in task_events), 1)
                    self.assertTrue(
                        all(e["task_id"] == task.name for e in task_events))
                    self.assertTrue(
                        all(
                            e.get("generation", summary["generation"]) ==
                            summary["generation"] for e in task_events))
                    predictions = [
                        e for e in task_events if e["event"] == "prediction"
                    ]
                    self.assertEqual(summary["prediction_count"],
                                     len(predictions))
                    self.assertTrue(
                        all((task / e["frame"]).exists() for e in predictions))
                    self.assertIn("TASK START",
                                  (task / "inference.txt").read_text())
                    self.assertIn("TASK END",
                                  (task / "inference.txt").read_text())
                if cancel_during_inference or reset_during_inference:
                    first_summary = json.loads(
                        (tasks[0] / "summary.json").read_text())
                    self.assertEqual(first_summary["late_prediction_count"], 1)
                    self.assertEqual(
                        first_summary["discarded_prediction_count"], 1)
                    self.assertEqual(
                        first_summary["end_reason"],
                        "simulation reset requested"
                        if reset_during_inference else "operator stop")
            finally:
                allow_reply.set()
                if process.poll() is None:
                    process.stdin.write("exit\n")
                    process.stdin.flush()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdin.close()
                output.close()
                server.shutdown()
                server_thread.join(timeout=2)
                for socket in (camera, debug, planner, pose, reset):
                    socket.close()
                context.term()
            self.assertEqual(process.returncode, 0,
                             (root / "console.log").read_text())

    def test_model_motion_then_stop_over_real_transports(self):
        self.exercise(cancel_during_inference=False)

    def test_operator_stop_works_before_delayed_model_reply(self):
        self.exercise(cancel_during_inference=True)

    def test_reset_during_inference_stays_idle_and_next_trial_runs(self):
        self.exercise(cancel_during_inference=False,
                      reset_during_inference=True)


if __name__ == "__main__":
    unittest.main()
