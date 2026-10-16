"""Non-blocking reset requests to the LightNav-owned simulation wrapper."""

from __future__ import annotations

import time
from uuid import uuid4

import zmq

DEFAULT_START_DISTANCE_M = 3.5
DEFAULT_RESET_ENDPOINT = "tcp://127.0.0.1:5562"


class TabletopResetClient:
    """Keep the control loop sending IDLE while the simulator resets."""

    def __init__(self, endpoint: str, timeout_s: float = 8.0) -> None:
        if not endpoint.startswith("tcp://127.0.0.1:"):
            raise ValueError("simulation reset requires a loopback endpoint")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.context = zmq.Context()
        self.socket = None
        self.request_id = None
        self.deadline = 0.0

    @property
    def pending(self) -> bool:
        return self.socket is not None

    def request(self, **start_options) -> str:
        if self.pending:
            raise RuntimeError("simulation reset is already pending")
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.endpoint)
        self.request_id = uuid4().hex
        self.deadline = time.monotonic() + self.timeout_s
        self.socket.send_json({
            "action": "reset",
            "request_id": self.request_id,
            "expires_at": time.time() + 2.0,
            **start_options
        })
        return self.request_id

    def poll(self) -> dict | None:
        if not self.pending:
            return None
        try:
            reply = self.socket.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            if time.monotonic() < self.deadline:
                return None
            reply = {
                "ok": False,
                "error": "simulation reset timed out; check terminal 1"
            }
        except ValueError:
            reply = {"ok": False, "error": "invalid simulation reset reply"}
        if not isinstance(reply, dict) or (
                reply.get("ok") and reply.get("request_id") != self.request_id):
            reply = {"ok": False, "error": "mismatched simulation reset reply"}
        self.socket.close()
        self.socket = None
        return reply

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
        self.socket = None
        self.context.term()
