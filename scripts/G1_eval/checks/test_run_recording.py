from __future__ import annotations

import errno
import tempfile
import time
import unittest
from pathlib import Path

import msgpack
import numpy as np

from Emerge.config.schema import Config
from Emerge.runtime.protocol import RunEvent
from Emerge.runtime.protocol import RunRequest
from Emerge.runtime.recording import RunVideoRecorder
from Emerge.runtime.recording import _FFmpegVideoWriter
from Emerge.runtime.recording import decode_recording_frames
from Emerge.runtime.storage import RunArtifacts


def _envelope(array: np.ndarray) -> dict[bytes, object]:
    return {
        b"nd": True,
        b"type": array.dtype.str.encode(),
        b"shape": array.shape,
        b"data": array.tobytes(),
    }


class RunRecordingTests(unittest.TestCase):

    def test_decodes_both_recording_views(self):
        ego = np.full((1, 1, 480, 640, 3), 17, dtype=np.uint8)
        third = np.full((480, 640, 3), 29, dtype=np.uint8)
        raw = msgpack.packb({
            "observation": {
                "video": {
                    "ego_view": _envelope(ego)
                }
            },
            "third_person_rgb": _envelope(third),
        })

        decoded_ego, decoded_third = decode_recording_frames(raw)

        self.assertEqual(decoded_ego.shape, (480, 640, 3))
        self.assertEqual(decoded_third.shape, (480, 640, 3))
        self.assertEqual(int(decoded_ego[0, 0, 0]), 17)
        self.assertEqual(int(decoded_third[0, 0, 0]), 29)

    def test_terminal_transcript_reassembles_stream_without_duplicate_message(
            self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "run"
            artifacts = RunArtifacts(directory,
                                     RunRequest(message="move safely"))
            try:
                events = [
                    ("run.started", {
                        "message": "move safely"
                    }),
                    ("assistant.delta", {
                        "message_id": "m1",
                        "text": "Move"
                    }),
                    ("assistant.delta", {
                        "message_id": "m1",
                        "text": " complete."
                    }),
                    ("assistant.message", {
                        "message_id": "m1",
                        "text": "Move complete."
                    }),
                    ("tool.started", {
                        "tool_call_id": "t1",
                        "tool": "execute_robot_action"
                    }),
                    ("tool.completed", {
                        "tool_call_id": "t1",
                        "tool": "execute_robot_action",
                        "duration_ms": 10,
                    }),
                    ("run.finished", {
                        "run_status": "completed",
                        "finish_reason": "stop"
                    }),
                ]
                for seq, (kind, data) in enumerate(events, 1):
                    artifacts.event(
                        RunEvent(
                            run_id="run",
                            session_id="session",
                            seq=seq,
                            type=kind,
                            data=data,
                        ))
            finally:
                artifacts.close()

            text = (directory /
                    "terminal_output.txt").read_text(encoding="utf-8")
            self.assertEqual(text.count("Move complete."), 1)
            self.assertIn("[TOOL] execute_robot_action · completed · 10ms",
                          text)
            self.assertIn("[RUN] completed · stop", text)

    def test_ffmpeg_writer_produces_playable_mp4(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "view.mp4"
            writer = _FFmpegVideoWriter(path, fps=10.0, ffmpeg_binary="ffmpeg")
            for value in (0, 64, 128):
                writer.write(np.full((480, 640, 3), value, dtype=np.uint8))
            error = writer.close()

            self.assertIsNone(error)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_run_recorder_writes_two_videos_and_manifest(self):
        import zmq

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            endpoint = f"ipc://{directory / 'frames.ipc'}"
            context = zmq.Context()
            publisher = context.socket(zmq.PUB)
            try:
                publisher.bind(endpoint)
            except zmq.ZMQError as exc:
                publisher.close(linger=0)
                context.term()
                if exc.errno == errno.EPERM:
                    self.skipTest(
                        "sandbox does not permit binding a ZeroMQ IPC socket")
                raise
            recorder = RunVideoRecorder(directory, endpoint=endpoint, fps=10.0)
            recorder.start()
            try:
                time.sleep(0.2)  # Allow the PUB/SUB subscription to settle.
                ego = np.full((1, 1, 480, 640, 3), 17, dtype=np.uint8)
                third = np.full((480, 640, 3), 29, dtype=np.uint8)
                raw = msgpack.packb({
                    "observation": {
                        "video": {
                            "ego_view": _envelope(ego)
                        }
                    },
                    "third_person_rgb": _envelope(third),
                })
                for _ in range(8):
                    publisher.send(raw)
                    time.sleep(0.04)
                time.sleep(0.2)
            finally:
                manifest = recorder.stop()
                publisher.close(linger=0)
                context.term()

            self.assertEqual(manifest["status"], "completed")
            self.assertGreater(manifest["videos"]["first_person"]["frames"], 0)
            self.assertGreater(manifest["videos"]["third_person"]["frames"], 0)
            self.assertGreater((directory / "first_person.mp4").stat().st_size,
                               0)
            self.assertGreater((directory / "third_person.mp4").stat().st_size,
                               0)

    def test_recording_config_accepts_camel_case(self):
        config = Config.model_validate({
            "runRecording": {
                "enabled": True,
                "endpoint": "ipc:///tmp/test.ipc",
                "fps": 5.0,
                "ffmpegBinary": "ffmpeg",
            }
        })
        self.assertTrue(config.run_recording.enabled)
        self.assertEqual(config.run_recording.endpoint, "ipc:///tmp/test.ipc")
        self.assertEqual(config.run_recording.fps, 5.0)


if __name__ == "__main__":
    unittest.main()
