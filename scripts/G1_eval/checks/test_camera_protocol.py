"""Protocol regression for the host-visible room camera and yaw."""
import math
import time
import unittest

import msgpack

from robot.navigation.lightnav_sonic.official01_observation import (
    decode_observation,
)


class CameraProtocolTest(unittest.TestCase):

    def test_room_payload_preserves_image_and_yaw(self):
        payload = {
            "observation": {
                "video": {
                    "ego_view": {
                        b"nd": True,
                        b"kind": b"",
                        b"type": "|u1",
                        b"shape": [1, 1, 480, 640, 3],
                        b"data": bytes(480 * 640 * 3),
                    }
                }
            },
            "wall_time_s": time.time(),
            "sim_time_s": 1.0,
            "base_quat": [math.sqrt(0.5), 0.0, 0.0,
                          math.sqrt(0.5)],
        }
        frame = decode_observation(msgpack.packb(payload, use_bin_type=True))
        self.assertAlmostEqual(frame.yaw_rad, math.pi / 2)
        self.assertTrue(frame.jpeg.startswith(b"\xff\xd8"))
        self.assertTrue(frame.fresh(1))
        del payload["base_quat"]
        frame = decode_observation(msgpack.packb(payload, use_bin_type=True))
        self.assertIsNone(frame.yaw_rad)
