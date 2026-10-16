import unittest
from unittest.mock import Mock
from unittest.mock import patch

from robot.navigation.lightnav_sonic import lightnav_ws


def _response(*, seq=4, stop=False, actions=None, rc=0):
    if actions is None:
        actions = [[0.25, 0.0, 0.1], [0.5, 0.0, 0.2]]
    return {
        "action": "next",
        "data": {
            "rc": rc,
            "seq": seq,
            "actions": {
                "step": 42,
                "actions": actions
            },
            "stop": stop,
            "latency_ms": 12.5,
            "raw_text": "<traj_1>",
        },
    }


class LightNavResponseTests(unittest.TestCase):

    def test_valid_response_preserves_first_waypoint(self) -> None:
        reply = lightnav_ws.parse_next_response(_response(), expected_seq=4)
        self.assertEqual(reply.first_waypoint, (0.25, 0.0, 0.1))
        self.assertFalse(reply.stop)
        self.assertEqual(reply.latency_ms, 12.5)

    def test_stop_response_is_valid(self) -> None:
        reply = lightnav_ws.parse_next_response(_response(stop=True))
        self.assertTrue(reply.stop)

    def test_sequence_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(lightnav_ws.LightNavProtocolError,
                                    "sequence mismatch"):
            lightnav_ws.parse_next_response(_response(seq=5), expected_seq=4)

    def test_server_error_is_rejected(self) -> None:
        with self.assertRaisesRegex(lightnav_ws.LightNavProtocolError,
                                    "rc=500"):
            lightnav_ws.parse_next_response(_response(rc=500))

    def test_incomplete_waypoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(lightnav_ws.LightNavProtocolError,
                                    "exactly three"):
            lightnav_ws.parse_next_response(_response(actions=[[0.1, 0.2]]))

    def test_non_finite_waypoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(lightnav_ws.LightNavProtocolError,
                                    "finite"):
            lightnav_ws.parse_next_response(
                _response(actions=[[float("inf"), 0.0, 0.0]]))


class ClientTimeoutTests(unittest.TestCase):

    def test_receive_timeout_is_bounded(self):
        client = lightnav_ws.LightNavWsClient("ws://test",
                                              request_timeout_s=0.5)
        socket = Mock()
        socket.recv.side_effect = TimeoutError("no response")
        client._socket = socket
        with self.assertRaises(TimeoutError):
            client.predict(b"jpeg", "walk forward")
        socket.recv.assert_called_once_with(timeout=0.5)

    def test_failed_session_closes_socket(self):
        client = lightnav_ws.LightNavWsClient("ws://test",
                                              request_timeout_s=0.5)
        socket = Mock()
        socket.recv.side_effect = TimeoutError("no login reply")
        with patch("websockets.sync.client.connect", return_value=socket):
            with self.assertRaises(TimeoutError):
                client.connect()
        socket.close.assert_called_once()
        self.assertIsNone(client._socket)

    def test_invalid_timeouts_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                lightnav_ws.LightNavWsClient("ws://test",
                                             request_timeout_s=value)


if __name__ == "__main__":
    unittest.main()
