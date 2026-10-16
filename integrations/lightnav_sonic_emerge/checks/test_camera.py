"""Check simulation transport adaptation without opening robot sockets."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from integrations.lightnav_sonic_emerge import camera
from integrations.lightnav_sonic_emerge import navigation_backend
from integrations.lightnav_sonic_emerge import observation_writer
from integrations.lightnav_sonic_emerge.bridge import official01_observation
from integrations.lightnav_sonic_emerge.bridge import runner


class CameraTransportTest(unittest.TestCase):

    def test_official01_adapter_preserves_jpeg_and_staleness(self):
        source = mock.Mock()
        source.latest.side_effect = [SimpleNamespace(jpeg=b"rgb"), None]
        with mock.patch.object(official01_observation,
                               "Official01Subscriber",
                               return_value=source) as constructor:
            subscriber = camera.create_camera("tcp://fixture", "ego_view",
                                              "official01_rgb")
        constructor.assert_called_once_with("tcp://fixture")
        self.assertEqual(subscriber.latest(0.75), b"rgb")
        self.assertIsNone(subscriber.latest(0.75))
        subscriber.close()
        source.close.assert_called_once()

    def test_sonic_transport_preserves_named_view(self):
        with mock.patch.object(runner, "SonicCameraSubscriber") as constructor:
            subscriber = camera.create_camera("tcp://fixture", "head")
        constructor.assert_called_once_with("tcp://fixture", "head")
        self.assertIs(subscriber, constructor.return_value)

    def test_backend_uses_configured_camera_transport(self):
        config = navigation_backend.NavigationConfig(
            lightnav_server="ws://fixture", camera_transport="official01_rgb")
        backend = navigation_backend.NavigationBackend(
            config, mock.Mock(spec=observation_writer.EgoObservationWriter))
        with mock.patch.object(camera, "create_camera") as factory:
            backend._camera_factory("tcp://fixture", "ego_view")
        factory.assert_called_once_with("tcp://fixture", "ego_view",
                                        "official01_rgb")

    def test_invalid_transport_is_rejected_before_opening_sockets(self):
        with self.assertRaises(ValueError):
            camera.create_camera("tcp://fixture", "ego_view", "unknown")
        with self.assertRaises(ValueError):
            camera.create_camera("tcp://fixture", "wrong", "official01_rgb")
        with self.assertRaises(ValueError):
            navigation_backend.NavigationConfig(lightnav_server="ws://fixture",
                                                camera_transport="unknown")
