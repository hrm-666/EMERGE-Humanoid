"""The read-only viewer must distinguish fresh camera frames from frozen images."""

import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from PIL import Image

from integrations.lightnav_sonic_emerge.bridge import ego_viewer
from integrations.lightnav_sonic_emerge.bridge import official01_observation


class EgoViewerTests(unittest.TestCase):

    def make_viewer(self):
        viewer = ego_viewer.EgoViewer.__new__(ego_viewer.EgoViewer)
        viewer.root = Mock()
        viewer.subscriber = Mock()
        viewer.image = Mock()
        viewer.status = Mock()
        viewer.args = ego_viewer.build_parser().parse_args([])
        viewer.photo = None
        viewer.last_timestamp = None
        return viewer

    def test_fresh_frame_display_and_snapshot_then_stale_clears_image(self):
        viewer = self.make_viewer()
        buffer = io.BytesIO()
        Image.new("RGB", (640, 480), "red").save(buffer, format="JPEG")
        frame = official01_observation.ObservationFrame(buffer.getvalue(),
                                                        time.time(), 1.5)
        viewer.subscriber.latest.return_value = frame
        with tempfile.TemporaryDirectory() as directory:
            viewer.args.snapshot_path = Path(directory) / "latest.jpg"
            with patch(
                    "integrations.lightnav_sonic_emerge.bridge.ego_viewer.ImageTk.PhotoImage"
            ) as photo:
                viewer.refresh()
                viewer.image.configure.assert_called_with(
                    image=photo.return_value)
                self.assertEqual(viewer.args.snapshot_path.read_bytes(),
                                 frame.jpeg)
                viewer.refresh()
                self.assertEqual(photo.call_count, 1)
            viewer.subscriber.latest.return_value = None
            viewer.subscriber.error = ""
            viewer.refresh()
            viewer.image.configure.assert_called_with(image="")
            self.assertIsNone(viewer.photo)
            self.assertIsNone(viewer.last_timestamp)
            self.assertIn("STALE",
                          viewer.status.configure.call_args.kwargs["text"])
            self.assertEqual(viewer.args.snapshot_path.read_bytes(), frame.jpeg)


if __name__ == "__main__":
    unittest.main()
