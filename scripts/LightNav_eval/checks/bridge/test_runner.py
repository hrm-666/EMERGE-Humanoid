import base64
import math
import unittest

from robot.vla.lightnav_sonic import feedback as runner


class RunnerInputTests(unittest.TestCase):

    def test_extracts_direct_camera_base64(self) -> None:
        jpeg = b"\xff\xd8mock-jpeg\xff\xd9"
        self.assertEqual(
            runner.extract_jpeg_from_camera_message(
                {"ego_view": base64.b64encode(jpeg).decode()}, "ego_view"),
            jpeg)

    def test_extracts_nested_camera_base64(self) -> None:
        jpeg = b"\xff\xd8mock-jpeg\xff\xd9"
        payload = {"images": {"ego_view": base64.b64encode(jpeg).decode()}}
        self.assertEqual(
            runner.extract_jpeg_from_camera_message(payload, "ego_view"), jpeg)

    def test_missing_camera_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ego_view"):
            runner.extract_jpeg_from_camera_message({}, "ego_view")

    def test_wxyz_yaw_uses_mujoco_world_convention(self) -> None:
        half = math.sqrt(0.5)
        self.assertAlmostEqual(runner.yaw_from_wxyz((half, 0.0, 0.0, half)),
                               math.pi / 2.0)

    def test_zero_quaternion_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero norm"):
            runner.yaw_from_wxyz((0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
