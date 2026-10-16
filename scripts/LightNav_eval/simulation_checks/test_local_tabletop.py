"""Check the ordinary-user launch keeps the original scene and loopback boundary."""

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.LightNav_eval import local_tabletop
from scripts.LightNav_eval import tabletop


class LocalTabletopTests(unittest.TestCase):

    def test_original_scene_and_support_with_no_privileged_command(self):
        with patch.dict("os.environ", {"DISPLAY": ":1"}):
            command, _ = local_tabletop.build_command("sim", [])
        self.assertTrue(local_tabletop.is_loopback_scene(command))
        self.assertIn("scripts.LightNav_eval.tabletop_sim", command)
        self.assertNotIn("--no-elastic-band", command)
        self.assertNotIn("sudo", command)
        self.assertEqual(command[command.index("--duration") + 1], "0")

    def test_scene_and_sonic_require_explicit_loopback(self):
        with patch.dict("os.environ", {"DISPLAY": ":1"}):
            command, _ = local_tabletop.build_command("sim", [])
        self.assertFalse(
            local_tabletop.is_loopback_scene(
                [item if item != "lo" else "eth0" for item in command]))
        self.assertTrue(
            local_tabletop.is_loopback_sonic(["/bin/g1_deploy_onnx_ref", "lo"]))
        self.assertFalse(
            local_tabletop.is_loopback_sonic(
                ["/bin/g1_deploy_onnx_ref", "eth0"]))

    def test_motion_rejects_foreign_endpoints_or_missing_simulation(self):
        args = tabletop.build_parser().parse_args(
            ["--simulation-loopback", "--enable-motion"])
        with self.assertRaisesRegex(RuntimeError, "127.0.0.1"):
            local_tabletop.require_loopback_simulation(args)
        args.observation_endpoint = "tcp://127.0.0.1:5560"
        with patch("scripts.LightNav_eval.local_tabletop.Path.iterdir",
                   return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "running as this user"):
                local_tabletop.require_loopback_simulation(args)

    def test_evaluation_needs_no_display_and_starts_no_controller(self):
        with patch.dict("os.environ", {"DISPLAY": ""}):
            command, _ = local_tabletop.build_command("evaluate",
                                                      ["--task-dir", "trial"])
        self.assertIn("scripts.LightNav_eval.tabletop_evaluate", command)
        self.assertNotIn("--interface", command)
        self.assertNotIn("--enable-motion", command)
        self.assertEqual(command[-2:], ["--task-dir", "trial"])

    def test_original_scene_and_interface_cannot_be_overridden(self):
        for extra in (["--interface",
                       "eth0"], ["--model",
                                 "other.xml"], ["--no-elastic-band"]):
            with self.assertRaises(ValueError):
                local_tabletop.build_command("sim", extra)

    def test_model_source_and_checkpoint_use_external_lightnav_root(self):
        with patch.dict("os.environ",
                        {"LIGHTNAV_ROOT": "/tmp/lightnav-model-source"}):
            command, env = local_tabletop.build_command("server", [])
        self.assertIn("/tmp/lightnav-model-source/src", env["PYTHONPATH"])
        self.assertEqual(command[command.index("--model_path") + 1],
                         "/tmp/lightnav-model-source/checkpoints/LightNav-0")
        root = Path(__file__).resolve().parents[3]
        self.assertIn(str(root), env["PYTHONPATH"].split(":"))

    def test_bridge_import_does_not_load_controller_dependencies(self):
        import subprocess
        import sys
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; import scripts.LightNav_eval.local_tabletop; "
            "assert 'robot.plugins' not in sys.modules; assert 'Emerge' not in sys.modules"
        ],
                                capture_output=True,
                                text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
