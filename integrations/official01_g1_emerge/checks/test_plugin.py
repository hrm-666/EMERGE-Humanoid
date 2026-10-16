from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Emerge.agent.skills import SkillsLoader
from Emerge.agent.tools.embodied import EmbodiedActionTool
from robot.drivers import load_driver
from robot.plugins import register_plugin


class Official01PluginTest(unittest.TestCase):
    def test_main_agent_discovers_skill_and_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            names = {item["name"] for item in SkillsLoader(workspace).list_skills()}
            schema = str(EmbodiedActionTool(workspace).parameters)
        self.assertIn("official01-g1-control", names)
        for action in ("navigate", "infer_gr00t", "run_fixed_action", "move_micro"):
            self.assertIn(action, schema)

    def test_manifest_registers_unique_driver(self):
        plugin_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {"Emerge_PLUGIN_HOME": temp}
        ):
            spec = register_plugin(plugin_root)
            self.assertEqual(spec.driver_name, "official01_g1")
            driver = load_driver(
                "official01_g1", runtime="dry-run", publish_observations=False
            )
            self.assertEqual(driver.runtime, "dry-run")
            self.assertTrue(driver.get_profile_path().is_file())


if __name__ == "__main__":
    unittest.main()
