"""Verify built-in registration without an installed external plugin."""
import os
import tempfile
import unittest
from unittest.mock import patch

from robot.drivers import load_driver
from robot.drivers.lightnav_sonic_driver import SonicLightNavDriver


class DriverRegistrationTest(unittest.TestCase):

    def test_load_without_external_registry(self):
        with tempfile.TemporaryDirectory() as plugin_home:
            with patch.dict(os.environ, {"Emerge_PLUGIN_HOME": plugin_home}):
                with tempfile.TemporaryDirectory() as workspace:
                    with load_driver(
                            "lightnav_sonic_g1",
                            workspace=workspace,
                            elastic_band_release_file=None,
                    ) as driver:
                        self.assertIsInstance(driver, SonicLightNavDriver)
                        self.assertTrue(driver.get_profile_path().is_file())
