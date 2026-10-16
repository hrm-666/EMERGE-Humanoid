"""Uniquely named external-plugin entry point.

EMERGE's plugin registry imports top-level modules from each registered plugin
directory.  A unique module name avoids colliding with other plugins that also
use a generic ``driver.py`` entry point.
"""

from integrations.official01_g1_emerge.driver import Official01G1Driver

__all__ = ["Official01G1Driver"]
