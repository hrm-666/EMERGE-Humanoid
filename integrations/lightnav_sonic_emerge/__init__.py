"""LightNav simulation integration; load the EMERGE driver only on demand.

Standalone bridge/simulator processes use their own Python environments and
must not import the Controller or its Python 3.11+ plugin dependencies.
"""

__all__ = ["SonicLightNavDriver"]


def __getattr__(name):
    if name == "SonicLightNavDriver":
        from .driver import SonicLightNavDriver
        return SonicLightNavDriver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
