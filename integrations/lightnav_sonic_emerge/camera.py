"""Read-only RGB transports shared by navigation and workspace observations."""

from __future__ import annotations

from typing import Protocol


class CameraSubscriber(Protocol):
    """Provide the newest usable JPEG without blocking for another frame."""

    def latest(self, max_age_s: float) -> bytes | None:
        """Return current RGB as JPEG, or None when unavailable or stale."""

    def close(self) -> None:
        """Release the read-only subscription."""


class _Official01Camera:
    """Adapt the existing apple-table RGB envelope to the JPEG interface."""

    def __init__(self, endpoint: str) -> None:
        try:
            from .bridge.official01_observation import Official01Subscriber
        except ImportError:  # External driver plugins use top-level imports.
            from bridge.official01_observation import Official01Subscriber
        self._subscriber = Official01Subscriber(endpoint)

    def latest(self, max_age_s: float) -> bytes | None:
        frame = self._subscriber.latest(max_age_s)
        return frame.jpeg if frame is not None else None

    def close(self) -> None:
        self._subscriber.close()


def create_camera(endpoint: str,
                  camera_name: str,
                  transport: str = "sonic_jpeg") -> CameraSubscriber:
    """Open one simulation camera without activating locomotion.

    Args:
        endpoint: ZMQ publisher endpoint.
        camera_name: Named view in SONIC packets; ego_view for official01_rgb.
        transport: sonic_jpeg or official01_rgb.

    Returns:
        A subscriber owned by the caller, which must close it.

    Raises:
        ValueError: The transport or official01 view is unsupported.
    """
    if transport == "official01_rgb":
        if camera_name != "ego_view":
            raise ValueError("official01_rgb only publishes ego_view")
        return _Official01Camera(endpoint)
    if transport != "sonic_jpeg":
        raise ValueError(f"Unsupported camera transport: {transport!r}")
    try:
        from .bridge.runner import SonicCameraSubscriber
    except ImportError:  # External driver plugins use top-level imports.
        from bridge.runner import SonicCameraSubscriber
    return SonicCameraSubscriber(endpoint, camera_name)
