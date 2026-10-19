"""Run the existing room with bounded viewer refresh and measured diagnostics.

Physics and cameras remain owned by the original P5 runner. GUI synchronization
is limited to 30 Hz instead of occurring on all 500 physics steps per second.
Only robot proprioception and simulation timing are added to diagnostics.
"""
from __future__ import annotations

import signal
import time
from typing import Any
from typing import Callable


class RateLimitedViewer:
    """Forward viewer operations while bounding expensive GUI synchronization."""

    def __init__(self,
                 viewer: Any,
                 clock: Callable[[], float] = time.monotonic):
        self._viewer = viewer
        self._clock = clock
        self._next_sync = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._viewer, name)

    def sync(self) -> None:
        now = self._clock()
        if now >= self._next_sync:
            self._viewer.sync()
            self._next_sync = now + 1.0 / 30.0


class DiagnosticPublisher:
    """Add measured robot and timing fields without altering policy images."""

    def __init__(self, publisher: Any, runner: Any):
        self._publisher = publisher
        self._runner = runner
        self._wall_start = time.monotonic()
        self._sim_start = float(runner.data.time)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._publisher, name)

    def maybe_publish(self,
                      adapter,
                      diagnostics=None,
                      third_person_renderer=None):
        runner = self._runner
        elapsed = max(time.monotonic() - self._wall_start, 1e-6)
        fields = dict(diagnostics or {})
        fields.update(
            robot_base_xyz=runner.data.xpos[adapter.root_body_id].tolist(),
            support_enabled=bool(runner.elastic_band is not None and
                                 runner.elastic_band.enable),
            simulation_realtime_factor=(float(runner.data.time) -
                                        self._sim_start) / elapsed,
        )
        return self._publisher.maybe_publish(adapter, fields,
                                             third_person_renderer)


def main() -> int:
    # The launcher selects the original dependency directory in PYTHONPATH.
    import p5_sim_observation_server as runtime

    class IntegrationRunner(runtime.P5MuJoCoRunner):
        """Retain P5 physics and camera contracts with bounded GUI overhead."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.viewer is not None:
                self.viewer = RateLimitedViewer(self.viewer)
            self.observation_publisher = DiagnosticPublisher(
                self.observation_publisher, self)
            self._support_request = None
            signal.signal(signal.SIGUSR1, self._request_support)
            signal.signal(signal.SIGUSR2, self._request_support)

        def _request_support(self, signum, _frame):
            # Signal handlers only enqueue; MuJoCo state changes on its owner.
            self._support_request = signum == signal.SIGUSR1

        def step(self):
            request = self._support_request
            if request is not None:
                self._support_request = None
                if self.elastic_band is not None:
                    if request:
                        # Support at the current position, never pull back to start.
                        self.elastic_band.point = self.data.xpos[
                            self.band_body_id].copy()
                    self.elastic_band.enable = request
                    print(f"ElasticBand enable: {request} (test signal)",
                          flush=True)
            super().step()

    runtime.P5MuJoCoRunner = IntegrationRunner
    return runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
