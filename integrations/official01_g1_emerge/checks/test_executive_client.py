from __future__ import annotations

import subprocess
import unittest

from integrations.official01_g1_emerge.executive_client import (
    ExecutiveClientError,
    ExecutiveProcessClient,
)


class ExecutiveSudoPreflightTest(unittest.TestCase):
    def test_simulation_preflight_prompts_through_controller_terminal(self):
        calls = []

        def sudo_runner(argv, *, check):
            calls.append((argv, check))
            return subprocess.CompletedProcess(argv, 0)

        client = ExecutiveProcessClient(
            "/tmp/control-root", runtime="sim", sudo_runner=sudo_runner,
            stdin_is_tty=lambda: True,
        )
        client._authenticate_simulation_sudo()
        self.assertEqual(calls, [(["sudo", "-v"], False)])

    def test_noninteractive_simulation_has_clear_recovery_message(self):
        calls = []
        client = ExecutiveProcessClient(
            "/tmp/control-root", runtime="sim",
            sudo_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
            stdin_is_tty=lambda: False,
        )
        with self.assertRaisesRegex(ExecutiveClientError, "interactive terminal"):
            client._authenticate_simulation_sudo()
        self.assertEqual(calls, [])

    def test_dry_run_does_not_request_sudo(self):
        calls = []
        client = ExecutiveProcessClient(
            "/tmp/control-root", runtime="dry-run",
            sudo_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
            stdin_is_tty=lambda: False,
        )
        self.assertEqual(client.runtime, "dry-run")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
