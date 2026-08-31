"""Command-line surface tests for the packaged native runtime."""

from __future__ import annotations

import subprocess
import sys


def test_cli_exposes_required_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "smolvla_mlx.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for command in ("convert", "test", "bench", "predict"):
        assert command in completed.stdout
