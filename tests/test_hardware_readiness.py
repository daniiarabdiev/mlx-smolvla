"""Documents-only Stage H operator handoff contracts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_latency_smoke_script_is_a_serve_only_operator_entrypoint() -> None:
    path = Path("scripts/serve_latency_smoke.py")
    source = path.read_text(encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(path), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--output" in completed.stdout
    assert "--host" in completed.stdout
    assert "--port" in completed.stdout
    assert "serve_forever" in source
    assert "latency_log=args.output" in source
    assert "assert " not in source


def test_hardware_runbook_has_exact_gate_commands_safety_and_evidence() -> None:
    runbook = Path("HARDWARE_RUNBOOK.md").read_text(encoding="utf-8")

    assert "ARM SESSION CONFIRMED" in runbook
    assert "Absent that exact line" in runbook
    assert "scripts/serve_latency_smoke.py" in runbook
    assert "--output .cache/hardware/first-contact-latency.jsonl" in runbook
    assert "python -m lerobot.async_inference.robot_client" in runbook
    assert "--robot.type=so101_follower" in runbook
    assert "--robot.max_relative_target=1.0" in runbook
    assert "--robot.disable_torque_on_disconnect=true" in runbook
    assert "--actions_per_chunk=1" in runbook
    assert "--fps=5" in runbook
    assert "hand on the power switch" in runbook
    assert "torque and speed limits" in runbook
    assert "kill the server, then power off" in runbook
    assert "Hardware validation status: NOT RUN" in runbook
    for evidence in (
        "latency JSONL",
        "server log",
        "client log",
        "observed motion",
        "rollback",
    ):
        assert evidence in runbook
