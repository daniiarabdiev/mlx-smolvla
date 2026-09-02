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
    runbook = Path("docs/HARDWARE_RUNBOOK.md").read_text(encoding="utf-8")

    assert "ARM SESSION CONFIRMED" in runbook
    assert "Absent that exact line" in runbook
    assert "scripts/serve_latency_smoke.py" in runbook
    assert "--output .cache/hardware/first-contact-latency.jsonl" in runbook
    assert "examples/bring_your_own_robot/hiwonder_so101_client.py" in runbook
    assert "--no-motion" in runbook
    assert "--single-action" in runbook
    assert "--hardware-safety-profile" in runbook
    assert "500 ms action watchdog" in runbook
    assert "60 seconds" in runbook
    assert "one-public-unit" in runbook
    assert "hand on the power switch" in runbook
    assert "torque and speed limits" in runbook
    assert "kill the server, then power off" in runbook
    assert "follower read path and 60-second no-motion loop" in runbook
    assert "motion has not run" in runbook
    assert "Raw `lerobot/smolvla_base` is suitable for no-motion diagnosis only" in runbook
    assert ".cache/hardware/server-venv" in runbook
    assert ".cache/hardware/client-venv" in runbook
    for evidence in (
        "latency JSONL",
        "server log",
        "client telemetry",
        "observed motion",
        "rollback",
    ):
        assert evidence in runbook


def test_public_hardware_status_is_explicitly_no_motion_only() -> None:
    first_contact = Path("hardware/FIRST_CONTACT.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    prose = " ".join(first_contact.split())

    assert "no-motion protocol complete; physical motion blocked" in first_contact
    assert "No torque-enable or goal-position write occurred" in prose
    assert "single-action nor bounded- continuous stage ran" in prose
    assert "drives a real SO-101 from a MacBook” is **not** evidenced" in first_contact
    assert "Raw `lerobot/smolvla_base` output is not a physical-action interface" in readme
    assert "single-action and bounded-continuous motion are not" in readme
