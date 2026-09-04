# Bring your own robot client

`mlx-smolvla serve` owns model execution and speaks LeRobot 0.6.1's four-RPC
async-inference protocol. A robot-specific client owns observation capture,
calibration, safety limits, actuation, and shutdown.

Prefer LeRobot's maintained `lerobot.async_inference.robot_client` when its
safety behavior is sufficient for your robot. For another robot class, adapt
only the client boundary:

1. Produce observations with the checkpoint's configured image/state keys and
   a task string.
2. Use the pinned LeRobot protobuf service and `Ready`, `SendObservations`,
   `GetActions`, and `Stop` semantics documented in `docs/ARCHITECTURE.md`.
3. Validate action shape/range and apply independent joint, rate, watchdog,
   session-duration, and exception-cleanup limits before any actuator write.
4. Complete a no-motion protocol before a single bounded action.

## Hiwonder SO-101 client

[`hiwonder_so101_client.py`](hiwonder_so101_client.py) is the inspected,
fail-closed client for the operator's Hiwonder HX-30HM SO-101 fork. It imports
the vendor checkout by path but never installs into, configures, calibrates, or
edits that checkout. Its default `--no-motion` mode runs the complete camera,
state, four-RPC, and action-envelope loop for 60 seconds without enabling
torque or writing a goal position.

Install the optional client dependencies in a repository-local environment:

```bash
uv venv --python 3.12 .cache/hardware/client-venv
uv pip install --python .cache/hardware/client-venv/bin/python -e '.[hardware]'
.cache/hardware/client-venv/bin/python \
  examples/bring_your_own_robot/hiwonder_so101_client.py --help
```

The motion modes are deliberately harder to invoke. They require an exact
follower serial/port match, an existing calibration, an operator-attested
hardware-limit profile, a start pose inside the 10%-inset calibration, and a
local checkpoint with effective six-axis state/action statistics. Every valid
target is rate-limited from live servo readback; malformed or out-of-domain
chunks hold, and every exit path returns slowly and verifies torque off.

Do not infer actuation readiness from the shipped code or completed no-motion
test. Follow the exact [hardware runbook](../../docs/HARDWARE_RUNBOOK.md),
[client design](../../hardware/CLIENT_DESIGN.md), and current
[first-contact status](../../hardware/FIRST_CONTACT.md). The 2026-09-02
no-motion protocol passed; later checks also verified camera framing and a
supported inset pose. The exact operator-attested low-controller profile and
the final physical checklist still block the first physical action.
