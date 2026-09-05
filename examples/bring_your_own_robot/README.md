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
   `SendPolicyInstructions`, and `GetActions` semantics documented in the
   [architecture](../../docs/ARCHITECTURE.md).
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
chunks hold, and exit cleanup attempts a bounded return and verifies torque off.
A return can exhaust its step cap; torque-off cleanup is still required.

Do not infer actuation readiness from the shipped code or completed no-motion
test. Follow the exact [hardware runbook](../../docs/HARDWARE_RUNBOOK.md),
[client design](../../hardware/CLIENT_DESIGN.md), and current
[first-contact status](../../hardware/FIRST_CONTACT.md). The 2026-09-02
no-motion protocol passed. The later 2026-09-04 session passed a final no-motion
loop, one valid guarded action, and a two-chunk continuous run, with exact
return and verified torque-off cleanup for the powered runs under the temporary
10% torque profile. A separate 20-chunk attempt is inconclusive under reduced
torque: exact return was not met, while torque-off cleanup was verified. These
results do not establish sustained operation or reliable task completion. Each
new powered session requires fresh live authorization, controller-profile
readback, and the runbook’s physical preflight.
