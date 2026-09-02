# Bring your own robot client

`mlx-smolvla serve` owns model execution and speaks LeRobot 0.6.1's four-RPC
async-inference protocol. A robot-specific client owns observation capture,
calibration, safety limits, actuation, and shutdown.

Prefer LeRobot's maintained `lerobot.async_inference.robot_client` when your
robot is supported. For another robot class, adapt only the client boundary:

1. Produce observations with the checkpoint's configured image/state keys and
   a task string.
2. Use the pinned LeRobot protobuf service and `Ready`, `SendObservations`,
   `GetActions`, and `Stop` semantics documented in `docs/ARCHITECTURE.md`.
3. Validate action shape/range and apply independent joint, rate, watchdog,
   session-duration, and exception-cleanup limits before any actuator write.
4. Complete a no-motion protocol before a single bounded action.

No generic hardware client is shipped yet because the required physical-driver
interfaces have not been inspected in an authorized session. Do not infer
hardware readiness from the software loopback tests; follow
[`docs/HARDWARE_RUNBOOK.md`](../../docs/HARDWARE_RUNBOOK.md) and the
[`hardware/FIRST_CONTACT.md`](../../hardware/FIRST_CONTACT.md) status.
