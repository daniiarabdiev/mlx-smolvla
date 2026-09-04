# Hiwonder SO-101 client design

## Decision

The inspected vendor checkout contains a LeRobot async-inference client and the
same four protobuf RPCs, but that generic client does not implement all safety
requirements in the public-release brief. In particular, it does not require
exact controller-limit readback, a 10%-inset calibrated envelope, a slow return
to the start pose, or the repository's fixed session caps.

The project therefore ships a narrow standalone client at
[`examples/bring_your_own_robot/hiwonder_so101_client.py`](../examples/bring_your_own_robot/hiwonder_so101_client.py).
It runs from this repository's `hardware` extra, imports the read-only vendor
robot/camera classes from a caller-supplied checkout, and speaks the audited
LeRobot 0.6.1 `Ready`, `SendPolicyInstructions`, `SendObservations`, and
`GetActions` protocol.

## Dependency boundary

- The base `mlx_smolvla` import remains free of Torch, Transformers, and
  LeRobot.
- The optional `hardware` extra pins `lerobot[async,hardware]==0.6.1` on Python
  3.12+ for protobuf/gRPC, serial, camera, and test-compatible protocol types.
- Vendor source is selected in a fresh client process before any `lerobot`
  import. A different already-imported LeRobot tree is rejected.
- The vendor checkout is never installed, upgraded, edited, configured, or
  calibrated by this project.
- A lightweight pickle-compatible protocol shim avoids importing the vendor's
  unrelated policy stack. Cross-process tests prove compatibility with the
  real LeRobot 0.6.1 `RemotePolicyConfig`, `TimedObservation`, and
  `TimedAction` globals.

## Follower-only open path

The factory validates the checkout and exact follower port, instantiates the
six-joint Hiwonder follower, and then performs only:

1. `bus.connect(handshake=True)`;
2. read `Torque_Enable` and require six zeroes;
3. read controller calibration and require exact agreement with the existing
   six-joint calibration file;
4. connect only the two selected cameras;
5. read one complete observation and reject missing, malformed, non-finite, or
   all-black frames.

It never calls the vendor robot's `connect()`, `configure()`, `calibrate()`, or
`send_action()` methods. The leader port is not passed to the process and is
never opened.

## Graduated modes

| Mode | Fixed limits | Writes |
| --- | --- | --- |
| `--no-motion` | exactly 60 s, 5 Hz, 500 ms RPC watchdog | no torque or actuator writes |
| `--single-action` | at most 20 chunks; rejected chunks hold, then the first valid action stops the session | gated motion only |
| `--continuous` | at most 90 s or 20 chunks, whichever comes first | gated motion only |

Motion modes additionally require:

- a local checkpoint with finite, positive six-axis `observation.state` and
  `action` mean/std tensors;
- a current-session verified JSON safety profile whose serial matches the selected
  follower port;
- exact readback equality for nine controller safety registers while torque is
  still off;
- integer position mode (0) on every joint, and an integer startup-force value
  no greater than the smaller of its persistent and temporary torque caps;
- a start pose already inside the 10%-inset calibrated range.

Only after those checks does the adapter read each raw present position and its
raw `Min_Position_Limit`/`Max_Position_Limit`, require every present value to
lie inside the inclusive controller range, and write those exact raw values to
`Goal_Position`. It then requires exact goal and fresh-present equality,
unchanged raw limits, and unchanged values for all nine safety registers,
position mode, and startup force.

The first `Goal_Position` write is the arming boundary. In the default
`explicit-torque` mode it must leave all six torque bits at zero before the
client calls `enable_torque()` and verifies six ones. The explicit
`goal-write` mode is limited to a controller already observed to auto-enable on
that write; it requires six ones after preload and never calls
`enable_torque()`. Every exception after the write or explicit-enable attempt
runs the verified all-six torque-off path. This prevents both stale-goal
arming and a second enable command on the observed controller. Every action
must be shape `(1, 6)`, finite, and inside the
vendor driver's full public domain (body joints −180°–180°, gripper 0–100). Valid
values are clipped to the 10%-inset calibration and rate-limited from current
servo readback by the stricter of 2% of calibrated span or one public unit.
Invalid values hold. Three consecutive 500 ms action timeouts terminate the
session.

The server/model setup RPC is allowed up to 600 seconds because checkpoint
loading can be long, but it always finishes before the safety session can
enable torque. The 500 ms timeout applies only to observation/action traffic.

## Writes and cleanup

The only actuator register the client can write is `Goal_Position`. Its first
write copies raw present encoder values after torque-off and raw-range checks;
because that write can energize the tested controller, all subsequent
verification is inside torque-off-on-error cleanup. Later writes require the
armed state. The
Hiwonder manual states that the position-mode `Time` field is not applicable,
so the client does not pretend `Goal_Time` controls speed. It enforces a
minimum 200 ms command/dwell interval while independently requiring
verified low hardware limits established under the operator's authorization.

On a normal cap, watchdog stop, exception, SIGINT, or SIGTERM, a motion session
attempts—in order—to hold current position, return to the recorded start pose
through fresh-readback steps bounded by the same one-public-unit envelope,
disable torque with all-zero readback, disconnect cameras, and close the
follower bus. Each return step uses the 1000 ms return dwell and the number of
steps cannot exceed the session chunk cap. Cleanup continues after individual
failures and reports every failure. A no-motion session skips all
hold/return/torque operations and closes I/O without a write.

## Telemetry boundary

Client JSONL is created with exclusive-create semantics and mode `0600`. It
contains configuration, counts, reason aggregates, cadence, and median/p95
latencies. Exact payload keys for images, frames, observations, state, tasks,
positions, targets, and actions are refused. Server JSONL uses its existing
exclusive recorder. Neither logger stores robot or policy payload values.
