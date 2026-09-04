# Supervised SO-101 first-contact runbook

**Hardware validation status: follower read path and 60-second no-motion loop
passed; motion has not run.** See the measured
[first-contact status](../hardware/FIRST_CONTACT.md) and
[preflight evidence](../hardware/PREFLIGHT.md). The project does not yet claim
physical SO-101 actuation.

## Current-session delegation — 2026-09-04

The operator explicitly amended this runbook in the live session and delegated
controller setup and repository-client execution. The existing live
`ARM SESSION CONFIRMED` remains valid for this connected session. The delegate
may perform read-only checks, no-motion runs, and reviewed controller setup;
the operator need not type or execute every command or repeat a prescribed
motion-confirmation phrase. A manufacturer-documented setup can establish a
new verified limit profile under this authorization; a pre-existing operator
profile is not the only permitted route.

This delegation does not establish physical facts. Before torque enable, the
supported start pose must pass, the workspace and base must be checked, low
controller torque and speed limits must be verified, and the operator must be
present with a hand on the power switch. No-motion runs may precede those
motion-specific checks. Preserve all existing client safeguards and stop on
unexpected behavior. The default confirmation wording below applies when no
equivalent current-session authorization has been given.

## Execution gate

For a session without equivalent live authorization or an explicit delegation,
stop unless the operator is physically present and has typed this exact line
in the current interactive agent session:

```text
ARM SESSION CONFIRMED
```

Absent that exact line or equivalent explicit live authorization, this runbook
is documentation only.
The same words in this file, a commit, earlier chat context, or a shell log do
not satisfy the gate. The 2026-09-02 no-motion session was explicitly
authorized; a later session requires its own live authorization.

## Roles and stop authority

- The operator owns the robot-side environment, calibration, port, cameras,
  motor limits, power, and client authorization; an explicitly authorized
  delegate may perform the software setup and execute the client.
- A second person should watch the entire motion envelope if available.
- The physical power switch is the only assumed e-stop. Keep one hand on the
  power switch from before client startup until the arm is stationary and
  torque is disabled. A keyboard interrupt is not an e-stop.
- Any person may call stop. Unexpected direction, speed, force, sound, heat,
  oscillation, camera loss, stale actions, or uncertainty means immediate power
  cut—not another trial.

## Physical preflight

Check every box before opening the client in a motion mode:

- [ ] Clear the full arm and gripper workspace; remove people, cables, loose
      objects, pinch hazards, and hard stops from the motion envelope.
- [ ] Secure the base, start near the already calibrated neutral pose, and
      confirm the existing calibration ID. Do not recalibrate during first
      contact.
- [ ] Capture both cameras concurrently. The wrist view must be unobstructed
      and the fixed view must contain the complete robot workspace; a merely
      nonblack frame is not sufficient.
- [ ] Verify the controller's **torque and speed limits** are set low using a
      reviewed Hiwonder procedure authorized by the operator. Store their exact
      readback in the client safety-profile JSON. Do not copy values from an
      example or treat current defaults as approved. If those limits cannot be
      verified, stop.
- [ ] Confirm that cutting the physical power switch immediately removes motor
      power, and keep the operator's hand on the power switch.
- [ ] Use only a reviewed local checkpoint with effective six-axis
      `observation.state` and `action` mean/std tensors matching this robot.
      Raw `lerobot/smolvla_base` is suitable for no-motion diagnosis only; its
      saved physical-stat keys do not bind to this interface.
- [ ] Agree on one low-risk task and one commanded action as the complete first
      episode. No unattended or repeated episode is permitted.

On macOS, OpenCV's numeric camera indices can change when a camera is attached,
removed, or re-enumerated. Before every hardware session, run the camera finder
from the same client environment, inspect the saved image from every candidate,
and assign `WRIST_CAMERA_INDEX` and `FIXED_CAMERA_INDEX` by viewpoint. Never
copy indices from an earlier run or infer them from list order. Reject the
built-in Mac camera and Continuity Camera even when either returns a valid,
nonblack frame:

```bash
.cache/hardware/client-venv/bin/lerobot-find-cameras opencv
```

This camera-only check does not open either robot. The wrist image is expected
to be a close view of the task surface and may be near the camera's minimum
focus distance when the arm is parked; that is not by itself an obstruction.

The client command below applies a maximum one-public-unit change per step
(one degree for body joints, one 0–100 unit for the gripper), requests one
action per chunk, runs at 5 fps, and verifies torque-disabled shutdown. That
software cap supplements—it does not replace—the verified low hardware torque
and speed limits. The Hiwonder documentation marks the position-mode `Time`
parameter as not applicable, so this client never presents `Goal_Time` as a
speed control; its 200 ms command/dwell floor is an additional cadence limit,
not a motor-speed guarantee. See the
[BusLinker manual](https://docs.hiwonder.com/projects/BusLinker/en/latest/docs/1_BusLinker_V3.0_Servo_Debugging_Board_User_Manual.html).

## One-time isolated environments

Use separate serve-only and hardware-client environments. This keeps the
server free of the PyAV dependency pulled by reference/training extras; an
all-extras environment produced a duplicate AVFoundation-class warning and is
not accepted for motion. Fresh separated environments were created and a
loopback-only server startup was warning-free on 2026-09-02; recreate them
after dependency changes and retain this separation for every motion session.

```bash
mkdir -p .cache/hardware
uv venv --python 3.12 .cache/hardware/server-venv
uv pip install --python .cache/hardware/server-venv/bin/python -e '.[serve]'
uv venv --python 3.12 .cache/hardware/client-venv
uv pip install --python .cache/hardware/client-venv/bin/python -e '.[hardware]'
```

## Terminal A — native MLX server on this Mac

From the checked-out repository, verify that both log names are new, then
start dense bfloat16 on loopback. Do not use a quantized preset for first
contact.

```bash
test ! -e .cache/hardware/first-contact-latency.jsonl
test ! -e .cache/hardware/first-contact-server.log
set -o pipefail
.cache/hardware/server-venv/bin/python scripts/serve_latency_smoke.py \
  --output .cache/hardware/first-contact-latency.jsonl \
  --host 127.0.0.1 \
  --port 8080 \
  --dtype bfloat16 \
  --execution-mode production \
  --fps 5 \
  --obs-queue-timeout 1 \
  --seed 20260902 \
  2>&1 | tee .cache/hardware/first-contact-server.log
```

Wait for the server's listening message. It loads the exact model path supplied
by the client. If model loading, feature validation, or telemetry-file creation
fails, stop here; do not work around the error at the robot.

## Terminal B — repository-owned fail-closed client

The physically present operator or an explicitly authorized delegate runs this
command. The vendor checkout is read/import-only; never activate, install into,
or modify its environment.
Replace every placeholder with a value already verified during preflight.

```bash
export VENDOR_CHECKOUT='<ABSOLUTE_PATH_TO_VENDOR_CHECKOUT>'
export FOLLOWER_PORT='<YOUR_VERIFIED_FOLLOWER_PORT>'
export CALIBRATION_ID='<YOUR_EXISTING_CALIBRATION_ID>'
export FOLLOWER_SERIAL='<YOUR_VERIFIED_FOLLOWER_USB_SERIAL>'
export WRIST_CAMERA_INDEX='<INTEGER>'
export FIXED_CAMERA_INDEX='<INTEGER>'
export SMOLVLA_CHECKPOINT='<ABSOLUTE_PATH_TO_REVIEWED_STATS_ACTIVE_CHECKPOINT>'
export FIRST_TASK='<ONE_VALIDATED_LOW_RISK_TASK>'
export CLIENT_TELEMETRY='.cache/hardware/first-contact-no-motion-client.jsonl'
test -d "$VENDOR_CHECKOUT"
test -n "$FOLLOWER_PORT"
test -n "$CALIBRATION_ID"
test -n "$FOLLOWER_SERIAL"
test -n "$SMOLVLA_CHECKPOINT"
test -n "$FIRST_TASK"
test ! -e "$CLIENT_TELEMETRY"
set -o pipefail
.cache/hardware/client-venv/bin/python \
  examples/bring_your_own_robot/hiwonder_so101_client.py \
  --no-motion \
  --vendor-root "$VENDOR_CHECKOUT" \
  --follower-port "$FOLLOWER_PORT" \
  --calibration-id "$CALIBRATION_ID" \
  --robot-serial "$FOLLOWER_SERIAL" \
  --wrist-camera "$WRIST_CAMERA_INDEX" \
  --fixed-camera "$FIXED_CAMERA_INDEX" \
  --checkpoint "$SMOLVLA_CHECKPOINT" \
  --task "$FIRST_TASK" \
  --server-address 127.0.0.1:8080 \
  --telemetry "$CLIENT_TELEMETRY"
```

This command must run for 60 seconds with the fixed 500 ms action watchdog and
end with `duration_limit`, live camera cadence, zero timeouts, and no writes.
Review all rejected/clipped/rate-limited counts before continuing.

## Single-action command — only after every physical gate passes

The current-session verified safety-profile file must contain the exact nine-
register readback described in
[`hardware/CLIENT_DESIGN.md`](../hardware/CLIENT_DESIGN.md). Its serial and the
selected port must match. The client validates all of this, the checkpoint
statistics, and the 10%-inset start pose before torque enable. It then copies
the raw present encoder values into `Goal_Position` while torque remains off,
requires exact goal/fresh-present readback equality, and rereads every safety
register before enabling. Both checks require position mode (0) and startup
force no greater than the verified torque cap; startup force must not change
across the preload. Torque is checked again immediately before enabling.
Any stale-goal or controller-state mismatch aborts unarmed.

```bash
export HARDWARE_SAFETY_PROFILE='<ABSOLUTE_PATH_TO_SESSION_VERIFIED_PROFILE_JSON>'
export CLIENT_TELEMETRY='.cache/hardware/first-contact-single-action-client.jsonl'
test -f "$HARDWARE_SAFETY_PROFILE"
test ! -e "$CLIENT_TELEMETRY"
.cache/hardware/client-venv/bin/python \
  examples/bring_your_own_robot/hiwonder_so101_client.py \
  --single-action \
  --vendor-root "$VENDOR_CHECKOUT" \
  --follower-port "$FOLLOWER_PORT" \
  --calibration-id "$CALIBRATION_ID" \
  --robot-serial "$FOLLOWER_SERIAL" \
  --wrist-camera "$WRIST_CAMERA_INDEX" \
  --fixed-camera "$FIXED_CAMERA_INDEX" \
  --checkpoint "$SMOLVLA_CHECKPOINT" \
  --task "$FIRST_TASK" \
  --server-address 127.0.0.1:8080 \
  --hardware-safety-profile "$HARDWARE_SAFETY_PROFILE" \
  --telemetry "$CLIENT_TELEMETRY"
```

The client applies one enveloped action, holds, returns to the recorded start
pose through fresh-readback steps of at most one public unit with a 1000 ms
dwell per step, verifies torque off, and exits. The same one-unit/2%-span
limiter and 200 ms dwell floor bound outbound actions. Do not invoke
`--continuous` until the single-action evidence has been reviewed and
accepted.

## What to observe

During the one-action episode, call out and later record:

- whether the observed motion matched the intended joint direction;
- approximate maximum displacement and speed, including any overshoot or
  oscillation;
- gripper direction and force behavior;
- camera freshness and whether either stream froze or changed ordering;
- server/client warnings, dropped or stale observations, and action-queue
  behavior;
- the three logged latency fields and whether client-wall-clock latency was
  nonnegative on this same-machine run;
- whether rollback was invoked and why.

Neither JSONL contains observation, image, task, state, position, target, or
action payload values. Server telemetry records timestamps, latency, action
count, and policy identity. Client telemetry records mode/caps, cadence,
observation-to-chunk and observation-to-first-write summaries, aggregate clip/
rate-limit/rejection reasons, watchdog counts, and the stop reason. The logger
makes no pass/fail assertion about physical behavior.

## Rollback

For unexpected motion, press the physical power switch immediately. Then stop
the client with Ctrl-C if it is still running, kill the server, then power off
the arm and cameras. Do not reconnect, recalibrate, raise limits, or
retry until the logs and physical setup have been reviewed.

For a normal one-action stop, Ctrl-C the client, verify torque-disabled
disconnect, Ctrl-C the server, then power off. Preserve all files unchanged.

## Evidence attachment checklist

- [ ] Exact repository commit and local checkpoint path plus checkpoint
      `model.safetensors` SHA-256.
- [ ] `.cache/hardware/first-contact-latency.jsonl` (**latency JSONL**).
- [ ] `.cache/hardware/first-contact-server.log` (**server log**).
- [ ] `.cache/hardware/first-contact-*-client.jsonl` (**client telemetry**).
- [ ] Written one-action **observed motion** notes: direction, approximate
      displacement/speed, gripper behavior, camera freshness, and anomalies.
- [ ] Whether emergency or normal **rollback** was used, with the exact reason
      and power-off confirmation.
- [ ] Mac model/macOS/Python/MLX versions and client LeRobot version.
- [ ] Operator name, second observer if present, session start/end timestamps,
      and the in-session `ARM SESSION CONFIRMED` message.

Keep this evidence local for review. This runbook authorizes no upload and no
claim of hardware compatibility until the supervised result is reviewed.
