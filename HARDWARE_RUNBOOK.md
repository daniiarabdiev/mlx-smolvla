# Supervised SO-101 first-contact runbook

**Hardware validation status: NOT RUN.** This document and its telemetry path
were prepared with software-only loopback tests. No robot directory, serial
port, camera, motor, or physical arm was accessed.

## Execution gate

Stop unless the operator is physically present and has typed this exact line
in the current interactive agent session:

```text
ARM SESSION CONFIRMED
```

Absent that exact line in the live session, this runbook is documentation only.
The same words in this file, a commit, earlier chat context, or a shell log do
not satisfy the gate. The agent must never execute the client commands below or
inspect `~/robot/so101`; they belong to the operator.

## Roles and stop authority

- The operator owns the robot-side environment, calibration, port, cameras,
  motor limits, power, and every client command.
- A second person should watch the entire motion envelope if available.
- The physical power switch is the only assumed e-stop. Keep one hand on the
  power switch from before client startup until the arm is stationary and
  torque is disabled. A keyboard interrupt is not an e-stop.
- Any person may call stop. Unexpected direction, speed, force, sound, heat,
  oscillation, camera loss, stale actions, or uncertainty means immediate power
  cut—not another trial.

## Physical preflight

Check every box before opening the client:

- [ ] Clear the full arm and gripper workspace; remove people, cables, loose
      objects, pinch hazards, and hard stops from the motion envelope.
- [ ] Secure the base, start near the already calibrated neutral pose, and
      confirm the existing calibration ID. Do not recalibrate during first
      contact.
- [ ] Verify the controller's established **torque and speed limits** are set
      low using the operator's known-good hardware procedure. LeRobot 0.6.1's
      client does not expose a global torque/current or profile-speed limit;
      do not invent register values. If those limits cannot be verified, stop.
- [ ] Confirm that cutting the physical power switch immediately removes motor
      power, and keep the operator's hand on the power switch.
- [ ] Use only a local checkpoint whose provenance and intended task have
      already been reviewed. Do not substitute a newly downloaded or unknown
      checkpoint at the arm.
- [ ] Agree on one low-risk task and one commanded action as the complete first
      episode. No unattended or repeated episode is permitted.

The client command below adds a **1.0-degree maximum relative target** per
joint, requests one action per chunk, runs at 5 fps, and disables torque on
disconnect. That software cap supplements—it does not replace—the verified
low hardware torque and speed limits.

## Terminal A — native MLX server on this Mac

From the checked-out `smolvla_mlx` repository, verify that both log names are
new, then start the dense-bf16 default on loopback. Do not use a quantized
preset for first contact.

```bash
uv sync --extra serve --frozen
test ! -e .cache/hardware/first-contact-latency.jsonl
test ! -e .cache/hardware/first-contact-server.log
set -o pipefail
uv run --extra serve python scripts/serve_latency_smoke.py \
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

## Terminal B — operator-owned LeRobot client

Only the physically present operator runs this from their existing,
known-working `~/robot/so101` environment. Replace the four values once; the
guards make empty placeholders fail before `RobotClient` connects.

```bash
cd ~/robot/so101
source .venv/bin/activate
export FOLLOWER_PORT='<YOUR_VERIFIED_FOLLOWER_PORT>'
export CALIBRATION_ID='<YOUR_EXISTING_CALIBRATION_ID>'
export LEROBOT_CAMERA_CONFIG='<YOUR_EXISTING_LEROBOT_CAMERA_CONFIG>'
export SMOLVLA_CHECKPOINT='<ABSOLUTE_PATH_TO_REVIEWED_LOCAL_CHECKPOINT>'
export FIRST_TASK='<ONE_VALIDATED_LOW_RISK_TASK>'
export CLIENT_LOG='logs/smolvla-first-contact-client.log'
test -n "$FOLLOWER_PORT"
test -n "$CALIBRATION_ID"
test -n "$LEROBOT_CAMERA_CONFIG"
test -n "$SMOLVLA_CHECKPOINT"
test -n "$FIRST_TASK"
test ! -e "$CLIENT_LOG"
set -o pipefail
python -m lerobot.async_inference.robot_client \
  --policy_type=smolvla \
  --pretrained_name_or_path="$SMOLVLA_CHECKPOINT" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$CALIBRATION_ID" \
  --robot.cameras="$LEROBOT_CAMERA_CONFIG" \
  --robot.use_degrees=true \
  --robot.max_relative_target=1.0 \
  --robot.disable_torque_on_disconnect=true \
  --actions_per_chunk=1 \
  --chunk_size_threshold=0.0 \
  --task="$FIRST_TASK" \
  --server_address=127.0.0.1:8080 \
  --policy_device=cpu \
  --client_device=cpu \
  --fps=5 \
  2>&1 | tee "$CLIENT_LOG"
```

Watch for the first commanded action only. Immediately press Ctrl-C in the
client after that single short horizon, even if motion looked correct. Confirm
the client disconnected with torque disabled before moving a hand away from
the power switch. Do not extend the run during first contact.

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

The latency JSONL contains no observation payload or action values. For each
successfully returned chunk it records the client observation timestamp,
timestep, server receipt/chunk-ready UTC times, monotonic server-receive-to-
chunk latency, inference latency, action count, and policy configuration. The
client-wall-clock value is meaningful only when client and server clocks are
synchronized; the monotonic server value is the robust local measure. The
logger makes no pass/fail assertion about the robot.

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
- [ ] `$CLIENT_LOG` from the operator environment (**client log**).
- [ ] Written one-action **observed motion** notes: direction, approximate
      displacement/speed, gripper behavior, camera freshness, and anomalies.
- [ ] Whether emergency or normal **rollback** was used, with the exact reason
      and power-off confirmation.
- [ ] Mac model/macOS/Python/MLX versions and client LeRobot version.
- [ ] Operator name, second observer if present, session start/end timestamps,
      and the in-session `ARM SESSION CONFIRMED` message.

Keep this evidence local for review. This runbook authorizes no upload and no
claim of hardware compatibility until the supervised result is reviewed.
