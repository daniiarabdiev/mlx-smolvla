# SO-101 first-contact status

**Status: no-motion protocol complete; physical motion blocked.**

The operator supplied `ARM SESSION CONFIRMED` in the live task on 2026-09-02.
The follower-only serial path, both cameras, native MLX server, and two
60-second no-motion loops were exercised. No torque-enable or goal-position
write occurred, and all six torque bits read zero after the runs. The leader
device was detected but never opened.

This is real-hardware protocol evidence, but it is not evidence that the
project drives a physical SO-101: neither the single-action nor bounded-
continuous stage ran. The precise physical blockers are in
[`PREFLIGHT.md`](PREFLIGHT.md).

## Connected surface

- Apple M5 Pro, 48 GB, macOS 26.6.2, Python 3.12.13, MLX 0.32.2 on
  `Device(gpu, 0)`.
- One Hiwonder SO-101 follower with six calibrated HX-30HM motors; exact USB
  serial retained only in ignored local evidence.
- Camera index 1 mapped from the wrist to `observation.images.camera1`.
- Camera index 2 mapped from the fixed view to
  `observation.images.camera2`.
- Checkpoint slot `observation.images.camera3` supplied by empty-camera
  padding.

## No-motion results

Both runs used dense bfloat16 production inference at a requested 5 Hz, one
action per chunk, a 500 ms action watchdog, and a fixed 60-second client cap.

| Checkpoint surface | Result | Observations / processed chunks | Camera sample FPS | Observation→chunk median / p95 | Clamps | Rate limits | Held invalid chunks | Timeouts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw `lerobot/smolvla_base` | duration cap | 295 / 295 | 4.914 | 149.313 / 151.139 ms | 14 | 84 | 281 | 0 |
| same weights + audited SO-101 stats | duration cap | 295 / 294 | 4.915 | 149.746 / 152.502 ms | 785 | 1,318 | 12 | 0 |

The raw base checkpoint's high rejection count is explained by its ineffective
saved physical statistics: its stored keys do not bind to
`observation.state`/`action`, and known base golden gripper outputs are negative
normalized values rather than 0–100 driver values. The client now refuses that
checkpoint in either motion mode.

The stats-active artifact uses the identical base weights plus pinned SO-101
state/action statistics. Twelve of 294 processed chunks still crossed the full
driver domain and were held; all other chunks were simulated through the
calibration clamp and current-position rate limiter. The extra final
observation arrived at the duration boundary and was discarded without error
or write, exercising the boundary-race regression.

Server-side stats-active medians were 147.563 ms receive-to-chunk and 147.212 ms
inference; p95 values were 150.086 ms and 149.713 ms. Client and server logs
are private, no-clobber JSONL under ignored `.cache/hardware/`; hashes are in
[`PREFLIGHT.md`](PREFLIGHT.md).

## Anomalies and blockers

- Both camera streams are live, but the wrist image is obstructed/too close and
  the fixed camera does not show the robot workspace.
- `shoulder_lift` and `elbow_flex` are near calibrated endpoints and outside
  the required 10%-inset start envelope.
- Controller torque/current/velocity/acceleration readbacks have not been
  established as low by an operator-known procedure; no accepted safety
  profile exists.
- No workspace-clear/base-secure/hand-on-power checklist was recorded for a
  motion attempt.

The server-environment anomaly was closed after these runs: a fresh `.[serve]`
environment contained no PyAV and its loopback-only startup/shutdown emitted
no duplicate AVFoundation-class warning. Motion work must continue to use that
separate environment rather than the all-extras development environment.

## Verdict

The claim “exchanges live camera/state observations and MLX action chunks with
a connected SO-101 on a MacBook, with motor writes suppressed” is evidenced.
The broader claim “drives a real SO-101 from a MacBook” is **not** evidenced and
must not be published. Resume only with the single-action gate after every
blocker above is cleared; bounded continuous remains gated on that result.
