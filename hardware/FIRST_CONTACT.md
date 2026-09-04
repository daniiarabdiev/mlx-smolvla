# SO-101 first-contact status

**Status: no-motion protocol complete; physical motion blocked.**

The operator supplied `ARM SESSION CONFIRMED` in the live task on 2026-09-02.
The follower-only serial path, both cameras, native MLX server, and four
60-second no-motion loops were exercised across the original and 2026-09-03
follow-up sessions. No torque-enable or goal-position write occurred, and all
six torque bits read zero after the runs. The leader device was detected but
never opened.

This is real-hardware protocol evidence, but it is not evidence that the
project drives a physical SO-101: neither the single-action nor bounded-
continuous stage ran. The precise physical blockers are in
[`PREFLIGHT.md`](PREFLIGHT.md).

## Connected surface

- Apple M5 Pro, 48 GB, macOS 26.6.2, Python 3.12.13, MLX 0.32.2 on
  `Device(gpu, 0)`.
- One Hiwonder SO-101 follower with six calibrated HX-30HM motors; exact USB
  serial retained only in ignored local evidence.
- In the corrected current enumeration, camera index 1 mapped from the wrist
  to `observation.images.camera1`.
- Camera index 0 mapped from the fixed view to
  `observation.images.camera2`.
- Camera index 2 was the built-in Mac camera and was excluded.
- Checkpoint slot `observation.images.camera3` supplied by empty-camera
  padding.

## No-motion results

All four runs used dense bfloat16 production inference at a requested 5 Hz, one
action per chunk, a 500 ms action watchdog, and a fixed 60-second client cap.

| Checkpoint surface | Result | Observations / processed chunks | Camera sample FPS | Observation→chunk median / p95 | Clamps | Rate limits | Held invalid chunks | Timeouts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw `lerobot/smolvla_base` | duration cap | 295 / 295 | 4.914 | 149.313 / 151.139 ms | 14 | 84 | 281 | 0 |
| same weights + audited SO-101 stats | duration cap | 295 / 294 | 4.915 | 149.746 / 152.502 ms | 785 | 1,318 | 12 | 0 |
| stats-active after dual-camera startup fix | duration cap | 294 / 293 | 4.895 | 152.080 / 154.139 ms | 639 | 1,443 | 1 | 0 |
| stats-active after camera-identity correction | duration cap | 293 / 292 | 4.876 | 148.907 / 150.512 ms | 758 | 1,426 | 15 | 0 |

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

The first 2026-09-03 follow-up incorrectly treated camera index 2 as the fixed
UVC view; it was the built-in Mac camera. After correcting the roles to fixed
index 0 and wrist index 1, both camera startup orders passed three of three
trials at 640x480/30. The unchanged repository client completed the fourth
no-motion run. Its server receive-to-chunk median/p95 was 146.642/148.338 ms
and inference median/p95 was 146.239/147.978 ms. Numeric indices remain
session-local and require visual validation after device changes.

## Anomalies and blockers

- Both intended UVC streams open together and return nonblack frames. The
  fixed view contains the task surface; the wrist view is an unobstructed,
  close desk view that is soft at the parked camera's focus distance. The
  operator confirmed the framing, so the camera blocker is closed.
- After the failed post-teleoperation read, the operator manually moved and
  mechanically supported the torque-free arm. The latest lift/elbow read was
  -20.396/62.989 degrees, and every joint passed the 10%-inset start envelope.
  All six torque bits remained zero.
- Controller torque/current/velocity/acceleration readbacks have not been
  established as low by an operator-known procedure; no accepted safety
  profile exists. The same read found `Acceleration=254` and
  `Maximum_Acceleration=254` on all six controllers.
- No workspace-clear/base-secure/hand-on-power checklist was recorded for a
  motion attempt.

This explains why `robot teleops` can work while MLX motion remains blocked:
the owner command deliberately enters the vendor configuration/torque/write
path and streams leader targets, whereas the MLX path has additional
start-envelope and exact-profile gates before its first torque-enable. The
read-only recheck confirmed both cameras and all-zero torque, so this is not a
camera enumeration or dead-motor failure.

The supported-pose read also found that the retained shoulder-lift and elbow
goals differed from present position by 84.396 and 33.495 degrees. No torque
was enabled. Failure-first software hardening now preloads and verifies raw
present positions as goals while torque is off, then rechecks torque-off before
enable. Return-to-start now uses bounded one-unit readback steps rather than a
single direct command. The new behavior passes 96 hardware/readiness tests,
482 fast tests, and all 773 tests, but remains unexercised on moving hardware
until the low-profile and physical gates below are satisfied.

The server-environment anomaly was closed after these runs: a fresh `.[serve]`
environment contained no PyAV and its loopback-only startup/shutdown emitted
no duplicate AVFoundation-class warning. Motion work must continue to use that
separate environment rather than the all-extras development environment.

## 2026-09-04 reconnection preflight

The operator supplied fresh live `ARM SESSION CONFIRMED` at source
`c228095285b33121c52c624d80127d238a4bb584`. The follower identity and existing
calibration matched. At `2026-09-04T14:22:04Z`, all six torque bits were zero
before and after read-only access, every joint was inside the 10%-inset
envelope, and lift/elbow measured -73.143/39.868 degrees. Mechanical support
and the final physical checklist still require operator attestation.

Fresh camera discovery identifies fixed index 0 and wrist index 1; index 2 is
the built-in camera and is excluded. Both intended cameras returned images,
but the current wrist view points toward the operator and the fixed view does
not show the arm's working area. This session's framing needs adjustment; the
earlier camera-identity and framing successes remain historical facts. A fourth
camera candidate returned no usable frame and is excluded without a retry.

The controller readbacks still match the earlier unapproved settings, including
`Acceleration=254` and `Maximum_Acceleration=254` on every motor. No approved
low-limit profile was supplied. No concurrent capture measurement, fresh
60-second no-motion loop, torque-enable, or physical action ran in this session.
The four earlier no-motion results are unchanged.

All 696 vendor tracked files and 32 operator-wrapper files remained byte
identical. The reviewed stats-active checkpoint passes its six-axis check and
retains its recorded model hash. Private frames, serials, and raw readbacks
remain under ignored `.cache/hardware/session-20260904T141839Z-b560dkhe/`.

## Current verdict

The claim “exchanges live camera/state observations and MLX action chunks with
a connected SO-101 on a MacBook, with motor writes suppressed” is evidenced.
The broader claim “drives a real SO-101 from a MacBook” is **not** evidenced and
must not be published. Resume only with the single-action gate after every
blocker above is cleared and a fresh no-motion check passes; bounded continuous
remains gated on the reviewed single-action result.
