# SO-101 first-contact status

**Status: bounded physical integration passed on 2026-09-04. The final
60-second no-motion check, one valid single action, and a two-chunk continuous
run passed with verified torque-off shutdown. A separate 20-chunk attempt
is inconclusive under reduced torque; its exact return-to-start requirement
was not met under the temporary low-torque profile.**

The operator supplied `ARM SESSION CONFIRMED` in the live task on 2026-09-02.
The follower-only serial path, both cameras, native MLX server, and four
60-second no-motion loops were exercised across the original and 2026-09-03
follow-up sessions. During those first four runs, no torque-enable or
goal-position write occurred, and all six torque bits read zero afterward. The
leader device was detected but never opened.

The later 2026-09-04 powered session adds limited physical-actuation evidence.
It proves the guarded client can execute one valid bounded action and a short
continuous stage, return exactly for those accepted runs, and disable torque.
It does not prove reliable task completion or sustained 20-chunk operation.
Sections below are chronological; the final verdict supersedes earlier pending
status recorded before motion.

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

## Historical anomalies and pre-motion blockers

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

## 2026-09-04 camera-adjustment follow-up

At source `3b726505c72e2e6c1c0c41bd770abac79d307a3f`, new images after the
operator's first adjustment showed fixed 0 including the desk and part of the
arm, but blurred and without a verified complete workspace. Wrist 1 still
faced sideways toward the operator. After a further wrist adjustment, a fresh
640x480 capture at `2026-09-04T14:52:50Z` shows the gripper, yellow ball, and
tabletop: wrist task-surface framing is corrected. An operator hand remains
near the gripper, so this does not attest a cleared motion envelope. The fixed
view still needs widening/focus; concrete positioning guidance was provided.
Built-in 2 and candidate 3's near-dark non-workspace image remain excluded.

Read-only follower access at `2026-09-04T14:48:09Z` matched the existing
calibration, passed every numeric inset check, and measured lift/elbow at
-73.055/39.868 degrees. All torque bits stayed zero and the port closed with
zero motor/torque writes. The nine controller readbacks are unchanged. The
operator reports no known approved low-limit profile; calibration does not
establish one. No concurrent capture-rate gate, fresh no-motion policy loop,
or physical actuation ran. All 696 vendor tracked files and 32 operator-wrapper
files remain unchanged.

Private fresh evidence and hashes are recorded in [PREFLIGHT.md](PREFLIGHT.md).

## 2026-09-04 fixed-camera framing and concurrent capture

After the operator adjusted the fixed camera using a live QuickTime preview,
new camera-only captures at source `cad455604299b36eb8c9a2883c5598bb58aa46a6`
show the follower arm and yellow-ball tabletop area in improved focus. The
wrist view shows the gripper and nearby ball. Current visual framing is
resolved. A loose cable crosses the tabletop and a mouse remains in the working
area; this does not attest physical workspace clearance.

Both cameras streamed together at 640x480 without timeouts. Independent
consumers measured fixed 20.10 FPS and wrist 6.34 FPS over approximately eight
seconds, with maximum read gaps of 66.98/165.93 ms. Both exceeded the intended
5 FPS control rate in this short camera-only probe, but neither achieved the
requested 30 FPS. The fresh 60-second robot/server no-motion loop remains
pending. Both camera handles closed; neither robot nor vendor checkout was
accessed, and no motor command ran. Approved low limits and physical
attestation remain open. Details and evidence hashes are in
[PREFLIGHT.md](PREFLIGHT.md).

## 2026-09-04 delegated commissioning and fresh no-motion checks

The operator explicitly delegated setup and client execution, amending the
old runbook authority wording. At source `06d366ff9f2336e3be1d2858d528237b38b8f3ac`,
the unchanged client completed two more 60-second stats-active no-motion runs
using fixed 0/wrist 1 and the separate server/client environments:

| Attempt | Observations / processed chunks | Duration | Sampled FPS | Observation→chunk median / p95 | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold server | 292 / 291 | 60.031 s | 4.864 | 161.235 / 163.926 ms | 1 |
| Warm repeat | 295 / 294 | 60.088 s | 4.909 | 165.462 / 168.631 ms | 0 |

The cold attempt missed the first action deadline and does not pass the
zero-timeout gate. The warm repeat passes it without changing the watchdog or
client. Both runs rejected zero chunks; the warm repeat counted 437 clipped
joint values and 1,358 rate-limited values. Its final server chunk arrived at
the duration boundary and was not processed. All motor writes were suppressed.

At 16:52 UTC, a fresh read matched calibration and found model 777, firmware
3.13, position mode 0, no status alarms, and six torque bits off. The elbow
now measured 86.110°, outside its 77.187° upper inset start margin. Fresh images
show the follower folded near the monitor and a cable near the gripper. This
is a new physical setup issue; it does not invalidate the earlier passing poses.

Hiwonder's own pinned source confirms the HX-30HM register map. Under the
delegated setup authority, three temporary SRAM controls were staged and
exactly read back on all six motors: acceleration 1, speed 56, and torque limit
100. These represent 100 encoder steps/s², about 4.9°/s at the motor shaft, and
10% of the documented torque scale. Startup force 32, position mode 0, the
remaining profile controls, and raw present/goal positions were unchanged.
Torque stayed off; neither a goal-position nor torque-enable write occurred.
These are commissioning settings, not a demonstrated gravity-hold result or a
universal safe preset. The private profile must be re-established after a reset
and must not be raised automatically.

The client has also been hardened to reread the complete controller profile
after raw-goal preload, require integer position mode 0, and reject startup
force above the torque cap or changed across preload. Failure-first tests
cover the previously unguarded cases; focused hardware/readiness checks pass
104/104. Full software verification is recorded in the continuation progress.
The supported inset pose, workspace/base/power checks, first action, and bounded
continuous stage remain open. Device handles and the MLX server are closed.
Detailed source links and private evidence hashes are in [PREFLIGHT.md](PREFLIGHT.md).

## 2026-09-04 17:31 UTC — adjusted position verified

At the operator's request, a fresh read-only check found every joint inside
the inset envelope, with lift/elbow at -54.330/33.187 degrees and zero measured
drift over two seconds. The 16:52 elbow failure is resolved. Existing
calibration and the exact reduced controller profile still match; mode 0,
startup force 32, no alarms, and six torque-off bits were verified.

The fixed view contains the raised follower and tabletop. The wrist view now
points upward at the operator/ceiling, so its mount needs to face the gripper
and task surface before the final inference/motion checks. No motor write was
attempted and all handles closed. A passing numeric pose does not establish
workspace/base/power readiness. Evidence hashes are in [PREFLIGHT.md](PREFLIGHT.md).

## 2026-09-04 final powered validation

Fresh follower identity, existing calibration, controller profile, status,
pose, and camera checks passed before motion. The temporary SRAM profile was
acceleration 1, goal velocity 56, and torque limit 100 on all six controllers;
mode was 0, startup force was 32, status had no alarms, and all torque bits were
zero. The fixed view contained the complete arm and work area, and the wrist
view faced the gripper and table. The start pose was inside every 10%-inset
bound and showed zero drift over two seconds.

The first arming attempt exposed a controller-specific behavior. Writing each
current raw `Present_Position` back to `Goal_Position` changed all six torque
bits from zero to one in about 12 ms without an explicit enable call. The old
guard rejected that state and verified all-six torque-off cleanup; no policy
action ran and there was no displacement. An offline raw audit confirmed that
all six present values were within the controller's raw minimum and maximum.
The client was then changed to check those limits before its first write and to
support two explicit arming modes. The tested unit used `goal-write`, which
requires the observed all-six enabled state and never enables twice. The
portable `explicit-torque` default still requires torque to remain off after
preload before one explicit enable. Every post-write error follows the verified
disable path.

The final no-motion run passed for 60.002 seconds with 295 observations and 295
processed chunks at 4.917 sampled FPS. Observation-to-chunk median/p95 was
167.734/173.328 ms, with zero timeouts and zero writes. It recorded 29 rejected
hold chunks, 404 clipped values, and 1,499 rate-limited values. The first two
single-action trials then held their rejected timestep-zero chunks and returned
without displacement. That finding led to the bounded single-action rule: wait
through rejected holds, stop after the first valid non-hold action, and retain
the 20-chunk hard attempt cap.

The successful single-action run processed two chunks: one rejected hold and
one valid action. Its commanded per-joint delta was
`(+1, -1, -1, -1, +1, -1)` public units, within the one-unit limit. It stopped
with `action_limit`, returned to the exact recorded start, showed zero
post-run drift, and read all six torque bits as zero. There were no timeouts.

A subsequent 20-chunk continuous attempt moved the arm gradually toward the
task object. Its exact return requirement was not met: the 20-step cleanup cap
was exhausted while several gravity-loaded joints stopped following one-degree
return targets under torque limit 100. This run exited nonzero and is retained
as inconclusive under reduced torque for sustained operation. The observed
return problem is consistent with insufficient gravity-holding torque, but
the retained logs do not isolate the cause or establish that all chunks were
valid and no serving or policy fault occurred. Cleanup still disabled all six
torque bits, and an independent check found the stopped pose inside every inset bound,
with zero drift, matching calibration/profile readbacks, clear camera views,
and no status alarm. The controller limits were not raised.

The accepted bounded-continuous stage used a two-chunk ceiling from that safe
pose. It processed one rejected current-position hold followed by one valid
one-unit action, stopped at `chunk_limit`, returned exactly, and passed an
independent final pose/profile/calibration/status check with zero two-second
drift and all six torque bits off. It exited zero in 6.514 seconds with no
timeouts, one clipped value, and six rate-limited values. The owned loopback
server was stopped and its port was closed after validation. Exact private
record hashes are in [`PREFLIGHT.md`](PREFLIGHT.md).

## Current verdict

The claim “executes a guarded single action and a short bounded-continuous run
on a connected SO-101 from a MacBook, with exact torque-off shutdown” is
evidenced. The evidence is limited to one valid action and two continuous
chunks under the temporary 10% torque profile. The 20-chunk attempt is
inconclusive under reduced torque, and its exact return requirement remains
unmet. Sustained 20-chunk operation and reliable
task completion are not evidenced. A nominal-profile rerun remains pending;
it does not supersede this attempt or its recorded outcome.
Future powered sessions require fresh authorization, physical preflight, and
profile readback; the client must never raise the torque limit automatically.
