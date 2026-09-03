# Hiwonder SO-101 preflight evidence

Date: 2026-09-02; follow-up: 2026-09-03

Host: Apple M5 Pro MacBook Pro (`Mac17,8`), 48 GB, macOS 26.6.2

Runtime: Python 3.12.13, MLX 0.32.2, `Device(gpu, 0)`

Hardware-client source: initial protocol
`404467190aeceea91f58cb98076148ab1aa0c0df`; dual-camera follow-up
`fbd34ed9f1bd3da095c5c7ee3bdc15d4f2bf795c`

## Scope and result

The operator supplied `ARM SESSION CONFIRMED` in the live task. This authorized
the follower-only read path, both cameras, and the graduated protocol. Serial
identifiers and private checkout paths remain only in ignored local telemetry;
they are redacted from this public report.

Read-only serial/calibration preflight and two 60-second no-motion loops
completed. No torque-enable or goal-position write was issued. Motion remains
blocked because the camera views are not operationally framed, the start pose
is outside the tightened envelope, and no operator-verified low hardware-limit
profile exists.

## Serial and calibration

- Two BusLinker serial devices were enumerated. The follower was identified by
  matching its USB serial to the existing follower calibration and reading all
  six present positions. The separate leader device was never opened.
- Existing calibration ID: `hiwonder_follower`.
- The loaded calibration covered exactly the six expected joints and matched
  the controller's read-back calibration.
- `Torque_Enable` read `0` for all six joints before the protocol runs and
  again at `2026-09-02T18:01:09Z` after them.
- No vendor `connect()`, `configure()`, `calibrate()`, or `send_action()` path
  was called. The repository client used `bus.connect(handshake=True)`, reads,
  camera connects, and `bus.disconnect(disable_torque=False)` only.

| Joint | Raw calibration | Public range | 10%-inset motion range | Present | Start gate |
| --- | ---: | ---: | ---: | ---: | --- |
| shoulder_pan | 1364–2618 | −55.121°–55.121° | −44.097°–44.097° | 6.857° | pass |
| shoulder_lift | 369–2753 | −104.791°–104.791° | −83.833°–83.833° | −102.066° | **fail** |
| elbow_flex | 1343–3538 | −96.484°–96.484° | −77.187°–77.187° | 95.868° | **fail** |
| wrist_flex | 841–3134 | −100.791°–100.791° | −80.633°–80.633° | 74.242° | pass |
| wrist_roll | 0–4095 | −180.000°–180.000° | −144.000°–144.000° | −2.066° | pass |
| gripper | 1698–3371 | 0–100 | 10–90 | 14.824 | pass |

The lift and elbow positions make the motion start gate fail by construction.
They must be moved manually while torque is disabled and then re-read before a
single-action run.

## Hardware-limit readback

These are observations, not approved settings and not a safety profile.

| Register | Five body joints | Gripper | Verdict |
| --- | ---: | ---: | --- |
| `Max_Torque_Limit` | 1000 | 500 | not operator-attested low |
| `Torque_Limit` | 1000 | 500 | not operator-attested low |
| `Protection_Current` | 3000 | 250 | not operator-attested low |
| `Overload_Torque` | 80 | 25 | not operator-attested low |
| `Acceleration` | 0 | 0 | meaning not assumed |
| `Goal_Velocity` | 0 | 0 | meaning not assumed |
| `Moving_Velocity` | 1 | 1 | read-only status/config observation |
| `Maximum_Velocity_Limit` | 65 | 65 | not operator-attested low |
| `Maximum_Acceleration` | 254 | 254 | maximum/default, not low |

The project deliberately does not invent safe values. Before motion, an
operator must establish low values using a known-good Hiwonder procedure and
provide a JSON profile containing the exact expected readback. The client
checks all nine registers and torque-off before it can enable torque.

## Cameras

Both cameras were captured concurrently for five seconds at 640×480.

| Index / role | Frames | Elapsed | Sustained FPS | Nonblack | Visual verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 / wrist → `camera1` | 43 | 5.0145 s | 8.575 | 43/43 | live, but obstructed/too close to identify the workspace |
| 2 / fixed → `camera2` | 150 | 5.0054 s | 29.968 | 150/150 | live, but aimed at the room rather than the robot workspace |

The checkpoint's third slot, `observation.images.camera3`, is supplied by the
existing empty-camera padding path. Captured frames were reviewed locally and
were not committed because one contains a bystander.

## 2026-09-03 follow-up

The operator reported another successful manual teleoperation test and
adjusted the cameras, then requested a complete retry. The setup record already
confirmed earlier 60 Hz owner-run teleoperation with all joints and the gripper
working; the follow-up therefore treated device health separately from the
stricter autonomous-motion gates.

- The follower resolved to its existing role, all six motor IDs responded at
  12.2-12.6 V with no status alarm, and all six torque bits read zero before
  and after access. The leader was not opened.
- A reproducible camera-order defect was isolated: starting the wrist stream
  first caused the fixed stream to report 15 FPS instead of the requested 30.
  Starting the fixed stream first allowed both cameras to accept 640x480 at
  30 FPS. Commit `fbd34ed` implements only that connection-order change and
  adds a regression test; camera keys and model feature ordering are unchanged.
- Through the corrected repository client, both fresh 640x480 frames were
  finite and nonblack. Visual review still failed the motion gate: the wrist
  view was heavily blurred and too close to identify the task workspace, while
  the fixed view showed the room but not the complete robot workspace. The
  private frames remain ignored and untracked.
- The post-teleoperation pose remained outside the inset start envelope:

| Joint | Present | 10%-inset motion range | Start gate |
| --- | ---: | ---: | --- |
| shoulder_pan | 6.593 deg | -44.097 deg-44.097 deg | pass |
| shoulder_lift | -103.912 deg | -83.833 deg-83.833 deg | **fail** |
| elbow_flex | 95.780 deg | -77.187 deg-77.187 deg | **fail** |
| wrist_flex | 49.187 deg | -80.633 deg-80.633 deg | pass |
| wrist_roll | -1.802 deg | -144.000 deg-144.000 deg | pass |
| gripper | 15.302 | 10-90 | pass |

- Manual teleoperation restored `Acceleration=254` on all six controllers;
  the other recorded maximum/current/torque values remained non-low defaults.
  This reinforces rather than clears the operator-attested low-profile gate.
- The corrected client completed another stats-active 60-second no-motion run:

| Result | Observations / chunks | Camera sample FPS | Observation-to-chunk median / p95 | Clamps | Rate limits | Held invalid chunks | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| duration cap | 294 / 293 | 4.895 | 152.080 / 154.139 ms | 639 | 1,443 | 1 | 0 |

Server receive-to-chunk latency was 149.774 ms median / 151.755 ms p95;
inference was 149.379 / 151.239 ms. No motor or torque write occurred. The
ignored client and server JSONL SHA-256 values are respectively
`4804d38aca9c02d85a726bb7a2314f59c8370f1c9ad300fb5d153b6c00f13117`
and `707fc290ace9cf4788f5e5001c9b86ec7a46a326c1b09c793063a0b0660b780d`.

## Server and model mapping

- The native server listened only on `127.0.0.1:8080` and reported MLX GPU as
  the default device.
- After the no-motion runs, fresh ignored Python 3.12 environments were created
  independently for `.[serve]` and `.[hardware]`. PyAV is absent from the
  server environment, its CLI import completed, and a loopback-only server
  startup/shutdown emitted no duplicate AVFoundation-class warning. This
  resolves the software-environment anomaly seen in the earlier all-extras
  run; future hardware work must keep using the separated environments.
- Physical wrist camera → `observation.images.camera1`.
- Physical fixed camera → `observation.images.camera2`.
- `observation.images.camera3` → empty-camera padding.
- Raw `lerobot/smolvla_base` has ineffective physical state/action statistics
  because its saved keys do not bind to `observation.state` and `action`. It is
  allowed only for no-motion diagnosis.
- `reference/artifacts/stats-active-base` uses the same base-model weights
  (SHA-256 `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`)
  plus audited SO-101 state/action statistics (SHA-256
  `06f188b71fa32351be661210758e40042dce3b3a0fcbf9ae89a1b7e4b1bdf797`).
  It is the only local base artifact considered eligible for a future motion
  attempt, after all physical gates clear.

## Preserved local evidence

The ignored `.cache/hardware/` directory contains the no-clobber client and
server JSONL. It contains no image, state, task, position, target, or action
payload.

| File | SHA-256 |
| --- | --- |
| `stage-a-no-motion-v3-client.jsonl` | `df7d59cb45f7023193d7973b38e7bf23813b25cfadea933b9c5addcf48c6ad67` |
| `stage-a-no-motion-v3-server.jsonl` | `e8daa487f27243b20c2171e609d5039e70b74560a03b791eef409c13bb7515a0` |
| `stage-a-stats-no-motion-client.jsonl` | `15e6aa28dd9e7a617aef8605e73a71a43cf3ff9f6dd326de127870bf6d6c14e3` |
| `stage-a-stats-no-motion-server.jsonl` | `6ccbbded994ef9196ebb6d5eae957f980e11a2cbd699fcdfd3d1eb35b988f2be` |

The vendor checkout remained at commit
`a24998f7ba3c77ea445b48c92ad15c14a50e492a`. Its pre-existing dirty files were
not modified. The composite of sorted SHA-256 lines for all 696 tracked files
was identical before and after access:
`d280efa881ab9e412cd071bbef38d8d9ec5050e484b49b5a2397df2a39bdb764`.

## Hard blockers before motion

1. Re-aim and secure both cameras again: the adjusted wrist view is nonblack
   but remains blurred/too close, and the adjusted fixed view still omits the
   robot workspace. Repeat visual review through the corrected dual-camera
   client.
2. With torque disabled, manually place the lift and elbow inside their
   10%-inset ranges, preferably near the calibrated neutral pose; repeat the
   position readback.
3. Establish and attest low controller torque/current/velocity/acceleration
   limits using an operator-known procedure; capture their exact readback in a
   safety-profile JSON file.
4. Clear the motion envelope, secure the base, and keep a hand on the physical
   power switch for the entire single-action attempt.
