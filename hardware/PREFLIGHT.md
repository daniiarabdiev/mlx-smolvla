# Hiwonder SO-101 preflight evidence

Date: 2026-09-02; follow-ups: 2026-09-03 and 2026-09-04

Host: Apple M5 Pro MacBook Pro (`Mac17,8`), 48 GB, macOS 26.6.2

Runtime: Python 3.12.13, MLX 0.32.2, `Device(gpu, 0)`

Hardware-client source: initial protocol
`404467190aeceea91f58cb98076148ab1aa0c0df`; dual-camera follow-up
`fbd34ed9f1bd3da095c5c7ee3bdc15d4f2bf795c`; camera-identity recheck
`61b3cf2edd40af25c46aac0f8e30c7c2ebd2c0fa`; stale-goal/gradual-return
hardening `b2b97e1255f721f591ef115528216bd5526798fe`

## Scope and result

The operator supplied `ARM SESSION CONFIRMED` in the live task. This authorized
the follower-only read path, both cameras, and the graduated protocol. Serial
identifiers and private checkout paths remain only in ignored local telemetry;
they are redacted from this public report.

Read-only serial/calibration preflight and four 60-second no-motion loops
completed across the original and follow-up sessions. No torque-enable or
goal-position write was issued. The corrected camera-identity recheck closed
the camera blocker. A later supported-pose read cleared the inset start-pose
blocker but found a hazardous stale goal, prompting the software hardening
recorded below. Motion remains blocked because no operator-verified low
hardware-limit profile exists and the physical motion checklist has not been
attested.

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

## Original camera capture (role labels superseded)

The measurements below are retained as the original audit record. The second
row was incorrectly labeled as the fixed camera; later visual review confirmed
that index 2 was the built-in Mac camera. See the camera-identity correction
below for the current verified roles.

Both cameras were captured concurrently for five seconds at 640×480.

| Index / role | Frames | Elapsed | Sustained FPS | Nonblack | Visual verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 / wrist candidate | 43 | 5.0145 s | 8.575 | 43/43 | close desk view |
| 2 / built-in Mac, incorrectly labeled fixed | 150 | 5.0054 s | 29.968 | 150/150 | room-facing built-in view; excluded |

The checkpoint's third slot, `observation.images.camera3`, is supplied by the
existing empty-camera padding path. Captured frames were reviewed locally and
were not committed because one contains a bystander.

## 2026-09-03 follow-up

This subsection preserves the first follow-up diagnosis for auditability. Its
camera-order conclusion was disproved by the later identity recheck below.

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

## 2026-09-03 camera-identity correction

The operator recognized that the purported fixed-camera frame was the
MacBook's built-in camera. Fresh labeled captures from every live OpenCV index
confirmed the error. In the current enumeration, index 0 is the fixed view of
the task surface, index 1 is the wrist view pointed down at the desk, and index
2 is the built-in Mac camera and must be excluded. These numeric indices are
session-local and must be visually revalidated after device changes.

- The working `robot teleops` command validates the leader/follower motor path
  but does not instantiate any camera. Its separate `robot teleop-cams` command
  instantiates only the configured wrist camera.
- With the two verified UVC views, both wrist-first and fixed-first startup
  passed three of three trials at 640x480/30. The earlier 15 FPS failure came
  from opening the built-in camera under the wrong role, so it is not evidence
  of a UVC startup-order defect.
- A concurrent five-second capture produced 101/101 nonblack fixed frames in
  5.014055 seconds (20.143 sustained FPS) and 56/56 nonblack wrist frames in
  5.081585 seconds (11.020 sustained FPS). Both devices negotiated
  640x480/30; the control loop samples at 5 FPS.
- The fixed frame contains the robot task surface. The wrist frame is an
  unobstructed close desk view and is soft because the parked camera is near
  its minimum focus distance; it is not physically covered. Together with the
  operator's framing confirmation, this closes the camera preflight blocker.
- The unchanged repository client then completed a fourth stats-active
  60-second no-motion run with the corrected roles: 293 observations, 292
  chunks, 4.876 sampled FPS, 148.907/150.512 ms client
  observation-to-chunk median/p95, 15 held out-of-domain chunks, and zero
  timeouts. Server receive-to-chunk median/p95 was 146.642/148.338 ms and
  inference was 146.239/147.978 ms. No motion latency was recorded because no
  actuator or torque write occurred; an independent post-run read found all
  six torque bits zero.
- The ignored corrected client and server JSONL hash respectively to
  `dd25a59cf95e631d9313192e6c5e26878039c432ca207f8dc1a43dafad095e67`
  and `6a80220e9f92386ee334b403ae0664e55039c4935f734db5fedeba20a7908c9b`.
  The fixed and wrist evidence frames hash respectively to
  `dcb192c435d045fc5d3652a855668825bce621f9bc276d835af20704cebdfac1`
  and `d49e349b5a26e314e1e62d79090c3c6a71b570b0331878d4f25404437b784dd3`.
- From independent idle preflights, `make test-fast` passed 479/479 selected
  tests with 291 slow tests deselected in 99.28 seconds, and `make test` passed
  all 770/770 tests in 639.60 seconds. Neither run reported a skip or xfail.

## 2026-09-03 post-teleoperation motion-gate recheck

At `2026-09-03T14:09:05Z`, after the operator reported another successful
manual teleoperation, the follower-only path was opened read-only with the
corrected fixed-0/wrist-1 roles. The leader was not opened and no register,
torque, or position write was issued.

- Both intended cameras returned finite, nonblack 480x640x3 frames.
- All six `Torque_Enable` values were zero before and after the read.
- The current position was 6.505, -93.275, 96.484, 61.407, -1.714, and 14.644
  in public joint order. `shoulder_lift` remains outside -83.833 to 83.833,
  and `elbow_flex` remains outside -77.187 to 77.187, so the immutable inset
  start-pose gate still fails.
- `Acceleration` and `Maximum_Acceleration` both read 254 on every controller.
  The remaining torque/current/velocity register groups matched the earlier
  non-low/default observations; no operator-attested profile exists.

The working teleoperation command and the guarded MLX client intentionally
have different admission criteria. `robot teleops` calls the vendor
`SOFollower.connect()`, whose configuration path writes acceleration and
maximum acceleration 254, then sends leader targets at 60 Hz; its command does
not set `max_relative_target`. The MLX client deliberately skips vendor
configuration writes, refuses to enable torque without an exact
operator-verified nine-register profile, requires the start pose inside the
10%-inset calibration envelope, caps each step, and self-terminates. Teleop
success therefore proves that the motors and leader/follower path work, but it
does not clear the separate autonomous-motion gate.

## 2026-09-03 supported pose and stale-goal hardening

The operator manually moved and then mechanically supported the torque-free
follower. A read-only recheck measured 6.242, -20.396, 62.989, 44.176, -1.802,
and 23.013 in public joint order. Every joint passed the 10%-inset start
envelope and all six torque bits remained zero.

The same read exposed a distinct arming hazard: the retained
`Goal_Position` differed from the present shoulder-lift position by 84.396
degrees and from the present elbow position by 33.495 degrees. Enabling torque
against those stale targets could have caused a large immediate move. No
hardware write or torque-enable was attempted.

Failure-first tests reproduced both missing protections. Commit `b2b97e1`
now requires an exact raw `Present_Position` to `Goal_Position` preload and
goal/fresh-present readback match while torque is off, followed by another
torque-off check before enable. Cleanup now returns through fresh-readback
steps bounded by the same maximum one-public-unit action envelope and a 1000
ms dwell, rather than one direct start-pose command. The hardware/readiness
suite passes 96/96, the fast suite passes 482/482 selected tests with 291
deselected in 99.16 seconds, and the full suite passes 773/773 in 660.64
seconds. These changes are software-verified but have not yet enabled torque
or moved the connected follower.

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
| `camera-remap-no-motion-v1-client.jsonl` | `dd25a59cf95e631d9313192e6c5e26878039c432ca207f8dc1a43dafad095e67` |
| `camera-remap-no-motion-v1-server.jsonl` | `6a80220e9f92386ee334b403ae0664e55039c4935f734db5fedeba20a7908c9b` |

The vendor checkout remained at commit
`a24998f7ba3c77ea445b48c92ad15c14a50e492a`. Its pre-existing dirty files were
not modified. The composite of sorted SHA-256 lines for all 696 tracked files
was identical before and after access:
`d280efa881ab9e412cd071bbef38d8d9ec5050e484b49b5a2397df2a39bdb764`.

## Hard blockers before motion

1. Keep the arm mechanically supported. The latest pose passed the inset
   envelope, but it must be re-read immediately before arming and must still
   pass without anyone holding it.
2. Establish and attest low controller torque/current/velocity/acceleration
   limits using an operator-known procedure; capture their exact readback in a
   safety-profile JSON file.
3. Clear the motion envelope, secure the base, and keep a hand on the physical
   power switch for the entire single-action attempt.

## 2026-09-04 authorized reconnection read

Fresh live `ARM SESSION CONFIRMED` authorized this session's read-only
preflight at source `c228095285b33121c52c624d80127d238a4bb584`.

- USB identity matched the configured follower; the leader was detected only
  through enumeration and never opened. Existing follower calibration matches
  controller readback. All six torque bits were zero before and after reads;
  no actuator or torque write occurred, and the serial port closed normally.
- At `2026-09-04T14:22:04Z`, every joint passed the numeric 10%-inset envelope.
  Lift/elbow measured -73.143/39.868 degrees. This does not attest mechanical
  support, workspace clearance, base security, or the physical power cut.
- Exact nine-register readbacks still match the previously observed unapproved
  settings, including acceleration and maximum acceleration 254 on all six
  controllers. An operator-approved low-limit profile has not been supplied.
- Fresh labeled camera discovery produced images for fixed 0, wrist 1, and
  excluded built-in 2. The current wrist image points toward the operator;
  the fixed image shows the computer desk without the arm's working area.
  Framing needs physical correction in this session. The prior identity fix
  and prior framing pass are not reclassified as failures. Candidate 3 yielded
  no usable image and is excluded; it was not retried.
- Concurrent camera-rate measurement and a new 60-second no-motion run were
  deferred at the physical preflight gate. Neither motion stage has run.
- The setup scripts resolve the operator command/configuration directory to a
  separate vendor Git checkout at the already recorded `a24998f` revision.
  Its 696 tracked files and all 32 operator-wrapper files were unchanged after
  access. No vendor environment was activated, installed into, or modified.
- The reviewed stats-active checkpoint still passes six-axis validation and
  its model retains SHA-256
  `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.

Private session evidence is under
`.cache/hardware/session-20260904T141839Z-b560dkhe/`. The readback JSON hashes
to `d90877883566e557aa86552f133a8ca15d02923fbcefff517f2d663f1a8bd550`;
the camera-discovery log hashes to
`643a5b8f0dc4bb5a384ebdfa6289375bbef28ff765551aa8328d5efee1f912d4`.
Frames and identifiers remain ignored and untracked.

Before continuing, power off for any manual adjustment, frame both cameras on
the cleared task area, establish the approved low-controller-limit profile,
and complete the physical checklist. Re-read the supported pose and limits,
then follow the runbook's fresh no-motion and separately confirmed motion
sequence. No failed gate is bypassed by reducing software step size.
