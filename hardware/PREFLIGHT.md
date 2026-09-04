# Hiwonder SO-101 preflight evidence

Date: 2026-09-02; follow-ups: 2026-09-03 and 2026-09-04

Host: Apple M5 Pro MacBook Pro (`Mac17,8`), 48 GB, macOS 26.6.2

Runtime: Python 3.12.13, MLX 0.32.2, `Device(gpu, 0)`

Hardware-client source: initial protocol
`404467190aeceea91f58cb98076148ab1aa0c0df`; dual-camera follow-up
`fbd34ed9f1bd3da095c5c7ee3bdc15d4f2bf795c`; camera-identity recheck
`61b3cf2edd40af25c46aac0f8e30c7c2ebd2c0fa`; stale-goal/gradual-return
hardening `b2b97e1255f721f591ef115528216bd5526798fe`; final powered-validation
source was based on pushed `cf01267f82d518285f81775cdb85294d1a6b1e1f`
plus the controller-arming and bounded-single-action changes recorded below.

## Scope and result

The operator supplied `ARM SESSION CONFIRMED` in the live task. This authorized
the follower-only read path, both cameras, and the graduated protocol. Serial
identifiers and private checkout paths remain only in ignored local telemetry;
they are redacted from this public report.

Read-only serial/calibration preflight and four 60-second no-motion loops
completed across the original and follow-up sessions. No torque-enable or
goal-position write was issued during those first four runs. The corrected
camera-identity recheck closed the camera blocker. A later supported-pose read
cleared the inset start-pose blocker but found a hazardous stale goal,
prompting the software hardening recorded below.

The final 2026-09-04 session then established the temporary reduced controller
profile, completed the physical checklist, passed another 60-second no-motion
run, executed one valid guarded action, and passed a two-chunk continuous run.
A separate 20-chunk attempt failed exact return-to-start under the same low
torque profile while still completing verified torque-off cleanup. The final
section is the current result and supersedes the historical pending gates.

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

## 2026-09-04 camera-adjustment and calibration follow-up

The same authorized live session continued at source
`3b726505c72e2e6c1c0c41bd770abac79d307a3f`. Fresh discovery after the operator's
first camera adjustment showed fixed 0 including the desk and part of the arm,
but blurred and without a verified complete workspace. Wrist 1 still faced
sideways toward the operator. Built-in 2 is excluded; candidate 3 now returns
a near-dark non-workspace image and remains excluded.

After further operator adjustment, camera-only capture at
`2026-09-04T14:52:50Z` shows the gripper, yellow ball, and tabletop in wrist 1
at 640x480. The wrist's task-surface framing is corrected. The operator's hand
is visible near the gripper, so workspace clearance is not established. The
fixed camera still needs a sharp view of the complete arm and reachable table
area; the operator received positioning and focus guidance. These captures
do not constitute a concurrent camera-rate gate or the fresh no-motion loop.

The operator also reported having calibrated the arm previously. Read-only
follower access at `2026-09-04T14:48:09Z` matched USB identity and the existing
calibration. Every joint passed the numeric inset envelope; lift/elbow read
-73.055/39.868 degrees. Torque remained zero on all six joints, no motor or
torque write occurred, and the port closed. No calibration was changed or
repeated. The nine controller readbacks match the prior unapproved settings.

The operator does not know an established low-limit profile. The inspected
operator setup documents calibration, teleoperation, and gripper-specific
protection, but does not establish the required six-joint low-limit profile.
Normal vendor configuration sets acceleration values to 254. The
[Hiwonder BusLinker manual](https://docs.hiwonder.com/projects/BusLinker/en/latest/docs/1_BusLinker_V3.0_Servo_Debugging_Board_User_Manual.html)
documents the torque-limit scale as 1000 for 100% stall torque, separate
acceleration/speed controls, and ignored Time in position mode. These explain
the readbacks; they do not supply an approved low-limit configuration for this
assembled arm. No values were guessed, written, or turned into an attested
profile. All 696 vendor tracked files and 32 operator-wrapper files remain
unchanged. No motion ran.

Fresh ignored evidence:
`.cache/hardware/camera-adjustment-20260904T144524Z-utqorp1q/`.
Follower JSON SHA-256:
`afc2accc75465f3eda99feeeed177aedc5886a418fbafaf7a2cd30feec89fb9a`.
Camera-discovery log SHA-256:
`229fe3a9644b74ca8058fdf7cda0a17520b49b91762a4065199dacbcebf34864`.
The corrected wrist image is preserved under `wrist-after-second-adjustment/`,
SHA-256 `aa244b858cd808fde458404d0c700ed9f9b5a3fa4014433e38e11eb8ac8cd9ac`.
Raw images, device identifiers, calibration hash, and readbacks stay private.

## 2026-09-04 fixed-camera framing resolved; concurrent capture measured

The operator used QuickTime's live preview to adjust the fixed camera, closed
the preview, and requested a recheck in the same authorized live session.
Camera-only checks at source `cad455604299b36eb8c9a2883c5598bb58aa46a6` used
the existing client environment and session-local fixed 0/wrist 1. New images
show the follower and tabletop task area in improved focus, and the wrist's
gripper/nearby yellow-ball view. The current framing issue is resolved. The
loose cable across the table and mouse in the working area must be cleared;
visual framing does not attest workspace/base/power-cut or supported-pose safety.

The first paired-consumer measurement at 15:16 UTC delivered 51 frames per
camera in 8.105 seconds (6.29 paired FPS), with no timeouts. To distinguish the
individual camera rates, a second measurement at 15:17 UTC consumed the two
streams independently while both remained open. Both requested 640x480/30:

| Camera | Frames | Measured duration | Consumed FPS | Maximum read gap | Timeouts |
| --- | --- | --- | --- | --- | --- |
| Fixed 0 | 161 | 8.009 s | 20.10 | 66.98 ms | 0 |
| Wrist 1 | 51 | 8.046 s | 6.34 | 165.93 ms | 0 |

Each consumed frame had a distinct pixel hash. Both rates exceeded the planned
5 FPS control cadence in this short probe, but **actual 30 FPS was not
demonstrated**. No cause for the reduced rates is asserted. This camera-only
measurement does not replace the required 60-second no-motion robot/server
loop. All camera handles closed; neither robot, serial port, nor vendor
checkout was accessed. No motor write, fresh pose/calibration/limit read,
policy inference, or physical action occurred. The low-limit profile and
physical attestation remain open; no guard or configured limit changed.

Private evidence: `.cache/hardware/fixed-adjustment-20260904T151528Z-45fdealv/`.
Paired capture summary SHA-256:
`ec3b804f48d8cb8823952ba66c3c666004074eeebc8002b4bc25cb897f88c91e`.
Independent cadence summary SHA-256:
`556c7d1c3a125342f38a93a5a540e2e93df1ab7c8981db63be6f99f2a2cb1a4d`.
Final fixed/wrist image SHA-256 values:
`4e0563f2c4b0cdef21027105d18c1938bacf9202a99eb81101d29ae38fe3a304` /
`4c07f586ffe7d5357e894cdd1457502a5c4489d8749e9b40513a596e35eb79de`.
Images and raw capture records remain ignored and untracked.

## 2026-09-04 delegated commissioning; reduced settings verified

The operator explicitly amended the runbook and delegated controller setup and
client execution. The existing live session authorization remains valid; the
old operator-only command requirement and prescribed motion phrase are not
current blockers. Physical readiness must still be established.

Two fresh 60-second stats-active no-motion runs used fixed 0/wrist 1, 5 Hz,
the unchanged 500 ms watchdog, and the separate environments. The first reached
the duration cap in 60.031 s with 292 observations/291 processed chunks,
161.235/163.926 ms client median/p95, one timeout/hold, 415 clipped joint values,
1,330 rate-limited values, and zero rejected chunks. The server's first logged
timestep was 1: the first request missed its deadline. This attempt fails the
zero-timeout acceptance requirement and is retained.

The warm repeat passed in 60.088 s with 295 observations/294 processed chunks,
4.909 sampled FPS, 165.462/168.631 ms client median/p95, zero timeouts, zero
holds/rejections, 437 clipped joint values, and 1,358 rate-limited values. The
server logged all 295 chunks; the client discarded the final one at the
duration boundary. Server receive-to-chunk median/p95 was 163.196/166.264 ms;
inference was 162.828/165.845 ms. Neither run wrote actuator or torque registers.
An earlier launcher used the resolved base-interpreter symlink and failed its
package import before device access; the corrected launcher preserves the
client-venv executable path. No environment was installed or modified.

The 16:52 UTC read matched the existing calibration and found all six HX-30HM
model numbers 777, firmware 3.13, position modes 0, startup-force values 32,
status alarms 0, and torque bits 0. Lift/elbow were -73.670/86.110 degrees;
the elbow exceeds its 77.187-degree inset upper bound. The subsequent fixed
view shows the folded follower near the monitor and a cable across the table;
the wrist sees the gripper, ball, and nearby cable. Physical support and
clearance remain unverified, so the client must not arm in this pose.

The register definitions are manufacturer-published, not inferred from a
third-party motor: the pinned
[Hiwonder HX-30HM table](https://github.com/Hiwonder-official/hiwonder-SoArm-101/blob/a24998f7ba3c77ea445b48c92ad15c14a50e492a/src/lerobot/motors/hiwonder/tables.py)
adopts the
[STS/SMS register layout](https://github.com/Hiwonder-official/hiwonder-SoArm-101/blob/a24998f7ba3c77ea445b48c92ad15c14a50e492a/src/lerobot/motors/feetech/tables.py).
The
[BusLinker manual](https://docs.hiwonder.com/projects/BusLinker/en/latest/docs/1_BusLinker_V3.0_Servo_Debugging_Board_User_Manual.html)
documents the speed/acceleration units, maximum-output torque scale, and
ignored position-mode Time control. This supports temporary SRAM setup without
changing EEPROM or factory limits.

| Temporary register | Address / bytes | All-six staged value | Documented interpretation |
| --- | --- | ---: | --- |
| Acceleration | 41 / 1 | 1 | 100 encoder steps/s² |
| Goal_Velocity | 46 / 2 | 56 | 56 encoder steps/s, about 4.9°/s at the shaft |
| Torque_Limit | 48 / 2 | 100 | 10% of the 1000-point output-torque scale |

Each write ran with six verified torque-off bits, matched identity/calibration
and firmware, and exact readback. Startup force 32 and mode 0 stayed unchanged;
the cap exceeds startup force and is below the existing persistent maxima.
The remaining profile registers, raw present positions, and retained goals
were unchanged. No goal-position or torque-enable command was sent. Every
handle closed. The profile loader accepts the exact private nine-register
profile and its serial; its procedure explicitly records delegated setup and
the absence of a validated holding/motion result. No automatic torque increase
or restoration is permitted. SRAM values must be freshly verified after any
power cycle or other robot configuration.

The reviewed client change now rechecks the entire profile after raw-goal
preload and validates integer position mode plus startup force before and after
that write. The startup force must fit the smaller torque cap and stay
unchanged across preload. All existing inset, raw-goal equality, rate, watchdog,
duration, return, and torque-off checks remain. This code is unit-tested but
has not yet been exercised with torque enabled.

Private evidence: `.cache/hardware/commissioning-20260904T163953Z-wg7zy750/`.

| Record | SHA-256 |
| --- | --- |
| Cold no-motion client | `72b660f630d9a3768c5f50ade7330dcd06789f87d50b689730f64572513f07f4` |
| Warm no-motion client | `cdef2730eced0408176af34ebacb87523860b12da87ab1f69a5ee7246af1b6cc` |
| Combined server latency | `cd7594b1b4f448e4dadf7dba44d7830e01a3e719d0306d242bc0870531db3fc4` |
| Read-only controller/pose | `3d1c915f1c352cbb2ac639c25c477c16b7e71ebdfe0b2106a1254652ea9938d2` |
| Kinematics staging | `475777534c16e0494af0d32af4557d389c3a036ea86d64abb5cf63686ff2ec9a` |
| Torque staging | `498c638f24c67674e6471b12d08976f8bb591baec160cae44f0a7c71815790d5` |
| Session profile | `594f9d81d5b16347736241c74a5b8f60e818456a6706b3f1077907204a2dd036` |
| Fixed image | `2b6a8cf1320799fb35cd522a9d5f11eab6fdcd0de859434e40a852eac46e61f3` |
| Wrist image | `1d075eac71f3e1aca31672b5648e8a5a24dea89d1b1c7c50bfe153dd232a6c23` |

Raw identifiers, frames, controller records, and the profile remain ignored.

The final controller-guard source passes 104 focused checks in 3.59 seconds,
497 fast tests (301 slow tests deselected) in 96.91 test seconds / 99.57 wall
seconds, and all 798 full-suite tests in 723.25 test seconds / 726.60 wall
seconds. No skips, expected failures, selection changes, or tolerance changes
were introduced. Both final lanes used the same runtime/test source hashes,
with no competing project compute or inherited test overrides at preflight.
The fast lane remains below the unchanged 120-second wall-time gate. These
are software results; they do not establish physical actuation or holding.

| Verification record | SHA-256 |
| --- | --- |
| Final focused log | `d9a19b3c2bf2801aba340a3f14828a2cb1146f0bbdba65b8525facaf3c7d8b54` |
| Final fast log | `a835531a8ff8c37c8ad65bcddb4df26d6a40a2600661aaa7f3d6d8fd03110220` |
| Final fast wall-time result | `d38fc51fd5fedcc772388433d039e42e5085a69c76cc538ef43aaf85953099d3` |
| Full-suite log | `0125036e0092868e2bc582abc138780090daa312eecdbe44ee7a5327567acef7` |
| Full-suite wall-time result | `53327a526de9528445651a21edd0ee74d277171932680b9e44decd050e06466f` |
| Final documentation/hygiene/distribution/readiness log | `fbeae43a2d744c9ea15df7c3de101a4d02ca614a43c4a0c8c13cb39a15d63606` |

After the final status and live-pose updates, all 28 documentation, repository
hygiene/link, distribution, and hardware-readiness checks passed in 19.58
test seconds / 25.25 wall seconds. Runtime/test hashes still match the full
suite, and all 696 vendor tracked files and 32 operator-wrapper files remain
byte-identical. Changed-document links resolve and private identifiers remain
absent from the diff.

## 2026-09-04 17:31 UTC adjusted-pose recheck

The operator adjusted the arm and requested a new live check. Exact follower
identity/calibration and all nine profile readbacks match. All six positions
pass the unchanged inset envelope: lift/elbow are -54.330/33.187 degrees, with
zero measured drift over two seconds. This resolves the 16:52 pose failure.
All controllers report model 777, firmware 3.13, mode 0, startup force 32,
status 0, and torque off. Temperatures are 41–45°C, measured voltage 12.2–12.6 V,
and current/velocity read zero. These readbacks do not prove a holding trial.

Both cameras returned concurrent 640x480 frames. The fixed view contains the
raised follower and tabletop; the wrist now points at the operator/ceiling.
Requested a power-off mount adjustment toward the gripper/task surface while
retaining the passing joint pose. Final physical readiness and no-motion
checks still precede any arming. All writes were blocked by the read-only
probe, none was attempted, and all serial/camera handles closed.

Private evidence: `.cache/hardware/pose-recheck-20260904T173059Z-xj1676sr/`.

| Record | SHA-256 |
| --- | --- |
| Read-only pose | `83b357e8a596eec1d4a5f9e50e2dfd0762d3d2c268925c00996ac154755b8290` |
| Combined check summary | `9063d2cf1a5f4a42791c7f0fc8361213faac988d34b18c9a4cb5a2a7233cfe65` |
| Fixed image | `30aa1aff92da02a4f022f4474a8c492f61c40d447a7175351360b5955416d984` |
| Wrist image | `3b4491e95037d54c338e2b92298554ba893bddf8b3b72f0901ff354222c358c9` |

## 2026-09-04 final no-motion and powered motion evidence

The operator completed the remaining physical setup and authorized the powered
retry. Fresh checks matched the follower identity, existing calibration, and
all profile registers. Temporary SRAM values were acceleration 1, goal
velocity 56, and torque limit 100 on all six motors. Mode 0, startup force 32,
status 0, and torque 0 also matched. The fixed camera showed the whole arm and
clear task area; the wrist camera showed the gripper and table. Every joint was
inside the 10%-inset range and the pre-motion pose showed no two-second drift.

### Goal-write arming discovery

On this exact controller, writing the six raw present positions back to
`Goal_Position` changed all six `Torque_Enable` bits from 0 to 1 in about
12 ms without calling the host's explicit enable method. The original guard
failed closed because it expected torque to remain zero. It wrote no policy
action, produced no displacement, disabled all six motors with exact zero
readback, closed the adapter, and exited nonzero. An independent offline audit
confirmed that every raw present value was within its current raw controller
minimum and maximum.

The corrected client checks raw minimum/maximum limits before the first write,
rechecks them afterward, and treats the goal write as the arming boundary. Its
explicit `goal-write` mode requires all six torque bits to become one and never
calls the enable method twice. The default `explicit-torque` path instead
requires six zeroes after preload before enabling once. Failure injection
covers automatic enable, invalid raw limits, partial explicit-enable failure,
and post-enable read failure; every post-write failure attempts verified
all-six torque-off cleanup.

Private evidence:
`.cache/hardware/motion-ready-20260904T174510Z-3tiyf6ed/`.

| Record | SHA-256 |
| --- | --- |
| Failed arming I/O trace | `e6b1022724ce124aff49bd76dad2de0449090d588da1446feb946dbb0f93836a` |
| Failed arming telemetry | `81b0e6826914c451644d32da2b618b8463e1c42b65df33f315860c4c303090ca` |
| Raw preload/range audit | `966a5a0dff8bd5e7a31d2c5ba8a2d3b64a6d01542788b168daccec7a64c004c7` |

### Final no-motion result

The fresh stats-active run completed 60.001971 seconds with 295 observations
and 295 processed chunks at 4.916505 sampled FPS. Observation-to-chunk
median/p95 was 167.734/173.328 ms. It had zero timeouts and zero writes; its
simulation recorded 29 rejected hold chunks, 404 clipped values, and 1,499
rate-limited values. First and last positions were identical.

### Single valid action

Two initial single-action invocations each received an invalid timestep-zero
chunk, wrote a current-position hold, returned exactly, and disabled torque.
A ten-prompt read-only policy probe produced the same invalid timestep-zero
shape and made no hardware write. The mode was therefore corrected to wait
through rejected holds while retaining a 20-chunk attempt cap and stopping
immediately after its first valid non-hold action.

The accepted retry processed two observations/chunks: one rejected hold and
one valid target with per-joint deltas `(+1, -1, -1, -1, +1, -1)` public
units. It stopped with `action_limit`, recorded one clipped and six
rate-limited values, had zero timeouts, returned to its exact recorded start,
and passed an independent zero-drift/profile/calibration/status check with all
six torque bits zero.

### Continuous attempts

The 20-chunk attempt produced gradual motion toward the task object under the
same one-unit bounds. It exited nonzero because several gravity-loaded joints
stopped following one-degree return targets under torque limit 100 and the
20-step return cap expired before exact start. The failure is retained as a
sustained-operation limitation. Cleanup still read all six torque bits as
zero, and the stopped pose was inside all inset bounds with zero drift, exact
profile/calibration readback, live cameras, and no status alarms. No limit was
raised and no automatic retry occurred.

From that independently checked safe pose, a two-chunk continuous stage passed:
one rejected current-position hold followed by one valid one-unit action. It
stopped with `chunk_limit`, returned exactly, exited zero in 6.513631 seconds,
and recorded zero timeouts, one clipped value, and six rate-limited values. A
final independent check found exact final pose
`(24.439560, -40.527473, 72.747253, 25.274725, -1.450549, 12.671847)`,
zero drift over two seconds, all inset/profile/calibration/status gates passing,
and all torque bits zero. The owned server was stopped and its loopback port
was closed.

Private powered-session evidence:
`.cache/hardware/powered-retry-20260904T5gzCJz/`.

| Record | SHA-256 |
| --- | --- |
| Final no-motion telemetry | `ed2cb9116974d5ed5da742c1fbb361ad7c4cf612117b7ef95819052446f0e3b3` |
| Final no-motion I/O trace | `866b590f645dd66698fe5226363585c1c4b4b80c44a8081136d9e8ec8e174470` |
| Successful single-action telemetry | `b2d367f302f7868a67fb7b9c2730e8e8072bafb23ad4f4199f380564066f02c1` |
| Successful single-action I/O trace | `8472dcdac833c128835ac7b0156fc505fcafb8440398bc95ed659600062eaed4` |
| Post-single independent check | `5f8c4ae90eea972fa5b46deeb92ecc49eda2872ccef9f00dd9b219ecd01123fb` |
| Failed 20-chunk telemetry | `c24d21ce3e640ce4fd5edac878c79b46a021283be9448861b3a6fe086e844865` |
| Failed 20-chunk I/O trace | `4a8a5b9dc4b23f1463893307554e999e350ff5e0973b6093b81fb3b30c944368` |
| Failed 20-chunk process result | `85f0653742f20f991be437b040abcecc8455392707db09ce9a402f6a2cdc4bda` |
| Post-failure independent check | `874b410152b55a6563ce53961d33b008dd3b9fd7e2b4f72b55c5205349498419` |
| Passing two-chunk telemetry | `edcd92744c9fd588a59810515a31525a6357a76bb08501ecd91936ab7b87185e` |
| Passing two-chunk I/O trace | `e3e1dba4bfac1b90f2114bbceb3f9500a994040532649715fd416084062451b0` |
| Final independent check | `d063c7ae96543a3548ef2757eec823228b72d6a91ec6f894fc6c02eb5027f2eb` |
| Server latency telemetry | `c2081b3e5344409e97af6d76626f86e7c15a93dd80775aa95d2f198e97cd5271` |
| Server log | `72718a8040fa27fc6e1bcfb2246d2dfad41cfca1b040fae356841468d332993f` |
| Server stop record | `ca31bdf9f48d9b7717866844c70a250a724f9c8074ae23d173559d785dd4a27b` |

This closes the required bounded first-contact integration gate. It does not
close the sustained 20-chunk return limitation or establish reliable
pick-and-place success.

## 2026-09-04 final current-source verification

With no competing test, training, server, hardware-client, or benchmark process,
the fast preflight measured 79.66% idle CPU. The unchanged `make test-fast`
target collected 803 tests, deselected the same 301 slow tests, and passed all
502 selected tests in 106.44 pytest seconds / 109.48 complete-command wall
seconds, below the two-minute requirement.

A second idle preflight measured 80.5% idle CPU. The unchanged `make test`
target passed all 803 tests in 752.18 pytest seconds / 755.65 wall seconds. It
reported no skip, xfail, or failure. The final-source controller and session
slice passes 109/109 in 3.90 seconds. No hardware was accessed during software
verification.

Private verification directory:
`.cache/release-final-20260904-3Fy3sOul/`.

| Record | SHA-256 |
| --- | --- |
| Fast idle preflight | `116b01044da706148a30974f36bcef733709d859b76299e45225e0a8e6bfdcde` |
| Full idle preflight | `ca893756c4fb4469914450faac242aa83ea3c213f45ce81b3800e4c9fd1cdc78` |
| Fast-lane log | `ab17c29c8e18ceb1ce16db5db0abeea5473be1b28159841136466f56e70bf5ff` |
| Full-suite log | `02647746229200ade22b18b1b699f06675249a080b9a5270c1480e785a9c4f01` |

The final public-release, repository-hygiene, distribution, and hardware-
readiness slice passed 28/28 in 16.20 pytest seconds / 19.50 wall seconds. Its
log SHA-256 is
`593e1c3152fc06e554ef0027895cfab83f1c67876e335f534b33b8fc8ef9a0bb`.
Lock validation resolves 122 packages and the active 100-package environment
passes dependency checking.
