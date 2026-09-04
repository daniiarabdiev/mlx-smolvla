# Hardware and v0.1.0 release continuation

Date: 2026-09-04. Starting source: `7fdf2fc2f35f41a776250f71c613039d0c41b6f3`.

Continue the approved [release brief](BRIEF_PUBLIC_RELEASE.md) using the later
[status](STATUS_PUBLIC_RELEASE.md), [runbook](../HARDWARE_RUNBOOK.md), and
operator's current handoff. The rename, compatibility work, native runtime,
serving, packaging, and trained-checkpoint reference repair are complete.
Training stays a research preview; broader studies, strict Metal improvements,
and active CI are outside this continuation.

The continuation began offline. The operator later connected the hardware and
supplied fresh live `ARM SESSION CONFIRMED` on 2026-09-04. Read-only follower
identity/calibration and numeric pose checks passed. Subsequent adjustment
resolved camera framing at that point; the 17:31 pose change later altered
the wrist view. The prior short concurrent capture measured fixed
20.10/wrist 6.34 FPS at 640x480 without timeouts (30 FPS requested, not achieved).
The operator then explicitly delegated setup and client execution. Two new
60-second no-motion runs completed: the cold-server attempt had one initial
timeout; the warm repeat had zero timeouts and zero actuator writes. Temporary
low speed, acceleration, and torque settings have now been staged and exactly
read back with torque off. At 16:52 UTC the elbow was outside the inset start
envelope. The 17:31 UTC read resolves that numeric pose failure: every joint
passes, with no two-second drift and the exact low profile still matched.
The wrist camera was subsequently aimed at the table, and final
camera/support/clearance/profile/pose checks passed. A fresh no-motion run, one
valid single action, and a two-chunk continuous stage passed with torque-off
shutdown. A separate 20-chunk attempt failed exact return under the 10% torque
profile and remains a documented sustained-operation limitation.

## 1. Software baseline and checkpoint

- [x] Verify clean `main`, canonical SSH remote, and exact local/remote HEAD;
  preserve newer work and existing artifacts. GitHub is private; no `v0.1.0`
  tag exists at kickoff.
- [x] Read the required current and historical evidence. Four no-motion loops,
  corrected camera roles/framing, and the supported inset pose remain recorded
  successes; their session-local checks must be repeated after reconnecting.
- [x] Record an idle preflight, unchanged fast/full test results, repository
  hygiene/link checks, and candidate/protected-evidence hashes in fresh ignored
  logs. Save results in `PROGRESS.md` and both current status files. The fast
  lane passed 489 tests in 106.30 seconds wall; the full suite passed 790 tests;
  final documentation/distribution checks passed 25 tests. Commit/push this
  verified documentation milestone and confirm the remote before handoff.

## 2. Fresh connected preflight — operator required

- [x] Obtain `ARM SESSION CONFIRMED` in the live session before device access
  or reading/executing the vendor checkout. Historical or pasted confirmations
  do not authorize a new session.
- [x] Preserve a before/after hash of the vendor's tracked files. Use the
  existing separate `.cache/hardware/server-venv` and `client-venv`; never edit,
  install into, upgrade, or otherwise change the vendor checkout/environment.
- [x] Identify only the follower; verify its existing calibration and six
  torque-off readbacks. Never open the leader or recalibrate during this work.
- [x] Visually identify both intended UVC cameras from fresh labeled frames,
  exclude unrelated cameras, correct their framing, and check concurrent
  capture. The short probe measured 20.10/6.34 FPS without timeouts; it does not
  demonstrate actual 30 FPS or replace the no-motion policy loop.
- [x] Re-read the mechanically supported pose against the unchanged 10%-inset
  envelope immediately before proceeding to arming.
- [x] Under the operator's explicit delegation, establish and exactly read back
  a manufacturer-documented temporary commissioning profile: SRAM acceleration
  1, speed 56, torque limit 100 on all six controllers. Persistent/factory
  settings, startup force 32, mode 0, present/goal positions, and torque-off
  were checked. This establishes reduced settings, not gravity-hold ability
  or physical safety; those require the supported supervised trial.
- [x] Confirm workspace clearance, secure base, physical power cut, and hand
  on power. Follow [HUMAN_TASKS.md](HUMAN_TASKS.md); keep private profile,
  serials, images, and telemetry under ignored local storage.

## 3. Graduated validation — delegated execution with operator present

- [x] Select the reviewed stats-active checkpoint matching the robot. Run a
  fresh 60-second `--no-motion` check using new, non-overwriting server/client
  logs. Require the duration cap, fresh cameras, zero timeouts and writes,
  torque-off readback, and review of rejection/clamp/rate-limit counts.
- [x] Close the controller-state gap before arming: reread the entire profile
  after raw-goal preload, require integer position mode 0, and reject excessive
  or changed startup force. Failure-first fault injection demonstrates these
  checks; the focused hardware/readiness suite passes 104 tests. The final
  final fast lane now passes 502 selected tests in 109.48 wall seconds, and the
  full suite passes all 803 tests in 755.65 wall seconds, without skips or
  expected failures.
- [x] Once every physical prerequisite actually holds, verify the fresh
  supported pose and operator readiness. The current-session delegation
  supersedes the old prescribed motion phrase; it does not supply those facts.
- [x] The authorized delegate runs one `--single-action` through the guarded
  client. Review direction, displacement, speed, gripper behavior, camera
  freshness, telemetry, gradual return, and torque-off. Preserve the existing
  stale-goal preload/readback, clamps, caps, and watchdogs. The passing run
  waited through one rejected hold and stopped after its first valid action.
- [x] Only after the one-action result is accepted, the authorized delegate runs
  `--continuous` with the existing 90-second/20-chunk cap, whichever comes
  first, and the same shutdown checks. The accepted stage used a two-chunk
  ceiling and returned exactly. A separate run at the 20-chunk ceiling failed
  exact return but disabled torque; keep that sustained result marked failed.

## 4. Final source, tag, and artifacts — after hardware passes

- [x] Record measured results in `hardware/FIRST_CONTACT.md` and
  `hardware/PREFLIGHT.md`; update README, CHANGELOG, status, and only genuinely
  resolved human tasks. Safe actuation establishes integration, not reliable
  pick-and-place task success.
- [x] Run the full suite, idle fast-lane timing, link/hygiene checks, and
  relevant source/distribution checks. The final lanes pass 803 full, 502 fast,
  109 focused, and 28 documentation/hygiene tests.
- [ ] Review and commit/push the final source and evidence; verify the remote
  before creating/pushing annotated `v0.1.0`.
- [ ] Build a new sdist and CPython 3.11/3.12/3.13 wheels from that tag into a
  new directory. Repeat archive/Twine checks and all seven applicable fresh
  install environments described in [DIST_MANIFEST.md](../evidence/DIST_MANIFEST.md).
  Record exact hashes and tag/source provenance in the manifest and commit/push
  that evidence. Preserve prior artifacts; never upload the untagged candidate.
- [ ] Obtain separate explicit authorization before changing GitHub visibility,
  uploading to PyPI/Hub, or creating the GitHub Release. Recheck PyPI name
  availability immediately before any authorized publication. Converted-weight
  uploads and a demonstration video remain optional.

Do not declare public release readiness while a required hardware or final
verification gate remains open. Normal source/evidence commits and pushes are
authorized; they do not authorize publication or public attribution.
