# Hardware and v0.1.0 release continuation

Date: 2026-09-04. Starting source: `7fdf2fc2f35f41a776250f71c613039d0c41b6f3`.

Continue the approved [release brief](BRIEF_PUBLIC_RELEASE.md) using the later
[status](STATUS_PUBLIC_RELEASE.md), [runbook](../HARDWARE_RUNBOOK.md), and
operator's current handoff. The rename, compatibility work, native runtime,
serving, packaging, and trained-checkpoint reference repair are complete.
Training stays a research preview; broader studies, strict Metal improvements,
and active CI are outside this continuation.

The operator reports that the hardware is not connected. This blocks fresh
hardware validation and the final release gates, while software preparation
can proceed. No live device or motion authorization has been supplied.

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

- [ ] Obtain `ARM SESSION CONFIRMED` in the live session before device access
  or reading/executing the vendor checkout. Historical or pasted confirmations
  do not authorize a new session.
- [ ] Preserve a before/after hash of the vendor's tracked files. Use the
  existing separate `.cache/hardware/server-venv` and `client-venv`; never edit,
  install into, upgrade, or otherwise change the vendor checkout/environment.
- [ ] Identify only the follower; verify its existing calibration and six
  torque-off readbacks. Never open the leader or recalibrate during this work.
- [ ] Visually identify both intended UVC cameras from fresh labeled frames,
  exclude the built-in/Continuity cameras, check concurrent capture, and re-read
  the mechanically supported pose against the unchanged 10%-inset envelope.
- [ ] Have the operator establish the exact approved low-controller-limit
  profile through their known-good procedure; capture all nine registers for
  all six joints and validate exact readbacks. Observed defaults, software
  step caps, and dwell times are not approval or motor-speed limits.
- [ ] Confirm workspace clearance, secure base, physical power cut, and hand
  on power. Follow [HUMAN_TASKS.md](HUMAN_TASKS.md); keep private profile,
  serials, images, and telemetry under ignored local storage.

## 3. Graduated validation — operator runs the client

- [ ] Select the reviewed stats-active checkpoint matching the robot. Run a
  fresh 60-second `--no-motion` check using new, non-overwriting server/client
  logs. Require the duration cap, fresh cameras, zero timeouts and writes,
  torque-off readback, and review of rejection/clamp/rate-limit counts.
- [ ] Once every physical prerequisite actually holds, obtain the separate
  live statement:

  ```text
  MOTION PREREQUISITES CONFIRMED: cameras framed, arm neutral, low limits profiled, workspace clear, base secure, hand on power.
  ```

- [ ] The operator runs one `--single-action` through the existing guarded
  client. Review direction, displacement, speed, gripper behavior, camera
  freshness, telemetry, gradual return, and torque-off. Preserve the existing
  stale-goal preload/readback, clamps, caps, and watchdogs unchanged.
- [ ] Only after the one-action result is accepted, the operator runs
  `--continuous` with the existing 90-second/20-chunk cap, whichever comes
  first, and the same shutdown checks. Unexpected motion or uncertainty means
  the runbook's physical stop and review, never an automatic retry.

## 4. Final source, tag, and artifacts — after hardware passes

- [ ] Record measured results in `hardware/FIRST_CONTACT.md` and
  `hardware/PREFLIGHT.md`; update README, CHANGELOG, status, and only genuinely
  resolved human tasks. Safe actuation establishes integration, not reliable
  pick-and-place task success.
- [ ] Run the full suite, idle fast-lane timing, link/hygiene checks, and
  relevant installed-package checks. Review and commit/push the final source
  and evidence; verify the remote before creating/pushing annotated `v0.1.0`.
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
