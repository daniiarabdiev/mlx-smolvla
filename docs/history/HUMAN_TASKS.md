# Human Tasks

## Open — complete the hardware gate before announcing v0.1.0

- The follower/camera preflight and four 60-second no-motion loops completed
  across 2026-09-02 and 2026-09-03. The latest run used the corrected
  session-local camera mapping: fixed index 0 and wrist index
  1; index 2 was the built-in Mac camera, not the fixed UVC view. First clear
  the remaining physical prerequisites below, then complete the
  fresh no-motion, single-action, and bounded-continuous stages. Hardware
  motion claims require reviewed passing measurements for all three modes in
  `hardware/FIRST_CONTACT.md`. The annotated `v0.1.0` tag additionally requires
  final software verification and a reviewed, committed, pushed final source
  and evidence checkpoint. Publication and announcement remain separately
  authorized actions after the complete release gates pass.

## Open — clear the remaining physical prerequisites for one action

- **Current availability:** the operator reconnected the hardware and supplied
  fresh live `ARM SESSION CONFIRMED` on 2026-09-04. Read-only follower and camera
  preflight ran; no torque or actuator write occurred. The separate motion
  prerequisites are still unconfirmed. Follow the
  [continuation plan](PLAN_HARDWARE_RELEASE_CONTINUATION.md).
- Before any device access or vendor-checkout read/execute, the physically
  present operator must supply `ARM SESSION CONFIRMED` in the live session.
  The completed historical session below and the pasted handoff do not satisfy
  this new gate. Freshly verify follower identity/calibration, both camera
  viewpoints, and the supported pose after reconnecting; never open the leader.
- **Current camera framing:** fresh images identify fixed 0 and wrist 1, with
  built-in 2 excluded. After further adjustment, the 14:52 UTC wrist image
  shows the gripper, yellow ball, and tabletop; its task-surface framing is
  corrected. Fixed 0 now includes part of the arm and desk, but remains blurred
  without a verified complete workspace. With motor power off, raise/back up
  the fixed camera and tilt it toward the task surface until the complete arm
  and reachable table area fit in one sharp view. Secure the camera and recheck.
  Clear the operator's hand from the gripper area before any motion.
- **Last pose status:** all six joints passed the numeric inset envelope on
  2026-09-04 at 14:48 UTC; lift/elbow read −73.055°/39.868°. Existing calibration
  again matches the arm. Mechanical support has not been
  freshly attested. Re-support, re-read, and pass the pose immediately before
  arming; do not use this snapshot as persistent clearance.
- The operator reports no known approved low-limit profile. Reviewed setup
  code and manufacturer documentation explain the controls but do not establish
  a low-limit profile for this assembled arm. Calibration does not establish
  low motor torque or speed; the observed settings remain unapproved.
- Using the operator's known-good Hiwonder/ServoStudio procedure, establish low
  torque, current, velocity, and acceleration limits. Do not copy the observed
  defaults in `PREFLIGHT.md`. Save the exact readback for every joint and every
  required register as a JSON safety profile outside the tracked tree, then
  validate it without opening hardware:

  ```sh
  .cache/hardware/client-venv/bin/python -c \
    'from mlx_smolvla.hardware_safety import load_hardware_safety_profile; import sys; print(load_hardware_safety_profile(sys.argv[1]))' \
    '<ABSOLUTE_PATH_TO_OPERATOR_VERIFIED_PROFILE_JSON>'
  ```

- Clear the motion envelope, secure the base, test the physical power cut, and
  keep one hand on the switch. Use the already validated serve-only/client-only
  environment split in `docs/HARDWARE_RUNBOOK.md`; do not substitute the
  all-extras development environment.
- Ordinary `robot teleops` success does not replace these steps: that command
  writes the vendor's 254 acceleration defaults and streams leader targets,
  while the autonomous MLX client must verify the separate inset-pose and
  exact low-profile gates before enabling torque.
- The client now prevents stale-goal jumps by preloading and exactly verifying
  the current raw position as the goal while torque is off. Outbound and
  return motion are both bounded to gradual one-public-unit steps. This does
  not replace the low-limit or physical checks.
- After reconnecting and completing preflight, the operator must run a fresh
  60-second `--no-motion` check from the runbook before any single action.
  Use new server/client log paths, require zero writes/timeouts and verified
  torque-off, and review camera freshness and rejected/clamped/rate-limited
  chunks. The earlier four successful runs remain historical evidence.
- After all listed items are physically true, send this exact new in-session
  statement before a single-action attempt:

  ```text
  MOTION PREREQUISITES CONFIRMED: cameras framed, arm neutral, low limits profiled, workspace clear, base secure, hand on power.
  ```

- Then follow only the `--single-action` command in the runbook. Do not run
  `--continuous` until the one-action direction, displacement, speed, gripper,
  cameras, telemetry, return-to-start, and torque-off results are reviewed.

## Done — correct and verify camera role mapping

- **Status:** done on 2026-09-03. The frame previously called the fixed camera
  was the built-in Mac camera. Fresh labeled captures established the current
  OpenCV roles as fixed index 0 and wrist index 1; index 2 is excluded.
- Both intended UVC cameras negotiated 640x480/30 and returned nonblack frames
  concurrently. The fixed view contains the task surface. The wrist view is
  pointed at the desk and is soft at its parked close-focus distance, not
  obstructed. The operator confirmed the framing.
- The corrected mapping completed a 60-second native MLX no-motion loop with
  293 observations, 292 chunks, 4.876 sampled FPS, zero timeouts, and all six
  torque bits zero afterward. Camera indices must still be visually rechecked
  after device changes as documented in `docs/HARDWARE_RUNBOOK.md`.

## Open — make the GitHub repository public

- After the hardware gate and final verification pass, review the tracked tree.
  Obtain separate explicit authorization to make the repository public before
  running this command; the continuation's commit/push authorization does not
  cover visibility changes:

  ```sh
  gh repo edit daniiarabdiev/mlx-smolvla \
    --visibility public \
    --accept-visibility-change-consequences
  ```

- Verify the public, logged-out view before sharing the link. Do not run this
  command while the hardware blocker remains open.

## Open — publish the Python distributions

- After the supervised hardware evidence is committed and `v0.1.0` is tagged,
  rebuild from that tag into a new directory. Do not publish the current
  untagged candidate merely because its smoke matrix passed:

  ```sh
  FINAL_RELEASE_SOURCE="$PWD/.cache/release-v0.1.0-source"
  FINAL_RELEASE_ARTIFACTS="$PWD/.cache/release-v0.1.0-artifacts"
  UV_CACHE_DIR="$PWD/.cache/uv"
  UV_PYTHON_INSTALL_DIR="$PWD/.cache/uv-pythons"
  export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
  test ! -e "$FINAL_RELEASE_SOURCE" && test ! -e "$FINAL_RELEASE_ARTIFACTS"
  git worktree add --detach "$FINAL_RELEASE_SOURCE" v0.1.0
  mkdir -p "$FINAL_RELEASE_ARTIFACTS"
  (cd "$FINAL_RELEASE_SOURCE" && \
    MACOSX_DEPLOYMENT_TARGET=14.0 uv build --sdist \
      --out-dir "$FINAL_RELEASE_ARTIFACTS" && \
    MACOSX_DEPLOYMENT_TARGET=14.0 uv build --wheel --python 3.11 \
      --out-dir "$FINAL_RELEASE_ARTIFACTS" && \
    MACOSX_DEPLOYMENT_TARGET=14.0 uv build --wheel --python 3.12 \
      --out-dir "$FINAL_RELEASE_ARTIFACTS" && \
    MACOSX_DEPLOYMENT_TARGET=14.0 uv build --wheel --python 3.13 \
      --out-dir "$FINAL_RELEASE_ARTIFACTS")
  uvx --from twine twine check "$FINAL_RELEASE_ARTIFACTS"/*
  ```

- Repeat the archive inspection and all seven fresh-install environments,
  including base offline predictions/`doctor`, serving/quantization, hardware
  imports without device access, cache compatibility, reference/training
  exact-weight checks, and hash capture from
  [`../evidence/DIST_MANIFEST.md`](../evidence/DIST_MANIFEST.md). Refresh that
  manifest with the tag-built bytes and commit/push it before uploading.
- Obtain separate explicit authorization for the PyPI upload after the final
  artifact set and manifest are reviewable. Recheck that the name is still
  available immediately before that authorized publication:

  ```sh
  python -c 'import urllib.request; urllib.request.urlopen("https://pypi.org/pypi/mlx-smolvla/json")'
  ```

  A `404` means the name still appears unclaimed; any successful response means
  stop and resolve the naming conflict.
- Supply the operator's PyPI credential securely through the environment,
  outside this repository and shell history. With explicit upload authorization
  and that credential already configured, upload only the final
  manifest-matched artifacts:

  ```sh
  uv publish \
    .cache/release-v0.1.0-artifacts/mlx_smolvla-0.1.0*
  ```

- Never write the token to a file, shell history, issue, log, or commit.

## Open — create the GitHub Release

- Only after a verified `v0.1.0` tag exists, PyPI publication succeeds, and
  separate explicit authorization to create the GitHub Release is supplied:

  ```sh
  gh release create v0.1.0 \
    .cache/release-v0.1.0-artifacts/mlx_smolvla-0.1.0* \
    --repo daniiarabdiev/mlx-smolvla \
    --title 'mlx-smolvla v0.1.0' \
    --notes-file CHANGELOG.md
  ```

## Optional — publish converted weights deliberately

- The software release does not need redistributed weights. Only with separate
  explicit Hub-upload authorization, reviewed licensing/provenance, and a
  complete converted checkpoint, use a new model repository and upload the audited
  conversion plus its source revision/name map—not local caches or datasets:

  ```sh
  hf upload <USER>/mlx-smolvla-converted /path/to/audited-conversion . \
    --repo-type model
  ```

## Optional — add the hardware video after validation

- After the hardware report passes, place the reviewed ≤20-second, ≤8-MiB file
  at `docs/media/so101-first-contact.webp`, update the README slot, rerun both
  test lanes, and commit it with the matching first-contact evidence.

## Done — finish the GitHub repository rename

- **Status:** done — at `2026-09-02T11:53:16Z`, read-only SSH lookup of
  `git@github.com:daniiarabdiev/mlx-smolvla.git` returned `Repository not
  found`, while the existing `origin` still fetched `main` at `a77e820` from
  `git@github.com:daniiarabdiev/smolvla-mlx.git`.
- **Action:** in GitHub, confirm the repository's exact name is
  `mlx-smolvla` and that the current SSH identity has read/write access. Then
  verify from this Mac:

  ```sh
  git ls-remote git@github.com:daniiarabdiev/mlx-smolvla.git HEAD refs/heads/main
  ```

- **Why:** the rename brief forbids changing `origin` before the new endpoint
  is fetchable. The local package, import, CLI, cache, and public text can be
  migrated independently; pushes continue safely to the existing `origin`
  until this endpoint resolves.
- **Resolution:** the Stage B checkpoint push returned GitHub's authoritative
  moved-repository notice. A subsequent `git ls-remote` of the new endpoint
  resolved `main` to `da1bb4d`; `origin` was then changed to the new SSH URL,
  fetched successfully, and matched local `HEAD` exactly.

## Done — check the `mlx-smolvla` PyPI name

- **Status:** done — both the official JSON endpoint and Simple Repository API
  returned HTTP 404 at `2026-09-02T11:48Z` and again during closing
  verification at `2026-09-02T14:45:08Z`.
- **Result:** `mlx-smolvla` appears unclaimed on public PyPI at the time of the
  read-only check. This is not a reservation; the operator must check again
  immediately before publishing.
- **Sources:** <https://pypi.org/pypi/mlx-smolvla/json> and
  <https://pypi.org/simple/mlx-smolvla/>.

## Done — confirm the supervised hardware session

- **Status:** done — the operator supplied the exact gate in the live task on
  2026-09-02. Follower serial/calibration reads, both cameras, and three bounded
  no-motion MLX loops then ran across 2026-09-02 and 2026-09-03. The leader was
  not opened and no motor or torque write occurred.
- **Gate supplied:**

  ```text
  ARM SESSION CONFIRMED
  ```

- **Remaining scope:** this resolved device-read/no-motion authorization, not
  the failed physical prerequisites listed above. Motion and the public
  hardware claim remain blocked.

## Done — provide the normative release brief

- **Status:** done — supplied by the operator and committed as
  `docs/history/BRIEF_RELEASE.md` on 2026-09-01.
- **Needed file:** `docs/history/BRIEF_RELEASE.md`, exactly as approved during the release
  planning work.
- **Action:** place `BRIEF_RELEASE.md` at the repository root or attach its
  complete text in this task.
- **Why:** `docs/history/BRIEF_FULL.md` requires the exact P0-1 through P1-4 and P2 package
  specifications and explicitly forbids re-deriving them. Searches of the
  repository, Git history, attachments, Downloads, unreachable objects, and
  the initially empty GitHub remote did not recover it.
- **Resolution:** the specification blocker is closed. Stage R, then Stage Q
  and Stage H, may proceed in the order amended by `docs/history/BRIEF_T3B.md`.
