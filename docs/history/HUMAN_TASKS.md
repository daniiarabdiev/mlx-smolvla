# Human Tasks

## Open — complete the hardware gate before announcing v0.1.0

- The follower/camera preflight and three 60-second no-motion loops completed
  across 2026-09-02 and 2026-09-03. The latest run includes the corrected
  fixed-before-wrist camera startup order. First clear the physical
  prerequisites below, then complete the
  single-action and bounded-continuous stages. Only after
  `hardware/FIRST_CONTACT.md` contains passing results for all three graduated
  modes may the release candidate claim hardware motion support, receive the
  `v0.1.0` tag, or be announced.

## Open — clear the physical prerequisites for one action

- Power the follower off before touching or repositioning the cameras or arm.
  Re-aim and secure the wrist camera so its full workspace is visible, and aim
  the fixed camera at the complete arm workspace. Remove bystanders from both
  views. Then repeat the concurrent five-second camera check from
  [`../../hardware/PREFLIGHT.md`](../../hardware/PREFLIGHT.md); a nonblack but
  obstructed frame does not pass. The 2026-09-03 adjustment made both streams
  nonblack, but visual review still found the wrist view blurred/too close and
  the fixed view outside the robot workspace.
- With torque disabled, manually place `shoulder_lift` and `elbow_flex` near
  their calibrated neutral positions. At minimum, readback must be inside
  −83.833°–83.833° for lift and −77.187°–77.187° for elbow. Do not recalibrate
  or command the motors to reach that pose.
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
- After all four items are physically true, send this exact new in-session
  statement before a single-action attempt:

  ```text
  MOTION PREREQUISITES CONFIRMED: cameras framed, arm neutral, low limits profiled, workspace clear, base secure, hand on power.
  ```

- Then follow only the `--single-action` command in the runbook. Do not run
  `--continuous` until the one-action direction, displacement, speed, gripper,
  cameras, telemetry, return-to-start, and torque-off results are reviewed.

## Open — make the GitHub repository public

- After the hardware gate and final verification pass, review the tracked tree
  and run:

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

- Repeat the archive inspection, four base fresh installs, offline predictions,
  `doctor`, serve-extra loopback, and hash capture from
  [`../evidence/DIST_MANIFEST.md`](../evidence/DIST_MANIFEST.md). Refresh that
  manifest with the tag-built bytes and commit/push it before uploading.
- Recheck that the name is still available immediately before publication:

  ```sh
  python -c 'import urllib.request; urllib.request.urlopen("https://pypi.org/pypi/mlx-smolvla/json")'
  ```

  A `404` means the name still appears unclaimed; any successful response means
  stop and resolve the naming conflict.
- Configure the operator's PyPI credential outside this repository, then upload
  only the final manifest-matched artifacts:

  ```sh
  UV_PUBLISH_TOKEN='<PYPI_TOKEN>' uv publish \
    .cache/release-v0.1.0-artifacts/mlx_smolvla-0.1.0*
  ```

- Never write the token to a file, shell history, issue, log, or commit.

## Open — create the GitHub Release

- Only after a verified `v0.1.0` tag exists and PyPI publication succeeds:

  ```sh
  gh release create v0.1.0 \
    .cache/release-v0.1.0-artifacts/mlx_smolvla-0.1.0* \
    --repo daniiarabdiev/mlx-smolvla \
    --title 'mlx-smolvla v0.1.0' \
    --notes-file CHANGELOG.md
  ```

## Optional — publish converted weights deliberately

- The software release does not need redistributed weights. If licensing and
  provenance are reviewed and a complete converted checkpoint is intentionally
  published later, use a new model repository and upload only the audited
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
