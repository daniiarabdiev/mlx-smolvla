# Human Tasks

## Open — complete the hardware gate before announcing v0.1.0

The operator supplied live `ARM SESSION CONFIRMED`, connected the hardware,
adjusted the cameras, and explicitly delegated controller setup and client
execution on 2026-09-04. The current-session amendment in the
[runbook](../HARDWARE_RUNBOOK.md) supersedes the old operator-only command and
prescribed motion-phrase requirements. No repeated authorization phrase is
needed for this session.

Two fresh 60-second no-motion runs completed after the camera adjustment.
The cold-server attempt had one initial timeout; the warm repeat passed with
295 observations, 294 processed chunks, zero timeouts, and no actuator writes.
The four earlier successful runs remain historical evidence.

Low controller settings are now established under that delegation: temporary
SRAM acceleration 1, speed 56, and torque limit 100 on all six joints, with
exact readbacks and all torque bits off. This is a new commissioning profile,
not a claim that its gravity-hold behavior has been physically validated. The
profile and raw evidence remain private. Power cycling or other configuration
can reset these settings; the delegate must freshly establish and verify them
before any arming attempt. No automatic torque increase is permitted.

## Open — physical setup for the first supervised action

- The **17:31 UTC** recheck passes all six inset position checks after the
  operator's adjustment. Lift/elbow read **-54.330/33.187°**, with no measured
  two-second drift. This resolves the prior 16:52 elbow failure. Existing
  calibration and the exact reduced controller profile match; all torque
  bits are off, mode is 0, startup force is 32, and status has no alarms.
- Fresh images show the raised white follower and tabletop in the fixed view,
  but the wrist camera now faces the operator/ceiling. With motor power off,
  turn that camera mount downward toward the gripper and tabletop near the
  yellow ball while keeping the passing joint pose. Keep the cable outside
  the moving envelope. Camera identity remains fixed 0, wrist 1; built-in 2
  is excluded. The earlier
  short independent rate measurement was 20.10/6.34 FPS at 640x480; actual
  30 FPS was not demonstrated.
- Support the arm so it will not fall when torque is off. Secure the base and
  clear the working envelope.
  Restore power with hands clear and remain beside the physical power switch.
  Tell the delegate when the physical setup is ready for a new read-only
  camera, pose, and controller check. This is an outstanding physical action,
  not a missing software-execution authorization.
- The delegate must recheck the supported inset pose and session profile,
  obtain a fresh passing no-motion result for the final setup, and then run
  only one guarded `--single-action` while the operator stays ready to cut
  power. Preserve the exact controller checks, raw-goal preload, one-unit step
  bounds, watchdog, gradual return, and torque-off verification. Unexpected
  motion or uncertainty requires a physical stop and review.
- Review direction, displacement, speed, gripper behavior, camera freshness,
  telemetry, return-to-start, and torque-off before the bounded `--continuous`
  stage. Safe actuation validates integration, not reliable pick-and-place.

The version tag additionally requires final software verification and a
reviewed, committed, pushed source/evidence checkpoint. Fresh tag-built
artifacts and their complete install matrix must precede publication. GitHub
visibility changes, PyPI/Hub uploads, release creation, and announcement remain
separately authorized actions after those gates pass.

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
