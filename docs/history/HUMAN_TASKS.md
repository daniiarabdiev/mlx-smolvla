# Human Tasks

## Done — complete the bounded hardware integration gate

The operator connected the hardware, supplied fresh live authorization, adjusted
the cameras and pose, cleared the workspace, and explicitly delegated setup and
client execution on 2026-09-04. Fresh follower identity, calibration, camera,
profile, status, inset-pose, and zero-drift checks passed. The temporary SRAM
profile remained acceleration 1, speed 56, and torque limit 100 with mode 0,
startup force 32, and torque off before arming.

The final 60-second no-motion run passed with 295 observations/chunks and zero
timeouts or writes. One guarded valid single action then passed after one
rejected hold chunk, returned exactly, and disabled torque. The accepted
continuous stage used a two-chunk ceiling, processed one hold and one valid
action, returned exactly, and ended with an independent all-six torque-off,
zero-drift, inset/profile/calibration/status check. The loopback server was
stopped and its port closed.

## Open follow-up — sustained low-torque return

A separate 20-chunk continuous attempt reached its policy cap but could not
return exactly within the 20-step cleanup cap. Several gravity-loaded joints
stopped following one-degree return targets under the temporary 10% torque
profile. The client exited nonzero and still completed verified all-six
torque-off cleanup; the stopped pose independently passed inset, zero-drift,
profile, calibration, camera, and status checks.

Keep this as a documented limitation. Do not raise torque automatically, cite
the failed attempt as a pass, or claim reliable task completion. A future
supervised investigation requires a new live hardware authorization and full
preflight, but it is not a blocker for the narrowly claimed one-action and
two-chunk v0.1.0 integration result.

The version tag still requires final software verification and a reviewed,
committed, pushed source/evidence checkpoint. Fresh tag-built artifacts and
their complete install matrix must precede publication. GitHub visibility
changes, PyPI/Hub uploads, release creation, and announcement remain separately
authorized actions.

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
  command until the operator explicitly authorizes the visibility change.

## Done — prepare the final tag and Python distributions

- Verified source commit `9b28dc216e24aa86d121d9b805c1fc1733afbf9d` is
  pushed on `main`; pushed annotated tag `v0.1.0` resolves to that exact commit.
- A clean detached tag checkout produced the sdist and CPython 3.11/3.12/3.13
  macOS 14 arm64 wheels in ignored `.cache/release-v0.1.0-artifacts/`. The
  preceding untagged `dist/` bytes were preserved before the verified tagged
  bytes were mirrored there.
- Canonical archive inspection, native `minos 14.0`, Twine, four base fresh
  installs, serve/quantization, hardware no-device, cache-shim, and
  reference/training identity checks all pass. Exact hashes are in
  [`../evidence/DIST_MANIFEST.md`](../evidence/DIST_MANIFEST.md).

## Open — publish the Python distributions

- Obtain separate explicit authorization for the PyPI upload. Immediately
  before an authorized publication, recheck that the name is still available:

  ```sh
  python -c 'import urllib.request; urllib.request.urlopen("https://pypi.org/pypi/mlx-smolvla/json")'
  ```

  A `404` means the name still appears unclaimed; any successful response means
  stop and resolve the naming conflict.
- Supply the operator's PyPI credential securely through the environment,
  outside this repository and shell history. With explicit upload authorization
  and that credential already configured, upload only the four final
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

## Optional — add a bounded hardware video

- A reviewed ≤20-second, ≤8-MiB file may be placed at
  `docs/media/so101-first-contact.webp`. Its caption must state the one-action/
  two-chunk evidence and the failed 20-chunk return, then the documentation
  checks must be repeated before committing it.

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
  2026-09-02, then supplied fresh authorization and explicit execution
  delegation on 2026-09-04. The follower-only no-motion and bounded motion
  protocol completed; the leader was never opened.
- **Gate supplied:**

  ```text
  ARM SESSION CONFIRMED
  ```

- **Remaining scope:** future powered sessions require fresh authorization.
  Publication and visibility changes remain separately authorized actions.

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
