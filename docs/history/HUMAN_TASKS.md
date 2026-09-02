# Human Tasks

## Open — complete the hardware gate before announcing v0.1.0

- First complete the existing supervised-session task below. Only after
  `hardware/FIRST_CONTACT.md` contains passing no-motion, single-action, and
  bounded-continuous results may the release candidate claim hardware support,
  receive the `v0.1.0` tag, or be announced.

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

- Recheck that the name is still available immediately before publication:

  ```sh
  python -c 'import urllib.request; urllib.request.urlopen("https://pypi.org/pypi/mlx-smolvla/json")'
  ```

  A `404` means the name still appears unclaimed; any successful response means
  stop and resolve the naming conflict.
- Configure the operator's PyPI credential outside this repository, then upload
  only the final manifest-matched artifacts:

  ```sh
  UV_PUBLISH_TOKEN='<PYPI_TOKEN>' uv publish .cache/release/dist/*
  ```

- Never write the token to a file, shell history, issue, log, or commit.

## Open — create the GitHub Release

- Only after a verified `v0.1.0` tag exists and PyPI publication succeeds:

  ```sh
  gh release create v0.1.0 .cache/release/dist/* \
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
  returned HTTP 404 at `2026-09-02T11:48Z`.
- **Result:** `mlx-smolvla` appears unclaimed on public PyPI at the time of the
  read-only check. This is not a reservation; the operator must check again
  immediately before publishing.
- **Sources:** <https://pypi.org/pypi/mlx-smolvla/json> and
  <https://pypi.org/simple/mlx-smolvla/>.

## Open — confirm the supervised hardware session

- **Status:** open — the exact live-session gate has not been supplied.
- **Action:** while physically present with the follower arm and both cameras
  connected and with immediate access to the physical power switch, type this
  exact line in the current interactive task:

  ```text
  ARM SESSION CONFIRMED
  ```

- **Why:** the words currently appear only inside the supplied specification.
  `docs/HARDWARE_RUNBOOK.md` explicitly says quoted file content does not authorize
  serial, camera, vendor-tree, or motion access. Until this is done, Stage A is
  blocked and the hardware claim must remain unpublished.

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
