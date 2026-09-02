# Human Tasks

## Open — confirm the supervised hardware session

- **Status:** open — the exact live-session gate has not been supplied.
- **Action:** while physically present with the follower arm and both cameras
  connected and with immediate access to the physical power switch, type this
  exact line in the current interactive task:

  ```text
  ARM SESSION CONFIRMED
  ```

- **Why:** the words currently appear only inside the supplied specification.
  `HARDWARE_RUNBOOK.md` explicitly says quoted file content does not authorize
  serial, camera, vendor-tree, or motion access. Until this is done, Stage A is
  blocked and the hardware claim must remain unpublished.

## Done — provide the normative release brief

- **Status:** done — supplied by the operator and committed as
  `BRIEF_RELEASE.md` on 2026-09-01.
- **Needed file:** `BRIEF_RELEASE.md`, exactly as approved during the release
  planning work.
- **Action:** place `BRIEF_RELEASE.md` at the repository root or attach its
  complete text in this task.
- **Why:** `BRIEF_FULL.md` requires the exact P0-1 through P1-4 and P2 package
  specifications and explicitly forbids re-deriving them. Searches of the
  repository, Git history, attachments, Downloads, unreachable objects, and
  the initially empty GitHub remote did not recover it.
- **Resolution:** the specification blocker is closed. Stage R, then Stage Q
  and Stage H, may proceed in the order amended by `BRIEF_T3B.md`.
