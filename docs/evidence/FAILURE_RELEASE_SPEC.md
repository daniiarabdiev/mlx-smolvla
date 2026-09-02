# Stage R Blocker — Missing `BRIEF_RELEASE.md`

## Resolution

**Resolved 2026-09-01.** The operator supplied the complete normative
`BRIEF_RELEASE.md` at the repository root, together with `BRIEF_T3B.md`. The
missing-specification blocker is closed. This file remains as the historical
record of why Stage R, Q, and H previously stopped; Stage R is now eligible to
proceed under the supplied package definitions.

## Status

At the time of this failure record, Stage R could not be certified because its
normative package specification, `BRIEF_RELEASE.md`, had not been supplied with
`BRIEF_FULL.md` and was absent from the project. The release requirements were
not reconstructed from guesses.

The operator-provided remote setup was completed safely because the GitHub
repository was empty: the existing verified `main` history was pushed to
`git@github.com:daniiarabdiev/smolvla_mlx.git`. This does not establish that
P0-1 or the rest of Stage R meet the missing acceptance criteria.

## Searches performed

1. Repository and Git history: no tracked, untracked, historical, or reachable
   `BRIEF_RELEASE.md`; no commits contain its P0/P1 package markers.
2. Supplied attachments and Downloads: the only matching content is the
   attached `BRIEF_FULL.md`, which references but does not embed the release
   package definitions.
3. GitHub origin: `git ls-remote` showed an empty repository before the initial
   push, so there was no remote branch from which to recover the brief.
4. Unreachable local objects: the inspected historical trees predate the
   release brief and contain only the v0.1 files.

## Consequence and continuation

- P0-2, P0-3, P1-1, P1-2, P1-3, P1-4, and the re-homed Stage Q definitions
  remain blocked.
- `RELEASE READY` must not be written to `STATUS_FULL.md` until the exact brief
  is restored and every package gate is executed or separately
  FAILURE-documented.
- Stage T0 has no Stage R dependency, so the training-readiness work proceeds
  while this input is outstanding.
