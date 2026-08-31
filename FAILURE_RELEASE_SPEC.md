# Stage R Blocker — Missing `BRIEF_RELEASE.md`

## Status

Stage R cannot be certified because its normative package specification,
`BRIEF_RELEASE.md`, was not supplied with `BRIEF_FULL.md` and is absent from
the project. The release requirements must not be reconstructed from guesses.

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
