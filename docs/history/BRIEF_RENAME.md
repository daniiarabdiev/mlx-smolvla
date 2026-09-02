# BRIEF_RENAME.md — canonical `mlx-smolvla` identity amendment

This operator amendment is normative before Stage D of
`BRIEF_PUBLIC_RELEASE.md`. The canonical identities are:

- GitHub repository: `daniiarabdiev/mlx-smolvla`
- PyPI distribution and CLI: `mlx-smolvla`
- Python import package: `mlx_smolvla`
- cache environment variable: `MLX_SMOLVLA_CACHE`
- default cache directory: `~/.cache/mlx_smolvla`

The package directory must be moved with `git mv`. Public class names remain
unchanged. For one release, the cache resolver must accept the legacy
`SMOLVLA_MLX_CACHE` environment variable and emit a warning. All package
metadata, entry points, imports, tests, active documentation, build files,
workflows, agent guidance, and the prepared Hub model card must use the new
identity.

Before changing `origin`, prove read access to
`git@github.com:daniiarabdiev/mlx-smolvla.git`. Check the PyPI distribution
name read-only and record an operator task if it is occupied.

Remove the erroneous public claim that this was the first Apple Silicon or MLX
inference port of SmolVLA. Historical records under `docs/history/` remain
unaltered and their index must record the correction dated 2026-09-02. The
README must acknowledge `tokimoa/smolvla-mlx` as an earlier independent port,
give its Hugging Face Hub link and Hub upload date, state factual differences
without comparative performance claims, and preserve the repository's own
evidence boundaries.

After the rename, run the complete suite, rebuild every release artifact under
the new names, repeat every fresh-install smoke, refresh `DIST_MANIFEST.md`,
commit, and push.
