# Hosted packaging and prepared Trusted Publishing

The hosted packaging lane is separate from the seeded numerical suite.
`.github/workflows/ci.yml` builds an sdist and a wheel on macOS 15 ARM64 for
Python 3.11, 3.12, and 3.13, with a macOS 14.0 deployment target. A fourth
Python 3.12 job resolves the wheel's direct dependencies at their lower bounds.
Every job checks distribution contents and installs the wheel in a fresh venv.

`tests/packaging/check_artifacts.py` uses only Python's standard library for
archive inspection. Its optional `--python` interpreter runs installed imports
from an empty temporary directory containing hostile `training` and `reference`
packages. It also checks the train CLI import closure without installing the
reference extras. Hugging Face access is disabled and cache paths are empty.
The checker runs `doctor` when MLX reports a Metal device; otherwise it prints
an explicit “NOT RUN” diagnostic for that check. These are package/import
checks, not checkpoint inference or numerical parity evidence.

The existing `.github/workflows/macos-15.yml` remains disabled. Its local
evidence, model assets, and capacity requirements do not fit standard hosted
runners. Neither new workflow invokes `make test-fast` or the complete suite.

## Release workflow

`.github/workflows/release.yml` is **dispatch-only**. Pushing `v0.1.2` or any
other tag cannot trigger publication. Its default `publish: false` builds and
validates artifacts for an already reviewed, existing `vMAJOR.MINOR.PATCH` tag.
The source job validates that the tag equals the package version, records the
exact commit, and builds the sdist once. All three Python wheel jobs build from
that same sdist and run the archive and installed checks. The final macOS job
checks the complete four-file set and that installing the cp313 wheel preserves
preinstalled NumPy 2.x and Transformers versions. Resolver conflicts fail the
job; no downgrade is silently accepted.

Publishing additionally requires both `publish: true` and the repository
variable `PYPI_TRUSTED_PUBLISHING_ENABLED` set to the literal string `true`.
Leave that variable unset until the publisher is registered and the workflow is
ready for an unpublished version. The Linux publish job alone receives
`id-token: write`; it uses the `pypi` environment and the PyPA action with
attestations enabled. It downloads the checked artifacts and does not rebuild
or execute repository code. There is no API-token secret or token fallback.

Configure the PyPI project's GitHub Trusted Publisher with:

| Field | Value |
| --- | --- |
| Owner | `daniiarabdiev` |
| Repository | `mlx-smolvla` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

Create the matching GitHub environment and use the desired branch/tag
restrictions. Required human reviewers are optional; this workflow introduces
no mandatory approval click. Dispatch against the appropriate workflow ref
and supply the existing reviewed tag as the `tag` input. Already published
versions cannot be reused, and this workflow intentionally does not tolerate
duplicate uploads. The current local Keychain credential remains available
for authorized local releases and is not revoked by this setup.

This file describes prepared infrastructure. It does not claim that PyPI
publisher registration, a hosted build, or an attested upload has completed.
Local publication does not acquire Trusted Publishing attestations merely by
including this workflow in the source tree.

## Local artifact check

After installing the built wheel and its base dependencies in a fresh venv:

```sh
python tests/packaging/check_artifacts.py \
  --wheel /absolute/path/to/mlx_smolvla-0.1.2-cp312-cp312-macosx_14_0_arm64.whl \
  --sdist /absolute/path/to/mlx_smolvla-0.1.2.tar.gz \
  --expected-version 0.1.2 \
  --python /absolute/path/to/venv/bin/python
```

The wheel argument may alternatively come from `MLX_SMOLVLA_WHEEL`. The
standalone entry point deliberately avoids the repository's pytest conftest,
which imports MLX and reference helpers before collecting packaging tests.

## Verified upstream references

Runner and action choices were checked on 2026-09-05. Actions are pinned to
commit IDs resolved from their official repositories: checkout v6, setup-python
v6, upload/download-artifact v4, and PyPA's release/v1 branch. Update pins
through normal review; they are not floating “latest” references.

- [GitHub-hosted runner architecture and capacity](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [Official macOS 15 ARM64 image](https://github.com/actions/runner-images/blob/main/images/macos/macos-15-arm64-Readme.md)
- [PyPA publishing action: Linux runtime and Trusted Publishing](https://github.com/pypa/gh-action-pypi-publish)
- [PyPI: producing attestations](https://docs.pypi.org/attestations/producing-attestations/)
- [PyPI: adding a Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
