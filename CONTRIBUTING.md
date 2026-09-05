# Contributing

Contributions are welcome when they preserve the project's evidence-backed
runtime and numerical contracts.

## Development setup

Use an Apple Silicon Mac with Python 3.12 or 3.13 for the complete reference
lane:

```bash
git clone https://github.com/daniiarabdiev/mlx-smolvla.git
cd mlx-smolvla
uv sync --all-extras
make test-fast
```

Run `make test` before proposing a change that affects model behavior,
conversion, serving, training, packaging, or evidence. The full suite uses
locally cached pinned model and dataset assets and can take several minutes.

## Numerical-gate policy

Acceptance tolerances are immutable after they have been used to judge an
implementation. Never loosen a gate to make a change pass. Diagnose a failure,
record it with reproducing evidence, and fix the implementation. Tests may not
be skipped or marked as expected failures to hide a regression.

The base `mlx_smolvla` runtime must remain free of imports from Torch,
Transformers, and LeRobot. Those dependencies belong only to explicitly
selected reference, serving, or training paths. Production remains the default
Metal execution mode; strict remains the explicit CPU parity mode.

## Proposing another checkpoint target

Open a feature request before implementation and include:

1. The exact Hub repository and immutable revision.
2. Its SmolVLA/SmolVLM architecture and configuration differences.
3. A complete proposed weight-name mapping and tensor inventory.
4. Real, pinned observations representative of the checkpoint's preprocessing.
5. Deterministic and statistical acceptance gates written before comparison.
6. A plan for runtime dependency isolation, conversion, offline reload, and a
   fresh-install smoke.

Reuse the nearest implementation under `mlx_smolvla/`; keep PyTorch reference
generation under `reference/`; add artifacts only through the documented
evidence workflow.

## Pull requests

Keep changes scoped, explain the user-visible behavior, list exact commands and
results, and link any new evidence. Do not include generated model weights,
datasets, caches, machine paths, credentials, or uploads. By participating, you
agree to the [code of conduct](.github/CODE_OF_CONDUCT.md).


## Dependency and release contracts

Ranges in `pyproject.toml` are the install contract. `uv.lock` is the evidence
lane: regenerate numerical evidence only from the lockfile. Reference/LeRobot
extras remain pinned because the protocol and comparison audits are specific
to those versions. MLX remains within the verified 0.32.0–0.32.2 range; the
0.32.0/0.32.1 CPU fallback remains supported. Pre-1.0 safetensors and tokenizers
stay within their tested minor lines until additional compatibility is checked.

Lab sources retain `training/` and `reference/` paths to preserve provenance
identifiers. Import them as `mlx_smolvla._lab.training` and
`mlx_smolvla._lab.reference`; distributions expose only the `mlx_smolvla`
top-level package. `_build_backend.py` belongs in the sdist, never the wheel.

README links use absolute URLs pinned to the release tag so PyPI links work
and each release cites the evidence as it existed at that release. Existing
result documents remain immutable; publish new validation separately.
