# Public release checklist

This checklist adapts current upstream guidance to `mlx-smolvla`. It is a
release gate, not a substitute for the numerical and runtime contracts in the
repository evidence.

## Repository and community health

- [x] Keep the root README concise, actionable, and linked to deeper evidence.
  Source: [GitHub repository best practices](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories).
- [x] Ship an explicit license, contribution guide, code-of-conduct reference,
  citation metadata, security policy, and issue forms in their conventional
  locations. Source: [GitHub community profiles](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/about-community-profiles-for-public-repositories).
- [x] Give vulnerability reporters a private path and state which release line
  receives security fixes. Source: [GitHub security quickstart](https://docs.github.com/en/code-security/getting-started/quickstart-for-securing-your-repository).
- [x] Keep workflows least-privileged, secret-free for tests, and disabled until
  their documented hardware requirements are available. Source: [GitHub secure
  repository guidance](https://docs.github.com/en/code-security/getting-started/quickstart-for-securing-your-repository).
- [x] Remove operator paths, machine identifiers, caches, build products, and
  private operating instructions from the public tracked surface. Source:
  [GitHub repository best practices](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories).

## Python distribution

- [x] Set the public version and complete `pyproject.toml` metadata, including
  README, Python range, license expression, classifiers, keywords, and canonical
  project URLs. Source: [PyPA `pyproject.toml` guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
- [x] Build both an sdist and platform wheels from a clean, pushed
  release-candidate source checkpoint, inspect their file lists, install them
  into clean environments, and exercise the installed CLI rather than the
  checkout. Source: [PyPA packaging flow](https://packaging.python.org/en/latest/flow/).
- [x] After the supervised hardware gate and final verification clear, create
  the annotated tag on the committed, pushed final source. Build a fresh sdist
  and CPython 3.11/3.12/3.13 wheels from that tag into a new directory; repeat
  the fresh-install smoke matrix and record exact hashes in
  [DIST_MANIFEST.md](../evidence/DIST_MANIFEST.md). Preserve prior artifacts;
  the existing untagged candidate must not be uploaded.
- [x] Render-check the long description and validate every distribution with
  `twine check` before publication. Source: [PyPA PyPI-friendly README guide](https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/).
- [ ] Optional future automation: use PyPI trusted publishing with narrowly
  scoped GitHub permissions. Active CI and trusted publishing are not v0.1.0
  prerequisites; an explicitly authorized operator upload may use credentials
  supplied securely outside the repository and shell history. Source:
  [PyPA GitHub Actions publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).

## First-use experience

- [x] Put requirements, installation, a minimal Python example, and one CLI
  command before architecture detail. Source: [MLX-LM README](https://github.com/ml-explore/mlx-lm/blob/main/README.md).
- [x] Separate base installation from optional serving/training dependencies and
  show explicit feature commands. Source: [LeRobot installation guide](https://github.com/huggingface/lerobot/blob/main/docs/source/installation.mdx).
- [x] Document checkpoint conversion/loading, quantization opt-ins, and agent
  guidance without making them prerequisites for the basic path. Source:
  [MLX-VLM README](https://github.com/Blaizzy/mlx-vlm/blob/main/README.md).
- [x] Link contribution and citation instructions from the first page and keep
  research/evidence detail in dedicated documents. Sources: [LeRobot README](https://github.com/huggingface/lerobot/blob/main/README.md) and [LeRobot contributing guide](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md).

## Project-specific release gates

- [x] `make test-fast` passes in under two minutes and `make test` passes with no
  skips or expected failures.
- [x] The strict deterministic and production statistical tolerances remain
  unchanged; every performance or correctness number in the README resolves to
  committed evidence.
- [x] Runtime import isolation still proves that `mlx_smolvla` does not import
  Torch, Transformers, or LeRobot.
- [x] `mlx-smolvla doctor` succeeds from every freshly installed artifact and
  reports the compatibility verdict, Metal state, extras, and cache details.
- [x] The sdist and every Python 3.11-3.13 wheel use canonical
  `mlx-smolvla` / `mlx_smolvla` archive paths, distribution metadata, imports,
  and CLI entry points, and pass offline prediction plus loopback serving
  smoke tests where their extras apply. Stable format-v1 evidence identifiers
  are the documented compatibility exception.
- [x] The tracked-root allowlist, internal-link check, personal-detail scan, and
  artifact manifest all pass from the final tree.
- [x] Hardware claims match the committed first-contact evidence: one valid
  guarded action and a two-chunk continuous run passed, while the failed
  20-chunk exact return remains a disclosed sustained-operation limitation.
- [x] No package, model, release, or documentation artifact is uploaded as part
  of automated preparation; publication remains an operator action.
