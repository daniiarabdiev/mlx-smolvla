# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-02

Initial public release candidate.

### Added

- Native MLX conversion and inference for the pinned SmolVLA base checkpoint,
  with production Metal and strict CPU execution modes.
- Fixed deterministic and statistical parity gates covering the vision stack,
  connector, truncated SmolVLM language prefix, action expert, flow matching,
  normalization, and postprocessing.
- Optional LeRobot 0.6.1 protocol serving, VLM weight-only quantization presets,
  and native MLX LoRA/full-training surfaces.
- `mlx-smolvla doctor` diagnostics and verified MLX 0.32.0-0.32.2 compatibility
  on Apple Silicon macOS 14 or newer.
- Standard 500-tensor LeRobot checkpoint export, PyTorch/LeRobot round-trip
  loading, exact-resume smokes, and committed training/correctness evidence.

### Known limitations

- Physical SO-101 integration has not yet passed the separately authorized
  hardware-validation protocol; no real-robot claim is made in this release
  candidate.
- Native MLX training is a preview: lockstep primitives and exported-checkpoint
  round trips pass, while the deterministic parity gap for an MLX-trained
  checkpoint remains documented and unresolved.

Evidence and reproducibility details are indexed in
[`docs/evidence/`](docs/evidence/README.md).

[0.1.0]: https://github.com/daniiarabdiev/mlx-smolvla/releases/tag/v0.1.0
