# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-09-05

### Fixed

- Correct the packaged PyPI description to show current installation instructions
  instead of pre-publication wording. Runtime behavior is unchanged.

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

### Fixed

- Preserve fp32 native-trained weights when loading the optional PyTorch
  reference, by selecting the destination dtype before strict state loading.
  The retained 56-case LoRA export now passes every original fixed limit;
  [repair evidence](docs/evidence/TRAINED_PARITY_REPAIR.md) preserves the prior
  verdicts and records the separate validation.
- Corrected Hiwonder camera guidance to assign session-local roles by visual
  preflight and exclude built-in or Continuity cameras; both startup orders
  pass with the two intended UVC devices.
- Prevented torque-enable against retained servo goals by preloading and
  exactly verifying fresh raw positions while torque remains off. Outbound and
  return-to-start commands are both bounded to gradual one-public-unit steps.
- Recheck the complete controller-limit profile after goal preload, require
  integer position mode, and reject startup force above the torque cap or
  changed during arming preparation.
- Treat the first goal-position write as an arming boundary. Raw present values
  must pass controller minimum/maximum checks first; explicit-torque and
  observed goal-write arming modes require exact all-six readback and verified
  torque-off cleanup on every post-write failure.
- Let single-action mode hold rejected chunks within its 20-chunk attempt cap
  and stop immediately after the first valid non-hold action.

### Known limitations

- Connected Hiwonder SO-101 state/camera capture, one guarded valid action, and
  a two-chunk continuous run passed with exact return and torque-off shutdown.
  A separate 20-chunk attempt failed exact return under the temporary 10%
  torque profile while still disabling torque. Sustained motion and reliable
  task completion are not claimed. See the
  [measured evidence](hardware/FIRST_CONTACT.md).
- Raw `lerobot/smolvla_base` output lacks effective generic-key physical
  statistics and is restricted to no-motion diagnostics by the fail-closed
  client. Motion requires a reviewed stats-active checkpoint and an
  operator-attested hardware-limit profile.
- Native MLX training remains a research preview: the retained expert-only
  LoRA export passes post-repair strict parity, but full fine-tuning has only
  code/smoke coverage and no robot task-success claim is made.

Evidence and reproducibility details are indexed in
[`docs/evidence/`](docs/evidence/README.md).

[0.1.0]: https://github.com/daniiarabdiev/mlx-smolvla/releases/tag/v0.1.0

[0.1.1]: https://github.com/daniiarabdiev/mlx-smolvla/releases/tag/v0.1.1
