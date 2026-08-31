# Progress

## 2026-08-31 — project start

- What: accepted the native SmolVLA MLX v0.1 specification and created the
  design and implementation plan.
- Evidence: design and plan self-reviews found 0 placeholders and 0 unresolved
  interface inconsistencies.
- Decision: use a reference-first hybrid port, reusing compatible `mlx-vlm`
  code and vendoring only where verified behavior requires it.
- Open questions: the architecture questions in `BRIEF.md` Section 10 will be
  resolved from the installed LeRobot source and checkpoint.
- Next: complete Phase 0 bootstrap and reference discovery.

## 2026-08-31 — Phase 0 bootstrap

- What: created the Python 3.12 uv project, in-repo cache routing, MLX runtime
  dependency set, package skeleton, Make targets, and required state files.
- Environment: macOS 26.5.2 (25F84), Apple M5 Pro, 51,539,607,552 bytes
  unified memory, Python 3.12.13, uv 0.11.25, MLX 0.32.2, mlx-vlm 0.6.17.
- Device evidence: `mx.default_device()` returned `Device(gpu, 0)` and
  `mx.metal.is_available()` returned `True`.
- Test evidence: `make test` collected 2 tests; 2 passed, 0 failed, 0 skipped
  in 0.02 seconds. The isolated subprocess loaded none of `torch`, `lerobot`,
  or `transformers`.
- Decision: exact resolved package versions are committed in `uv.lock`.
- Open questions: none for bootstrap.
- Next: install the optional reference lane and discover the exact mainline
  LeRobot SmolVLA sources, checkpoint revision, and usable SO-101 dataset.

## 2026-08-31 — Phase 0 reference discovery

- What: pinned and discovered the installed mainline reference without relying
  on remembered import paths; queried immutable Hub revisions and safetensor
  metadata without downloading the model weights.
- Reference: LeRobot 0.6.1, PyTorch 2.11.0, Transformers 5.5.4. Exact source
  paths are recorded in `ARCHITECTURE.md`.
- Checkpoint evidence: `lerobot/smolvla_base` revision
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`; 500 tensors and 450,046,176
  parameters from the Hub safetensor header.
- Dataset evidence: `lerobot/svla_so101_pickplace` revision
  `f641879e22172be7e8161d5e6c1503c2d2feb657`; 50 episodes, 11,939 frames,
  two 480x640 camera streams, 6-D state, 6-D action, and a language-task table.
- Test evidence: `tests/test_reference_discovery.py` collected 3 tests; 3
  passed, 0 failed, 0 skipped in 8.19 seconds.
- Decision: use the official SO-101 dataset named in the brief; no fallback is
  required. PyTorch is explicitly pinned to 2.11.0 in the reference extra.
- Open questions: the runtime architecture and preprocessing details remain
  for the Phase 1 source/runtime audit.
- Next: download the pinned checkpoint into the repository cache, load the
  policy on CPU fp32, and execute one real observation.
