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
