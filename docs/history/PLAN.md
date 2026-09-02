# Execution Plan

The detailed test-first implementation plans are
`docs/superpowers/plans/2026-08-31-smolvla-mlx-v0.1.md` and
`docs/superpowers/plans/2026-08-31-native-rmsnorm-extension.md`.

- [x] Phase 0 — bootstrap the pinned PyTorch reference lane and regenerate
  deterministic real-frame goldens.
- [x] Phase 1 — audit the checkpoint/reference architecture and record reuse
  decisions.
- [x] Phase 2 — convert every checkpoint tensor to a strict MLX name map.
- [x] Phase 3 — reproduce preprocessing, vision, connector, prefix cache,
  action expert, and Euler flow at the fixed fp32/bf16 tolerances.
- [x] Phase 4 — validate deterministic and 50-frame statistical parity; ship
  the queue-backed policy API.
- [x] Phase 5 — measure native Metal inference and record the benchmark.
- [x] Phase 6 — ship the CLI, pinned dependency metadata, source/wheel
  artifacts, clean-install proof, and real dataset prediction proof.

## Completion record

v0.1 met the `BRIEF.md` Definition of Done on 2026-08-31. The final exact-tree
verification was `make test`: 179 passed in 147.26 seconds. `PROGRESS.md` and
`STATUS.md` contain the numerical, packaging, golden-reproducibility, and
fresh-environment evidence.
