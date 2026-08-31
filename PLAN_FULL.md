# Full-Scope Execution Plan

This plan tracks `BRIEF_FULL.md` without replacing the detailed, test-first
plans under `docs/superpowers/plans/`. Every completed package ends in a passing
focused test, a regression check appropriate to its risk, a commit, and a push
to `origin/main`.

## Kickoff and protected baseline

- [x] Read the permanent rules, v0.1 brief, full-scope brief, completion status,
  and recent progress.
- [x] Run the exact v0.1 baseline: 179 tests passed in 158.71 seconds.
- [x] Verify more than 40 GiB free: 553 GiB was available at kickoff.
- [x] Connect the empty GitHub repository as `origin` and push the existing
  verified `main` history without replacing it.
- [x] Search for the normative `BRIEF_RELEASE.md` through the repository,
  history, attachments, Downloads, unreachable objects, and remote.
- [x] Record the missing release specification in
  `FAILURE_RELEASE_SPEC.md` and `HUMAN_TASKS.md`.

## Stage R — Release

**State:** blocked on the exact `BRIEF_RELEASE.md` package definitions.

- [ ] Execute P0-1 reconciliation using its normative acceptance criteria.
- [ ] Execute P0-2 and P0-3.
- [ ] Execute P1-1, P1-2, P1-3, and P1-4.
- [ ] Rerun the protected v0.1 and import-isolation gates.
- [ ] Write `RELEASE READY` in `STATUS_FULL.md` only after every Stage R package
  is green or independently FAILURE-documented.

## Stage T0 — Training-readiness audit

**Dependencies:** none. **State:** complete.

- [x] Add a training-only optional extra and isolated `training/` package
  without changing base-runtime imports.
- [x] Build a differentiable full-path random-weight smoke harness.
- [x] Prove finite gradients for every reference-trainable parameter and record
  step latency, peak memory, parameter counts, and disk before/after.
- [x] Exclude the inference-only native RMSNorm/CPU primitive from autodiff and
  document the differentiable MLX training path.
- [x] Inventory the current bridge loader, episode split requirements, and T1
  serialized-draw format.
- [x] Deliver `TRAINING_FEASIBILITY.md`, focused tests, full regression, commit,
  and push.

## Stage T1 — Gradient parity at step zero

**Dependency:** T0.

- [ ] Serialize one real reference batch, sampled timesteps, noise, scalar loss,
  and every trainable gradient with a deterministic manifest.
- [ ] Implement the exact differentiable MLX flow-matching loss over identical
  draws.
- [ ] Gate loss relative difference at `≤ 1e-4` and every trainable tensor at
  gradient relative L2 `≤ 1e-2` plus cosine similarity `≥ 0.999`.
- [ ] Report the worst five gradient tensors, rerun protected gates, commit, and
  push.

## Stage T2 — Optimizer lockstep

**Dependency:** T1.

- [ ] Match reference AdamW semantics and cosine-with-warmup schedule exactly.
- [ ] Execute 25 CPU/fp32 steps over identical serialized batches and draws.
- [ ] Gate every step's loss relative difference at `≤ 1e-3` and final
  per-tensor parameter drift relative L2 at `≤ 5e-3`.
- [ ] Commit and push a passing result or write the required failure analysis;
  T3 remains eligible when T1 passed.

## Stage T3 — MLX LoRA fine-tune

**Dependency:** T1.

- [ ] Add configurable LoRA to the used VLM attention/MLP linears and expert.
- [ ] Train on Metal/bf16 using a fixed whole-episode held-out split of at least
  15%, with metrics kept locally as CSV.
- [ ] Fit the run into the two-hour budget using measured step time and record
  any step-count reduction from the 3,000-step/batch-8 default.
- [ ] Export a merged standard safetensors checkpoint loadable by MLX and the
  PyTorch reference.
- [ ] Gate held-out MAE at `≤ 0.9 ×` base over at least 50 unseen samples,
  Torch/MLX round-trip ratio in `[0.95, 1.05]`, and the unchanged inference
  parity ladder.
- [ ] Write `TRAINING ALPHA` only after all three gates pass, then commit and
  push.

## Stage T4 — Training UX and full fine-tune

**Dependency:** T3.

- [ ] Add lazy `smolvla-mlx train` dispatch with dataset/path, steps, batch,
  learning rate, LoRA/full, output, resume, CSV metrics, and last-three
  checkpoint retention.
- [ ] Gate 100 uninterrupted steps against 50 + resume + 50 with per-tensor
  maximum absolute difference `≤ 1e-6`.
- [ ] Run the full-fine-tune path within the two-hour budget and apply T3's
  outcome gates without extending the budget silently.

## Stage T5 — Training documentation and benchmark

**Dependency:** T3.

- [ ] Record traceable LoRA/full and bf16/fp32 steps/s, per-1k wall-clock, and
  peak memory in `BENCHMARK.md`.
- [ ] Add an evidence-backed “Fine-tune on your Mac” README workflow and exact
  commands.

## Stage Q — Quality extras

**Dependency:** Stage R. **State:** blocked with the missing release brief.

- [ ] Execute the normative PyTorch-MPS comparison, bf16-latency investigation,
  quantization experiment, and CI packages from `BRIEF_RELEASE.md`.
- [ ] Follow each package with the full suite, a commit, and a push.

## Stage H — Hardware readiness

**Dependency:** Stage R P1-4. **Execution mode:** documents and server-side
observation logging only.

- [ ] Write `HARDWARE_RUNBOOK.md` with conservative first-contact and rollback
  instructions for the operator.
- [ ] Add a serve-side latency smoke script that cannot open serial ports or
  issue robot commands.
- [ ] Never read or execute `~/robot/so101`, import the vendor fork, open a
  serial port, or cause motion.
- [ ] Treat live-arm execution as unavailable unless the operator types the
  exact line `ARM SESSION CONFIRMED` during that live session.

## Final handoff

- [ ] Run the complete suite and audit import isolation, disk budget, artifacts,
  secrets, Git status, and remote synchronization.
- [ ] Update `STATUS_FULL.md`, `PROGRESS.md`, and `HUMAN_TASKS.md` with exact
  evidence and unresolved items.
- [ ] Commit and push the coherent final state.
