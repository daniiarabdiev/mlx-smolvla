# BRIEF_T3B.md — amendment to BRIEF_FULL: unblock, T3 second attempt, finish every phase

This amends `BRIEF_FULL.md`. Everything not amended here stays as written
there and in `BRIEF_RELEASE.md`. `AGENTS.md` governs at all times.
`FAILURE_LORA_FINETUNE.md` is not reopened, reinterpreted, or edited; the
original T3 run remains failure-documented under its original gates.

---

## 0. Kickoff message (the operator pastes this as the first prompt)

> Read `AGENTS.md`, `BRIEF_FULL.md`, `BRIEF_RELEASE.md`, `BRIEF_T3B.md`,
> `STATUS_FULL.md`, and `FAILURE_LORA_FINETUNE.md` before doing anything.
> If `BRIEF_RELEASE.md` is still absent, write that to `HUMAN_TASKS.md` and
> proceed with the T3B stages only; never re-derive Stage R.
> Then: (1) run `make test` and record the baseline in `PROGRESS.md`;
> (2) write `PLAN_T3B.md` with the stage order in BRIEF_T3B Section 4;
> (3) execute, appending to `PROGRESS.md` with numbers, committing after every
> passing test, pushing after every completed package.
> The derived gate in Section 2 must be computed, written, and hashed before
> any MLX-versus-PyTorch comparison of the new checkpoint; a floor computed
> after seeing MLX numbers is invalid and must be discarded.
> Never loosen a fixed tolerance, never touch `~/robot/so101` or hardware,
> never upload. Finish with a full-suite run, a commit, a push, and an
> updated `STATUS_FULL.md`.

---

## 1. Stage T3B-1 — Reference self-consistency floor on the failed checkpoint (diagnostic)

Purpose: measure how much the PyTorch reference disagrees *with itself* on the
T3 checkpoint under arithmetic perturbation, on the same 56 frozen cases and
stored noise. Informational only; it changes nothing about the original T3
verdict.

- Perturbation set, all on the exported T3 checkpoint, same 56 cases, same
  stored `1×50×32` noise, same processor tensors:
  - (a) baseline: CPU fp32 exactly as the goldens were generated;
  - (b) CPU fp32 with `torch.set_num_threads(1)`;
  - (c) CPU fp32 with the maximum thread count;
  - (d) MPS device with fallback enabled for unsupported ops;
  - (e) CPU float64: model and inputs cast to double.
- For each case and each perturbation p ≠ (a), compute the normalized action
  chunk and its max-abs difference from (a). Report the envelope
  `F = max over cases and p` and, separately, `F64` (the (e)-only component).
- Report next to them, purely as context, the original MLX-versus-(a) value
  (`0.17762404680252075`). Write the table to `SELF_CONSISTENCY_T3.md`.
- Gate: table exists with hashes of every input; no verdict.

---

## 2. Stage T3B-2 — Prospective parity procedure for trained checkpoints

Purpose: define, before the new checkpoint exists, how deterministic parity is
judged for a fine-tuned model whose velocity field may be ill-conditioned.

- Fixed gates carried over unchanged: image preprocessing max-abs ≤ 1e-5;
  state preprocessing max-abs ≤ 1e-6; held-out MAE ratio fine/base ≤ 0.9;
  Torch/MLX held-out MAE ratio within [0.95, 1.05].
- **Derived deterministic gate.** For a trained checkpoint C:
  1. Run the Section 1 perturbation set on C with PyTorch only, on the frozen
     cases and stored noise; compute `F(C)` as in Section 1.
  2. Write `F(C)`, the per-perturbation table, and the input hashes to
     `.cache/training/<run>/floor.json`, hash it, and record the hash and a
     timestamp in `PROGRESS.md` *before* any MLX inference of C is compared.
  3. Threshold: normalized action-chunk max-abs (MLX versus (a)) ≤
     `max(0.005, 3 × F(C))`.
  Rationale, stated once: MLX is one more reduction order among several; it
  is allowed to differ from the reference by a small multiple of the
  reference's own spread, and never by less than the original fixed
  tolerance, which still applies whenever the model is well-conditioned.
- The multiplier 3 and the fallback 0.005 are fixed by this brief and are not
  adjustable afterward.
- Deliverable: `PARITY_PROCEDURE_TRAINED.md` and the evaluator implementing
  it with tests, including a test that the procedure refuses to run if the
  floor file's timestamp is later than any MLX comparison artifact.

---

## 3. Stage T3B-3 — Expert-only LoRA fine-tune, 3,000 updates (prospectively frozen)

- Freeze policy: match the reference stack's default SmolVLA fine-tuning
  freeze policy exactly (expect: vision encoder frozen, prefix decoder frozen,
  expert trained). Verify it in the installed reference config and record it
  in `ARCHITECTURE.md`. LoRA targets: attention and MLP linears of the expert
  only. Rank 8, alpha 16, dropout 0 — identical to the original T3 run so the
  two are comparable.
- Budget identical to the original run: 3,000 updates, effective batch 8,
  same learning-rate schedule, same seed and serialized draws where the
  original run serialized them, same held-out episode split, same 56 frozen
  cases. Freeze and hash the configuration before starting.
- Same export path (merged fp32, standard LeRobot layout, correct stats),
  same byte-level export audit.
- Gates, in this order: preprocessing (fixed); held-out improvement (fixed);
  Torch/MLX round trip (fixed); floor computed and hashed (Section 2 step 2);
  derived deterministic gate (Section 2 step 3).
- Report-only: side-by-side table with the original T3 run — held-out MAE,
  round-trip ratio, `F`, MLX-versus-reference max-abs, the amplification
  curve across Euler steps, wall-clock, peak memory. This table is the
  "which layers to adapt" section of the eventual write-up.
- Milestones: all gates pass → write `TRAINING ALPHA` to `STATUS_FULL.md`.
  Fixed gates pass but the derived gate fails → write
  `TRAINING ALPHA (STATISTICAL)` and a `FAILURE_LORA_FINETUNE_B.md` with the
  same three-hypothesis discipline as before.

---

## 4. Dependency and ordering amendments

- **T4 and T5 depend on T3B-3's fixed outcome gates** (held-out improvement
  and round trip), not on the derived deterministic gate. The deterministic
  result is documented honestly in T5 either way. Rationale: those gates
  prove the training pipeline works; the deterministic gate measures the
  trained model's conditioning.
- **T4 full fine-tune is code and smoke only.** Implement `--full`, validate
  with a 100-update smoke run (loss decreases; resume-exactness gate holds
  for the full path as well as LoRA), and stop. The long full fine-tuning
  run is deferred to a separate brief by operator decision.
- **Parallelization rule.** Launch the T3B-3 training run in the background
  (`nohup`, log to the run directory, checkpoint cadence as in T4's spec)
  and work on Stage R packages that do not measure latency while it trains:
  P0-1 license, P0-2, P0-3, P1-2, P1-3. Do not run P1-1 production-path
  timing, `make bench`, or any Stage Q benchmark while training or floor
  computation is running; check for a live training process before every
  timing measurement and record that it was idle.
- **Order for this run:** T3B-1 → T3B-2 → start T3B-3 training → Stage R
  non-timing packages → T3B-3 evaluation and gates → remaining Stage R
  (P1-1, P1-4) → T4 → T5 → Q → H (documents only).
- Everything else in `BRIEF_FULL.md` Sections 1, 3, and 4 stands: protected
  baseline, 40 GB disk floor with training keeping the last three
  checkpoints, no credentials, documents-only hardware stage with the
  `ARM SESSION CONFIRMED` gate, final full suite plus commit plus push.
