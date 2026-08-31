# Stage T3 LoRA Fine-Tune Implementation Plan

> Execute package by package with red-green-refactor discipline. Preserve all
> inference/T0/T1/T2 gates, repository-local caches, immutable thresholds, and
> one commit/push per passing package.

**Goal:** Train a native MLX LoRA model on a deterministic whole-episode split,
export merged standard LeRobot weights and correct statistics, and pass all
three T3 outcome gates.

**Architecture:** Keep `smolvla_mlx/` unchanged. Put LoRA, the Torch-to-NumPy
data bridge, stats-aware policy composition, run loop, export, reference
round-trip evaluator, and evidence writer under `training/`. Use rank-8/
alpha-16 adapters on exactly 229 linears, fp32 adapter masters over bf16 base
weights, gradient accumulation of eight, and a 56-case held-out manifest.

**Required direct-command environment:**

```bash
HF_HOME="$PWD/.cache/hf" \
UV_CACHE_DIR="$PWD/.cache/uv" \
SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx"
```

`make` already exports these values.

---

## Package 1: LoRA topology, trainable set, and loss gradients

**Files:**

- Create: `training/lora.py`
- Create: `tests/test_training_lora.py`
- Modify: `training/model.py`

1. Write failing contracts for LoRA initialization math, dtype, configurable
   rank/alpha, exact 229-target insertion, exact 458-tensor trainable set,
   zero-init output identity, nonzero adapter gradients on the real T1 batch,
   and merge equivalence/no surviving wrappers.
2. Implement the minimal Torch-free module traversal/insertion/merge layer.
   Extend only the training composition loader to accept bfloat16; preserve the
   fp32 T1 default and all base-package behavior.
3. Run focused T0/T1/T2/model/import-isolation tests, `git diff --check`, then
   commit/push `phase-10: add native MLX LoRA topology`.

## Package 2: Whole-episode bridge and train-only statistics

**Files:**

- Create: `training/dataset.py`
- Create: `tests/test_training_dataset.py`

1. Write failing tests for the exact seeded 42/8 split, no overlap, 16%
   holdout, deterministic episode-aware order, train-only Parquet statistics,
   no held-out-row leakage, eight distinct microbatches, and a fixed bridge
   case matching the pinned reference processor tensor-for-tensor.
2. Implement a lazily imported LeRobot/Torch bridge returning owned NumPy
   dataclasses. Preserve source episode/frame/absolute identities and expose a
   deterministic 56-case held-out specification.
3. Run the focused real-data cases plus import isolation, commit/push
   `phase-10: add deterministic training data bridge`.

## Package 3: Stats-aware policy and merged standard export

**Files:**

- Create: `training/preprocessing.py`
- Create: `training/export.py`
- Create: `training/reference_export.py`
- Create: `tests/test_training_export.py`

1. Write failing tests for exact state/action normalization, standard processor
   tensor loading, inverse source-name mapping, convolution layout reversal,
   all-500 atomic fp32 export, manifest checksums, no adapter names, and strict
   MLX/Torch loads. Use a temporary modified checkpoint for round-trip tests;
   do not wait for the long run.
2. Implement the training-only processor wrapper and merged exporter. Generate
   LeRobot processors through its native API with train-only stats and portable
   tokenizer identity.
3. Verify local strict loads in both frameworks, focused inference regression,
   and import isolation; commit/push `phase-10: export merged LeRobot checkpoints`.

## Package 4: Accumulating Metal trainer, benchmark, and frozen run budget

**Files:**

- Create: `training/finetune.py`
- Create: `scripts/finetune_lora.py`
- Create: `tests/test_training_finetune.py`
- Modify: `Makefile`

1. Write failing small-model contracts for deterministic gradient
   accumulation/division, one clip/update per eight microbatches, CSV schema,
   atomic run manifest, bfloat16 frozen base/fp32 adapters, and step-budget
   arithmetic.
2. Implement the loop using the proven T2 optimizer. Add disk/memory guards,
   exception-safe metrics flushing, periodic local adapter checkpoints, and
   final merged export.
3. Benchmark 3 warm-up plus 10 real effective-batch updates on Metal, record
   median/peak memory and freeze `min(3000, floor(6900/median))` before the
   full run. Commit/push the passing machinery and frozen budget with
   `phase-10: add measured Metal LoRA trainer`.

## Package 5: Real fine-tune and held-out outcome gate

**Files:**

- Create locally: `.cache/training/t3/metrics.csv`
- Create locally: `.cache/training/t3/run.json`
- Create locally: `.cache/training/t3/export/`
- Create: `training/evaluation.py`
- Create: `scripts/check_lora_finetune.py`
- Create: `tests/test_training_evaluation.py`
- Modify: `Makefile`

1. Freeze and hash 56 held-out cases/noise before training; evaluate the base
   MLX checkpoint with train-only statistics.
2. Run the measured step budget once. Merge/export the final adapters, load the
   export strictly in MLX, and evaluate the same 56 cases. Require fine MAE
   `<= 0.9 * base MAE` without extending the run.
3. Load the exact export in Torch and score the same cases/noise. Require the
   Torch/MLX MAE ratio in `[0.95, 1.05]`. Write a complete atomic T3 JSON
   report, run focused gates, commit/push `phase-10: prove LoRA fine-tune outcome`.

## Package 6: Stats-active parity and T3 closure

**Files:**

- Create: `LORA_FINETUNE.md`
- Modify: `PLAN_FULL.md`
- Modify: `PROGRESS.md`
- Modify: `STATUS_FULL.md`
- Modify: `HUMAN_TASKS.md` only for a genuinely new human dependency

1. Run the exported checkpoint through stats-active Torch/MLX preprocessing,
   normalized-flow, and physical-action comparisons on the fixed held-out
   manifest at unchanged `5e-3` fp32 max-absolute tolerance. Persist every
   boundary and worst case.
2. Run fresh `make lora-finetune-check`, `make test`, `uv lock --check`, import
   isolation, no-skip/mock/xfail audit, `git diff --check`, disk floor, secret
   scan, and local/remote identity checks.
3. Only if all three immutable gates pass, write `TRAINING ALPHA` and complete
   T3. Otherwise write `FAILURE_LORA_FINETUNE.md` with the frozen evidence.
   Commit/push `phase-10: complete LoRA fine-tune gate`.
