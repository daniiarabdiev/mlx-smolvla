# Stage T2 Optimizer Lockstep Implementation Plan

> Execute package by package with red-green-refactor discipline. Preserve the
> full inference/T0/T1 gates, repository-local caches, immutable tolerances,
> direct `main` workflow authorized by the operator, and one push per passing
> package.

**Goal:** Match the actual SmolVLA AdamW, clipping, and cosine-with-warmup
semantics over 25 identical-draw CPU/fp32 updates and pass every immutable T2
loss/final-parameter gate.

**Architecture:** Keep `smolvla_mlx/` unchanged. Add framework-neutral MLX
optimizer semantics under `training/`, a reference-only PyTorch capture module,
one manifest-backed optimizer artifact, and one strict MLX lockstep report.
Reuse T1's fixed batch, initial-parameter identity, model composition, CPU
autodiff scope, and artifact machinery.

**Required direct-command environment:**

```bash
HF_HOME="$PWD/.cache/hf" \
UV_CACHE_DIR="$PWD/.cache/uv" \
SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx"
```

`make` already exports these values.

---

## Package 1: Exact schedule, clipping, and AdamW semantics

**Files:**

- Create: `training/optimizer.py`
- Create: `tests/test_training_optimizer.py`

### Step 1: Write failing cross-framework contracts

Require:

- the first 25 MLX schedule values to equal the installed LeRobot scheduler's
  values for a 100,000-step horizon;
- global-norm and clipped gradients to match PyTorch on a multi-tensor fp32
  example where clipping is active;
- one and 25 AdamW updates on real small tensors to match PyTorch while using
  nontrivial decay, epsilon, both betas, and the schedule; and
- explicit audited defaults (`1e-4`, `(0.9, 0.95)`, `1e-8`, `1e-10`, clip 10,
  warmup 1,000, decay 30,000, floor `2.5e-6`, horizon 100,000).

Run the new file and record the missing-module red result.

### Step 2: Implement the smallest exact training-only optimizer layer

Add an immutable config, the exact scheduler function, PyTorch-style global
clipping, and an AdamW wrapper with decoupled decay and bias correction. Use no
Torch/LeRobot imports. Synchronize and expose state only as needed by later
checkpoint/resume work.

### Step 3: Verify and checkpoint

Run the optimizer cases plus T1 objective/gradient/model/import-isolation
regressions. Run `git diff --check`, commit with
`phase-9: match reference optimizer semantics`, and push.

---

## Package 2: Reference 25-step optimizer artifact

**Files:**

- Create: `training/reference_lockstep.py`
- Create: `scripts/make_optimizer_goldens.py`
- Create: `tests/test_training_reference_lockstep.py`
- Modify: `Makefile`

### Step 1: Write the failing artifact contract

Require source pins, T1 manifest binding, fixed 25-step/100,000-horizon
metadata, exact optimizer/scheduler fields, 330 payloads, 25 finite losses,
the measured LR sequence, active clip coefficients, serialized draws, and 155
finite final fp32 parameters.

### Step 2: Capture the actual reference loop

Load the real T1 case once. Prove its prepared arrays and initial parameters
equal T1. Build the checkpoint's own AdamW/scheduler presets, seed once, and
for each step perform actual forward, backward, global clip, optimizer step,
zero-grad, then scheduler step. Serialize draws/metrics and final canonical
parameters atomically with disk guards.

### Step 3: Generate, verify, and checkpoint

Add `make optimizer-goldens`, generate the official ignored artifact twice,
require the same manifest hash, run focused real reference tests and import
isolation, then commit/push with
`phase-9: capture reference optimizer lockstep`.

---

## Package 3: Native MLX 25-step lockstep gate

**Files:**

- Create: `training/lockstep.py`
- Create: `scripts/check_optimizer_lockstep.py`
- Create: `tests/test_training_lockstep.py`
- Modify: `Makefile`

### Step 1: Write failing gate tests

Require strict artifact linking and initial parameter equality, exact draw/LR
consumption, all 25 loss comparisons, all 155 final parameter comparisons,
worst-five views, and immutable limits `1e-3` / `5e-3`.

### Step 2: Implement lockstep

Load the fp32 checkpoint on MLX CPU, validate T1/T2 artifacts, reuse T1's
processed batch, and execute 25 synchronized value/grad/clip/update operations
inside one exception-safe differentiable primitive scope. Compare losses and
final parameters in NumPy float64 and write a complete atomic JSON report.

### Step 3: Run the gate and checkpoint

Run `make optimizer-lockstep` and the full focused T2/T1 training set. Apply
the three-hypothesis failure protocol without threshold changes if needed.
Commit/push a pass with `phase-9: prove optimizer lockstep`.

---

## Package 4: T2 evidence and protected closure

**Files:**

- Create: `OPTIMIZER_LOCKSTEP.md`
- Modify: `PLAN_FULL.md`
- Modify: `PROGRESS.md`
- Modify: `STATUS_FULL.md`
- Modify: `HUMAN_TASKS.md` only for a real new human dependency

Record the exact 25 losses, maximum loss error, worst five final parameter
drifts, artifact/report SHA-256 values, runtime, peak memory, disk before/after,
and every optimizer semantic checked.

Run fresh:

```bash
make optimizer-lockstep
make test
HF_HOME="$PWD/.cache/hf" UV_CACHE_DIR="$PWD/.cache/uv" \
SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx" uv lock --check
git diff --check
```

Audit import isolation, no skips/mocks/`xfail`, unchanged thresholds, artifact
hash links, at least 40 GiB free, no secret-like staged files, and exact local
to remote synchronization. Mark T2 complete only on all 25/155 gates; otherwise
link `FAILURE_OPTIMIZER_LOCKSTEP.md`. Commit/push the coherent state with
`phase-9: complete optimizer lockstep gate`.
