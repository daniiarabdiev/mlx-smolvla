# Stage T1 Gradient-Parity Implementation Plan

> Execute this plan package by package with red-green-refactor discipline. Keep
> all caches under the repository, preserve the protected inference suite, and
> push each completed package directly to the operator-approved `main` branch.

**Goal:** Capture the actual Torch SmolVLA step-zero loss and every selected
gradient on one fixed real training batch, reproduce them with end-to-end MLX
CPU autodiff over identical serialized draws, and pass the immutable T1 gates.

**Architecture:** Keep `smolvla_mlx/` unchanged. Add framework-neutral artifact
IO and parity metrics under `training/`, a Torch/LeRobot-only capture module that
is imported explicitly, a scoped pure-MLX CPU primitive adapter, and a strict
checkpoint-backed MLX parity runner. Reuse the existing canonical conversion and
model components rather than maintaining a second architecture.

**Pinned stack:** Python 3.12.13, MLX 0.32.2, LeRobot 0.6.1, Torch 2.11.0,
NumPy 2.2.6, safetensors 0.8.0, checkpoint/dataset/VLM revisions from
`reference.discovery`.

**Required command environment:** `make` exports these already. Prefix every
direct `uv` command with:

```bash
HF_HOME="$PWD/.cache/hf" \
UV_CACHE_DIR="$PWD/.cache/uv" \
SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx"
```

---

## Package 1: Manifest-backed training artifacts

**Files:**

- Create: `training/data.py`
- Create: `tests/test_training_data.py`

### Step 1: Write failing artifact tests

Add real temporary-directory tests that require:

- safe relative logical names;
- contiguous `.npy` output with `allow_pickle=False`;
- sorted `manifest.json` entries containing path, shape, dtype, byte count, and
  SHA-256;
- `metadata.json` containing the finalized manifest SHA-256;
- exact load/shape/dtype verification; and
- a hash mismatch after one saved payload is tampered with.

Run:

```bash
HF_HOME="$PWD/.cache/hf" UV_CACHE_DIR="$PWD/.cache/uv" \
SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx" \
uv run pytest tests/test_training_data.py -q
```

Expected red result: import failure because `training.data` does not exist.

### Step 2: Implement the smallest framework-neutral store

Implement `TrainingArtifactWriter` and `TrainingArtifact` using NumPy, JSON,
SHA-256, and atomic temporary-file replacement. Reject absolute paths,
traversal, duplicate names, object arrays, incomplete metadata, and mismatched
payloads. Do not import Torch, LeRobot, or Transformers.

### Step 3: Verify and checkpoint

Run the focused test plus import isolation:

```bash
HF_HOME="$PWD/.cache/hf" UV_CACHE_DIR="$PWD/.cache/uv" \
SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx" \
uv run pytest tests/test_training_data.py tests/test_import_isolation.py -q
git diff --check
```

Commit and push:

```bash
git add training/data.py tests/test_training_data.py
git commit -m "phase-8: add training artifact manifests"
git push origin main
```

---

## Package 2: Exact masking and comparison metrics

**Files:**

- Modify: `training/objective.py`
- Modify: `training/model.py`
- Modify: `training/gradients.py`
- Modify: `tests/test_training_objective.py`
- Modify: `tests/test_training_model.py`
- Create: `tests/test_training_gradients.py`

### Step 1: Write failing objective and metric tests

Add cases proving:

- 32-wide predictions are cropped to six physical dimensions;
- `action_is_pad=True` timesteps contribute neither numerator nor denominator;
- the denominator is `valid_timesteps * action_dim`, including the all-padded
  clamp-to-one case;
- the random T0 batch carries an all-false `(1, 50)` temporal mask;
- relative L2 and cosine use float64 accumulation and match hand-computed
  arrays; and
- shape mismatch, non-finite values, and zero-norm references fail loudly.

Run the three files and record the behavioral failures before implementation.

### Step 2: Implement exact semantics

Extend `TrainingBatch` with `action_is_pad`. Add the optional temporal mask to
`masked_velocity_mse` and pass it from `training_loss`. Add immutable metric
dataclasses/helpers to `training.gradients`; do not embed the gate thresholds in
generic math helpers.

### Step 3: Verify and checkpoint

Run:

```bash
HF_HOME="$PWD/.cache/hf" UV_CACHE_DIR="$PWD/.cache/uv" \
SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx" \
uv run pytest tests/test_training_objective.py tests/test_training_model.py \
  tests/test_training_gradients.py tests/test_training_audit.py -q
git diff --check
```

Commit and push with `phase-8: match training loss masks and metrics`.

---

## Package 3: Scoped differentiable CPU primitives

**Files:**

- Modify: `training/differentiable.py`
- Create: `tests/test_training_differentiable.py`

### Step 1: Write failing primitive/dispatch tests

Require pure-MLX fp32 RMSNorm, split-half RoPE, last-axis softmax, and SiLU to
have finite nonzero VJPs on CPU. Snapshot every patched runtime callable and
prove the context changes the intended aliases only while active, restores them
after normal exit, and restores them after an intentional exception. Require a
clear error when invoked outside an MLX CPU stream.

Expected red result: the new functions/context are absent.

### Step 2: Implement the isolated adapter

Add the three missing pure operations beside the existing differentiable
RMSNorm. Implement an exception-safe locked context manager with lazy runtime
imports. Patch the runtime module aliases and `ReferenceRMSNorm.__call__` only
inside the scope, forbid nested activation, and restore in reverse order.

### Step 3: Verify runtime restoration and checkpoint

Run the new tests together with all exact inference primitive tests and import
isolation:

```bash
HF_HOME="$PWD/.cache/hf" UV_CACHE_DIR="$PWD/.cache/uv" \
SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx" \
uv run pytest tests/test_training_differentiable.py tests/test_rmsnorm.py \
  tests/test_rope.py tests/test_softmax.py tests/test_silu.py \
  tests/test_import_isolation.py -q
```

Commit and push with `phase-8: add scoped CPU autodiff primitives`.

---

## Package 4: Actual reference training capture

**Files:**

- Create: `training/reference.py`
- Create: `scripts/make_training_goldens.py`
- Create: `tests/test_training_reference.py`
- Modify: `Makefile`

### Step 1: Write the failing real-batch contract

Add a real pinned-data test for the fixed episode/frame that asserts:

- model-ready pixels `(2, 3, 512, 512)` and masks `(2, 1)`;
- tokens/mask `(1, 48)`, padded state `(1, 32)`, actions `(1, 50, 32)`, and
  temporal mask `(1, 50)`;
- all 50 actions are valid;
- dataset-stat normalization is active for state/action;
- identifiers are episode/frame/absolute index `0/100/100`; and
- the source-to-canonical selected parameter map is exactly 155 unique names
  and 99,880,992 scalars.

Run and record the missing-module red result.

### Step 2: Implement reference preparation and capture

In `training.reference`, lazily/explicitly own all Torch and LeRobot imports.
Resolve deltas through the installed policy config and dataset metadata, mirror
the train-loop uint8 conversion and collation, apply the fixed rename/stat
overrides, and expose a typed prepared case.

Implement capture of the actual policy forward/backward with seed `20260831`.
Use a real action-output hook for predicted velocity; serialize model-ready
batch tensors, exact draws, flow boundaries, loss, every canonical parameter,
and every gradient. Validate finite/nonzero gradients and the exact masked-loss
reconstruction before finalizing metadata.

Add a script and `training-goldens` Make target. The script defaults to
`.cache/training/gradient_goldens` and prints manifest/artifact hashes and size.

### Step 3: Verify focused behavior and generate the official artifact

Run:

```bash
make test TESTS="tests/test_training_reference.py tests/test_training_data.py tests/test_import_isolation.py -q"
make training-goldens
```

Then load the artifact through `TrainingArtifact`, verify every entry, check the
manifest metadata hash, record disk before/after and artifact size, and rerun
the focused tests against the completed artifact.

Commit source/tests/Makefile only (the `.cache` artifact remains ignored) and
push with `phase-8: capture reference training gradients`.

---

## Package 5: Checkpoint-backed MLX parity gate

**Files:**

- Modify: `training/model.py`
- Create: `training/parity.py`
- Create: `scripts/check_gradient_parity.py`
- Create: `tests/test_training_parity.py`
- Modify: `Makefile`

### Step 1: Write failing composition and gate tests

Add tests requiring:

- `SmolVLATrainingModel.from_pretrained` to consume the existing strict fp32
  converted checkpoint tree and expose the exact 155 selected canonical names;
- every selected MLX parameter to equal its serialized Torch parameter before
  differentiation;
- the serialized batch/draws to construct one `TrainingBatch` without sampling;
- a parity report containing all 155 metrics and both worst-five views; and
- immutable thresholds `1e-4`, `1e-2`, and `0.999` to pass on the real artifact.

Run the focused tests and record the absent API failure.

### Step 2: Implement strict loading and parity

Add a component-injection constructor/classmethod to the training model while
preserving its random default for T0. In `training.parity`, load and integrity-
check the artifact, set MLX to CPU/fp32, activate the scoped differentiable
primitives, run `nn.value_and_grad`, synchronize loss/all gradients, compare in
NumPy float64, and construct a JSON-compatible report.

Treat parameter mismatch, name mismatch, missing/extra gradients, non-finite
values, wrong device/dtype, or artifact-integrity errors as hard failures before
threshold evaluation. Always sort/report the five largest relative L2 values
and five lowest cosines.

Add `scripts/check_gradient_parity.py` and a `training-parity` Make target that
writes `.cache/training/t1-parity.json` atomically and exits nonzero when any
gate fails.

### Step 3: Run the immutable gate and checkpoint

Run:

```bash
make training-parity
make test TESTS="tests/test_training_parity.py tests/test_training_model.py \
tests/test_training_objective.py tests/test_training_gradients.py \
tests/test_training_differentiable.py tests/test_import_isolation.py -q"
git diff --check
```

Do not alter a threshold if the gate fails. Follow the three-hypothesis failure
protocol and either fix the implementation or write `FAILURE_GRADIENT_PARITY.md`.
On pass, commit and push with `phase-8: prove step-zero gradient parity`.

---

## Package 6: T1 evidence, protected regression, and handoff

**Files:**

- Create: `GRADIENT_PARITY.md`
- Modify: `PROGRESS.md`
- Modify: `PLAN_FULL.md`
- Modify: `STATUS_FULL.md`
- Modify: `HUMAN_TASKS.md` only if the gate exposes a real human dependency

### Step 1: Record exact evidence

Record reference/MLX loss, relative loss, maximum relative L2, minimum cosine,
the five worst relative-L2 tensors, elapsed times, artifact/report SHA-256,
artifact size, and disk free before/after. Mark T1 complete only if all 155
tensors pass both gradient conditions.

Update T2 and T3 to ready when T1 passes. Do not write `TRAINING ALPHA`; that
milestone remains exclusive to T3's outcome and round-trip gates.

### Step 2: Run all final verification

Run fresh on the exact documentation tree:

```bash
make training-parity
make test
HF_HOME="$PWD/.cache/hf" UV_CACHE_DIR="$PWD/.cache/uv" \
SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx" uv lock --check
git diff --check
git status --short
```

Also verify:

- at least 40 GiB remains free;
- base runtime import isolation is still empty;
- no tolerance/skip/mock/`xfail` changed;
- no secret-like files are staged;
- the two artifact hashes match the report; and
- `origin/main` will contain the exact tested commit.

### Step 3: Final T1 commit and push

Commit the reports/status with `phase-8: complete gradient parity gate`, push
`main`, and verify local `HEAD` equals `refs/remotes/origin/main`.

The next eligible work is Stage T2 optimizer lockstep. Stage T3 is also eligible
after T1, even if T2 later becomes FAILURE-documented.
