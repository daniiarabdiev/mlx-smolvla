# Stage T0 Training-Readiness Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the full SmolVLA MLX architecture can complete a differentiable random-weight training step over every reference-trainable parameter on this M5 Pro, with measured resource evidence and no inference-runtime regression.

**Architecture:** Add an optional, separately packaged `training/` layer that composes the existing vision, connector, truncated language, state projection, and expert modules. It freezes the reference-frozen components, computes the exact one-step flow-matching objective on Metal, audits every selected gradient, and emits machine-readable evidence; native CPU compatibility primitives remain inference-only.

**Tech Stack:** Python 3.12, MLX 0.32.2, NumPy 2.2.6, pytest 9.1.1, uv, safetensors-compatible canonical parameter names.

**Spec:** `docs/superpowers/specs/2026-08-31-training-readiness-audit-design.md`

## Global Constraints

- The 179-test v0.1 suite, fixed inference tolerances, and runtime import-isolation contract remain unchanged.
- All persistent caches and audit JSON stay under `.cache/`; at least 40 GiB free is mandatory.
- `training/` may use MLX and optional reference dependencies, but importing `smolvla_mlx` must not import `training`, Torch, LeRobot, or Transformers.
- No mock model components, skipped tests, `xfail`, external metrics, uploads, robot files, serial ports, or hardware.
- The reference training selection is exactly `state_proj` plus the complete action expert; vision, connector, and language are frozen for T0.
- The physical action loss uses dimensions `0:6`; padded dimensions `6:32` do not contribute.
- The measured smoke uses one observation, two 512×512 cameras, 48 language tokens, a 50×32 action chunk, seed 0, and bf16 parameter storage with fp32 loss arithmetic.

---

### Task 1: Establish the optional training package boundary

**Files:**

- Create: `training/__init__.py`
- Modify: `pyproject.toml`
- Modify: `setup.py`
- Modify: `MANIFEST.in`
- Modify: `tests/test_distribution.py`
- Modify: `tests/test_import_isolation.py`
- Modify: `uv.lock`

**Interfaces:**

- Consumes: existing `smolvla-mlx` package metadata and import-isolation subprocess.
- Produces: importable but side-effect-free `training` package and optional extra named `train`; no new unconditional dependency.

- [x] **Step 1: Write the failing package-boundary tests**

Add these assertions to `tests/test_distribution.py`:

```python
from importlib.util import find_spec


def test_training_package_is_shipped_as_an_optional_surface() -> None:
    metadata = distribution("smolvla-mlx").metadata

    assert find_spec("training") is not None
    assert "train" in (metadata.get_all("Provides-Extra") or [])
```

Extend the subprocess in `tests/test_import_isolation.py` to import `training`
and assert both forbidden frameworks and training side effects are absent:

```python
import training

loaded = {name.split('.', 1)[0] for name in sys.modules}
print(json.dumps({
    "forbidden": sorted(loaded & {"torch", "lerobot", "transformers"}),
    "training_api": sorted(name for name in vars(training) if not name.startswith("_")),
}))
```

The parent assertion becomes:

```python
payload = json.loads(completed.stdout)
assert payload["forbidden"] == []
assert payload["training_api"] == []
```

- [x] **Step 2: Run the tests and verify the intended red state**

Run:

```bash
uv run --extra reference pytest tests/test_distribution.py tests/test_import_isolation.py -q
```

Expected: failure because `training` is not installed and the `train` extra is
not declared; existing runtime dependency assertions remain green.

- [x] **Step 3: Add the minimal optional package**

Create `training/__init__.py` with only a module docstring. Add this extra to
`pyproject.toml`:

```toml
train = []
```

Change `setup.py` to:

```python
packages=["smolvla_mlx", "training"],
```

Add to `MANIFEST.in`:

```text
recursive-include training *.py
```

Regenerate package metadata:

```bash
uv lock
```

- [x] **Step 4: Verify the package boundary is green**

Run:

```bash
uv run --extra reference pytest tests/test_distribution.py tests/test_import_isolation.py -q
uv lock --check
```

Expected: all focused tests pass; the unconditional runtime dependency set is
still exactly the six pinned v0.1 packages.

- [x] **Step 5: Record and commit the package boundary**

Append focused results to `PROGRESS.md`, check `git diff --check`, then commit:

```bash
git add training/__init__.py pyproject.toml setup.py MANIFEST.in uv.lock tests/test_distribution.py tests/test_import_isolation.py PROGRESS.md
git commit -m "phase-7: isolate training package (distribution tests pass)"
git push origin main
```

### Task 2: Define differentiable primitives and the exact flow objective

**Files:**

- Create: `training/differentiable.py`
- Create: `training/objective.py`
- Create: `tests/test_training_objective.py`
- Modify: `PROGRESS.md`

**Interfaces:**

- Produces: `differentiable_rms_norm(inputs, weight, eps) -> mx.array` and
  `flow_matching_inputs(actions, noise, timesteps) -> tuple[mx.array,
  mx.array]` plus `masked_velocity_mse(predicted_velocity, target_velocity,
  action_dim) -> mx.array`.
- Consumes: only MLX arrays; no runtime or reference framework imports.

- [x] **Step 1: Write failing tests for gradients and padded-action masking**

Create `tests/test_training_objective.py`:

```python
import mlx.core as mx


def test_differentiable_rms_norm_has_finite_input_and_weight_gradients() -> None:
    module = __import__("training.differentiable", fromlist=["differentiable_rms_norm"])
    fn = module.differentiable_rms_norm
    inputs = mx.array([[1.0, -2.0, 3.0]], dtype=mx.float32)
    weight = mx.ones((3,), dtype=mx.float32)
    value_and_grad = mx.value_and_grad(lambda x, w: mx.sum(fn(x, w, 1e-5)), argnums=(0, 1))

    value, (input_grad, weight_grad) = value_and_grad(inputs, weight)
    mx.eval(value, input_grad, weight_grad)

    assert bool(mx.all(mx.isfinite(input_grad)))
    assert bool(mx.all(mx.isfinite(weight_grad)))
    assert input_grad.shape == inputs.shape
    assert weight_grad.shape == weight.shape


def test_flow_objective_ignores_padded_action_dimensions() -> None:
    module = __import__("training.objective", fromlist=["flow_matching_inputs"])
    actions = mx.zeros((1, 2, 4), dtype=mx.float32)
    noise = mx.ones((1, 2, 4), dtype=mx.float32)
    timesteps = mx.array([0.25], dtype=mx.float32)
    prediction = mx.array([[[0.0, 1.0, 100.0, 100.0], [0.0, 1.0, 100.0, 100.0]]])

    noisy_actions, target = module.flow_matching_inputs(actions, noise, timesteps)
    loss = module.masked_velocity_mse(prediction, target, action_dim=2)
    mx.eval(loss, noisy_actions, target)

    assert mx.allclose(noisy_actions, mx.full(actions.shape, 0.25))
    assert mx.allclose(target, mx.ones(actions.shape))
    assert float(loss) == 0.5
```

- [x] **Step 2: Verify both tests fail for missing production behavior**

Run:

```bash
uv run pytest tests/test_training_objective.py -q
```

Expected: failures inside the test bodies because the two training modules do
not yet exist.

- [x] **Step 3: Implement the minimal differentiable formulas**

Create `training/differentiable.py`:

```python
from __future__ import annotations

import mlx.core as mx


def differentiable_rms_norm(inputs: mx.array, weight: mx.array, eps: float) -> mx.array:
    values = inputs.astype(mx.float32)
    variance = mx.mean(mx.square(values), axis=-1, keepdims=True)
    normalized = values * mx.rsqrt(variance + mx.array(eps, dtype=mx.float32))
    return normalized * weight.astype(mx.float32)
```

Create `training/objective.py` with validation and these cores:

```python
time = timesteps.astype(mx.float32)[:, None, None]
noisy_actions = time * noise.astype(mx.float32) + (1.0 - time) * actions.astype(mx.float32)
target_velocity = noise.astype(mx.float32) - actions.astype(mx.float32)
return noisy_actions, target_velocity

error = predicted_velocity.astype(mx.float32)[:, :, :action_dim] - target_velocity.astype(mx.float32)[:, :, :action_dim]
loss = mx.mean(mx.square(error))
return loss
```

Reject mismatched action/noise/prediction shapes, non-`[batch]` timesteps, and
`action_dim` outside `[1, padded_width]` with `ValueError`.

- [x] **Step 4: Verify the objective tests pass**

Run:

```bash
uv run pytest tests/test_training_objective.py -q
```

Expected: 6 tests pass with finite gradients, exact padding behavior, and the
shape/action-width rejection cases exercised.

- [x] **Step 5: Record and commit the objective**

Append the red/green evidence to `PROGRESS.md`, then commit and push:

```bash
git add training/differentiable.py training/objective.py tests/test_training_objective.py PROGRESS.md
git commit -m "phase-7: add differentiable flow objective (objective tests pass)"
git push origin main
```

### Task 3: Compose the full training model and reference trainable set

**Files:**

- Create: `training/model.py`
- Create: `training/gradients.py`
- Create: `tests/test_training_model.py`
- Modify: `PROGRESS.md`

**Interfaces:**

- Produces: `SmolVLATrainingModel(nn.Module)`, `TrainingBatch`,
  `make_random_audit_batch(seed)`, `training_loss(model, batch)`,
  `configure_reference_trainable(model) -> tuple[str, ...]`, and
  `canonical_parameter_name(name) -> str`.
- Consumes: existing fixed-dimension runtime modules and Task 2's flow
  objective.

- [x] **Step 1: Write the failing model-selection tests**

Create `tests/test_training_model.py` with a small real MLX container for the
selection contract and a deterministic batch-shape assertion:

```python
import mlx.core as mx
import mlx.nn as nn


class SmallComponents(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = nn.Linear(2, 2)
        self.connector = nn.Linear(2, 2)
        self.language = nn.Linear(2, 2)
        self.state_proj = nn.Linear(2, 2)
        self.expert = nn.Linear(2, 2)


def test_reference_selection_trains_only_state_projection_and_expert() -> None:
    module = __import__("training.gradients", fromlist=["configure_reference_trainable"])
    model = SmallComponents()

    names = module.configure_reference_trainable(model)

    assert names
    assert all(name.startswith(("state_proj.", "expert.")) for name in names)
    assert any(name.startswith("state_proj.") for name in names)
    assert any(name.startswith("expert.") for name in names)


def test_random_audit_batch_has_the_audited_shapes() -> None:
    module = __import__("training.model", fromlist=["make_random_audit_batch"])
    batch = module.make_random_audit_batch(seed=0)

    assert batch.processed.pixel_values.shape == (2, 3, 512, 512)
    assert batch.processed.input_ids.shape == (1, 48)
    assert batch.processed.state.shape == (1, 6)
    assert batch.actions.shape == (1, 50, 32)
    assert batch.noise.shape == (1, 50, 32)
    assert batch.timesteps.shape == (1,)
    mx.eval(batch.actions, batch.noise)
```

- [x] **Step 2: Verify the tests fail because the APIs are absent**

Run:

```bash
uv run pytest tests/test_training_model.py -q
```

Expected: two failures inside the test bodies for missing training modules.

- [x] **Step 3: Implement the component container and exact selection**

`SmolVLATrainingModel` owns these attributes with the existing constructors:

```python
self.vision = VisionEncoder()
self.connector = Connector()
self.language = TruncatedLanguageModel()
self.state_proj = nn.Linear(32, 960, bias=True)
self.expert = ActionExpert()
```

`TrainingBatch` is a frozen dataclass containing `ProcessedObservation`,
`actions`, `noise`, `timesteps`, and `action_dim=6`. The deterministic batch
uses `mx.random.seed(seed)`, two camera images, all-one masks, token IDs
`mx.arange(48)[None, :]`, six state values, and the fixed shapes in the test.

Implement selection in `training/gradients.py`:

```python
from mlx.utils import tree_flatten


def configure_reference_trainable(model: nn.Module) -> tuple[str, ...]:
    model.freeze()
    model.state_proj.unfreeze()
    model.expert.unfreeze()
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    if not names or not all(name.startswith(("state_proj.", "expert.")) for name in names):
        raise RuntimeError(f"unexpected reference trainable set: {names}")
    return names


def canonical_parameter_name(name: str) -> str:
    if name.startswith("expert.action_"):
        return name.removeprefix("expert.")
    return name
```

Implement `training_loss` by running vision → connector → padded state →
prefix build/encode, calling `flow_matching_inputs` to obtain `x_t` and the
target velocity, running expert denoise, and calling `masked_velocity_mse` on
the expert velocity. Cast the final MSE path to fp32.

- [x] **Step 4: Verify selection and batch construction pass**

Run:

```bash
uv run pytest tests/test_training_model.py tests/test_training_objective.py -q
```

Expected: 4 tests pass.

- [x] **Step 5: Record and commit the model composition**

Append evidence to `PROGRESS.md`, then commit and push:

```bash
git add training/model.py training/gradients.py tests/test_training_model.py PROGRESS.md
git commit -m "phase-7: compose differentiable training path (model tests pass)"
git push origin main
```

### Task 4: Run the real full-architecture differentiability and resource gate

**Files:**

- Create: `training/audit.py`
- Create: `tests/test_training_audit.py`
- Modify: `PROGRESS.md`

**Interfaces:**

- Produces: `TrainingAuditResult.as_dict()`,
  `summarize_gradients(parameters, gradients)`, and
  `run_training_readiness_audit(seed=0) -> TrainingAuditResult`.
- Consumes: Task 3's model, batch, selector, and scalar loss.

- [x] **Step 1: Write the failing audit-schema and full-smoke test**

Create `tests/test_training_audit.py`:

```python
def test_full_random_weight_training_step_has_finite_selected_gradients() -> None:
    module = __import__("training.audit", fromlist=["run_training_readiness_audit"])
    result = module.run_training_readiness_audit(seed=0)
    payload = result.as_dict()

    assert payload["device"].startswith("Device(gpu")
    assert payload["dtype"] == "bfloat16"
    assert payload["microbatch"] == 1
    assert payload["camera_count"] == 2
    assert payload["trainable_tensor_count"] > 0
    assert payload["trainable_scalar_count"] > 0
    assert payload["gradient_tensor_count"] == payload["trainable_tensor_count"]
    assert payload["all_gradients_finite"] is True
    assert payload["zero_norm_gradient_tensors"] == []
    assert payload["forward_ms"] > 0.0
    assert payload["forward_backward_ms"] > 0.0
    assert payload["peak_memory_bytes"] > 0
    assert payload["disk_free_before_bytes"] >= 40 * 1024**3
    assert payload["disk_free_after_bytes"] >= 40 * 1024**3
```

- [x] **Step 2: Run the integration test and verify the intended red state**

Run:

```bash
uv run pytest tests/test_training_audit.py -q
```

Expected: failure inside the test because `training.audit` is absent.

- [x] **Step 3: Implement synchronized audit measurement**

`run_training_readiness_audit` performs these exact operations:

```python
model = SmolVLATrainingModel()
model.set_dtype(mx.bfloat16)
selected_names = configure_reference_trainable(model)
batch = make_random_audit_batch(seed)
mx.eval(*[value for _, value in tree_flatten(model.parameters())])

forward_start = time.perf_counter()
forward_loss = training_loss(model, batch)
mx.eval(forward_loss)
forward_ms = (time.perf_counter() - forward_start) * 1_000.0

mx.reset_peak_memory()
value_and_grad = nn.value_and_grad(model, lambda current_batch: training_loss(model, current_batch))
step_start = time.perf_counter()
loss, gradients = value_and_grad(batch)
flat_gradients = tuple(tree_flatten(gradients))
mx.eval(loss, *[gradient for _, gradient in flat_gradients])
forward_backward_ms = (time.perf_counter() - step_start) * 1_000.0
```

Measure disk with `shutil.disk_usage(repo_root).free` before model construction
and after evaluated gradients. Compare flattened parameter and gradient names
exactly, count scalars from shapes, evaluate `mx.isfinite`, record zero norms,
capture `mx.get_active_memory()` and `mx.get_peak_memory()`, and include Python,
macOS, MLX, seed, shapes, device, and dtype in the frozen result dataclass.
Raise before returning if free disk falls below 40 GiB, names differ, a gradient
is non-finite, or any selected tensor has zero norm.

- [x] **Step 4: Run the real gate to green**

Run:

```bash
uv run pytest tests/test_training_audit.py -q
```

Expected: 1 full-architecture test passes; record its measured duration,
trainable tensor/scalar counts, peak memory, and loss in `PROGRESS.md`.

- [x] **Step 5: Commit and push the measured audit implementation**

Run `git diff --check`, then:

```bash
git add training/audit.py tests/test_training_audit.py PROGRESS.md
git commit -m "phase-7: audit full training gradients (audit test passes)"
git push origin main
```

### Task 5: Publish the feasibility artifact and protect the baseline

**Files:**

- Create: `scripts/training_feasibility.py`
- Create: `TRAINING_FEASIBILITY.md`
- Modify: `Makefile`
- Modify: `PLAN_FULL.md`
- Modify: `STATUS_FULL.md`
- Modify: `PROGRESS.md`

**Interfaces:**

- Produces: `make training-audit`, JSON at
  `.cache/training/t0-audit.json`, and the tracked feasibility decision record.
- Consumes: `run_training_readiness_audit(seed=0)`.

- [x] **Step 1: Write the failing standalone-script contract**

Add a subprocess test to `tests/test_training_audit.py`:

```python
import json
from pathlib import Path
import subprocess
import sys


def test_training_feasibility_script_writes_machine_readable_evidence(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [sys.executable, "scripts/training_feasibility.py", "--output", str(output), "--seed", "0"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["all_gradients_finite"] is True
    assert payload["gradient_tensor_count"] == payload["trainable_tensor_count"]
```

- [x] **Step 2: Verify the script test fails because the entrypoint is absent**

Run:

```bash
uv run pytest tests/test_training_audit.py::test_training_feasibility_script_writes_machine_readable_evidence -q
```

Expected: failure with Python unable to open `scripts/training_feasibility.py`.

- [x] **Step 3: Add the standalone entrypoint and Make target**

The script inserts the repository root into `sys.path`, parses `--output` and
`--seed`, calls the audit, creates only the output's parent under `.cache/`, and
writes sorted/indented JSON. Add:

```make
.PHONY: goldens test bench training-audit

training-audit:
	uv run python scripts/training_feasibility.py --seed 0 --output $(CURDIR)/.cache/training/t0-audit.json
```

- [x] **Step 4: Run the artifact command and focused tests**

Run:

```bash
make training-audit
uv run pytest tests/test_training_objective.py tests/test_training_model.py tests/test_training_audit.py tests/test_import_isolation.py -q
```

Expected: JSON is written under `.cache/training/`; every T0 and isolation test
passes. Record its SHA-256 and exact numeric fields.

- [x] **Step 5: Write the evidence-backed feasibility report**

Create `TRAINING_FEASIBILITY.md` with the JSON's exact machine/dtype/shape,
loss, gradient counts, zero/non-finite counts, forward and forward+backward
milliseconds, active/peak memory, disk before/after, and artifact SHA-256.
Document these decisions explicitly:

- native CPU compatibility primitives remain inference-only because the
  extension exposes no VJP;
- T0 uses differentiable MLX operations on Metal and T1 must add its pure-MLX
  CPU comparison path;
- the bridge begins from the existing pinned public dataset loader but crosses
  into `training/` as plain arrays;
- the T1 manifest fields are those in the Stage T0 design, with hashes and
  exact source revisions;
- microbatch/batch expansion consequences are derived from the measured
  resource result.

Mark Stage T0 complete in `PLAN_FULL.md`; update `STATUS_FULL.md` with the
measured evidence and Stage T1 as next only if the gate passed.

- [x] **Step 6: Run the complete protected verification**

Run:

```bash
make test
uv lock --check
git diff --check
```

Expected: all old and new tests pass with zero failures; lock and whitespace
checks exit 0.

- [x] **Step 7: Commit, push, and verify synchronization**

```bash
git add scripts/training_feasibility.py TRAINING_FEASIBILITY.md Makefile PLAN_FULL.md STATUS_FULL.md PROGRESS.md tests/test_training_audit.py
git commit -m "phase-7: complete training readiness audit (full suite passes)"
git push origin main
git status --short --branch
git rev-parse HEAD origin/main
```

Expected: clean `main...origin/main` and identical local/remote commit IDs.
