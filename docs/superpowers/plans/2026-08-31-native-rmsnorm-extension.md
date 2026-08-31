# Native CPU RMSNorm Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the fp32 CPU decoder meet the unchanged Section 6 parity bounds by reproducing PyTorch CPU RMSNorm arithmetic in a dependency-isolated MLX primitive.

**Architecture:** Keep the existing MLX model and its GPU execution path intact. Add a tiny compiled MLX extension that schedules a CPU-only lazy `Primitive` when the active MLX device is CPU, using PyTorch 2.11's cascade-sum reduction order over the final 960-wide dimension; on GPU, retain MLX's normal `fast.rms_norm` implementation. A pure-Python `ReferenceRMSNorm` module owns MLX weights and selects the correct backend without importing a reference framework.

**Tech Stack:** Python 3.12, MLX 0.32.2 C++ extension API, nanobind 2.15.0, CMake, Apple clang, setuptools with MLX's CMake extension helper, pytest, NumPy, optional reference-only PyTorch 2.11.0.

**Spec:** `BRIEF.md`, `AGENTS.md`, and `FAILURE_language.md`.

## Global Constraints

- Do not change any tolerance in `BRIEF.md` Section 6.
- `smolvla_mlx` must never import `torch`, `lerobot`, or `transformers`; the C++ extension links only MLX.
- Keep all build and model caches under this repository's `.cache/` directory.
- The reference contract is PyTorch CPU/fp32; CPU is the only path that needs its arithmetic reproduced.
- Do not force the model to CPU in production. GPU calls must continue using native MLX RMSNorm.
- Do not add mocks, skipped tests, or xfails.
- Record measured evidence in `PROGRESS.md` and commit every independently passing test using `phase-3: <what> (<test> passes)`.

## File Structure

- `pyproject.toml` and `setup.py` use MLX's supported setuptools/CMake extension build and declare matching native build requirements.
- `CMakeLists.txt` locates the active interpreter's MLX and nanobind CMake packages, then builds `smolvla_mlx._rmsnorm_native`.
- `smolvla_mlx/native/rmsnorm.h` defines the C++ operation and primitive interfaces.
- `smolvla_mlx/native/rmsnorm.cpp` implements the lazy primitive and the PyTorch CPU cascade-sum reduction.
- `smolvla_mlx/native/bindings.cpp` exposes the operation through MLX's `NB_DOMAIN mlx` nanobind domain.
- `smolvla_mlx/rmsnorm.py` defines `ReferenceRMSNorm`, the dependency-isolated MLX module used by the decoder.
- `smolvla_mlx/language.py` replaces `nn.RMSNorm` instances with `ReferenceRMSNorm`.
- `tests/test_rmsnorm.py` tests the direct source-derived normalization boundary.
- `tests/test_language.py` remains the immutable end-to-end decoder acceptance test.
- `tests/test_import_isolation.py` imports the new runtime module from a clean subprocess.
- `PROGRESS.md`, `STATUS.md`, and `FAILURE_language.md` record the result without suppressing any existing evidence.

### Task 1: Build and prove the dependency-isolated CPU primitive

**Files:**

- Create: `CMakeLists.txt`
- Create: `setup.py`
- Create: `smolvla_mlx/native/rmsnorm.h`
- Create: `smolvla_mlx/native/rmsnorm.cpp`
- Create: `smolvla_mlx/native/bindings.cpp`
- Create: `smolvla_mlx/rmsnorm.py`
- Create: `tests/test_rmsnorm.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_import_isolation.py`

**Interfaces:**

- Consumes: `mlx.core.array` activations shaped `[..., 960]`, float32 MLX weight shaped `[960]`, `eps: float`, and the active MLX stream.
- Produces: `smolvla_mlx.rmsnorm.ReferenceRMSNorm(width: int, eps: float)` with a standard MLX `weight` parameter and `__call__(x: mx.array) -> mx.array`.
- Produces: `_rmsnorm_native.rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array` for a CPU stream.

- [x] **Step 1: Write the failing direct-boundary test**

```python
def test_cpu_reference_rmsnorm_matches_pytorch_cpu_on_real_prefix(checkpoint_dir):
    from smolvla_mlx.rmsnorm import ReferenceRMSNorm
    prefix_embeddings = np.load("tests/golden/sample_000/prefix/embeddings.npy")
    source_weight = load_file(
        ".cache/smolvla_mlx/language-prefix-float32/model.float32.safetensors"
    )["language.layers.0.input_layernorm.weight"]
    module = ReferenceRMSNorm(960, eps=1e-5)
    module.weight = mx.array(source_weight)
    with mx.stream(mx.cpu):
        actual = module(mx.array(prefix_embeddings))
    expected = torch.rms_norm(torch.from_numpy(prefix_embeddings), (960,), torch.from_numpy(source_weight), 1e-5)
    np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=0.0, atol=1e-6)
```

- [x] **Step 2: Run the direct-boundary test and verify it fails because the runtime module is absent**

Run: `HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv SMOLVLA_MLX_CACHE=$PWD/.cache/smolvla_mlx uv run --extra reference pytest tests/test_rmsnorm.py -q`

Expected: FAIL during import of `smolvla_mlx.rmsnorm`.

- [x] **Step 3: Add the native build configuration**

```toml
[build-system]
requires = ["setuptools>=42", "cmake>=3.27", "mlx==0.32.2", "nanobind==2.15.0"]
build-backend = "setuptools.build_meta"
```

```cmake
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir OUTPUT_VARIABLE MLX_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
list(PREPEND CMAKE_PREFIX_PATH "${MLX_ROOT}")
find_package(nanobind CONFIG REQUIRED)
find_package(MLX CONFIG REQUIRED)
add_library(smolvla_rmsnorm STATIC smolvla_mlx/native/rmsnorm.cpp)
target_link_libraries(smolvla_rmsnorm PUBLIC mlx)
nanobind_add_module(_rmsnorm_native NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx smolvla_mlx/native/bindings.cpp)
target_link_libraries(_rmsnorm_native PRIVATE smolvla_rmsnorm mlx)
```

- [x] **Step 4: Implement one CPU primitive with PyTorch's documented reduction order**

```cpp
array rms_norm_cpu(const array& x, const array& weight, float eps, StreamOrDevice s = {}) {
  auto stream = to_stream(s);
  return array(x.shape(), float32, std::make_shared<PyTorchCPURMSNorm>(stream, eps), {x, weight});
}

void PyTorchCPURMSNorm::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  // Allocate output, register both inputs and output with mlx::cpu's encoder,
  // and dispatch one CPU closure over all flattened 960-element rows.
  // Each row squares in float32, runs the four-level cascade sum copied from
  // PyTorch v2.11.0 SumKernel.cpp, divides by 960, adds eps, applies 1/sqrt,
  // then multiplies every feature by its MLX weight.
}
```

`eval_gpu` must reject direct use with a clear error; `ReferenceRMSNorm` selects the standard MLX fast path before a GPU primitive can be created. The primitive must validate float32 input, matching final dimension, and float32 `[960]` weights rather than silently casting.

- [x] **Step 5: Add the Python module and extend import isolation coverage**

```python
class ReferenceRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((width,), dtype=mx.float32)
        self.width = width
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        if mx.default_device() == mx.cpu:
            return _rmsnorm_native.rms_norm(mx.contiguous(x), self.weight, self.eps)
        return mx.fast.rms_norm(x, self.weight, self.eps)
```

The clean subprocess test must import `smolvla_mlx.rmsnorm` and still assert all three forbidden reference packages are absent.

- [x] **Step 6: Rebuild the editable package and run the isolated tests**

Run: `HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv SMOLVLA_MLX_CACHE=$PWD/.cache/smolvla_mlx uv sync --extra reference --reinstall-package smolvla-mlx && uv run --extra reference pytest tests/test_rmsnorm.py tests/test_import_isolation.py -q`

Expected: every direct normalization and isolation assertion passes.

- [ ] **Step 7: Commit the independently verified extension**

```bash
git add .gitignore pyproject.toml CMakeLists.txt setup.py smolvla_mlx/native smolvla_mlx/rmsnorm.py tests/test_rmsnorm.py tests/test_import_isolation.py
git commit -m "phase-3: add CPU RMSNorm primitive (rmsnorm tests pass)"
```

### Task 2: Use the verified normalizer in the language decoder

**Files:**

- Modify: `smolvla_mlx/language.py`
- Modify: `tests/test_language.py`
- Modify: `PROGRESS.md`
- Modify: `STATUS.md`
- Modify: `FAILURE_language.md`

**Interfaces:**

- Consumes: `ReferenceRMSNorm` from `smolvla_mlx.rmsnorm`.
- Produces: unchanged public `TruncatedLanguageModel` API and state-dict names (`*.input_layernorm.weight`, `*.post_attention_layernorm.weight`, and `norm.weight`).

- [ ] **Step 1: Keep the existing failing decoder assertion as the acceptance test**

```python
assert np.max(np.abs(actual_array - expected_array)) <= 1e-3
```

No decoder tolerance or parameterization changes are permitted.

- [ ] **Step 2: Replace only the three RMSNorm construction sites**

```python
from smolvla_mlx.rmsnorm import ReferenceRMSNorm

self.input_layernorm = ReferenceRMSNorm(_HIDDEN_SIZE, eps=_RMS_NORM_EPS)
self.post_attention_layernorm = ReferenceRMSNorm(_HIDDEN_SIZE, eps=_RMS_NORM_EPS)
self.norm = ReferenceRMSNorm(_HIDDEN_SIZE, eps=_RMS_NORM_EPS)
```

No change to the MLP, attention, weight mapping, RoPE, cache, or masks belongs in this task.

- [ ] **Step 3: Run the immutable focused decoder suite**

Run: `HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv SMOLVLA_MLX_CACHE=$PWD/.cache/smolvla_mlx uv run --extra reference pytest tests/test_prefix.py tests/test_language.py tests/test_import_isolation.py -q`

Expected: all 51 collected focused tests pass. If a raw decoder check still fails, record its exact sample/layer/error in `FAILURE_language.md` and stop this implementation branch; do not introduce a second rounding algorithm.

- [ ] **Step 4: Update the progress and status records with measured results**

Append a dated `## 2026-08-31 — Phase 3 native CPU RMSNorm` entry that states
the exact collected/passed/failed totals from Step 3, the largest observed raw
decoder maximum-absolute error, whether the unchanged `1e-3` condition passed,
and either `Next: action projections and timestep embedding` or the exact
remaining native-kernel blocker. Do not replace those numeric values with
qualitative wording.

If the suite passes, remove the language-specific blocked statement from `STATUS.md` and retain the historical failure analysis as resolved evidence.

- [ ] **Step 5: Commit the decoder integration**

```bash
git add smolvla_mlx/language.py tests/test_language.py PROGRESS.md STATUS.md FAILURE_language.md
git commit -m "phase-3: match CPU decoder RMSNorm (language tests pass)"
```

### Task 3: Verify the installable artifact and guard the GPU path

**Files:**

- Modify: `tests/test_rmsnorm.py`
- Modify: `README.md`
- Modify: `PROGRESS.md`

**Interfaces:**

- Consumes: installed `smolvla_mlx._rmsnorm_native` extension and `ReferenceRMSNorm`.
- Produces: a wheel that includes the extension and a GPU call that remains a standard MLX RMSNorm operation.

- [ ] **Step 1: Add a non-reference GPU-path test**

```python
def test_reference_rmsnorm_keeps_gpu_execution_native_mlx() -> None:
    module = ReferenceRMSNorm(960, eps=1e-5)
    with mx.stream(mx.gpu):
        output = module(mx.ones((1, 1, 960), dtype=mx.float32))
        mx.eval(output)
    assert output.shape == (1, 1, 960)
```

- [ ] **Step 2: Build a wheel and verify its contents and import behavior**

Run: `HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv SMOLVLA_MLX_CACHE=$PWD/.cache/smolvla_mlx uv build && uv run python -c 'import smolvla_mlx.rmsnorm as r; print(r.ReferenceRMSNorm(960, 1e-5).width)'`

Expected: the wheel build succeeds and the installed runtime imports without a reference dependency.

- [ ] **Step 3: Run the complete suite and commit the packaging verification**

Run: `HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv SMOLVLA_MLX_CACHE=$PWD/.cache/smolvla_mlx make test`

Expected: all tests pass with no skips or xfails.

```bash
git add tests/test_rmsnorm.py README.md PROGRESS.md pyproject.toml uv.lock
git commit -m "phase-3: package native RMSNorm (full suite passes)"
```

## Self-Review

- Spec coverage: Task 1 preserves the no-reference-runtime rule and uses the CPU/fp32 golden contract. Task 2 makes the strict Section 6 decoder test the only acceptance gate. Task 3 proves the compiled artifact and preserves GPU-native execution.
- Placeholder scan: the plan contains exact paths, function names, commands, and acceptance evidence; it has no unresolved implementation markers.
- Type consistency: the C++ binding returns `mlx.core.array`, the Python module accepts and returns `mx.array`, and `ReferenceRMSNorm` retains the `weight` parameter names expected by strict conversion loading.

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-08-31-native-rmsnorm-extension.md`. The user explicitly asked for autonomous continuation, so this plan will be executed inline rather than delegated.
