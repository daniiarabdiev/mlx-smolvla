# SmolVLA MLX v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a native, installable MLX inference implementation of `lerobot/smolvla_base` with numerical parity against the pinned LeRobot PyTorch reference and measured performance on the operator's M5 Pro.

**Architecture:** Build a deterministic CPU reference-and-golden lane first, audit the installed model and checkpoint, then port one boundary at a time into focused MLX modules. Reuse compatible `mlx-vlm` code, vendor only the pieces needed for truncated decoder execution and key/value exposure, and keep all PyTorch-side dependencies outside the runtime package.

**Tech Stack:** Python 3.12, uv, MLX, mlx-vlm, safetensors, NumPy, Hugging Face Hub, tokenizers, Pillow, PyTorch CPU reference, mainline LeRobot, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-smolvla-mlx-design.md`

## Global Constraints

- The reference checkpoint is `lerobot/smolvla_base`; reference behavior and checkpoint metadata override hypotheses in `BRIEF.md`.
- Python is 3.11 or 3.12 under uv; this plan uses 3.12 and commits `uv.lock`.
- Runtime imports are limited to `mlx`, `mlx-vlm` or licensed vendored code, `safetensors`, `numpy`, `huggingface_hub`, `tokenizers`, and `pillow`.
- `torch`, `transformers`, and `lerobot` may be imported only below `reference/`, `scripts/`, and `tests/`.
- `HF_HOME`, `UV_CACHE_DIR`, and `SMOLVLA_MLX_CACHE` point inside `.cache/` for every development command.
- Reference goldens run on CPU in fp32 with fixed seeds; MPS is not a golden source.
- Numerical thresholds are copied from `BRIEF.md` Section 6 and may never be loosened.
- No model-component mocks, skipped tests, or `xfail` entries are permitted.
- No reads from `~/robot/so101`, no serial ports, no hardware, no Hub uploads, no secrets, and no pushes except to `origin`.
- Every passing task appends numerical evidence to `PROGRESS.md` and creates a commit using `phase-N: <what> (<test> passes)`.

## File map

- `pyproject.toml`, `uv.lock`, `.python-version`: package metadata and reproducible environments.
- `.codex/config.toml`, `.gitignore`, `Makefile`: repository isolation and repeatable commands.
- `smolvla_mlx/config.py`: validated checkpoint-derived configuration dataclasses.
- `smolvla_mlx/types.py`: observation, processed-input, prefix-cache, and action-chunk interfaces.
- `smolvla_mlx/cache.py`: end-user and development cache resolution.
- `smolvla_mlx/preprocessing.py`: exact camera, image, token, state, and action preprocessing.
- `smolvla_mlx/vision.py`: SigLIP-style vision stack.
- `smolvla_mlx/connector.py`: pixel shuffle and multimodal projection.
- `smolvla_mlx/language.py`: truncated SmolLM decoder with per-layer key/value capture.
- `smolvla_mlx/expert.py`: timestep, action projections, cross-attention, and expert blocks.
- `smolvla_mlx/flow.py`: verified timestep schedule and Euler integration.
- `smolvla_mlx/convert.py`: complete source-to-target weight mapping and safetensor conversion.
- `smolvla_mlx/policy.py`: model assembly, prefix reuse, action queue, and public policy API.
- `smolvla_mlx/cli.py`: convert, test, bench, and predict commands.
- `reference/discovery.py`: installed-version, source-path, checkpoint, and dataset discovery.
- `reference/policy.py`: pinned fp32 CPU reference adapter.
- `reference/goldens.py`: deterministic intermediate capture and manifest creation.
- `scripts/make_goldens.py`, `scripts/bench.py`: reproducible golden and benchmark entry points.
- `tests/conftest.py`: golden loading and fixed tolerance helpers.
- `tests/test_*.py`: isolation, conversion, module, integration, API, and packaging tests.
- `ARCHITECTURE.md`, `REUSE_DECISIONS.md`, `BENCHMARK.md`, `README.md`, `NOTICE`: durable technical evidence.
- `PLAN.md`, `PROGRESS.md`, `HUMAN_TASKS.md`, `STATUS.md`: execution state required by `AGENTS.md`.

---

### Task 1: Reproducible repository bootstrap

**Files:**
- Create: `.python-version`, `.codex/config.toml`, `.gitignore`, `Makefile`, `pyproject.toml`
- Create: `smolvla_mlx/__init__.py`, `smolvla_mlx/cache.py`
- Create: `tests/test_import_isolation.py`, `tests/test_cache.py`
- Create: `PLAN.md`, `PROGRESS.md`, `HUMAN_TASKS.md`, `STATUS.md`
- Create: `uv.lock`

**Interfaces:**
- Consumes: repository rules in `AGENTS.md` and dependency boundaries in `BRIEF.md`.
- Produces: `smolvla_mlx.cache.resolve_cache_dir(explicit: Path | None = None) -> Path`, an importable package, and repeatable `make goldens`, `make test`, and `make bench` commands.

- [x] **Step 1: Write failing bootstrap tests**

```python
def test_runtime_import_isolation():
    code = "import smolvla_mlx,sys; print(','.join(sorted(sys.modules)))"
    modules = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert not {"torch", "lerobot", "transformers"} & set(modules.split(","))

def test_explicit_cache_wins(tmp_path):
    assert resolve_cache_dir(tmp_path) == tmp_path.resolve()
```

- [x] **Step 2: Prove the tests fail before the package exists**

Run: `HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv uv run pytest tests/test_import_isolation.py tests/test_cache.py -v`

Expected: collection fails because `smolvla_mlx` is not importable.

- [x] **Step 3: Add package metadata, cache resolver, command targets, and state files**

```python
def resolve_cache_dir(explicit: Path | None = None) -> Path:
    candidate = explicit or os.environ.get("SMOLVLA_MLX_CACHE")
    return Path(candidate).expanduser().resolve() if candidate else Path.home() / ".cache" / "smolvla_mlx"
```

Use `uv python install 3.12`, initialize the package for Python `>=3.12,<3.13`, add the allowed runtime dependencies, add pytest as a development dependency, and write `.codex/config.toml` exactly as specified in `SETUP.md` Section 2. Make targets must prefix all three in-repo cache variables.

- [x] **Step 4: Lock, sync, and verify bootstrap tests**

Run: `UV_CACHE_DIR=$PWD/.cache/uv uv lock && make test`

Expected: both bootstrap tests pass and the subprocess reports none of the three forbidden modules.

- [x] **Step 5: Record and commit the evidence**

Append Python, uv, macOS, CPU, memory, and MLX versions plus the two passing test names to `PROGRESS.md`.

```bash
git add .python-version .codex .gitignore Makefile pyproject.toml uv.lock smolvla_mlx tests PLAN.md PROGRESS.md HUMAN_TASKS.md STATUS.md
git commit -m "phase-0: bootstrap repository (isolation tests pass)"
```

### Task 2: Installed reference and checkpoint discovery

**Files:**
- Create: `reference/__init__.py`, `reference/discovery.py`
- Create: `tests/test_reference_discovery.py`, `ARCHITECTURE.md`
- Modify: `pyproject.toml`, `uv.lock`, `PROGRESS.md`

**Interfaces:**
- Consumes: in-repo Hugging Face cache and the optional `reference` environment.
- Produces: `ReferenceDiscovery` with installed versions, exact source files, checkpoint revision/config, tensor inventory, parameter count, and selected dataset identifier.

- [x] **Step 1: Add pinned reference dependencies and write a failing discovery test**

```python
def test_discovery_finds_installed_smolvla(tmp_path):
    result = discover_reference(cache_dir=tmp_path)
    assert result.lerobot_version
    assert result.policy_source.is_file()
    assert result.config_source.is_file()
    assert result.checkpoint_id == "lerobot/smolvla_base"
```

Use uv's optional `reference` group for mainline LeRobot, PyTorch, Transformers, and datasets; regenerate `uv.lock` so exact versions are committed.

- [x] **Step 2: Run the discovery test and capture the missing implementation failure**

Run: `HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv uv run --extra reference pytest tests/test_reference_discovery.py -v`

Expected: import fails for `reference.discovery` or `discover_reference`.

- [x] **Step 3: Implement source discovery without remembered paths**

```python
@dataclass(frozen=True)
class ReferenceDiscovery:
    lerobot_version: str
    transformers_version: str
    torch_version: str
    policy_source: Path
    config_source: Path
    checkpoint_id: str
    dataset_id: str

def smolvla_sources() -> list[Path]:
    root = Path(importlib.util.find_spec("lerobot").submodule_search_locations[0])
    return sorted(path for path in root.rglob("*.py") if "smolvla" in path.name.lower())
```

Inspect candidates for the exported policy/configuration class names, record the exact installed package versions and source paths, query Hub metadata for the checkpoint revision and config, and verify candidate SO-101 datasets by loading their metadata and feature schema.

- [x] **Step 4: Run discovery and write the immutable evidence section**

Run: `HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv uv run --extra reference python -m reference.discovery --write ARCHITECTURE.md`

Expected: `ARCHITECTURE.md` names exact versions, source paths, checkpoint revision, model tensor count/parameter count, and a dataset with at least two image features plus state, action, and task text.

- [x] **Step 5: Verify and commit**

Run: `make test TESTS="tests/test_reference_discovery.py tests/test_import_isolation.py"`

Expected: discovery and runtime isolation both pass.

```bash
git add pyproject.toml uv.lock reference tests/test_reference_discovery.py ARCHITECTURE.md PROGRESS.md
git commit -m "phase-0: discover pinned reference (discovery tests pass)"
```

### Task 3: Reference policy smoke path and architecture audit

**Files:**
- Create: `reference/policy.py`, `scripts/inspect_reference.py`
- Create: `tests/test_reference_policy.py`, `REUSE_DECISIONS.md`
- Modify: `ARCHITECTURE.md`, `PROGRESS.md`

**Interfaces:**
- Consumes: `ReferenceDiscovery` and one real dataset observation.
- Produces: `ReferencePolicy.load(discovery)`, `ReferencePolicy.prepare(observation)`, `ReferencePolicy.predict(observation, noise)`, and a complete verified architecture report.

- [x] **Step 1: Write a failing real-reference smoke test**

```python
def test_reference_predicts_one_action_chunk(reference_case):
    policy, observation, noise = reference_case
    chunk = policy.predict(observation, noise=noise)
    assert chunk.shape == (policy.config.chunk_size, policy.config.action_dim)
    assert torch.isfinite(chunk).all()
    assert chunk.device.type == "cpu"
    assert chunk.dtype == torch.float32
```

- [x] **Step 2: Run the smoke test before the adapter exists**

Run: `HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv uv run --extra reference pytest tests/test_reference_policy.py -v`

Expected: failure naming the absent adapter or fixture.

- [x] **Step 3: Implement the CPU fp32 adapter and deterministic observation loader**

```python
class ReferencePolicy:
    def predict(self, observation: Mapping[str, object], noise: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(0)
        with torch.inference_mode():
            return self.policy.predict_action_chunk(observation, noise=noise).cpu().float()
```

Adapt the signature to the installed reference after inspecting it; keep fixed noise explicit, call `eval()`, force CPU/fp32, and assert no parameter or input resides on MPS.

- [x] **Step 4: Audit every `BRIEF.md` Section 3 hypothesis**

Run `scripts/inspect_reference.py` to record module nesting, tensor boundary shapes, exact layer counts, state placement, attention masks, preprocessing constants, normalization mode/statistics, Euler schedule/sign, action queue behavior, and callable signatures. Inspect installed `mlx-vlm` source for matching SmolVLM/Idefics components and record reuse/vendor/reimplement decisions plus license obligations.

- [x] **Step 5: Run the smoke test and commit audit evidence**

Run: `make test TESTS="tests/test_reference_policy.py tests/test_reference_discovery.py"`

Expected: one real observation produces a finite CPU fp32 action chunk of the checkpoint-configured shape.

```bash
git add reference scripts/inspect_reference.py tests/test_reference_policy.py ARCHITECTURE.md REUSE_DECISIONS.md PROGRESS.md
git commit -m "phase-1: audit reference architecture (reference smoke test passes)"
```

### Task 4: Deterministic golden capture and comparison helpers

**Files:**
- Create: `reference/goldens.py`, `scripts/make_goldens.py`
- Create: `tests/conftest.py`, `tests/test_goldens.py`
- Modify: `Makefile`, `.gitignore`, `PROGRESS.md`

**Interfaces:**
- Consumes: `ReferencePolicy`, eight or more deterministic dataset observations, fixed Gaussian noise, and audited hook points.
- Produces: `GoldenWriter.add(name: str, tensor: torch.Tensor | np.ndarray)`, `GoldenStore.load(name: str) -> np.ndarray`, test-side `GoldenCase.array(name) -> np.ndarray`, `GoldenCase.mx(name, dtype) -> mx.array`, `GoldenCase.observation() -> Mapping[str, object]`, and `tests/golden/manifest.json` with shape, dtype, and SHA-256.

- [x] **Step 1: Write failing manifest and reproducibility tests**

```python
def test_golden_writer_hashes_exact_bytes(tmp_path):
    writer = GoldenWriter(tmp_path)
    writer.add("sample_000/noise", np.arange(6, dtype=np.float32).reshape(2, 3))
    manifest = writer.finalize()
    assert manifest["sample_000/noise"]["shape"] == [2, 3]
    assert len(manifest["sample_000/noise"]["sha256"]) == 64

def test_manifest_is_stable(golden_regenerator):
    assert golden_regenerator() == golden_regenerator()
```

- [x] **Step 2: Run tests and observe missing writer/store failures**

Run: `uv run --extra reference pytest tests/test_goldens.py -v`

Expected: collection or attribute failures for the unimplemented golden API.

- [x] **Step 3: Implement atomic array capture and sorted manifest output**

```python
def add(self, name: str, value: TensorLike) -> None:
    array = np.ascontiguousarray(to_numpy(value))
    path = self.root / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    self.entries[name] = tensor_record(array, path)
```

Register audited forward hooks for preprocessed inputs, vision output, connector output, every used decoder hidden/KV pair, state embedding, every expert block output, every Euler velocity, normalized actions, and un-normalized actions. Write sample metadata and fixed seeds alongside the tensor manifest.

- [x] **Step 4: Generate goldens twice from a clean output directory**

Run: `make goldens && cp tests/golden/manifest.json .cache/manifest.first.json && make goldens && cmp .cache/manifest.first.json tests/golden/manifest.json`

Expected: `cmp` exits zero and the manifest covers at least eight real observations and every audited boundary.

- [x] **Step 5: Verify helper and reference tests, then commit code only**

Run: `make test TESTS="tests/test_goldens.py tests/test_reference_policy.py"`

Expected: all tests pass; generated golden arrays remain ignored.

```bash
git add reference/goldens.py scripts/make_goldens.py tests/conftest.py tests/test_goldens.py Makefile .gitignore PROGRESS.md
git commit -m "phase-0: capture deterministic goldens (golden tests pass)"
```

### Task 5: Checkpoint-derived configuration and exact preprocessing

**Files:**
- Create: `smolvla_mlx/config.py`, `smolvla_mlx/types.py`, `smolvla_mlx/preprocessing.py`
- Create: `tests/test_config.py`, `tests/test_preprocessing.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: checkpoint config, processor files, normalization statistics, observation dictionaries, and preprocessing goldens.
- Produces: `SmolVLAConfig.from_pretrained_files(path)`, `ProcessedObservation`, `SmolVLAPreprocessor.__call__(observation)`, `normalize_actions`, and `unnormalize_actions`.

- [x] **Step 1: Write failing config and real-golden preprocessing tests**

```python
def test_config_matches_audited_checkpoint(checkpoint_dir):
    config = SmolVLAConfig.from_pretrained_files(checkpoint_dir)
    assert config.action_dim == audited("action_dim")
    assert config.chunk_size == audited("chunk_size")
    assert config.num_steps == audited("num_steps")

@pytest.mark.parametrize("sample", range(8))
def test_preprocessing_matches_reference(sample, preprocessor, golden):
    actual = preprocessor(golden.observation(sample))
    np.testing.assert_allclose(actual.pixel_values, golden.array(sample, "pixel_values"), atol=1e-5, rtol=0)
    np.testing.assert_array_equal(actual.input_ids, golden.array(sample, "input_ids"))
    np.testing.assert_allclose(actual.state, golden.array(sample, "state_normalized"), atol=1e-6, rtol=0)
```

- [x] **Step 2: Run preprocessing tests and record the absent interfaces**

Run: `uv run pytest tests/test_config.py tests/test_preprocessing.py -v`

Expected: collection failures for `SmolVLAConfig` and `SmolVLAPreprocessor`.

- [x] **Step 3: Implement validated dataclasses and one preprocessing stage at a time**

```python
@dataclass(frozen=True)
class ProcessedObservation:
    pixel_values: mx.array
    pixel_attention_mask: mx.array
    input_ids: mx.array
    text_attention_mask: mx.array
    state: mx.array
```

Match audited camera ordering/empty-camera masking, Pillow resize and padding, processor mean/std, tokenizer IDs and fixed padding, and checkpoint normalization statistics. Reject missing required keys and non-finite inputs with field-specific errors.

- [x] **Step 4: Run focused tests after each preprocessing boundary**

Run: `uv run pytest tests/test_config.py tests/test_preprocessing.py -v`

Expected: exact token IDs, state max error at most `1e-6`, and normalized image max error at most `1e-5` for all samples.

- [x] **Step 5: Commit passing preprocessing evidence**

```bash
git add smolvla_mlx/config.py smolvla_mlx/types.py smolvla_mlx/preprocessing.py tests/test_config.py tests/test_preprocessing.py PROGRESS.md
git commit -m "phase-3: port preprocessing (preprocessing tests pass)"
```

### Task 6: Complete weight conversion

**Files:**
- Create: `smolvla_mlx/convert.py`
- Create: `tests/test_conversion.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: original checkpoint safetensors, audited target parameter tree, and cache directory.
- Produces: `ConversionReport`, `build_name_map(source_names, target_names)`, and `convert_checkpoint(source_dir, output_dir, dtype) -> ConversionReport`.

- [x] **Step 1: Write failing bijection, shape, checksum, and round-trip tests**

```python
def test_conversion_maps_every_tensor_once(checkpoint_dir, converted_dir):
    report = convert_checkpoint(checkpoint_dir, converted_dir, dtype="float32")
    assert report.unmapped_source == ()
    assert report.uninitialized_target == ()
    assert report.source_parameter_count == report.target_parameter_count
    assert len(report.source_names) == len(set(report.source_names))
```

- [x] **Step 2: Run conversion tests before implementation**

Run: `uv run pytest tests/test_conversion.py -v`

Expected: import or symbol failure for the converter.

- [x] **Step 3: Implement explicit auditable mapping and atomic output**

```python
@dataclass(frozen=True)
class ConversionReport:
    source_names: tuple[str, ...]
    target_names: tuple[str, ...]
    unmapped_source: tuple[str, ...]
    uninitialized_target: tuple[str, ...]
    source_parameter_count: int
    target_parameter_count: int
    checksums: Mapping[str, str]
```

Encode every audited rename/transpose explicitly, write `name_map.json`, retain fp32 source values, derive bf16 output from fp32, and refuse duplicate or shape-changing mappings unless the audited rule names the exact reshape/transpose.

- [x] **Step 4: Convert both dtypes and run full conversion checks**

Run: `uv run pytest tests/test_conversion.py -v`

Expected: zero unmapped source tensors, zero uninitialized targets, equal parameter counts, and stable per-tensor checksums.

- [x] **Step 5: Commit converter and evidence**

```bash
git add smolvla_mlx/convert.py tests/test_conversion.py PROGRESS.md
git commit -m "phase-2: convert checkpoint weights (conversion tests pass)"
```

### Task 7: Vision encoder and connector parity

**Files:**
- Create: `smolvla_mlx/vision.py`, `smolvla_mlx/connector.py`
- Create: `tests/test_vision.py`, `tests/test_connector.py`
- Create: `NOTICE`
- Modify: `REUSE_DECISIONS.md`, `PROGRESS.md`

**Interfaces:**
- Consumes: `SmolVLAConfig`, converted weights, batched `pixel_values`, pixel masks, and vision/connector goldens.
- Produces: `VisionEncoder.__call__(pixels, mask) -> mx.array` and `Connector.__call__(features) -> mx.array`.

- [ ] **Step 1: Write failing fp32/bf16 golden tests**

```python
@pytest.mark.parametrize("dtype,rel_l2,max_abs", [(mx.float32, 1e-3, 1e-3), (mx.bfloat16, 3e-2, None)])
def test_vision_matches_golden(dtype, rel_l2, max_abs, model_parts, golden):
    actual = model_parts.vision(golden.mx("pixel_values", dtype), golden.mx("pixel_mask"))
    assert_error(actual, golden.array("vision_features"), rel_l2=rel_l2, max_abs=max_abs)
```

- [ ] **Step 2: Prove vision and connector interfaces are absent**

Run: `uv run pytest tests/test_vision.py tests/test_connector.py -v`

Expected: collection failures for the new modules.

- [ ] **Step 3: Reuse or vendor the audited vision implementation and add connector logic**

Implement the exact patch embedding, positional handling, attention, normalization epsilon, activation, layer order, pixel shuffle, and projection found in Task 3. Batch all cameras in one MLX call. Preserve license headers for every vendored source file and enumerate it in `NOTICE`.

- [ ] **Step 4: Run focused parity tests in both dtypes**

Run: `uv run pytest tests/test_vision.py tests/test_connector.py -v`

Expected: fp32 relative L2 and max absolute error at most `1e-3`; bf16 relative L2 at most `3e-2` for all golden samples.

- [ ] **Step 5: Commit the passing model boundary**

```bash
git add smolvla_mlx/vision.py smolvla_mlx/connector.py tests/test_vision.py tests/test_connector.py NOTICE REUSE_DECISIONS.md PROGRESS.md
git commit -m "phase-3: port vision and connector (vision tests pass)"
```

### Task 8: Truncated language decoder and prefix cache

**Files:**
- Create: `smolvla_mlx/language.py`
- Create: `tests/test_language.py`, `tests/test_prefix.py`
- Modify: `smolvla_mlx/types.py`, `NOTICE`, `PROGRESS.md`

**Interfaces:**
- Consumes: token embeddings, connector output, state placement, prefix mask, converted decoder weights, and per-layer hidden/KV goldens.
- Produces: `PrefixCache(hidden: mx.array, keys: tuple[mx.array, ...], values: tuple[mx.array, ...], mask: mx.array)` and `TruncatedLanguageModel.encode_prefix(processed: ProcessedObservation, image_tokens: mx.array, stop_after: int | None = None) -> PrefixCache`.

- [ ] **Step 1: Write failing layer-by-layer and mask tests**

```python
def test_prefix_mask_is_exact(prefix_builder, golden):
    cache = prefix_builder(golden.processed(), golden.mx("connector_output"))
    np.testing.assert_array_equal(np.asarray(cache.mask), golden.array("prefix_mask"))

@pytest.mark.parametrize("layer", audited_decoder_layers())
def test_decoder_layer_matches_golden(layer, language, golden):
    cache = language.encode_prefix(golden.processed(), golden.mx("connector_output"), stop_after=layer + 1)
    assert_error(cache.hidden, golden.array(f"decoder/{layer}/hidden"), rel_l2=1e-3, max_abs=1e-3)
```

- [ ] **Step 2: Run tests before the truncated decoder exists**

Run: `uv run pytest tests/test_language.py tests/test_prefix.py -v`

Expected: missing symbol failures.

- [ ] **Step 3: Implement exact prefix assembly, 2D mask, RoPE, and layer cutoff**

```python
@dataclass(frozen=True)
class PrefixCache:
    hidden: mx.array
    keys: tuple[mx.array, ...]
    values: tuple[mx.array, ...]
    mask: mx.array
```

Expose each used layer's post-projection keys and values in the audited layout. Assert the configured cutoff is no greater than the checkpoint decoder depth and that key/value tuple lengths equal the used layer count.

- [ ] **Step 4: Run every layer and prefix test in fp32 and bf16**

Run: `uv run pytest tests/test_language.py tests/test_prefix.py -v`

Expected: exact masks; each fp32 boundary stays within `1e-3` relative L2 and max absolute error, and bf16 stays within `3e-2` relative L2.

- [ ] **Step 5: Commit decoder parity**

```bash
git add smolvla_mlx/language.py smolvla_mlx/types.py tests/test_language.py tests/test_prefix.py NOTICE PROGRESS.md
git commit -m "phase-3: port prefix decoder (language tests pass)"
```

### Task 9: Action expert and Euler flow integration

**Files:**
- Create: `smolvla_mlx/expert.py`, `smolvla_mlx/flow.py`
- Create: `tests/test_expert.py`, `tests/test_flow.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: `PrefixCache`, normalized state/noise, converted expert weights, timestep schedule, and expert/velocity goldens.
- Produces: `ActionExpert.__call__(cache, actions, state, timestep) -> mx.array`, `timestep_schedule(num_steps) -> mx.array`, and `euler_sample(expert, cache, state, noise) -> mx.array`.

- [ ] **Step 1: Write failing exact-schedule, block, and integration tests**

```python
def test_schedule_matches_reference(golden, config):
    np.testing.assert_array_equal(np.asarray(timestep_schedule(config.num_steps)), golden.array("timesteps"))

@pytest.mark.parametrize("step", audited_euler_steps())
def test_velocity_matches_golden(step, expert_fixture, golden):
    velocity = expert_fixture.velocity_at(step)
    assert_error(velocity, golden.array(f"flow/{step}/velocity"), rel_l2=1e-3, max_abs=1e-3)
```

- [ ] **Step 2: Run tests before expert and flow modules exist**

Run: `uv run pytest tests/test_expert.py tests/test_flow.py -v`

Expected: missing module or symbol failures.

- [ ] **Step 3: Implement projections, timestep embedding, expert masks/blocks, and audited Euler update**

```python
def euler_step(actions: mx.array, velocity: mx.array, dt: mx.array) -> mx.array:
    return actions + dt.astype(actions.dtype) * velocity
```

Replace the displayed sign only if Task 3's source audit proves the reference uses the opposite update; record the exact source lines in `ARCHITECTURE.md`. Apply the verified cross-attention layer alignment, action-token self-attention mask, padding, and output slicing.

- [ ] **Step 4: Run per-block, per-step, and final normalized-chunk tests**

Run: `uv run pytest tests/test_expert.py tests/test_flow.py -v`

Expected: every fp32 module boundary stays within `1e-3`; bf16 relative L2 stays within `3e-2`; fp32 normalized final chunk max absolute error stays within `5e-3`; bf16 stays within `5e-2`.

- [ ] **Step 5: Commit expert and flow parity**

```bash
git add smolvla_mlx/expert.py smolvla_mlx/flow.py tests/test_expert.py tests/test_flow.py ARCHITECTURE.md PROGRESS.md
git commit -m "phase-3: port action expert and flow (expert tests pass)"
```

### Task 10: Public policy API and deterministic end-to-end parity

**Files:**
- Create: `smolvla_mlx/policy.py`
- Create: `tests/test_policy_api.py`, `tests/test_end_to_end.py`
- Modify: `smolvla_mlx/__init__.py`, `PROGRESS.md`

**Interfaces:**
- Consumes: converted model components, `SmolVLAPreprocessor`, an observation mapping, and optional fixed noise.
- Produces: `SmolVLAMLX.from_pretrained(model_id, cache_dir=None, dtype=mx.bfloat16)`, `predict_action_chunk(observation, noise=None)`, `select_action(observation)`, `reset()`, and read-only `queued_actions: int`.

- [ ] **Step 1: Write failing action-queue, reset, prefix-reuse, and deterministic tests**

```python
def test_select_action_uses_and_clears_queue(policy, observation):
    first = policy.select_action(observation)
    assert policy.queued_actions == policy.config.n_action_steps - 1
    policy.reset()
    assert policy.queued_actions == 0
    assert first.shape == (policy.config.action_dim,)

@pytest.mark.parametrize("dtype,max_abs", [(mx.float32, 5e-3), (mx.bfloat16, 5e-2)])
def test_deterministic_chunk_matches_reference(dtype, max_abs, policy_factory, golden):
    chunk = policy_factory(dtype).predict_action_chunk(golden.observation(), noise=golden.mx("noise", dtype))
    np.testing.assert_allclose(np.asarray(chunk), golden.array("actions_normalized"), atol=max_abs, rtol=0)
```

- [ ] **Step 2: Run tests before policy assembly exists**

Run: `uv run pytest tests/test_policy_api.py tests/test_end_to_end.py -v`

Expected: missing `SmolVLAMLX` failures.

- [ ] **Step 3: Assemble the model and implement prefix reuse plus action queue**

```python
def select_action(self, observation: Mapping[str, object]) -> np.ndarray:
    if not self._queue:
        chunk = self.predict_action_chunk(observation)
        self._queue.extend(np.asarray(chunk[: self.config.n_action_steps]))
    return self._queue.popleft()
```

Compute preprocessing, vision, connector, and language prefix once per chunk. Pass the frozen prefix cache through every Euler step, evaluate MLX arrays before timing or returning, slice padded action dimensions, and un-normalize before queueing.

Extend `tests/test_conversion.py` with a full-model load assertion that compares the instantiated MLX parameter tree with `name_map.json` and proves every converted target tensor initializes exactly one model parameter.

- [ ] **Step 4: Run deterministic parity and API tests for all real golden samples**

Run: `uv run pytest tests/test_policy_api.py tests/test_end_to_end.py -v`

Expected: API behavior passes; fp32 and bf16 normalized action chunks stay within their fixed maximum absolute differences.

- [ ] **Step 5: Commit the public policy boundary**

```bash
git add smolvla_mlx/policy.py smolvla_mlx/__init__.py tests/test_policy_api.py tests/test_end_to_end.py PROGRESS.md
git commit -m "phase-4: add policy API (end-to-end tests pass)"
```

### Task 11: Statistical correctness gate

**Files:**
- Create: `tests/test_statistical.py`, `scripts/statistical_check.py`
- Modify: `PROGRESS.md`, `STATUS.md`

**Interfaces:**
- Consumes: at least 50 deterministic real dataset observations, ground-truth actions, PyTorch fp32 outputs, and MLX fp32/bf16 outputs.
- Produces: `StatisticalResult.from_json(path)` and a JSON evidence record with per-backend MAE and MLX/reference ratios.

- [ ] **Step 1: Write a failing ratio-gate test**

```python
@pytest.mark.slow
def test_mlx_is_not_worse_than_reference(statistical_result):
    assert statistical_result.sample_count >= 50
    assert statistical_result.mlx_fp32_mae <= 1.05 * statistical_result.torch_fp32_mae
    assert statistical_result.mlx_bf16_mae <= 1.05 * statistical_result.torch_fp32_mae
```

- [ ] **Step 2: Run before the evidence script exists**

Run: `uv run --extra reference pytest tests/test_statistical.py -v`

Expected: absent fixture or script failure.

- [ ] **Step 3: Implement deterministic sample selection and MAE reporting**

Use stable episode/index pairs, identical observations, checkpoint normalization, and ground-truth action horizons. Serialize sample IDs, individual absolute-error sums/counts, aggregate MAEs, and both ratios so the result is independently auditable.

```python
@pytest.fixture(scope="session")
def statistical_result() -> StatisticalResult:
    return StatisticalResult.from_json(Path(".cache/statistical.json"))
```

- [ ] **Step 4: Run the full 50-sample gate in fp32 and bf16**

Run: `HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv SMOLVLA_MLX_CACHE=$PWD/.cache/smolvla_mlx uv run --extra reference python scripts/statistical_check.py --samples 50 --output .cache/statistical.json && uv run --extra reference pytest tests/test_statistical.py -v`

Expected: both MLX MAEs are at most `1.05` times the PyTorch fp32 MAE.

- [ ] **Step 5: Commit the statistical gate and numerical evidence summary**

```bash
git add tests/test_statistical.py scripts/statistical_check.py PROGRESS.md STATUS.md
git commit -m "phase-4: validate statistical parity (statistical test passes)"
```

### Task 12: Benchmarking and parity-preserving optimization

**Files:**
- Create: `scripts/bench.py`, `tests/test_bench.py`, `BENCHMARK.md`
- Modify: `smolvla_mlx/policy.py`, `PROGRESS.md`

**Interfaces:**
- Consumes: warmed policy, fixed real observation, fp32/bf16 modes, machine metadata, and 50 measured runs.
- Produces: `BenchmarkResult`, `run_benchmark(policy, observation, measured_runs=50) -> BenchmarkResult`, and `BENCHMARK.md` with preprocessing, vision, prefix, expert-loop, total median/p95, and peak memory.

- [ ] **Step 1: Write failing benchmark-schema and warmup-exclusion tests**

```python
def test_benchmark_result_has_required_metrics(benchmark_result):
    assert benchmark_result.measured_runs == 50
    assert benchmark_result.total_ms.median > 0
    assert benchmark_result.total_ms.p95 >= benchmark_result.total_ms.median
    assert set(benchmark_result.stages) == {"preprocessing", "vision", "prefix", "expert"}
```

- [ ] **Step 2: Run tests before benchmark implementation**

Run: `uv run pytest tests/test_bench.py -v`

Expected: absent benchmark API failure.

- [ ] **Step 3: Implement synchronized measurements and machine report**

Evaluate MLX outputs before stopping each timer, run warmups outside the sample set, collect 50 samples per dtype, measure peak memory through MLX's available memory API, and record `sysctl`, macOS, MLX, Python, checkpoint revision, and git commit.

- [ ] **Step 4: Apply optimizations one at a time behind the full parity gate**

Batch cameras, reuse the prefix, remove Python work from the Euler inner loop, and use `mx.compile` only where output equality remains within the fixed thresholds. After each change run `tests/test_vision.py`, `tests/test_language.py`, `tests/test_expert.py`, and `tests/test_end_to_end.py` before retaining it.

- [ ] **Step 5: Measure a quiet final run and commit results**

Run: `make test && make bench`

Expected: the complete suite passes and `BENCHMARK.md` contains fp32/bf16 median, p95, stage split, and peak memory. The under-200-ms bf16 target is reported as a target, not converted into a correctness gate.

```bash
git add scripts/bench.py tests/test_bench.py smolvla_mlx/policy.py BENCHMARK.md PROGRESS.md
git commit -m "phase-5: benchmark native inference (benchmark tests pass)"
```

### Task 13: Packaging, CLI, documentation, and fresh-install proof

**Files:**
- Create: `smolvla_mlx/cli.py`, `tests/test_cli.py`, `tests/test_wheel.py`
- Create: `README.md`
- Modify: `NOTICE`
- Modify: `pyproject.toml`, `uv.lock`, `PROGRESS.md`, `STATUS.md`

**Interfaces:**
- Consumes: all public library APIs, converter, benchmark script, dataset loader, and package metadata.
- Produces: `smolvla-mlx convert|test|bench|predict`, built wheel, documented Python API, offline cache reuse, and final status.

- [ ] **Step 1: Write failing CLI and wheel-isolation tests**

```python
def test_cli_exposes_required_commands():
    result = subprocess.run([sys.executable, "-m", "smolvla_mlx.cli", "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    for command in ("convert", "test", "bench", "predict"):
        assert command in result.stdout

def test_built_wheel_imports_without_reference_dependencies(fresh_venv, wheel):
    fresh_venv.install(wheel)
    modules = fresh_venv.run("import smolvla_mlx,sys; print(','.join(sys.modules))")
    assert "torch" not in modules and "lerobot" not in modules and "transformers" not in modules
```

- [ ] **Step 2: Run before CLI entry points and wheel fixtures exist**

Run: `uv run pytest tests/test_cli.py tests/test_wheel.py -v`

Expected: missing CLI app or wheel fixture failures.

- [ ] **Step 3: Implement CLI commands and complete package metadata**

Build the CLI with the standard-library `argparse` module so no unapproved runtime dependency is added. Each command must return nonzero on failure and print the model ID, resolved cache, dtype, and output artifact. `predict --dataset <id> --index <n>` loads one real frame, constructs the observation mapping, runs `select_action`, and prints a JSON action vector.

- [ ] **Step 4: Document install, API, correctness, performance, licenses, and limitations**

Write README examples for `pip install`, `from_pretrained`, `select_action`, reset, offline cache reuse, each CLI command, the exact reference/checkpoint revisions, fixed tolerances, benchmark table, and v0.1 exclusions. Ensure `NOTICE` lists every vendored source and license.

- [ ] **Step 5: Run fresh-environment and full Definition-of-Done checks**

Run: `uv build && uv run pytest tests/test_cli.py tests/test_wheel.py -v && make test && git diff --check`

Expected: wheel installs into a clean environment, prediction works on a real frame, runtime isolation passes, no test is skipped or xfailed, and every `BRIEF.md` Section 9 checkbox has linked evidence in `PROGRESS.md`.

- [ ] **Step 6: Mark completion and commit**

Write `STATUS.md` with the exact line `DEFINITION OF DONE MET` only after all fresh checks succeed.

```bash
git add smolvla_mlx/cli.py tests/test_cli.py tests/test_wheel.py README.md NOTICE pyproject.toml uv.lock PROGRESS.md STATUS.md
git commit -m "phase-6: package v0.1 (full suite passes)"
```

### Task 14: Final repository audit

**Files:**
- Modify only if the audit finds an evidenced defect: files named by the failing check.

**Interfaces:**
- Consumes: clean repository, all committed evidence, optional `origin`.
- Produces: verified clean status, optional push, and a release-ready commit.

- [ ] **Step 1: Run the complete verification matrix from a clean checkout state**

Run: `make goldens && make test && make bench && uv build && git diff --check && git status --short`

Expected: byte-stable golden manifest, zero test failures/skips/xfails, benchmark artifact present, wheel built, no whitespace defects, and only ignored caches/build outputs outside git status.

- [ ] **Step 2: Inspect dependency and repository boundaries**

Run: `! rg -n "(^|[[:space:]])(import|from)[[:space:]]+(torch|lerobot|transformers)" smolvla_mlx`

Run: `! git ls-files | rg "(^|/)(\.cache|golden|.*\.safetensors|.*\.npy)(/|$)"`

Expected: both commands produce no prohibited tracked/runtime matches.

- [ ] **Step 3: Verify completion documentation against the specification**

Check every Section 9 item against its named test output or artifact, confirm `STATUS.md` has `DEFINITION OF DONE MET`, and confirm `HUMAN_TASKS.md` has no open item.

- [ ] **Step 4: Push only when an `origin` already exists**

Run: `git remote get-url origin && git push origin main`

Expected: if `origin` is absent, record that no push was attempted; if present, push succeeds without changing public metadata or adding attribution.
