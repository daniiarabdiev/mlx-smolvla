# SmolVLA MLX — Agent Handoff Report

**Status: v0.1 complete.** This report is the evidence-backed handoff for any
agent continuing the project. It records what was built, what was verified, and
the boundaries that must be preserved. It does **not** authorize starting the
deferred robot, training, or quantization work without a newly agreed scope.

## 1. Canonical workspace and repository state

| Item | Verified value |
| --- | --- |
| Canonical working tree | `/Users/dan/Desktop/workshop/robotics-mlx-contrib` |
| Branch | `main` |
| Completion commit before this report | `f1ae9d647a6f711305aa1f8cdb534dec33e160ef` (`f1ae9d6`) |
| Git remote | None configured |
| Completion status | `STATUS.md` says `DEFINITION OF DONE MET` |

An otherwise-empty, separately initialized Git directory exists at
`/Users/dan/Documents/ChatGPT/robotics-mlx-contrib`. Do not work there; it is
not the implementation repository.

The worktree was clean when this report was started. This document and its
`PROGRESS.md` entry are documentation-only changes.

## 2. What was delivered

`smolvla_mlx` is a native Apple Silicon / MLX inference implementation for the
audited `lerobot/smolvla_base` checkpoint. The runtime has no imports of
PyTorch, LeRobot, or Transformers.

Implemented capabilities:

- Strict checkpoint conversion and loading: all **500 tensors** and
  **450,046,176 parameters** are accounted for.
- Policy preprocessing for the audited SO-101 data: two real camera streams,
  language tokenization, state handling, and the checkpoint's observed
  normalization behavior.
- SmolVLM visual encoder, pixel-shuffle connector, and the first **16** VLM
  prefix-decoder layers with K/V caching.
- The 16-layer action expert, flow-matching Euler sampler, and action queue.
- Public `SmolVLAMLX` / `select_action` / `reset` policy API.
- Native command line interface:
  `smolvla-mlx convert`, `test`, `bench`, and `predict`.
- Installable source distribution and macOS arm64 wheel, including the native
  RMSNorm extension.

The core runtime is under `smolvla_mlx/`. Reference-only PyTorch / LeRobot code
lives under `reference/`, `scripts/`, and tests; it is deliberately not a
runtime dependency.

## 3. Pinned artifacts and audited architecture

| Artifact | Pinned value |
| --- | --- |
| Reference stack | LeRobot `0.6.1`, PyTorch `2.11.0`, Transformers `5.5.4` |
| Policy checkpoint | `lerobot/smolvla_base` at `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| Base VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Golden dataset | `lerobot/svla_so101_pickplace` at `f641879e22172be7e8161d5e6c1503c2d2feb657` |
| VLM / expert layers | 16 / 16 |
| VLM / expert widths | 960 / 720 |
| Camera input used in v0.1 | `observation.images.side`, `observation.images.up` |
| Flow schedule | 10 Euler steps, `t=1.0` to `0.1`, `dt=-0.1` |
| Action representation | internal `50 × 32`, output slice `50 × 6`, queue length 50 |

The model configuration exposes three camera slots, but the audited v0.1
workflow uses the two real camera streams above. The checkpoint declares
mean/std normalization, yet its saved keys do not match the actual SO-101
`observation.state` and `action` keys. The pinned implementation consequently
matches the reference's effective identity behavior for those runtime values.
Do not "fix" that without a new reference audit and parity evidence.

Detailed derivation and source paths are in `ARCHITECTURE.md`; reuse and license
decisions are in `REUSE_DECISIONS.md` and `NOTICE`.

## 4. Completion evidence

All of the following were run on the final exact v0.1 tree:

| Check | Result |
| --- | --- |
| Full suite | `make test`: **179 passed in 147.26 s** |
| Deterministic actions | Eight real observations pass fp32 max-abs `≤ 5e-3` and bf16 max-abs `≤ 5e-2` |
| Statistical gate | 50 real samples: fp32 MLX/reference action-MAE ratio `0.9999999969253671`; bf16 ratio `1.0000097740913103`; both below immutable gate `1.05` |
| Golden reproducibility | Two `make goldens` runs produced 2,160 tensors / 8 samples with identical manifest SHA-256 `8531e61b98506d5a43e0b1235de7aece578bf4efa66b007fa13f6cecd1ceb215` and metadata SHA-256 `469144c0a539a2c40a7f0ea33066f2be6da12d44bd94d2167913d97482460c56` |
| M5 Pro performance | fp32: 111.34 ms median / 111.80 ms p95 / 2.94 GiB peak; bf16: 131.12 ms median / 131.71 ms p95 / 2.44 GiB peak |
| Package builds | `UV_NO_CACHE=1 uv build` created source and macOS arm64 wheel artifacts; source archive contains native C++ build inputs and wheel contains `_rmsnorm_native` |
| Fresh installations | Both `pip install .` and wheel installation worked in clean Python 3.12 environments, with installed CLI and runtime-import isolation verified |
| Real installed CLI prediction | `smolvla-mlx predict --dataset lerobot/svla_so101_pickplace --episode 0 --index 0` returned `[-0.3573277, -0.3161944, -0.0685726, 0.0844326, 1.1378109, -0.8640175]` |

The performance evidence is from this M5 Pro with 48 GiB unified memory,
macOS 26.5.2, Python 3.12.13, and MLX 0.32.2. bf16 reduces peak memory but was
slower than fp32 under that MLX version; the bf16 under-200-ms target is still
met. See `BENCHMARK.md` for the detailed measurement record.

## 5. Runtime and packaging contract

The six exact runtime dependencies are:

```toml
huggingface-hub==1.29.0
mlx==0.32.2
numpy==2.2.6
pillow==12.3.0
safetensors==0.8.0
tokenizers==0.22.2
```

The optional `reference` extra is intentionally separate and contains
`lerobot[dataset,smolvla]==0.6.1` and `torch==2.11.0`. It is required for
reference tests and the dataset-backed child-process bridge, but not ordinary
native runtime import.

The package artifacts are in `dist/`:

- `smolvla_mlx-0.0.1.tar.gz`
- `smolvla_mlx-0.0.1-cp312-cp312-macosx_26_0_arm64.whl`

When rebuilding, use `UV_NO_CACHE=1 uv build`: `UV_CACHE_DIR` normally lives
inside the source tree, which causes `uv build` to reject it as a build cache.

## 6. Working rules that must remain intact

- Never loosen a numerical tolerance, skip a test, add an `xfail`, or replace
  model work with a mock. New behavior needs a red test first and independently
  verified numbers after the change.
- `smolvla_mlx/` must remain free of `torch`, `lerobot`, and `transformers`
  imports. The import-isolation tests enforce this contract.
- Keep the reference goldens CPU/fp32 and fixed-seed. Do not substitute MPS or
  Metal for the reference numerics.
- Route model, Hugging Face, and uv caches inside this repository:

  ```bash
  export HF_HOME="$PWD/.cache/hf"
  export UV_CACHE_DIR="$PWD/.cache/uv"
  export SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx"
  ```

- Preserve the existing caches rather than deleting or copying them. At
  handoff they contain roughly 965 MiB (`.cache/hf`), 1.9 GiB
  (`.cache/uv`), 65 GiB (`.cache/smolvla_mlx`), and 512 MiB (`tests/golden`).
  The golden directory is ignored by Git but required for the full suite.
- Do not touch `~/robot/so101`, serial ports, robot hardware, credentials, or
  external deployment/upload state. No hardware work was performed.
- Commit after each passing meaningful change. There is currently no Git
  remote, so no push is expected.

## 7. Known limitations and intentionally deferred work

### v0.1 scope boundary

The v0.1 Definition of Done is met. The following have **not** been started:

- Robot integration, including the deferred LeRobot async policy server and
  vendor-fork client.
- Training or fine-tuning parity (the prospective v0.2 goal).
- Quantization experiments beyond the accepted bf16 path.
- Support beyond the audited checkpoint and two-camera SO-101 input contract.

These are not defects in the delivered v0.1 inference port. They are separate
projects that need an explicit requirement, acceptance criteria, safety review,
and fresh plan before implementation.

### Metal strict-module-parity caveat

The strict immutable parity tests use focused MLX CPU compatibility primitives
to reproduce PyTorch CPU arithmetic exactly. Native Metal kernels are retained
for performance and successfully satisfy the accepted end-to-end bf16 behavior,
but they do not currently satisfy the much tighter raw CPU-reference module
comparisons for the Vision and Connector stages. Recorded evidence:

- Vision Metal fp32: relative L2 approximately `8.44e-3` to `9.73e-3`, max
  absolute difference `0.219` to `0.421` over the eight samples.
- Connector Metal: maximum absolute difference up to approximately `0.045`.

This is documented in `FAILURE_vision.md` and `FAILURE_connector.md`. It must
not be addressed by changing test tolerances. If exact Metal module parity is a
new goal, treat it as an explicit research/performance project and maintain the
existing v0.1 gates unchanged. `FAILURE_language.md` is historical: its CPU
compatibility issue was resolved and the final decoder suite passes.

## 8. Recommended first turn for a successor agent

1. Work only in the canonical repository listed above and inspect the state:

   ```bash
   cd /Users/dan/Desktop/workshop/robotics-mlx-contrib
   export HF_HOME="$PWD/.cache/hf"
   export UV_CACHE_DIR="$PWD/.cache/uv"
   export SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx"
   git status --short --branch
   git log --oneline -12
   ```

2. Read `AGENTS.md`, `BRIEF.md`, this report, `STATUS.md`, the final entries in
   `PROGRESS.md`, `ARCHITECTURE.md`, `REUSE_DECISIONS.md`, and the relevant
   `FAILURE_*.md` documents before changing code.

3. Establish a fresh baseline with:

   ```bash
   make test
   ```

4. If asked to extend the project, first make the requested scope explicit
   (for example: exact Metal parity, memory/speed optimization, quantization,
   server-only integration, or v0.2 training parity). Then write a new
   phase-by-phase plan with tests and acceptance thresholds before touching the
   current implementation.

5. Preserve the v0.1 end-to-end, statistical, import-isolation, packaging, and
   benchmark gates. Run the relevant focused test after every code change and
   the full suite before declaring any extension complete.

## 9. Most useful source files

| Need | Files |
| --- | --- |
| User-facing setup, API, CLI, correctness summary | `README.md`, `pyproject.toml`, `Makefile` |
| Exact audited model/data behavior | `ARCHITECTURE.md`, `reference/audit.py`, `reference/policy.py` |
| Rationale for reimplemented/vendored code | `REUSE_DECISIONS.md`, `NOTICE` |
| Runtime implementation | `smolvla_mlx/preprocessing.py`, `vision.py`, `connector.py`, `language.py`, `expert.py`, `flow.py`, `policy.py`, `convert.py`, `cli.py` |
| Native CPU compatibility primitives | `smolvla_mlx/rmsnorm.py`, `smolvla_mlx/native/` |
| Golden generation and verification | `scripts/make_goldens.py`, `scripts/statistical_check.py`, `tests/` |
| Performance evidence | `scripts/bench.py`, `BENCHMARK.md` |
| Completion and historical decision record | `STATUS.md`, `PROGRESS.md`, `FAILURE_*.md` |

## 10. Handoff conclusion

The original long-term direction—moving SmolVLA to an MLX-native inference
implementation that makes proper use of Apple Silicon and unified memory—has
been achieved for the defined v0.1 inference scope, with reproducible numerical
evidence and an installable package. The next agent should not restart or
re-port it; they should preserve this verified baseline and begin only a
separately authorized follow-on scope.
