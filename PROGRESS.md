# Progress

## 2026-08-31 — project start

- What: accepted the native SmolVLA MLX v0.1 specification and created the
  design and implementation plan.
- Evidence: design and plan self-reviews found 0 placeholders and 0 unresolved
  interface inconsistencies.
- Decision: use a reference-first hybrid port, reusing compatible `mlx-vlm`
  code and vendoring only where verified behavior requires it.
- Open questions: the architecture questions in `BRIEF.md` Section 10 will be
  resolved from the installed LeRobot source and checkpoint.
- Next: complete Phase 0 bootstrap and reference discovery.

## 2026-08-31 — Phase 0 bootstrap

- What: created the Python 3.12 uv project, in-repo cache routing, MLX runtime
  dependency set, package skeleton, Make targets, and required state files.
- Environment: macOS 26.5.2 (25F84), Apple M5 Pro, 51,539,607,552 bytes
  unified memory, Python 3.12.13, uv 0.11.25, MLX 0.32.2, mlx-vlm 0.6.17.
- Device evidence: `mx.default_device()` returned `Device(gpu, 0)` and
  `mx.metal.is_available()` returned `True`.
- Test evidence: `make test` collected 2 tests; 2 passed, 0 failed, 0 skipped
  in 0.02 seconds. The isolated subprocess loaded none of `torch`, `lerobot`,
  or `transformers`.
- Decision: exact resolved package versions are committed in `uv.lock`.
- Open questions: none for bootstrap.
- Next: install the optional reference lane and discover the exact mainline
  LeRobot SmolVLA sources, checkpoint revision, and usable SO-101 dataset.

## 2026-08-31 — Phase 0 reference discovery

- What: pinned and discovered the installed mainline reference without relying
  on remembered import paths; queried immutable Hub revisions and safetensor
  metadata without downloading the model weights.
- Reference: LeRobot 0.6.1, PyTorch 2.11.0, Transformers 5.5.4. Exact source
  paths are recorded in `ARCHITECTURE.md`.
- Checkpoint evidence: `lerobot/smolvla_base` revision
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`; 500 tensors and 450,046,176
  parameters from the Hub safetensor header.
- Dataset evidence: `lerobot/svla_so101_pickplace` revision
  `f641879e22172be7e8161d5e6c1503c2d2feb657`; 50 episodes, 11,939 frames,
  two 480x640 camera streams, 6-D state, 6-D action, and a language-task table.
- Test evidence: `tests/test_reference_discovery.py` collected 3 tests; 3
  passed, 0 failed, 0 skipped in 8.19 seconds.
- Decision: use the official SO-101 dataset named in the brief; no fallback is
  required. PyTorch is explicitly pinned to 2.11.0 in the reference extra.
- Open questions: the runtime architecture and preprocessing details remain
  for the Phase 1 source/runtime audit.
- Next: download the pinned checkpoint into the repository cache, load the
  policy on CPU fp32, and execute one real observation.

## 2026-08-31 — Phase 1 reference runtime audit

- What: loaded the immutable SmolVLA checkpoint on CPU fp32, pinned its base
  SmolVLM2 tokenizer/config snapshot, mapped one real two-camera SO-101 sample,
  ran a deterministic 50-action chunk, and generated an execution-derived
  architecture report plus complete safetensors header inventory.
- Reference pins: checkpoint `c83c3163b8ca9b7e67c509fffd9121e66cb96205`;
  base VLM `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; LeRobot 0.6.1 / PyTorch 2.11.0 /
  Transformers 5.5.4 / mlx-vlm 0.6.4.
- Architecture evidence: 450,046,176 parameters: 350,165,184 VLM and
  98,245,840 expert. Two 512x512 cameras produce 2x64x960 connector tokens;
  the fixed prefix is 1x177x960, 16 cache layers each expose K/V of 1x5x177x64,
  and the expert is 16 layers wide 720 with self-attention at even layers and
  VLM-K/V cross-attention at odd layers.
- Preprocessing evidence: bilinear aspect-preserving resize to 512x512 with
  left/top zero padding, then `x * 2 - 1`; task newline plus right-padded
  48-token encoding; state zero-padded from 6 to 32. The saved processor
  declares MEAN_STD but has only robot-prefixed action-stat keys, so on this
  unprefixed checkpoint path state normalization and action unnormalization are
  both effective identity transforms.
- Flow evidence: starts at t=1.0, uses 10 steps at dt=-0.1 through t=0.1, and
  applies `x_t = x_t + dt * v_t`. A unit-velocity characterization ends at
  -1.0000001192092896, consistent with float32 arithmetic.
- Reuse decision: vendor/adapt only the MLX-only focused vision/connector/text
  primitives from mlx-vlm under MIT notices; reimplement preprocessing,
  tokenization, policy assembly, expert, flow, cache semantics, and conversion
  from the audited behavior to preserve dependency isolation.
- Test evidence: `make test TESTS="tests/test_reference_discovery.py
  tests/test_reference_policy.py tests/test_reference_audit.py tests/test_cache.py
  tests/test_import_isolation.py"` collected 11 tests and passed 11/11 in
  45.37 seconds. The audit script writes JSON evidence and a 500-tensor
  Markdown inventory.
- Open questions: none that block deterministic golden capture. Exact MLX
  preprocessing/tokenizer behavior will be proven against saved tensors before
  the model port uses it.
- Next: capture byte-stable goldens at the audited module boundaries, then map
  every checkpoint tensor into the native MLX module tree.

## 2026-08-31 — Phase 0 deterministic golden capture

- What: implemented an atomic `.npy` golden writer/store, deterministic
  eight-episode sample plan, transparent residual-block hooks, and the
  `make goldens` reference capture command. Generated arrays include raw and
  preprocessed inputs, vision and connector features, all 16 VLM block outputs
  and K/V pairs, all 16 expert outputs for each of 10 Euler steps, action suffix
  embeddings, padded velocities, and final normalized/un-normalized actions.
- Coverage: `tests/golden/manifest.json` contains 2,160 tensors across 8 real
  frames from episodes 0, 7, 14, 21, 28, 35, 42, and 49. Each sample has 270
  named tensors: 16 VLM hidden outputs, 32 VLM cache K/V tensors, 170 expert
  tensors, 30 flow tensors, plus preprocessing and final-action boundaries.
- Correction: the per-step velocity remains `[1, 50, 32]` in the padded action
  space; only `SmolVLAPolicy._get_action_chunk` slices the final action chunk to
  the physical `[1, 50, 6]` dimension. The architecture report and tests now
  encode that distinction.
- Reproducibility evidence: `make goldens`, copied `manifest.json` and
  `metadata.json`, ran `make goldens` again, then `cmp` passed for both files.
  The output reports 2,160 tensors and 8 samples on both runs.
- Test evidence: `tests/test_goldens.py` collected 5 tests and passed 5/5 in
  17.02 seconds, including real reference capture and the CLI capture path.
- Decision: goldens are intentionally ignored because they are large generated
  derivatives. The writer's per-file SHA-256 manifest and fixed checkpoint,
  VLM, dataset, sample, and seed metadata make them locally regenerable.
- Open questions: none for golden capture.
- Next: implement checkpoint-derived native configuration and preprocessing,
  beginning with exact camera resize/padding and tokenizer parity against all
  saved golden inputs.

## 2026-08-31 — Phase 3 native configuration and preprocessing

- What: added dependency-isolated `SmolVLAConfig`, `ProcessedObservation`, and
  `SmolVLAPreprocessor` runtime interfaces. The native processor reads
  `tokenizer.json` through `tokenizers`, appends the required instruction
  newline, pads on the right with token ID 2 to 48 slots, and emits MLX arrays.
- Image evidence: a float32 NumPy bilinear implementation matches PyTorch
  `align_corners=False`; it aspect-resizes 480x640 frames to 384x512, pads
  128 rows on top (and left when needed), then applies `x * 2 - 1`.
  All eight golden cases match the reference pixels at max absolute tolerance
  1e-5.
- State/action evidence: observations remain 1x6 after the saved processor,
  then policy code will zero-pad state to 32 before projection. The checkpoint's
  mismatched statistics retain identity action normalize/unnormalize behavior,
  proven by an MLX action round-trip test.
- Isolation evidence: importing `smolvla_mlx`, `.config`, `.preprocessing`,
  and `.types` in a clean subprocess leaves `torch`, `lerobot`, and
  `transformers` absent from `sys.modules`.
- Test evidence: `make test TESTS="tests/test_config.py
  tests/test_preprocessing.py tests/test_import_isolation.py"` collected 11
  tests and passed 11/11 in 0.50 seconds. The 8 parametrized real-golden cases
  assert exact IDs/masks and fixed image/state tolerances.
- Open questions: native preprocessing currently uses NumPy for the exact
  bilinear arithmetic before moving the compact result to MLX. It is correct
  and dependency-light; Phase 5 benchmarking will decide whether a Metal resize
  optimization can preserve the same golden tolerance.
- Next: define the full native module parameter tree and explicit safetensors
  conversion map, with one source tensor mapped exactly once.

## 2026-08-31 — Phase 2 checkpoint conversion

- What: implemented a native MLX safetensors converter with a strict 500-to-500
  canonical name bijection. Target names are `vision.*`, `connector.*`,
  `language.*`, `expert.*`, and the five policy projections; the unused language
  head is retained as `language.lm_head` so source/target parameter equality is
  exact.
- Transform evidence: 499 tensors retain their original layout. The sole shape
  transform is the SigLIP patch convolution,
  `model...patch_embedding.weight` `[768, 3, 16, 16]` →
  `vision.embeddings.patch_embedding.weight` `[768, 16, 16, 3]`, matching MLX
  Conv2D's OHWI convention.
- Integrity evidence: each conversion emits fp32 or bf16 safetensors plus a
  500-record `name_map.json` carrying source and target shapes, transform name,
  raw source SHA-256, and raw converted SHA-256. Both modes report exactly
  450,046,176 source and target parameters, zero unmapped source tensors, and
  zero uninitialized target tensors.
- Artifacts: local ignored test conversions measured 1.7 GiB fp32 and 858 MiB
  bf16. This validates that the user-facing cache can retain either precision
  without reference dependencies.
- Test evidence: `make test TESTS="tests/test_conversion.py
  tests/test_import_isolation.py"` collected 4 tests and passed 4/4 in
  2.16 seconds. Tests validate fp32/bf16 output, every mapping's uniqueness,
  the special convolution layout, and native import isolation.
- Open questions: none for conversion. The next modules must retain the
  canonical tree names in `name_map.json`; loading them will turn any mismatch
  into an immediate, auditable error.
- Next: vendor/adapt the MIT MLX vision and connector primitives under the
  documented license boundary, load converted weights, and prove their outputs
  against the vision/connector goldens.

## 2026-08-31 — Phase 3 vision encoder and connector

- What: added dependency-isolated native MLX `VisionEncoder` and scale-4
  `Connector`, with the complete 12-layer 768-wide SigLIP-style tower,
  tanh-GELU, 1e-6 layer normalization, 32x32 patch positions, and the audited
  1024-to-64 pixel shuffle followed by the 960-wide projection. Both modules
  strictly load their canonical converted weight subtrees.
- Licensing: focused adaptations retain the mlx-vlm 0.6.4 MIT attribution in
  their source headers; `NOTICE` and `REUSE_DECISIONS.md` enumerate their
  upstream source paths and adaptations.
- Golden evidence: the original Section 6 thresholds remain unchanged. The
  32-case suite (eight real observations × fp32/bf16 storage × vision and
  connector) passed 32/32 in 42.10 seconds on MLX CPU, the same execution
  domain as the CPU/fp32 PyTorch goldens. fp32 sample-000 vision error was
  relative L2 `6.996e-06`, max absolute `3.338e-04`; connector output was
  elementwise equal. bf16 retains compact weights but uses fp32 activations,
  matching the reference's checkpoint-upcast golden methodology.
- Regression evidence: `make test` collected 61 tests and passed 61/61 in
  107.05 seconds, including reference, golden reproducibility, conversion,
  preprocessing, and dependency-isolation coverage after the new runtime
  modules were imported in a clean subprocess.
- Metal evidence: default GPU execution is functional but cannot meet the
  immutable CPU-reference threshold due to reproducible reduction-order
  differences. The verified envelopes and three ruled-out implementation
  hypotheses are recorded in `FAILURE_vision.md` and `FAILURE_connector.md`;
  no tolerance was relaxed and no test was skipped.
- Decision: use the MLX CPU backend for strict deterministic module parity
  while retaining GPU-capable native modules for later Phase 5 investigation.
  Future decoder/expert golden tests will follow the same CPU/fp32 reference
  contract until a Metal precision control or exact custom kernel is found.
- Open questions: Metal numerical parity remains open for performance work;
  it does not block correctness-path implementation on the audited reference
  device.
- Commit: `phase-3: port vision and connector (vision tests pass)`.
- Next: port the truncated language decoder, prefix assembly, and KV cache
  against the saved layer-by-layer goldens.

## 2026-08-31 — Phase 3 truncated language decoder precision boundary

- What: added a dependency-isolated native `TruncatedLanguageModel`, exact
  image/language/state prefix assembly, the cumulative prefix-LM 2-D mask,
  split-half RoPE at base 10,000, 15-query/5-KV grouped-query attention,
  explicit post-RoPE cache export, and a bounded decoder cutoff. The source
  checkpoint's actual language subtree is exactly layers `0..15`, so the native
  tree uses 16 layers and strict loading succeeds; it does not synthesize the
  16 absent layers advertised by the base-VLM config.
- Prefix/cache evidence: fp32 prefix embeddings and all masks/position IDs are
  exact. Across all eight real golden samples, exported fp32 K/V differ by at
  most `4.351139e-05` / `1.180172e-05` for keys/values, and the final normalized
  prefix output differs by at most `2.336502e-05` absolute. The 3-layer cutoff,
  cache length, and cache mask all pass.
- bf16 contract: the test boundary now correctly keeps observations and
  connector outputs in fp32 while varying only compact checkpoint storage.
  This follows the real runtime path and BRIEF Section 6: bf16 has a relative
  L2 bound of `3e-2`, not an extra maximum-absolute bound. All 25 bf16 focused
  cases pass under that unchanged contract; no tolerance was loosened.
- fp32 precision evidence: `uv run pytest tests/test_prefix.py
  tests/test_language.py -q` collected 50 cases. It passed 46 and failed only
  the raw decoder-residual maximum-absolute check for samples 004–007:
  `1.4648438e-03`, `3.2958984e-03`, `2.3193359e-03`, and `1.0986328e-03`
  respectively, versus the immutable `1e-3` limit. Every one of those cases
  remains comfortably within the required relative-L2 bound; the failure is
  solely the raw-output maximum absolute boundary.
- Root-cause trace: the incoming prefix is byte-identical. The first divergence
  occurs in MLX CPU RMS-normalization/reduction at roughly `5e-7` to `1e-6` per
  activation. With an identical normalized input, native and PyTorch MLP
  projections agree; the small norm difference is amplified by the high-gain
  SwiGLU path. In sample 005 layer 03, token 134 / feature 87, the native MLP
  result is `-1584.9042` versus PyTorch `-1584.9069`, producing `2.6855469e-03`
  raw error even though its layer K/V boundaries remain near `1e-5`.
- Ruled out: wrong prefix ordering/mask/state placement (exact); wrong layer
  depth/weight mapping (strict source tree loads); RoPE base/pairing/GQA/cache
  layout (K/V evidence above); delayed MLX materialization (unchanged); a
  direct reference-form RMS expression (unchanged); fp64 RMS reduction
  (improved several samples but sample 005 still reached `1.953125e-03`); and
  GEMM-reduced RMS, fp64 attention, and fp64 MLP variants (did not meet the
  fixed bound and sometimes worsened it).
- Upstream check: the installed MLX `0.32.2` is the current release, and its
  public RMSNorm source documents the CPU fallback as fp32
  `mean(square(x))` followed by `rsqrt`, with no exposed reduction-precision
  control. An in-place upgrade or supported runtime flag is therefore not an
  available resolution for this boundary.
- Decision: retain the source-semantic fp32 MLX implementation and the
  immutable test. `FAILURE_language.md` records the reproducible precision
  boundary; no test is skipped or xfailed. Later modules that require a
  Section-6-passing prefix decoder are blocked pending an MLX CPU reduction
  precision control or a justified native kernel that reproduces this reference
  arithmetic.
- Licensing/isolation: added the mlx-vlm MIT and LeRobot Apache-2.0 notices for
  `language.py`; the clean-subprocess isolation test now imports the decoder as
  part of its forbidden-framework check.
- Next: preserve this checkpointed native implementation and resolve the
  documented RMS-reduction precision boundary before integrating the expert.

## 2026-08-31 — Phase 3 native CPU RMSNorm boundary

- What: added a dependency-isolated MLX C++ extension with a lazy CPU
  `Primitive` that reproduces PyTorch 2.11's four-level cascade reduction over
  the model's fixed 960-wide RMSNorm dimension. `ReferenceRMSNorm` selects it
  only for an active CPU stream and keeps GPU execution on `mx.fast.rms_norm`.
- Arithmetic evidence: an isolated Apple-clang source translation matches all
  177 real prefix-row squared means bit-for-bit with PyTorch CPU. The packaged
  primitive then matched the complete weighted `torch.rms_norm` output for all
  169,920 real prefix values exactly.
- Build evidence: the MLX extension uses the project's supported CMake route
  with MLX 0.32.2 and its matching nanobind 2.15.0 ABI. The direct boundary
  test `tests/test_rmsnorm.py` collected 1 test and passed 1/1 in 0.63 seconds.
- Isolation evidence: a clean subprocess imports `smolvla_mlx.rmsnorm` with
  the rest of the runtime and leaves `torch`, `lerobot`, and `transformers`
  absent. The direct and isolation checks passed 2/2 in 0.74 seconds.
- Next: replace the decoder's three RMSNorm construction sites and rerun the
  immutable focused decoder suite.

## 2026-08-31 — Phase 3 exact CPU decoder arithmetic

- What: resolved the strict language-decoder precision boundary with focused,
  dependency-isolated MLX CPU primitives for the reference RMSNorm reduction,
  fixed-prefix RoPE, attention softmax, and SwiGLU SiLU activation. GPU paths
  retain the normal MLX kernels; the compatibility primitives are selected only
  for the CPU reference stream.
- Arithmetic evidence: the 177-token RoPE transform, masked 177-wide attention
  softmax, and 2,560-wide SiLU activation are each elementwise equal to the
  pinned PyTorch 2.11.0 CPU result. The softmax implementation uses PyTorch's
  four-lane pairwise reduction tree rather than a hardware horizontal-add
  shortcut, which removes the last one-ULP probability drift.
- Golden evidence: `uv run --extra reference pytest tests/test_rmsnorm.py
  tests/test_rope.py tests/test_softmax.py tests/test_silu.py
  tests/test_prefix.py tests/test_language.py tests/test_import_isolation.py -q`
  passed **57/57** in 7.62 seconds. The original prefix/language reproduction
  now passes all 50 cases under the unchanged Section 6 tolerances; no test was
  skipped, xfailed, or relaxed.
- Licensing: the compact ARM exponential adaptation is documented in `NOTICE`
  under SLEEF's Boost Software License 1.0. `REUSE_DECISIONS.md` records the
  bounded CPU-compatibility use case.
- Decision: the prefix decoder and KV cache are unblocked for the remaining
  policy implementation. `FAILURE_language.md` is retained as a resolved
  historical diagnosis rather than an active blocker.
- Commit: `phase-3: match CPU decoder arithmetic (decoder tests pass)`.
- Next: implement the SmolVLA action expert, flow-matching Euler loop, action
  queue, and public policy API against the already generated golden traces.

## 2026-08-31 — Phase 3 expert-width CPU RMSNorm

- What: extended the dependency-isolated CPU RMSNorm compatibility primitive
  from the VLM's 960 channels to the action expert's audited 720 channels. The
  primitive remains deliberately bounded to those two checkpoint widths and
  retains the same PyTorch-compatible cascade reduction.
- Test evidence: `uv run --extra reference pytest tests/test_rmsnorm.py -q`
  passed **3/3** in 1.02 seconds. The new 720-wide test uses the captured
  timestep/action embedding and an actual expert norm weight, and matches
  `torch.rms_norm` element-for-element.
- Next: load the expert weights into a native action-expert tree and add the
  first red projection, attention, and flow tests against the saved traces.

## 2026-08-31 — Phase 3 action expert and Euler flow

- What: implemented the dependency-isolated 720-wide `ActionExpert` and the
  native flow-matching sampler. The tree strictly loads all action/timestep
  projections, 16 expert blocks, final norm, and velocity head from the
  converted checkpoint.
- Architecture evidence: even-numbered blocks append 50 RoPE-positioned action
  K/V states to the corresponding frozen VLM cache for causal self-attention.
  Odd-numbered blocks project only the frozen 177-token VLM K/V cache for
  cross-attention, with query positions reset to zero exactly as in the
  reference. The cache is never mutated by denoising, which is the native
  equivalent of the reference cache crop after each step.
- Flow evidence: the native schedule is the ten fp32 values `1.0` through
  `0.1`; each update uses `x_t = x_t + (-0.1) * v_t`. The final 32-wide state
  is sliced only at the policy boundary, preserving the reference's padded
  velocity domain during all expert steps.
- Golden evidence: `uv run --extra reference pytest tests/test_rmsnorm.py
  tests/test_expert.py tests/test_flow.py tests/test_import_isolation.py -q`
  passed **39/39** in 15.44 seconds. This includes every action/timestep
  embedding across 8 real observations × 2 storage precisions, plus all 16
  expert-layer outputs, final expert norms, velocities, Euler states, and final
  normalized action chunks at all 10 steps for every sample. The Section 6
  tolerances remain unchanged.
- Isolation evidence: the clean subprocess now imports the expert and flow
  runtime modules and still loads none of `torch`, `lerobot`, or `transformers`.
- Next: assemble preprocessing, vision, prefix-cache, expert, unnormalization,
  and the 50-action FIFO into the public `SmolVLAMLX` API.

## 2026-08-31 — Phase 4 native policy API and deterministic parity

- What: added `SmolVLAMLX.from_pretrained`, `predict_action_chunk`,
  `select_action`, `reset`, `queued_actions`, and explicit converted-weight
  provenance. The loader supports a local checkpoint or Hub ID, reuses cached
  conversion artifacts, and strictly accounts for all 500 canonical tensors.
- Loader evidence: the native parameter tree and converted safetensors name set
  are identical in both storage modes. The test rejects both an omitted source
  tensor and a tensor attached under the wrong native prefix before inference
  begins.
- API evidence: prediction preprocesses one observation, evaluates vision and
  connector, creates exactly one VLM prefix cache, reuses that cache through
  ten denoising steps, then slices 32 padded action dimensions to the physical
  6-D normalized chunk. `select_action` postprocesses only when queueing and
  refills its FIFO with 50 actions only after it is empty; `reset()` clears it.
- Test evidence: `uv run --extra reference pytest tests/test_policy_api.py
  tests/test_end_to_end.py tests/test_import_isolation.py -q` passed **21/21**
  in 18.22 seconds. The end-to-end test covers 8 real frames × fp32/bf16
  storage at the fixed `5e-3` / `5e-2` maximum-absolute thresholds; queue and
  loader assertions run in both modes. The isolation subprocess imports the
  public policy and still loads no prohibited reference frameworks.
- Next: perform the 50-sample statistical accuracy check, record a benchmark,
  and add the packaging/CLI/README surface specified for v0.1.

## 2026-08-31 — Phase 4 statistical accuracy gate

- What: added a deterministic reference-lane statistical checker and a
  dependency-isolated JSON parser/gate. It evaluates the first physical action
  from the same noise-driven chunk against the ground-truth action at frame 0
  of each of the 50 SO-101 episodes, preserving the checkpoint's effective
  identity output normalization.
- Evidence artifact: `.cache/statistical.json` records 50 episode/frame IDs,
  noise seeds, per-backend absolute-error sums and element counts, aggregate
  MAEs, and MLX-to-reference ratios. Its atomic final values are PyTorch fp32
  MAE `55.783039437383415`, MLX fp32 MAE `55.78303926587105` (ratio
  `0.9999999969253671`), and MLX bf16 MAE `55.78358466590444` (ratio
  `1.0000097740913103`).
- Test evidence: `uv run --extra reference pytest tests/test_statistical.py
  -q` passed **1/1** in 0.01 seconds after the full checker completed. Both
  ratios remain below the immutable `1.05` gate; no tolerance was changed.
- Next: add reproducible performance measurement and a machine-readable
  benchmark report, then finish the user-facing packaging surface.

## 2026-08-31 — Phase 5 Metal timestep compatibility

- What: fixed the action timestep embedding's device boundary. CPU keeps the
  audited float64 construction before its fp32 cast for golden parity; Metal
  now uses the equivalent fp32 construction because MLX intentionally does not
  support float64 GPU arrays.
- Test evidence: `uv run --extra reference pytest tests/test_expert.py
  tests/test_flow.py -q` passed **36/36** in 14.08 seconds, including the new
  no-float64 Metal execution contract and the unchanged CPU golden suite.
- Metal smoke evidence: a real bf16 `SmolVLAMLX.predict_action_chunk` on the
  default `Device(gpu, 0)` returned a finite `[1, 50, 6]` result, took
  `2851.56 ms` on its cold run, and reported `1,868,069,105` bytes peak MLX
  memory. Against sample 000 it had relative L2 `0.00424612` and max absolute
  `0.0408914`, within the immutable bf16 end-to-end `5e-2` maximum bound.
- Next: make the warmed Metal path measurable over 50 runs with per-stage
  timing and an auditable benchmark report.

## 2026-08-31 — Phase 5 native Metal benchmark

- What: added synchronized stage-level timing with five excluded warmups and
  fifty measured real-observation runs per compact-weight mode. The benchmark
  evaluates MLX work at every boundary, reports median/p95s, and records the
  maximum of active and peak allocator memory after warmup.
- Machine evidence: Apple M5 Pro, 51,539,607,552-byte unified memory, macOS
  26.5.2, Python 3.12.13, and MLX 0.32.2. `BENCHMARK.md` records commit
  `72b1716d3cda72a26bced98721133fbd158a5919` and the complete output.
- Result evidence: fp32 median/p95 chunk latency is `111.34 / 111.80 ms`
  (preprocess `4.45`, vision+connector `46.12`, prefix `8.17`, expert loop
  `52.62` ms) at `2.94 GiB` peak MLX memory. bf16 is `131.12 / 131.71 ms`
  (stage medians `4.50 / 48.25 / 11.08 / 67.36` ms) at `2.44 GiB` peak. The
  bf16 latency target is met in this measured environment; its lower memory
  use is retained even though this MLX version's fp32 kernels are faster.
- Test evidence: `uv run pytest tests/test_bench.py
  tests/test_import_isolation.py -q` passed **2/2** in 0.21 seconds. The
  schema test verifies 50 measured samples, explicit warmup exclusion, all
  four required stages, percentile ordering, and peak-memory reporting.
- Next: add the CLI, README, wheel/fresh-install proof, and final audit.

## 2026-08-31 — Phase 6 CLI, source distribution, and standalone-script repair

- What: added the `smolvla-mlx` console entry point with `convert`, `test`,
  `bench`, and `predict` subcommands; added the project README; and made the
  source distribution include the CMake build file and native compatibility
  sources required to rebuild the extension.
- Build evidence: with all persistent caches routed under `.cache/`,
  `UV_NO_CACHE=1 uv build` produced both
  `dist/smolvla_mlx-0.0.1.tar.gz` and
  `dist/smolvla_mlx-0.0.1-cp312-cp312-macosx_26_0_arm64.whl`. The first build
  exposed the absent build inputs in the sdist; `MANIFEST.in` fixes the root
  cause rather than relying on an in-tree build.
- CLI/isolation evidence: `tests/test_cli.py` passed **1/1**, and the clean
  import-isolation test now imports the CLI along with every runtime module.
- Script repair evidence: the full-suite release check reproduced two
  standalone-script failures caused by Python setting `sys.path[0]` to
  `scripts/`. Both `make_goldens.py` and `inspect_reference.py` now insert the
  repository root before their first `reference` import, matching the working
  benchmark and statistical scripts. `uv run --extra reference pytest
  tests/test_goldens.py tests/test_reference_audit.py -q` passed **7/7** in
  35.28 seconds.
- Next: install the wheel into a clean virtual environment, exercise the
  installed console command against a real cached dataset frame, then run the
  complete final verification and repository audit.

## 2026-08-31 — Phase 6 fresh-install and real CLI prediction proof

- What: tightened the published runtime metadata to the six audited native
  dependencies and removed the unused `mlx-vlm` distribution, whose source
  logic is already vendored under the MIT attribution in `NOTICE`. All runtime
  versions are now exact pins in both `pyproject.toml` and `uv.lock`.
- Metadata evidence: the red/green installed-distribution tests first caught
  `mlx-vlm` as an unconditional requirement, then verified the final exact
  runtime set: `huggingface-hub 1.29.0`, `mlx 0.32.2`, `numpy 2.2.6`,
  `pillow 12.3.0`, `safetensors 0.8.0`, and `tokenizers 0.22.2`.
  `uv run pytest tests/test_distribution.py -q` passed **2/2** in 0.01
  seconds, and `uv lock --check` resolved 103 packages successfully.
- Fresh-install evidence: a new Python 3.12 virtual environment under
  `.cache/fresh-final.3AeWfA` completed the literal `pip install .` path,
  rebuilt the native CMake extension, and imported the installed
  `site-packages/smolvla_mlx` with no `torch`, `lerobot`, or `transformers`
  modules loaded. The installed console command exposed all four required
  subcommands and converted the real checkpoint into an 858 MiB bf16 MLX
  safetensors artifact.
- Real CLI evidence: after `pip install ".[reference]"` installed the pinned
  optional LeRobot 0.6.1/PyTorch 2.11.0 bridge, installed
  `smolvla-mlx predict --dataset lerobot/svla_so101_pickplace --episode 0
  --index 0` returned the real bf16 action
  `[-0.3573277, -0.3161944, -0.0685726, 0.0844326, 1.1378109, -0.8640175]`.
  Its dataset cache was preseeded from the already-downloaded public dataset;
  inference, conversion, and child-process extraction all ran in the fresh
  environment.
- Next: commit the dependency metadata/test/doc changes, run the full suite
  and build artifacts one final time, audit repository state, and write the
  Definition-of-Done status.

## 2026-08-31 — Phase 6 final artifact and reproducibility audit

- Golden reproducibility: `make goldens` regenerated all 2,160 tensors for
  the eight specified real observations twice. Both passes produced manifest
  SHA-256 `8531e61b98506d5a43e0b1235de7aece578bf4efa66b007fa13f6cecd1ceb215`
  and metadata SHA-256
  `469144c0a539a2c40a7f0ea33066f2be6da12d44bd94d2167913d97482460c56`.
- Full-suite evidence: `make test` passed **179/179** in 147.02 seconds after
  the release-metadata changes, including fp32/bf16 module parity,
  deterministic end-to-end, 50-frame statistical, standalone script, CLI,
  and import-isolation checks.
- Artifact evidence: `UV_NO_CACHE=1 uv build` successfully rebuilt the sdist
  and wheel from the sdist. The source archive contains `CMakeLists.txt`, the
  README, CLI, and all C++/header build inputs; the wheel contains the compiled
  `_rmsnorm_native` extension and console entry point.
- Packaging correction: the build initially warned that the source-only
  `smolvla_mlx.native` directory looked like an undeclared package. Setting
  `include_package_data=False` in `setup.py` retains those files in the sdist
  for compilation but omits them from the wheel. A second build has no
  setuptools package-discovery warning and retains the native extension.
- Wheel smoke evidence: the rebuilt wheel installed into a separate Python
  3.12 virtual environment using only the pinned core runtime. Outside the
  checkout, `smolvla-mlx --help` exposed all four commands;
  `smolvla_mlx._rmsnorm_native` loaded from `site-packages`; and the forbidden
  module set remained empty.
- Next: checkpoint this final packaging correction, rerun the full suite on
  the exact final tree, then update `PLAN.md` and `STATUS.md` to the v0.1
  Definition of Done.

## 2026-08-31 — v0.1 Definition of Done

- Final exact-tree verification: `make test` passed **179/179** in **147.26
  seconds** after the final `setup.py` artifact correction.
- Audit evidence: the tree was clean before the completion-document update;
  the runtime-source scan found no direct `torch`, `lerobot`, or `transformers`
  imports; the tracked-content scan found no token-shaped credentials; and no
  human task is open.
- Definition evidence: strict conversion, all fp32/bf16 module tests,
  deterministic action parity, the 50-frame statistical gate, real policy API,
  reproducible goldens, M5 Pro benchmark, source/wheel artifacts, clean source
  and wheel installs, and a real installed CLI dataset prediction are recorded
  above and in `BENCHMARK.md`/`STATUS.md`.
- Decision: mark `STATUS.md` with `DEFINITION OF DONE MET`. Future work is
  limited to the explicitly deferred v0.2/training, quantization, and robot-I/O
  scope in `BRIEF.md`.
