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

## 2026-08-31 — Agent handoff report

- What: added `HANDOFF_REPORT.md`, a self-contained record for a successor
  agent of the canonical workspace, final v0.1 evidence, pinned artifacts,
  runtime contract, safety boundaries, known Metal-parity caveat, and a safe
  first-turn checklist.
- Evidence: the report was based on the clean v0.1 completion commit
  `f1ae9d6`, `STATUS.md`, `BENCHMARK.md`, final `PROGRESS.md` entries,
  `ARCHITECTURE.md`, and the preserved failure analyses. It makes no code,
  model, cache, hardware, or package changes.
- Next: wait for an explicitly scoped follow-on request; do not start deferred
  Phase 7 robot integration, training parity, or quantization autonomously.

## 2026-08-31 — Full-scope kickoff, protected baseline, and origin

- Authority: the operator supplied `BRIEF_FULL.md`, explicitly expanded the
  scope to release, training, quality, and hardware-readiness documentation,
  and requested autonomous execution. The attached brief is now preserved at
  the repository root.
- Baseline evidence: `make test` passed **179/179** in **158.71 seconds** on
  Python 3.12.13 before continuation changes. Disk free was **553 GiB**, above
  the 40 GiB floor.
- GitHub evidence: `git ls-remote git@github.com:daniiarabdiev/smolvla_mlx.git`
  returned no refs. The remote was added as `origin`, and the existing verified
  `main` history through `458042b` was pushed without reinitializing the repo,
  replacing README, or force-pushing.
- Release-spec blocker: `BRIEF_RELEASE.md` is absent from the worktree, Git
  history, available attachments, Downloads, inspected unreachable trees, and
  the initially empty remote. `FAILURE_RELEASE_SPEC.md` records the searches;
  `HUMAN_TASKS.md` requests the exact file. The package gates are not guessed.
- Design decision: preserve v0.1 as an immutable inference baseline and add a
  thin, optional training compatibility layer under `training/`; detailed
  decomposition and the first T0 design are committed under
  `docs/superpowers/specs/`.
- Next: create the detailed test-first Stage T0 implementation plan, execute
  its differentiability/resource audit, then rerun the protected suite.

## 2026-08-31 — Stage T0 optional training package boundary

- Red evidence: the new distribution and import-boundary tests produced the
  intended **2 failures / 2 passes** because `training` did not exist and the
  `train` extra was absent.
- What: added a side-effect-free `training` package, declared an empty optional
  `train` extra, included the package in wheel/sdist metadata, and kept every
  reference framework out of unconditional dependencies and import state.
- Green evidence: with `HF_HOME`, `UV_CACHE_DIR`, and `SMOLVLA_MLX_CACHE`
  explicitly under this repository,
  `pytest tests/test_distribution.py tests/test_import_isolation.py -q`
  passed **4/4 in 0.62 seconds**; `uv lock --check` resolved all 103 packages
  in 3 ms.
- Cache correction: an earlier direct uv invocation inherited the user's
  pre-existing global uv cache and waited on its lock. Both agent-started
  processes were stopped before testing; the recorded green run used only the
  required repository-local caches.
- Next: add the pure-MLX differentiable RMSNorm and exact padded flow-matching
  objective through a new red/green cycle.

## 2026-08-31 — Stage T0 differentiable objective

- Red evidence: six focused tests failed because the training-only RMSNorm and
  flow-objective modules did not exist. The tests independently cover input and
  weight gradients, the literal flow interpolation/velocity target, exclusion
  of padded action dimensions, and invalid shape/action-width rejection.
- What: added a pure-MLX fp32 RMSNorm with autodiff support plus the exact
  `x_t = t * noise + (1 - t) * actions`, `u_t = noise - actions`, and physical-
  action-only MSE operations. The native inference compatibility extension is
  unchanged.
- Green evidence: repository-local-cache `pytest
  tests/test_training_objective.py -q` passed **6/6 in 0.02 seconds**. Both the
  input and RMSNorm-weight gradients were shape-matching and finite.
- Next: compose the existing full architecture behind a training container and
  prove its selected parameter tree is exactly `state_proj` plus the action
  expert.

## 2026-08-31 — Stage T0 full training-path composition

- Red evidence: four tests failed because the training component container,
  parameter selector, canonical name mapping, and deterministic audit batch did
  not exist.
- What: composed the existing 12-layer vision encoder, connector, 16-layer
  language prefix, state projection, and 16-layer expert under one optional MLX
  training module. The default selection freezes vision/connector/language and
  enables only `state_proj` plus the complete expert, matching the audited
  reference configuration.
- Behavior evidence: the deterministic batch is one observation with two
  `3×512×512` camera tensors, 48 language tokens, six state dimensions, and
  `50×32` action/noise tensors. A real bf16-storage full forward returned a
  finite positive scalar flow loss; no component was mocked.
- Green evidence: repository-local-cache `pytest tests/test_training_model.py
  tests/test_training_objective.py -q` passed **10/10 in 0.35 seconds**.
- Next: differentiate this full path, evaluate every selected gradient, and
  record latency, MLX memory, and disk measurements in the T0 audit result.

## 2026-08-31 — Stage T0 full differentiability and resource gate

- Red evidence: the integration test first failed because `training.audit` did
  not exist. The first complete implementation then exposed a reporting defect:
  MLX 0.32.2's namespace has no `__version__`; a focused regression test failed
  on that exact boundary before the implementation switched to installed
  distribution metadata.
- Gradient evidence: seed 0, bf16 parameter storage, microbatch 1, two
  `3×512×512` cameras, and a `50×32` action tensor produced loss
  **2.7361748218536377**. All **155/155** selected parameter tensors
  (**99,880,992 scalars**) had present, shape-matching, finite, nonzero
  gradients; maximum absolute gradient was **1.4765625**.
- Resource evidence: synchronized forward was **119.758 ms** and synchronized
  forward+backward was **181.849 ms**. Active MLX memory was
  **1,108,300,302 bytes** and peak was **2,510,577,166 bytes**. Disk free moved
  from **591,044,726,784** to **591,044,689,920 bytes**, remaining more than
  500 GiB above the 40 GiB gate.
- Green evidence: repository-local-cache `pytest tests/test_training_audit.py
  -q` passed **2/2 in 0.62 seconds**, including a second real full-architecture
  gradient step and the MLX-version regression test.
- Decision: the reference-default expert-plus-state training path is feasible
  on this M5 Pro. Native CPU compatibility calls remain inference-only; the
  training path uses differentiable MLX kernels.
- Next: add the standalone audit command and machine-readable artifact, write
  `TRAINING_FEASIBILITY.md`, then run the complete protected suite.

## 2026-08-31 — Stage T0 feasibility artifact and completion

- Artifact evidence: `make training-audit` wrote
  `.cache/training/t0-audit.json` with SHA-256
  `88dacde30996c2d9cbad90681204e2583c92f6d51f0f3747c6e37e57b709fd51`.
  The artifact independently recorded **155/155** finite, nonzero gradients,
  **99,880,992** trainable scalars, **196.799 ms** synchronized
  forward+backward, and **2,509,594,126 bytes** peak MLX memory.
- Focused evidence: all objective, model, audit, standalone-script, and runtime
  import-isolation cases passed **14/14 in 7.99 seconds**.
- Full regression evidence: `make test` passed **193/193 in 158.14 seconds** on
  the exact Stage T0 code and report tree. The original 179 inference/reference
  gates remain green; no tolerance, skip, mock, or `xfail` changed.
- Deliverable: `TRAINING_FEASIBILITY.md` records the native-RMSNorm exclusion,
  bridge-loader inventory, one-observation batching limitation, measured
  resource headroom, and exact T1 gradient-golden manifest design.
- Decision: Stage T0 is complete. `TRAINING ALPHA` is not claimed; that
  milestone requires the Stage T3 real LoRA outcome and round-trip gates.
- Next: begin Stage T1 with the actual pinned LeRobot training forward on one
  fixed public-dataset batch and fully serialized noise/timestep draws.

## 2026-08-31 — Stage T1 design audit

- What: traced the installed LeRobot 0.6.1 dataset, training loop, processor,
  SmolVLA forward, and trainable parameter tree before fixing the T1 design.
- Fixed case: public dataset episode 0, frame/absolute index 100, seed
  20260831. All 50 target actions are valid. The actual training path converts
  uint8 cameras to fp32/255, renames side/up to camera1/camera2, and injects the
  public dataset statistics into the checkpoint processor.
- Reference probe: CPU/fp32 loss 2.101923942565918 at sampled timestep
  0.8003060817718506; 155/155 gradients present, finite, and nonzero over
  99,880,992 scalars. Measured forward plus backward was 0.68445 seconds.
- Differentiable MLX CPU probe: loss 2.101925849914551, loss relative difference
  9.074298999059449e-07, worst gradient relative L2
  8.673578115837066e-06 (`action_time_mlp_in.weight`), and minimum cosine
  0.9999999999623879 across the strict 155-name canonical bijection.
- Decision: use a scoped training-only pure-MLX primitive adapter around the
  unchanged runtime modules. The probe clears the immutable T1 limits without a
  duplicate model or inference behavior change. The non-gating probe produced no
  repository artifact; official evidence will come from the manifest-backed T1
  scripts.
- Design: `docs/superpowers/specs/2026-08-31-gradient-parity-design.md`.
- Plan: `docs/superpowers/plans/2026-08-31-gradient-parity.md` fixes six
  independently testable and pushable packages from artifact IO through the
  official immutable gate and protected regression.
- Next: execute Package 1 with a failing manifest test, then progress through
  the plan without changing the pinned thresholds.

## 2026-08-31 — Stage T1 Package 1 artifact manifests

- Red evidence: `tests/test_training_data.py` first failed **8/8** because the
  planned `training.data` module did not exist.
- What: added framework-neutral atomic `.npy` artifact writing, sorted JSON
  manifests, per-payload SHA-256/shape/dtype/byte counts, manifest hashing in
  metadata, safe relative-name enforcement, complete reload verification, and
  explicit tamper detection. The module imports NumPy only; it does not import
  Torch, LeRobot, or Transformers.
- Green evidence: artifact and runtime import-isolation tests passed **9/9 in
  0.25 seconds**; `git diff --check` passed.
- Next: add red cases for temporal padding, physical action-width masking, and
  float64 gradient-comparison metrics.

## 2026-08-31 — Stage T1 Package 2 loss masks and metrics

- Red evidence: the new objective, batch, and comparison cases produced **11
  expected failures** while all 9 pre-existing focused cases stayed green.
- What: threaded `action_is_pad` through `TrainingBatch` and the full loss,
  cropped squared error to the physical action width, and matched LeRobot's
  valid-timestep × physical-dimension denominator with its clamp-to-one
  behavior. Added strict float64 relative-L2, cosine, maximum-difference, and
  relative-loss helpers with explicit invalid/zero-reference handling.
- Green evidence: objective, model, comparison, and real T0 audit regressions
  passed **23/23 in 2.87 seconds**; `git diff --check` passed.
- Next: replace the inference-only CPU primitive calls inside an exception-safe
  training scope and prove exact inference dispatch is restored afterward.

## 2026-08-31 — Stage T1 Package 3 scoped CPU autodiff

- Red evidence: all **4** new scoped primitive cases failed on the absent RoPE,
  softmax, SiLU, and context-manager APIs.
- What: added pure fp32 MLX VJPs for split-half RoPE, last-axis softmax, SiLU,
  and the existing RMSNorm, plus a locked CPU-only context that lazily redirects
  language/expert aliases and `ReferenceRMSNorm.__call__`. Nested activation is
  rejected and every callable is restored in reverse order after normal or
  exceptional exit.
- Protected evidence: scoped-autodiff, exact native RMSNorm/RoPE/softmax/SiLU,
  and import-isolation tests passed **12/12 in 2.22 seconds**; `git diff
  --check` passed. No `smolvla_mlx/` source changed.
- Next: reproduce the real LeRobot train-loop batch and serialize the actual
  Torch loss, draws, parameters, and all 155 gradients.

## 2026-08-31 — Stage T1 Package 4 reference gradient capture

- Red evidence: the real-batch contract initially produced **3 collection
  errors** because `training.reference` was absent, while the golden-integrity
  case failed because no artifact had been captured.
- What: mirrored LeRobot 0.6.1's actual uint8 collation, `/255` camera path,
  camera renaming, public-dataset statistics override, tokenization, padding,
  and trainable-parameter selection for episode/frame/absolute index
  **0/100/100**. Added an actual Torch CPU/fp32 forward/backward capture with
  seed **20260831**, exact serialized noise/timestep draws, flow boundaries,
  masked-loss reconstruction, and all canonical parameters and gradients.
- Artifact evidence: `.cache/training/gradient_goldens` contains **324**
  hash-verified tensors totaling **805,554,153 bytes**, including **155**
  trainable tensors and **99,880,992** trainable scalars. It records loss
  **2.101923942565918**, timestep **0.8003060817718506**, and dataset-statistics
  SHA-256 `4ae86bed785e0f98914812e87736e216139a22b43cf2b990e68384d85168c3c8`.
  Every captured gradient is finite and nonzero.
- Determinism evidence: two complete post-fix captures independently produced
  manifest SHA-256
  `b029a0ed66312e785cb8aa3f1db0affb16c9502ad7b5d0fe0feea3177bf8c145`.
- Debugging evidence: full artifact verification exposed NumPy's promotion of
  zero-dimensional arrays through `ascontiguousarray`. A focused regression
  first reproduced the incorrect `(1,)` shape; the artifact boundary now
  preserves true scalars without changing non-scalar contiguity.
- Green evidence: real preparation, complete artifact integrity, artifact IO,
  scalar preservation, and runtime import isolation passed **14/14 in 8.14
  seconds**. `git diff --check` passed. Disk free remained above **545 GiB**,
  far above the 40 GiB gate.
- Next: load the converted checkpoint through `SmolVLATrainingModel`, consume
  the serialized batch and draws under MLX CPU autodiff, and compare all 155
  gradients against the immutable T1 limits.

## 2026-08-31 — Stage T1 Package 5 checkpoint-backed parity gate

- Red evidence: the four new checkpoint/batch/threshold/gate cases failed
  **4/4** because `training.parity` and its required APIs did not exist.
- Structural gate: the strict native fp32 loader matched every one of the
  **155** selected MLX parameter tensors exactly to the serialized Torch
  values before differentiation; names, shapes, dtypes, and values are hard
  failures rather than tolerance checks.
- Numerical gate: reference loss **2.101923942565918** and MLX loss
  **2.101925849914551** differ by **9.074298999059449e-7**, below the immutable
  `1e-4` limit. All **155/155** gradients pass both limits: worst relative L2
  **8.673578115837066e-6** (`action_time_mlp_in.weight`) versus `1e-2`, and
  minimum cosine **0.9999999999623879** versus `0.999`.
- Artifact evidence: `make training-parity` wrote the complete 155-comparison
  report to `.cache/training/t1-parity.json` with SHA-256
  `f4da0c16771a462e45bd615728bc02a059633db19eb77883342203426cb4d634`.
  It binds golden manifest SHA-256
  `b029a0ed66312e785cb8aa3f1db0affb16c9502ad7b5d0fe0feea3177bf8c145`.
- Resource evidence: the final official MLX forward/backward took **1.161
  seconds**, total validation took **2.169 seconds**, peak MLX memory was
  **2,173,315,618 bytes**, and disk free remained above **542 GiB**.
- Green evidence: the complete checkpoint-backed parity plus protected
  training objective, gradient metrics, differentiable primitives, model, and
  import-isolation set passed **29/29 in 3.64 seconds**; `git diff --check`
  passed.
- Next: commit the executable gate, then write the T1 evidence report, update
  milestone state, and run the complete protected repository suite.

## 2026-08-31 — Stage T1 completion and protected regression

- Deliverable: `GRADIENT_PARITY.md` binds the pinned checkpoint, VLM, dataset,
  exact real batch, serialized draws, structural identity gate, immutable
  numerical gates, resource measurements, and reproduction commands.
- Worst-five relative-L2 evidence, reported regardless of pass:
  `action_time_mlp_in.weight` (**8.673578115837066e-6**, cosine
  **0.9999999999623879**),
  `expert.layers.1.input_layernorm.weight` (**6.656345209857855e-6**,
  **0.9999999999789950**),
  `expert.layers.1.self_attn.q_proj.weight` (**6.3123810635703525e-6**,
  **0.9999999999808424**),
  `expert.layers.9.input_layernorm.weight` (**5.851465222097282e-6**,
  **0.9999999999864958**), and
  `expert.layers.1.self_attn.k_proj.weight` (**5.559739433938906e-6**,
  **0.9999999999855081**).
- Final artifact evidence: `make training-parity` again passed **155/155** and
  wrote report SHA-256
  `f4da0c16771a462e45bd615728bc02a059633db19eb77883342203426cb4d634`;
  its embedded golden-manifest SHA matches the independently computed
  `b029a0ed66312e785cb8aa3f1db0affb16c9502ad7b5d0fe0feea3177bf8c145`.
- Full regression evidence: the exact post-T1 tree passed **224/224 tests in
  172.58 seconds**. This includes the complete v0.1 inference/reference ladder,
  real training artifacts, T0 audit, T1 identity/parity, and base import
  isolation.
- Closure evidence: `uv lock --check` resolved **103 packages**; `git diff
  --check` passed; no training test contains a skip, `xfail`, or mock escape
  hatch; **543 GiB** remains free; and local `HEAD` matched
  `refs/remotes/origin/main` before the documentation-only completion commit.
- Decision: Stage T1 is complete. Stage T2 optimizer lockstep and Stage T3 LoRA
  fine-tuning are both ready. `TRAINING ALPHA` remains reserved for T3's three
  outcome and round-trip gates.
- Next: audit the installed reference AdamW and cosine-with-warmup semantics,
  then write and execute the Stage T2 25-step lockstep design.

## 2026-08-31 — Stage T2 optimizer/scheduler design audit

- Upstream audit: the pinned SmolVLA config selects PyTorch AdamW with LR
  `1e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, decoupled weight decay `1e-10`,
  and global gradient-norm clipping at `10`. The loop orders backward → clip →
  optimizer step → zero-grad → scheduler step.
- Scheduler audit: LeRobot uses 1,000 warmup steps, 30,000 cosine-decay steps,
  and a `2.5e-6` floor. T2 freezes the first 25 updates of the default
  100,000-step configuration, giving LR **9.990009990009991e-8** at step 0 and
  **2.4975024975025017e-6** at step 24. A 25-step scheduler horizon was rejected
  because LeRobot's short-run scaling truncates warmup to zero and tests a
  different edge-case schedule.
- Clipping probe: the serialized T1 gradient tree has PyTorch global norm
  **43.20928955078125**, so the configured clip is materially active.
- Data decision: repeat the exact non-padded T1 real batch for the 25-step
  optimizer isolation test, while serializing a distinct sequential noise and
  timestep draw for every update.
- Design: `docs/superpowers/specs/2026-08-31-optimizer-lockstep-design.md`.
- Plan: `docs/superpowers/plans/2026-08-31-optimizer-lockstep.md` defines four
  test-first packages from scalar semantics through full protected closure.
- Next: write failing cross-framework scheduler, clipping, and AdamW tests,
  then implement the Torch-free MLX optimizer layer.

## 2026-08-31 — Stage T2 Package 1 optimizer semantics

- Red evidence: all **5** schedule/default/clipping/one-step/25-step cases failed
  because the planned `training.optimizer` module did not exist.
- What: added a Torch-free immutable SmolVLA optimizer config, exact LeRobot
  cosine-with-warmup LR function including its short-horizon branch,
  PyTorch-order multi-tensor fp32 global-norm clipping, and an MLX AdamW wrapper
  with decoupled pre-update decay and enabled first/second-moment bias
  correction.
- Cross-framework evidence: all first 25 default-run LR values match the
  installed LeRobot scheduler at zero tolerance; clipping matches
  `torch.nn.utils.clip_grad_norm_` on an active multi-tensor case; and both one
  and 25 updates match `torch.optim.AdamW` under deliberately enlarged decay,
  epsilon, and altered betas that make ordering errors observable.
- Green evidence: optimizer semantics plus protected objective, gradient,
  training-model, and base import-isolation cases passed **26/26 in 0.40
  seconds**; `git diff --check` passed.
- Next: capture the actual 25-step PyTorch checkpoint evolution with the T1
  batch, 25 serialized draw pairs, per-step optimizer metrics, and all 155
  final parameters.

## 2026-09-01 — Stage T2 Package 2 reference optimizer capture

- Red evidence: the four fixed-window/artifact/step/final-parameter cases first
  failed on the absent reference module and artifact.
- What: added a real PyTorch capture that proves the prepared batch and all 155
  initial selected parameters exactly equal T1, builds the checkpoint's own
  AdamW and scheduler presets, seeds once, and performs 25 actual
  forward/backward/clip/update/zero/schedule steps. It serializes 25 draw pairs,
  125 scalar step metrics, and all 155 final fp32 parameters.
- Debugging evidence: the first artifact showed that stochastic step 23 has
  gradient norm **6.328668594360352**, below the clip limit, while step 0 is
  **43.20928955078125**. The over-constrained test was corrected to verify the
  exact coefficient at every step and require clipping to be exercised; 24/25
  steps clip and one correctly uses coefficient 1.0. A tensor-detach warning
  was also removed at its scalar-conversion boundary.
- Artifact evidence: `.cache/training/optimizer_goldens` contains **330**
  payloads and is **399,852,897 bytes**. Reference loss moves from
  **2.101923942565918** to **0.5079650282859802** over the fixed-batch window.
  Two complete captures produced identical manifest SHA-256
  `88c3febc7da3e553bcb7c26f261721369ed1f56efd457887b7d43d50a077807c`,
  bound to T1 manifest
  `b029a0ed66312e785cb8aa3f1db0affb16c9502ad7b5d0fe0feea3177bf8c145`.
- Green evidence: the reference optimizer artifact contract passed **4/4 in
  0.34 seconds**; the wider reference/artifact/import-isolation set passed
  **18/18 in 6.78 seconds**; `git diff --check` passed. The final capture took
  **23.311 seconds**, with more than **542 GiB** disk free.
- Next: execute the same 25 draws and update order in MLX, compare every loss
  and all 155 final parameters, and emit the immutable lockstep report.

## 2026-09-01 — Stage T2 Package 3 native optimizer lockstep

- Red evidence: all **3** threshold/link/full-gate cases failed because
  `training.lockstep` did not exist.
- What: added strict T1/T2 artifact linkage, exact per-step draw loading, native
  MLX CPU/fp32 value/grad/global-clip/AdamW updates, all 25 loss comparisons,
  all 155 final-parameter comparisons, worst-five views, and an atomic
  standalone report. Names, pins, counts, initial values, dtypes, shapes,
  devices, and learning rates are hard pre-threshold gates.
- Immutable result: all **25/25** loss comparisons pass; the maximum relative
  difference is **1.3529624562582406e-6** at step 7 versus `1e-3`. All
  **155/155** final tensors pass; maximum relative-L2 drift is
  **2.8499913470883435e-8**
  (`expert.layers.1.self_attn.q_proj.weight`) versus `5e-3`.
- Supporting optimizer evidence: maximum gradient-norm relative difference is
  **4.9728077828962536e-5** and maximum clip-coefficient absolute difference is
  **3.2007518114496314e-5** across the 25 real full-model steps.
- Artifact evidence: `make optimizer-lockstep` wrote
  `.cache/training/t2-lockstep.json` with SHA-256
  `99076dabe396ef65086d7cae7e2edaa1ab9b7b16c356a3e9a49c49eec8312eae`,
  binding the expected T1 and optimizer manifests.
- Resource evidence: synchronized updates took **32.999 seconds**, total strict
  validation took **33.982 seconds**, peak MLX memory was **3,373,751,277
  bytes**, and disk free remained above **542 GiB**.
- Green evidence: the complete focused T2/T1 optimizer, artifact, parity,
  differentiable-primitive, and import-isolation set passed **21/21 in 37.11
  seconds**; `git diff --check` passed.
- Next: write `OPTIMIZER_LOCKSTEP.md`, update stage state, rerun the standalone
  report and full 236-test protected suite, then close and push T2.

## 2026-09-01 — Stage T2 completion and protected regression

- Deliverable: `OPTIMIZER_LOCKSTEP.md` records every audited AdamW, clipping,
  schedule, and update-order semantic; all 25 reference/MLX loss pairs; the
  five worst loss steps and final tensors; artifact links; reproduction
  commands; and measured resource use.
- Final gate evidence: a fresh `make optimizer-lockstep` passed **25/25** loss
  and **155/155** final-parameter gates. Worst loss steps are 7
  (**1.3529624562582406e-6**), 18 (**1.282254892855697e-6**), 9
  (**1.1715480594856972e-6**), 14 (**1.0663780020910487e-6**), and 24
  (**1.0560605024102234e-6**), all below `1e-3`.
- Worst final tensors are `expert.layers.1.self_attn.q_proj.weight`
  (**2.8499913470883435e-8**),
  `expert.layers.1.self_attn.o_proj.weight` (**1.939221166057099e-8**),
  `expert.layers.5.self_attn.o_proj.weight` (**1.796292933841857e-8**),
  `expert.layers.3.self_attn.q_proj.weight` (**1.785367358266342e-8**), and
  `expert.layers.9.self_attn.o_proj.weight` (**1.769763506243787e-8**), all
  below `5e-3`.
- Hash evidence: the final report SHA-256 is
  `da8cabf5eecf4379065771b3a74407c47290b8aee9c2d0a9756893b6dd87a6a4`;
  its embedded T1 and optimizer hashes match independently computed manifests
  `b029a0ed66312e785cb8aa3f1db0affb16c9502ad7b5d0fe0feea3177bf8c145`
  and `88c3febc7da3e553bcb7c26f261721369ed1f56efd457887b7d43d50a077807c`.
- Full regression evidence: the exact post-T2 tree passed **236/236 tests in
  189.07 seconds**, including the entire inference/reference ladder, T0, T1,
  both real artifacts, and the cumulative T2 optimizer gate.
- Closure evidence: `uv lock --check` resolved **103 packages**; `git diff
  --check` passed; no training test contains a skip, `xfail`, or mock escape;
  **542 GiB** remains free; caches remain repository-local; and local `HEAD`
  matched `refs/remotes/origin/main` before the documentation-only completion
  commit.
- Decision: Stage T2 is complete without failure documentation. T3 remains
  ready on its T1 dependency, now with both step-zero and optimizer evolution
  correctness evidence. `TRAINING ALPHA` is still reserved for T3's held-out
  improvement, export round-trip, and unchanged inference-parity gates.
- Next: audit LoRA insertion/export and whole-episode data split mechanics,
  benchmark a real Metal training step, then freeze the Stage T3 design and
  time-bounded run plan.

## 2026-09-01 — Stage T3 Packages 1-3 LoRA, data, and export boundaries

- Design: froze rank 8 / alpha 16 / dropout 0 adapters on exactly **229**
  linears: 112 used VLM attention/MLP projections, 116 expert projections, and
  the state projection. Only **458** fp32 adapter tensors are trainable over a
  frozen bf16 base. Install, gradients, and merge are count-checked.
- Gradient evidence: all adapter gradients are finite on the full native loss.
  Zero-B initialization makes all A gradients zero on update zero. Exactly five
  terminal VLM-layer B gradients are also structurally zero because SmolVLA
  consumes layer 15 K/V but not its final query/output/MLP hidden result; all
  other **224/229** B tensors receive signal.
- Data: seed `20260901` fixes held-out episodes
  `(2, 7, 21, 28, 31, 34, 35, 41)`, exactly **8/50 = 16%**. The remaining 42
  episodes contribute **10,011** train rows; 1,928 held-out rows contribute
  nothing to state/action statistics. Train-stat SHA-256 is
  `5aa5ab85e0c71c0adee97782be37907b0918050a8539bb3aab88fe392953948e`.
- Bridge: LeRobot's pinned delta timestamps, episode-aware sampler, collator,
  processors, video backend, and tokenizer produce owned NumPy microbatches.
  The fixed episode/frame `0/100` case matches every T1 model-ready tensor
  byte-for-byte; independent bridges reproduce the same shuffled identities.
- Export: merged native tensors reverse-map into all **500** standard LeRobot
  fp32 tensors (**450,046,176** scalars), reverse the patch-convolution layout,
  and save native processor files with train-only stats. A real temporary export
  loaded strictly in both MLX and Torch/LeRobot.
- Green evidence: Package 1's LoRA plus protected T1/T2 focus passed **40/40 in
  37.84 seconds**; Package 2's data/reference focus passed **19/19 in 6.88
  seconds**; Package 3's real export/conversion/inference focus passed **23/23
  in 14.50 seconds**. No base-package source changed.

## 2026-09-01 — Stage T3 Package 4 measured Metal trainer

- Trainer: added exact fp32 averaging over eight distinct one-sample
  microbatches, one global clip and AdamW update per effective batch, native
  Gaussian/Beta flow draws, durable per-update CSV metrics, atomic run state,
  local adapter checkpoint, fp32 merge, and standard export.
- Red/green evidence: all five initial contracts failed on the absent module;
  the implementation then passed **5/5 in 0.10 seconds**.
- Required real benchmark: three warm-up plus ten measured Metal updates at
  effective batch 8 had median **1.6376500835176557 seconds/update** (range
  **1.6308664999669418–1.69344437494874**), with peak MLX memory
  **2,478,791,461 bytes**.
- Frozen budget: `min(3000, floor(6900 / 1.6376500835176557))` is **3,000**,
  so no reduction is required. Estimated training time is **4,912.95 seconds
  (81.88 minutes)**, leaving approximately five minutes inside the two-hour
  envelope for export/evaluation. Benchmark report SHA-256 is
  `3598214cecd083cd3d5d143edd3edbe614dc899d30622152573e87dd104fe442`.
- Resource evidence: more than **539 GiB** remains free, above the 40 GiB
  floor. The run budget is now immutable; the held-out outcome cannot extend it.
- Next: execute the fresh 3,000-update run, export, and apply all three frozen
  held-out/round-trip/stats-active parity gates.

## 2026-09-01 — Stage T3 held-out evidence frozen before training

- Evaluation population: captured **56** cases, exactly seven spread through
  each of the eight unseen episodes. Every case includes raw uint8 camera
  frames, raw six-axis state, the physical first-action target, and one fixed
  `1x50x32` Gaussian flow draw from seed `20260902`.
- Integrity: the **280-array**, 100 MiB local artifact has manifest SHA-256
  `9cabca6cd21e8658a94e42980af3e91ecd8ff5ed5daca5f75eb7a1ebd1d261a3`
  and remains bound to train-stat hash
  `5aa5ab85e0c71c0adee97782be37907b0918050a8539bb3aab88fe392953948e`.
- Frozen baseline: fp32 MLX on the CPU compatibility lane, with those train-only
  stats active, scored physical first-action MAE **4.639846293521779** over
  **336** action elements. Its report SHA-256 is
  `211d6778b0530208ca2e81abe6f4002cc683e24d496a09ddbe39c100ebd4f7ce`.
- Immutable improvement target: the final MLX export must score at most
  **4.175861664169601** (`0.9 *` the unrounded base MAE). Case selection,
  noise, target, statistics, and baseline are now fixed before update one.
- Green evidence: all **5/5** evaluation artifact, metric, threshold-boundary,
  and baseline-binding cases passed in **0.17 seconds**.
- Next: run exactly 3,000 effective-batch updates from a fresh base; no result
  may change this population, baseline, budget, or threshold.
