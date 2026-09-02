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

## 2026-09-01 — Stage T3 interruption recovery and exact checkpoints

- Interruption evidence: the first foreground training process ended after
  update **1,375/3,000** before a final-only adapter save. Its complete metrics
  trace is preserved at
  `.cache/training/t3-interrupted-20260901-step1375/metrics.csv`, SHA-256
  `a41c8bf259e98c0ad909bf54398d4da8c3d073e056f4fea9ee92d0ad807dbca1`.
  A deterministic replay was deliberately stopped at update **210** to avoid
  repeating that risk; its metrics SHA-256 is
  `b75dba66a4e6a0f8731b69da86a6d925086ebee9dcf8bd78a0b2e4b90c4eea4e`.
- Red evidence: **5** new recovery contracts initially failed because metrics
  could not resume, flow RNG could not be advanced, model/optimizer
  checkpoints did not exist, cadence was absent, and the data bridge exposed
  no sampler state. A sixth retention contract then failed on the absent
  last-three policy.
- Implementation: checkpoints are atomically published after update 1, every
  100 updates, and the final update. They bind all adapter and AdamW tensors,
  schedule step, smoothed loss, elapsed time, peak memory, sampler offset,
  flow-draw count, and a SHA-256 of every trajectory-affecting setting. Resume
  validates file hashes and tensor contracts, restores the sampler directly,
  advances MLX RNG exactly, and preserves any post-checkpoint CSV tail in a
  recovery file before continuing. Only the three newest complete checkpoints
  are retained.
- Exact real-model proof: checkpointing after update 1 and rebuilding the full
  **458-trainable-tensor** LoRA model reproduced update 2 with identical loss
  **0.980276882648468**, identical gradient norm **2.3309686183929443**, and
  byte-identical final trainable digest
  `3e92cc1a6b52e1036248475280ec571ac4227e4355b753798ee39479ad2a8f36`.
- Green evidence: checkpoint, metrics, RNG, sampler, optimizer, LoRA, export,
  evaluation, and import-isolation focus passed **41/41 in 12.04 seconds**;
  the tighter checkpoint/data/optimizer set passed **22/22 in 0.84 seconds**;
  `git diff --check` passed. Disk free is **587 GiB**.
- Next: commit/push the resilience layer, launch the unchanged frozen
  3,000-update run, verify checkpoint publication at steps 1 and 100, and
  continue through export and all three outcome gates.

## 2026-09-01 — Stage T3 checkpoint crash-window hardening

- Review-driven red evidence: a valid checkpoint from a different run could
  count toward retention and displace one of the active run's last three; a
  finite but incorrect boundary `updates_per_second` value was also accepted.
  Both new regression tests failed before their fixes.
- Publication recovery: resume discovers the newest fully valid step directory
  even when `latest.json` is stale or absent, repairs the pointer, and validates
  config identity, file hashes, tensor names, shapes, dtypes, and optimizer
  step before mutating live state. A complete export published before a crash
  is checksum/inventory/metadata validated and reused.
- Retention and identity: pruning now considers only checkpoints matching the
  active run-config hash and current model/optimizer schemas. Valid other-run,
  invalid, partial, symlinked, temporary, and unrelated paths remain untouched.
  The run-config digest binds the exact converted base and name-map bytes plus
  the complete effective optimizer configuration.
- Metrics durability: the raw CSV is copied and fsynced before parsing; only
  the checkpointed prefix must be complete; the active file is atomically
  rewound; and every checkpoint boundary field, including derived throughput,
  must match before append resumes.
- Post-hardening real-model proof: saving update 1 and rebuilding/restoring the
  full **458-tensor** LoRA model reproduced update 2 exactly: loss
  **1.256011962890625**, gradient norm **1.3243852853775024**, model digest
  `9d5feac138050a282a0c1a5014986ff337f5242680ec1ad3bc5448dd8389ad2d`,
  and optimizer digest
  `5ff37c17a0e887aabf1e8f5f0e38d01d4d33fa5b2b7716450170674915111266`.
- Green evidence: checkpoint/data/optimizer/export/import-isolation focus passed
  **36/36 in 11.31 seconds**; the complete repository passed **278/278 in
  215.81 seconds**; `uv lock --check` resolved **103** packages; `git diff
  --check` passed; the two interrupted metrics traces retain their recorded
  SHA-256 values; `.cache/training/t3` remains absent; and **580 GiB** is free.
- Independent review: the final verdict was **Ready: Yes**, with no critical,
  important, or minor findings. The reviewer independently passed **36/36**
  targeted, **48/48** broader, and **278/278** full-suite cases on this tree.
- Next: commit/push this resilience package, then launch the frozen run and
  inspect checkpoints 1 and 100.

## 2026-09-01 — Stage T3 fixed 3,000-update run and exact finalization recovery

- Run: completed all **3,000** effective-batch-8 Metal updates in
  **4,956.693033957996 seconds**. Final loss is **0.15556271374225616**, final
  smoothed loss is **0.15856251862136114**, and peak MLX memory is
  **2,478,803,693 bytes**.
- Checkpoints: step 1 and every 100-step boundary were published; retention
  leaves complete steps **2,800, 2,900, and 3,000**. The step-3,000 model hash
  is `814e6f4b2a78a46b609aa7b48a28b4509f709d3e851e588dcd9a4bd2ca1408dc`
  and optimizer hash is
  `c9440be75315e04c1812ba18da0e0daccd2990fb0f6fdb1841d40ef7b01ffb5a`.
- Finalization recovery: the first final adapter save failed after training
  because MLX appended `.safetensors` to a temporary path ending in `.tmp`.
  The complete step-3,000 checkpoint protected the run. A regression test
  reproduced the suffix behavior, and the temporary path now ends in
  `.adapter.tmp.safetensors` before atomic replacement.
- Exact resume evidence: resume restored completed step 3,000, replayed no
  optimizer update, and exported successfully. The regenerated adapter is
  byte-identical to the pre-failure archived adapter, SHA-256
  `814e6f4b2a78a46b609aa7b48a28b4509f709d3e851e588dcd9a4bd2ca1408dc`.
  The run, metrics, and export-manifest SHA-256 values are respectively
  `c7c3b86361c0872e26f2088cbd33ada865cf450b6711a9b737ece933c1868c82`,
  `7f3a8c070f8102d7edc0afe5a9f4e5088321d1cdd21548fc21e9c772dbfafc2c`,
  and `70e8e6d33a52356ae873942a9569f619a9831e8ee3a90d0fcc594a3b6d17b0bb`.
- Export: all **500** tensors and **450,046,176** parameters are fp32 and load
  strictly in MLX and Torch/LeRobot. Reconstructing the adapter merge on the
  MLX GPU reproduced every exported source-layout tensor byte-for-byte:
  **0** mismatches and maximum absolute difference **0.0**.
- Next: apply all three immutable outcome gates to the frozen 56-case
  population without changing the run or selecting a checkpoint by result.

## 2026-09-01 — Stage T3 outcome failure preserved after three-hypothesis diagnosis

- Outcome report: `.cache/training/t3-outcome.json`, SHA-256
  `69c60ffd139543aa42258a7fdd86f9c0c71cbe700f9e9756d455917b98823bb5`,
  binds the frozen population, train statistics, adapter, run, metrics, and
  export manifest by SHA-256.
- Improvement passed: base MLX physical first-action MAE
  **4.639846293521779** fell to **2.1164464077779224**, ratio
  **0.4561458017979897** versus the immutable maximum **0.9**.
- Round trip passed: the strict Torch export scored MAE
  **2.101113587617874**, ratio **0.9927553940871358** to MLX versus required
  `[0.95, 1.05]`.
- Parity failed: preprocessing max absolute is only
  **3.5762786865234375e-7**, but normalized action-chunk max absolute is
  **0.0915878415107727** and standardized physical max absolute is
  **0.09158781915903091**, both above the unchanged **0.005** gate. The worst
  fixed case is episode 31, frame 0.
- Hypothesis 1 ruled out data/stat/preprocessing identity: tokens and masks are
  exact; forcing exact Torch preprocessed or connector arrays leaves the worst
  final difference at about **0.0916**. The original fp32 base passes the same
  stats-active evaluator at **8.996715223474894e-6**.
- Hypothesis 2 ruled out adapter merge/name/layout: all 500 GPU-merged tensors
  match the export byte-for-byte, and both framework loaders consume it
  strictly.
- Hypothesis 3 found no localized implementation defect: injecting the exact
  Torch prefix K/V reduces the final error to **0.008952289819717407**.
  Teacher-forced layer traces show small distributed framework reduction-order
  drift; the last two Euler steps amplify velocity differences from
  **0.2628980875015259** and **0.7516704201698303** into the final
  **0.0915878415107727** action difference.
- Decision: preserve the failed gate and write `FAILURE_LORA_FINETUNE.md`.
  `TRAINING ALPHA` is not written. T4 and T5 are dependency-blocked on T3; R,
  Q, and H remain blocked on the absent normative release brief. The remaining
  work has reached the full-scope stop condition after final verification,
  commit, and push.
- Fresh regression evidence: the outcome-evaluator and failure-closure tree
  passed **281/281 tests in 207.42 seconds** with zero failures, skips, or
  xfails reported. Current disk free is **571 GiB**.
- Fresh immutable-gate evidence: `make lora-finetune-check` re-evaluated all
  56 MLX cases, all 56 strict Torch cases, and the eight-episode parity set.
  It reproduced report SHA-256
  `69c60ffd139543aa42258a7fdd86f9c0c71cbe700f9e9756d455917b98823bb5`
  and exited nonzero as required because only the parity gate is false.

## 2026-09-01 — Stage T3 outcome-evidence hardening and full-manifest rerun

- Independent review demonstrated four integrity gaps in the first evaluator:
  case task/identity metadata was outside the recorded tensor-manifest hash;
  a modified baseline could be accepted with its new hash; parity selected one
  post-run case per episode and excluded raw physical P2 from the gate; and CSV
  plus symlink validation was structural rather than complete. Each fix began
  with a regression that failed on the prior implementation.
- The tensor manifest `9cabca6c...`, baseline `211d6778...`, and train-stat
  `5aa5ab85...` identities were frozen before update one. The full metadata
  digest `f49ee54a...` was first recorded during this post-run hardening pass;
  it is therefore treated as a retrospective file-integrity check, not a
  prospective commitment. Every metadata field is independently reconstructed
  from the precommitted selection algorithm and pinned dataset revision before
  acceptance; consumed dataset files must match the audited HF revision tree.
  Base, fine-MLX, Torch, and parity records must contain all 56 identities in
  order with internally consistent error sums and counts. Symlinked ancestors,
  nested dataset symlinks, and paths outside this repository's `.cache` are
  rejected.
- Training evidence validation now recomputes the run-configuration digest from
  every trajectory-affecting run field, reads metrics from one immutable byte
  snapshot, and verifies every row's frozen learning-rate schedule, nonnegative
  domains, smoothed-loss recurrence, derived throughput, and monotonic
  elapsed/peak values. The final row, optimizer LR, gradient norm, clip
  coefficient, and all 24,000 sample/draw counters are reconciled to the
  step-3,000 checkpoint. Adapter metadata, complete model/optimizer checkpoint,
  and processor statistics recomputed from the 42 frozen training episodes are
  also hash-validated. Evaluation now enforces the 40 GiB free-space floor
  before loading models.
- The export audit manifest now carries the run-config, evaluation tensor and
  metadata, frozen baseline, dataset revision, split, train-stat, and adapter
  hashes. Updating that JSON-only audit manifest changed its SHA-256 to
  `55ad6834cbb3acb9dd565a57296a274d78e7cdc863aa81c3e6ef25da8b66ba03`;
  all model, processor, adapter, optimizer, and checkpoint bytes remained
  unchanged.
- A final independent review found that MLX's path-keyed conversion cache could
  otherwise be stale while preserving valid tensor names. The outcome evaluator
  now validates all **500** cached native tensors against the hash-validated
  export before each MLX scoring pass: **499** fp32 tensors match by raw tensor
  checksum and the vision patch convolution matches the exact OIHW-to-OHWI
  transpose. The converted model and canonical name-map hashes are recorded in
  the outcome source chain. The generic BF16 validator also derives every
  rounded target value from the fp32 source and rejects a tampered cache even
  when its name-map checksum is rewritten.
- The fresh real gate evaluated 56 MLX MAE cases, the exact 56 Torch cases, and
  all 56 stats-active parity cases. Improvement remains **0.4561458017979897**
  and Torch/MLX remains **0.9927553940871358**, both passing. Parity remains a
  failure: image preprocessing max is **3.5762786865234375e-7** against
  **1e-5**, state preprocessing is **0.0** against **1e-6**, normalized max is
  **0.17762404680252075**, raw physical max is **6.632053375244141**, and
  standardized physical max is **0.17762437462806702** against the unchanged
  **0.005** action-parity gate.
- The new complete outcome report is
  `.cache/training/t3-outcome.json`, SHA-256
  `8b74faf8f9cc96341090f91cfa795ed874c838026416944e4b77a550ad91bc44`.
  It binds 15 exact source digests, including the dataset-revision tree, native
  conversion/model map, and final checkpoint metadata/model/optimizer. `make
  lora-finetune-check` exits nonzero only because the immutable parity gate is
  false; no threshold or population was changed to obtain a pass.
- Fresh full regression: **308/308 tests passed in 205.10 seconds**. The
  standalone runtime import-isolation proof passed **1/1**, `uv lock --check`
  resolved **103** packages, `git diff --check` passed, **570 GiB** is free,
  and local `HEAD` still matches `origin/main` at the prior durable checkpoint
  `f6099583d2c3538b52c520dd110b02a786834299` before this package is committed.

## 2026-09-01 — T3B and release-scope kickoff

- Specification checkpoint: the operator-supplied normative
  `BRIEF_RELEASE.md` and amendment `BRIEF_T3B.md` were committed before they
  were read as commit `ab14dbe` (`phase-11: add release and T3B
  specifications`).
- Required-read evidence: `AGENTS.md`, `BRIEF.md`, `BRIEF_FULL.md`,
  `BRIEF_RELEASE.md`, `BRIEF_T3B.md`, `STATUS_FULL.md`,
  `FAILURE_LORA_FINETUNE.md`, `FAILURE_RELEASE_SPEC.md`, `PLAN.md`,
  `PLAN_FULL.md`, the recent progress entries, and `HUMAN_TASKS.md` were read
  completely before execution planning.
- Blocker resolution: `FAILURE_RELEASE_SPEC.md` now preserves the historical
  failure while marking it resolved by operator supply; the corresponding
  `HUMAN_TASKS.md` entry is closed. Stage R, Q, and H are no longer blocked by
  an absent specification.
- Protected baseline: `make test` collected and passed **308/308 tests in
  226.41 seconds** (`real 231.59`, `user 196.93`, `sys 35.01`) with no T3B
  training or self-consistency-floor process running. Disk free is **570
  GiB**, above the mandatory 40 GiB floor.
- Execution plan: `PLAN_T3B.md` fixes the mandated T3B → Stage R → T4 → T5 →
  Stage Q → documentation-only Stage H sequence, package checkpoints, idle
  timing guards, and final verification. Next is T3B-1 test-first, without
  editing or reinterpreting the original T3 failure record.

## 2026-09-01 — T3B-1 failed-checkpoint PyTorch self-consistency floor

- Safety/order: no training or floor process was active at launch, **572 GiB**
  was free, and no benchmark or timing measurement ran during the floor. The
  original `FAILURE_LORA_FINETUNE.md` and every original tolerance remained
  byte-for-byte unchanged.
- Procedure: the prospectively frozen v3 plan launched nine fresh processes on
  the retained merged fp32 T3 export and exact 56 frozen cases/stored
  `1×50×32` noise: CPU fp32 baseline (default 6 threads), CPU fp32 at 1 and 18
  threads, five fixed MPS fp32 fallback slots, and CPU float64. Each complete
  `56×50×6` normalized action result was persisted and independently hashed
  before aggregation. All documented MPS and CPU thread-control variables were
  cleared before NumPy/Torch import; only fallback was enabled for MPS. Every
  worker used seed `20260901`, ordinary nondeterministic-algorithm mode, and
  `highest` float32 matmul precision.
- Result: single-thread CPU max-abs was
  **0.000024199485778808594**; 18-thread CPU was
  **0.000023752450942993164**; each of the five MPS fallback slots was
  **0.000022813677787780762**; float64 was
  **0.00003549918286283038**. Therefore the Section 1 envelope is
  **F = 0.00003549918286283038**, and the separately reported component is
  **F64 = 0.00003549918286283038**. The original MLX-versus-baseline value
  **0.17762404680252075** is present only as context; no new verdict was made.
- Evidence: `.cache/training/t3/floor.json` was atomically created at
  `2026-09-01T13:40:46.946734+00:00` with embedded timestamp
  `1788270046946734000` ns and later file mtime `1788270046951823007` ns.
  Its SHA-256 is
  `cba4a856f9c907d986ffc8703789673611e54bad983c2afd0a987830466f0585`.
  The combined input SHA-256 is
  `d31a0835867116d7bfbe63f6cd23666eecfdc0a660aba620730cd09320295299`;
  it binds 7 export files, 282 evaluation files, 5 pinned-dataset validation
  inputs, 10 tokenizer files, and 33 direct implementation, runtime-package,
  distribution-record, and lock inputs. The implementation tree SHA-256 is
  `1b73640f406b751cfd8bccb1061673d4f6888dbec69cb9ce7d8444175cceaad6`.
- Review hardening: a read-only checkpoint review identified incomplete
  runtime/provenance validation, a trusted group-tree digest, lexical-only
  cache containment, output/input overlap risk, and missing restart coverage.
  Regression tests first reproduced those gaps. The validator now reconstructs
  every group tree and variant artifact, enforces all nine fixed worker
  identities and exact runtime/environment evidence, resolves path ancestors
  and rejects overlaps/worker escapes, checks timestamp identity, and can
  atomically assemble completed workers without loading a model or restarting
  them. The hidden-worker bootstrap is import-safe for both `--worker NAME` and
  `--worker=NAME`, and CLI option abbreviations are disabled.
- Historical disclosure: preserved MPS observations span an uncontrolled v1
  result of **0.35390492528676987**, six clean v1 follow-ups at
  **0.000022813677787780762**, a sanitized v2 single-process result of
  **1.7168622612953186**, and a separate nondefault fast-math diagnostic of
  **1.8492990136146545**. They use different hashed procedures and are not
  included in v3 `F`. The five v3 MPS outputs were byte-identical, but the
  processes share one driver/device and are not treated as statistically
  independent or as establishing an upper bound.
- Test-first evidence: the new contract suite first failed **6/6** because the
  module and CLI did not exist, then its two added integrity cases failed until
  tree and variant-artifact hashing were implemented. The pre-v3 hardened tree
  passed **328/328 tests in 224.96 seconds**. The final floor-focused suite now
  passes **39/39**, including real model-free assembly from all nine artifacts.
  Independent post-write validation reloaded every artifact, recomputed `F` and
  `F64`, rehashed every current input, and proved creation-time/file-mtime order.
  The final exact tree passes **347/347 tests in 244.83 seconds**. `git diff
  --check`, byte-compilation, and the protected failure-document check pass;
  `FAILURE_LORA_FINETUNE.md` remains byte-for-byte unchanged. **573 GiB** remains
  free.
- Milestone: **T3B-1 COMPLETE — SELF-CONSISTENCY FLOOR RECORDED**. See
  `SELF_CONSISTENCY_T3.md`. Next is the prospective T3B-2 procedure and
  timestamp-enforcing evaluator, before any T3B checkpoint exists.

## 2026-09-01 — T3B-2 prospective trained-checkpoint parity procedure

- Prospective freeze: the procedure and constants were completed before any
  T3B checkpoint, T3B floor, or MLX-versus-PyTorch comparison existed. The fixed
  gates remain image preprocessing `<= 1e-5`, state preprocessing `<= 1e-6`,
  fine/base MLX held-out MAE ratio `<= 0.9`, and Torch/fine-MLX MAE ratio within
  `[0.95, 1.05]`. The deterministic threshold is exactly
  `max(0.005, 3 * F(checkpoint))`; neither constant is configurable.
- Chronology: a one-shot marker reads the real clock immediately before MLX
  work, binds one absent comparison path, and requires
  `floor.created <= floor.mtime < marker.created <= marker.mtime <=
  comparison.created <= comparison.mtime <= evaluation.created`. Floor, marker,
  comparison, and result installation are no-clobber operations. Outputs that
  overlap the raw floor bundle or a floor-bound exact input tree are rejected.
- Floor and input provenance: the evaluator descriptor-reads all 18 raw worker
  files and reconstructs every variant maximum, `F`, and `F64`. It then opens
  and rehashes every checkpoint, held-out case/noise, pinned-dataset, tokenizer,
  processor, and implementation input. Checkpoint and evaluation relative-file
  inventories must be exact. Hugging Face tokenizer symlinks are accepted only
  when their captured targets remain within the declared cache root, and link
  topology plus target bytes are revalidated before installation.
- Comparison evidence: all 56 frozen case identities and all 56 records for base
  MLX, fine MLX, Torch, and stats-active parity are required. Every aggregate,
  ratio, and gate is recomputed; derived overflow is rejected. The frozen base
  report body is checked against SHA-256 `211d6778...b5f7ce`. The exact trained
  export must include a bound `training_manifest.json`; its file inventory and
  tensor/scalar counts must match an actual canonical fp32 conversion. Semantic
  conversion validation operates on owner-only private copies of already
  descriptor-captured source/model/name-map bytes, closing path-swap races.
- Stable artifact behavior: every input snapshot binds device, inode, size,
  `mtime_ns`, SHA-256, and bytes within the evaluator run and is checked again
  before an atomic hard-link install. Existing or concurrently created marker,
  comparison, or evaluation artifacts are preserved rather than overwritten.
  The installed result contains complete evidence, separate fixed and derived
  decisions, and a standalone validator that recomputes all aggregates.
- Packaging: `setup.py` now discovers the complete `smolvla_mlx`, `training`,
  and `reference` package surfaces. A clean-source wheel test inspects the wheel,
  installs it into an isolated target, and imports the trained-parity evaluator
  from that installed artifact.
- Test/review evidence: the focused evaluator suite passes **52/52**. The
  self-consistency, distribution-wheel, and trained-parity suites pass
  **97/97 in 18.70 seconds**. Independent read-only review iteratively closed
  provenance, chronology, symlink, semantic-conversion, race, path-overlap, and
  numerical fail-open findings and returned explicit approval with no remaining
  Blocker, Important, or substantive Minor issue.
- Full checkpoint: with no training, floor, or benchmark process active,
  `make test` passed **402/402 tests in 230.01 seconds**. `git diff --check` and
  byte-compilation pass. `FAILURE_LORA_FINETUNE.md` remains byte-for-byte
  unchanged from commit `6c94ccd` with SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
  **567 GiB** remains free.
- Milestone: **T3B-2 COMPLETE — PROSPECTIVE PARITY PROCEDURE FROZEN**. See
  `PARITY_PROCEDURE_TRAINED.md`. Next is T3B-3 expert-only LoRA configuration,
  full-suite verification, and background launch.

## 2026-09-01 — T3B-3a expert-only LoRA pre-launch checkpoint

- Reference freeze audit: the installed LeRobot `0.6.1` defaults are
  `freeze_vision_encoder=True`, `train_expert_only=True`, and
  `train_state_proj=True`. The inspected configuration source SHA-256 is
  `2fb637cb428fa2fdf1d114646dcffaf4728216bfe5b7039d5d0cac4857ffc4e0`;
  the VLM-with-expert implementation source SHA-256 is
  `996d3b0c713c0ed42b383aa2cf89b2e6f9868e337747c480a64adaecdc1073cf`.
  `ARCHITECTURE.md` records how the T3B amendment deliberately excludes the
  separately controlled state projection while matching the frozen
  vision/prefix and trainable-expert boundary.
- Topology: the historical `legacy_full` path remains 229 adapters / 458
  tensors. The new `expert_only` path is exactly the four attention and three
  MLP linears in each of 16 expert layers: **112 adapters, 224 fp32 tensors,
  and 1,708,032 trainable scalars**. Tests prove the excluded vision,
  language/prefix, state, and action/time projection paths remain plain frozen
  linears; optimizer initialization validates state coverage for every adapter
  tensor; merge restores the plain model tree.
- Prospective commitment: `fixed_steps` mode commits exactly 3,000 updates at
  effective batch 8 and records `timing_measurements=false`; it cannot call the
  legacy budget benchmark. `launch.json` is self-hashed, atomically installed
  without clobbering, and binds the seeds, split/statistics, checkpoint and base
  conversion, optimizer schedule, all LoRA names/counts, reference-source
  hashes, direct/transitive training implementation plus native-extension
  hashes, and the fixed 56-case export/evaluation audit chain. The training
  process descriptor-reads the file without following symlinks, reconstructs
  it from live components, binds both file and semantic hashes into `run.json`,
  and repeats physical plus semantic validation immediately before update 1.
- Resume and launch safety: fixed expert-only training requires the exact
  prepared `OUTPUT/launch.json`; a fresh directory may contain only that file,
  `training.log`, and JSON `training.pid` identity. Existing checkpoint
  retention, interrupted-metrics recovery, sampler/flow-state restoration, and
  exact optimizer resume contracts remain covered. CLI prepare mode cannot
  enter training; fixed mode refuses budget-selection timing; background mode
  can bind PID, executable, working directory, and both launch digests before
  the run begins.
- Runtime and checkpoint hardening: the executable POSIX launcher now enters
  the repository virtual environment with `python -I -S` before Python startup
  can execute user-site hooks, preserves the caller's working directory and
  argument bytes, and is part of the implementation hash. Direct unisolated
  invocation is rejected. Runtime provenance binds guarded source/extension
  generations plus the installed-package inventory, then fails closed on any
  later guarded import. Retained-checkpoint namespace evidence binds every
  retained directory and file, not only the newest checkpoint, across
  publication and finalization.
- Real path preflight: the first disposable isolated run correctly stopped
  before update 1 when PyAV lazily requested `av.subtitles` after provenance
  freeze. The semantic audit now decodes and preprocesses one batch on its
  independent disposable bridge before freezing, revalidates every physical
  input afterward, and leaves the live bridge unconsumed. A freshly prepared
  run then completed **29 real Metal updates** before an intentional SIGINT.
  It published and bound step 1 with metadata SHA-256
  `30d9964b87df8bfefc9cbe14902fa36cc567aac7d6b71cdc23d7228c8853feaf`,
  model SHA-256
  `3aa626d834715474a243a29e050d63db9949730edf6d5880599dad0bc4f5832e`,
  and optimizer SHA-256
  `9a64a2512441eaf0076f642f1833bf849aa5436efe42cd2c79094652882a351a`.
  Its launch/configuration/run-configuration hashes are respectively
  `8cdec8fbfaaac5b9d0fd6ebb8e67f257ef976961a4982b695a42eb6a1fe7818f`,
  `83725054c498f78027aab840222921bb13f6ffe56d90b581a3e57e8ca8529ddc`,
  and `09895b216aff79ea3e26294aa4ef0484e5d316ee88eef7733782f95a9da62350`.
  Independent review rechecked the stopped PID, 29 metrics rows, run state,
  latest pointer, and checkpoint hashes and found no remaining source blocker.
- Final pre-launch verification: byte-compilation passed; the focused runtime,
  fine-tune, and dataset set passed **142/142 in 235.70 seconds**. With no
  training or floor process active at `2026-09-01T22:44:56Z`, `make test`
  collected and passed **536/536 tests in 490.35 seconds**. The two additional
  cases prove that a copied launcher preserves its caller's working directory
  under `-I -S` and that semantic reconstruction materializes the disposable
  decode path without consuming the live bridge.
- Protected evidence: `FAILURE_LORA_FINETUNE.md` remains byte-for-byte
  unchanged at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
  No budget-selection timing benchmark, floor computation, hardware access, or
  upload occurred. Next: commit and push these exact implementation bytes,
  then generate/hash and launch the canonical T3B run.

## 2026-09-02 — T3B-3 canonical launch and concurrent Stage R start

- Published implementation: the reviewed pre-launch tree was committed as
  `75b5361` (`phase-14: harden T3B launch and checkpoint integrity (536 tests
  pass)`) and pushed normally. Local `HEAD`, `origin/main`, and
  `git ls-remote origin refs/heads/main` all resolve to
  `75b5361b8269d0e8a946c3bc00c77560e472957a`.
- Canonical commitment: `.cache/training/t3b/launch.json` was created only
  after that push. Its file SHA-256 is
  `95f765137fe0f70c034561cf96e7f845f18ab31285f2051c945264f3ccdbea81`;
  configuration SHA-256 is
  `fe8a937e26b29b2914097ce652c1685d616eaa17f9193369e7c0df1770748fb3`;
  run-configuration SHA-256 remains
  `09895b216aff79ea3e26294aa4ef0484e5d316ee88eef7733782f95a9da62350`.
- Background ownership: this command host reaped a direct `nohup` child before
  it opened any artifact, so a detached `tmux` bootstrap was used to establish
  the documented `nohup` command as an orphaned process. An initial supervised
  attempt spelled cache paths absolutely and failed closed before `run.json`
  because the bridge commitment preserves literal cache-path spelling. Its
  valid prestart log was automatically preserved as
  `startup-recoveries/training-log-prestart-000001`. A diagnostic comparison
  proved the normal and supervised isolated manifests identical at **3,922
  implementation hashes / 3,763 guarded modules**. Relaunching the exact
  repository-relative command matched the prospective commitment.
- Running evidence: PID **69355**, parent PID **1**, working directory the
  repository root, isolated repository interpreter, and training-log inode
  **23558994** are bound in `training.pid` and `run.json`; process-identity
  SHA-256 is
  `49fb4fd9cfeacfff92c82b89e388c97bbc52d991761b5958d03fc63d36534b06`.
  The run passed update 1, published `checkpoints/step-000001`, and atomically
  bound metadata SHA-256
  `b17fb2415c96782587e7ae710526e35afbcd90596153b67c7a0e2a2e9fde8f44`,
  model SHA-256
  `3aa626d834715474a243a29e050d63db9949730edf6d5880599dad0bc4f5832e`,
  and optimizer SHA-256
  `9a64a2512441eaf0076f642f1833bf849aa5436efe42cd2c79094652882a351a`.
  It remained `running` and had already written metric step 10 when this
  milestone was recorded.
- Concurrent release P0-1: the SSH fetch/push origin is the operator-provided
  `git@github.com:daniiarabdiev/smolvla_mlx.git`, and the remote branch mirrors
  the canonical history. The complete canonical Apache License 2.0 text is now
  present as `LICENSE` (SHA-256
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`),
  `MANIFEST.in` explicitly includes both legal files, and `README.md` links
  both `LICENSE` and the existing third-party `NOTICE`. Static acceptance is
  complete; the next full-suite verification is deferred until training is
  absent so no test load contaminates training evidence.
- Guardrails: no budget benchmark, floor computation, parity comparison,
  hardware/serial access, or upload is running with training. The protected
  original-T3 failure record remains unchanged.

## 2026-09-02 — Stage R P0-2 public fine-tune discovery

- Search method: queried the public Hugging Face model API for `smolvla`,
  `SmolVLA SO101 fine-tuned LeRobot`, and `SmolVLA SO100 LeRobot policy`, then
  inspected the first 1,000 `smolvla` model records for a full standard LeRobot
  processor file set. Numerous SO-100/SO-101 weights exist, but nearly all
  expose only `config.json`, `model.safetensors`, and `train_config.json`; those
  cannot independently prove active saved normalization.
- Selected public target:
  `soonweihong0857/swhfypv3_smolvla_multitask_model`, model revision
  `5e2491c809ec892427f54db1eb23bf8c4bbbf770`. Its card identifies
  `lerobot/smolvla_base` as the base and
  `soonweihong0857/smolvla_multitask_data` as the training dataset. The model
  repository totals **1,197,819,154 bytes**, including the complete standard
  pre/postprocessor configs and safetensors; the linked dataset at revision
  `ec0062a53e0ae88d46a4341ab0695dfa9f03111b` totals **471,239,739 bytes**.
  Both are below the package's approximate 5 GiB cap.
- Compatibility: the config uses the audited
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` backbone, 16 VLM layers, 0.75
  expert-width multiplier, 32-wide state/action padding, 50-action chunks,
  and two 480×640 inputs named `observation.images.wrist_camera` and
  `observation.images.top_camera`. Its processor tensors contain active
  six-element `observation.state.mean/std` and `action.mean/std`; each records
  44,900 samples. This simultaneously exercises non-base camera names, active
  statistics, Hub-ID loading, and a real public fine-tune.
- Header-only verification at the pinned revision found exactly **500** model
  tensors and **450,046,176** parameters, matching the canonical architecture
  already handled by the strict converter without downloading the 1.2 GB
  payload. Its saved preprocessor explicitly requests right-side,
  `max_length=48` token padding and MEAN_STD for state/action; both processor
  safetensors expose float32 six-element state/action vectors. Extra dataset
  statistics are present and must be ignored rather than treated as an error.
- The linked revision is a LeRobot v3 dataset with **100 episodes**, **44,900
  frames**, **2 tasks**, and 30 fps observations. It provides exactly the two
  configured 480×640 cameras as AV1/PyAV video plus six-element state/action
  columns, so the planned frozen evaluation corpus can cover both tasks without
  inventing feature renames or sourcing inputs from a different dataset.
- Rejected examples: the exact-dataset candidate
  `lerobot-edinburgh-white-team/smolvla_svla_so101_pickplace` and the tagged
  `rancheng222/smolvla_so101_tie_bag` each fit below 1 GiB but omit all saved
  processor files, so they cannot satisfy the active-normalization target.
  `BookHou/smolvla-rgbd-unfrozen-b8-9280` has processors but no linked
  downloadable evaluation dataset in its metadata.
- Implementation boundary: the core inference preprocessor currently resolves
  active stats in configuration but still returns identity transforms. The
  training lane already contains a tested stats-aware prototype. After T3B
  exits, P0-2 will move strict mean/std loading into the dependency-light core,
  preserve the base checkpoint's identity behavior, improve config/observation
  diagnostics, create the reference stats-active target and second golden set,
  and then download/evaluate the selected public target at fixed tolerances.
  No large model/dataset download or implementation-file mutation occurred
  during training.

## 2026-09-02 — Stage R P0-3 read-only cache baseline

- Baseline sizes: `.cache/smolvla_mlx` is **77,380,960 KiB** (73.80 GiB),
  `.cache/hf` is **1,868,928 KiB**, and `.cache/training` was **3,296,196
  KiB** while the live T3B checkpoint tree was still growing.
- Largest retained categories are the canonical fp32 policy/source cache
  (**7,922,456 KiB**), general converted checkpoint cache (**3,516,584 KiB**),
  benchmark cache (**2,637,584 KiB**), the public base-checkpoint snapshot
  (**893,328 KiB**), and explicit inference/parity caches. Training goldens,
  original T3 evidence, and the live T3B tree are protected evidence rather
  than cleanup candidates.
- Regenerable debris: **23** top-level `debug-*` experiment trees total
  **37,803,716 KiB** (36.05 GiB). Small named scratch probes and
  `benchmark-debug` are also candidates, but the specialized non-debug
  conversion caches are retained because the full suite reuses them and they
  are expensive to regenerate.
- Planned safety contract: inventory is read-only; cleanup accepts only the
  exact repository `.cache/smolvla_mlx` root, refuses symlinked ancestry or
  candidates, uses a fixed basename allowlist/prefix policy, defaults to a dry
  run in its CLI, and cannot name training/checkpoint/evidence roots. Tests will
  cover root/path traversal, symlink, and retained-evidence refusal before any
  deletion. No cache entry was removed during this baseline or while T3B runs.
- The deletion allowlist is now intentionally narrower than the inventory:
  top-level directories matching `debug-*` plus the exact directory
  `benchmark-debug`, and nothing else. Small `*_probe` files are not worth
  broadening destructive scope; all non-debug conversions, policy/source
  caches, goldens, Hugging Face data, and every training/evidence path remain
  outside the cleanup command by construction.

## 2026-09-02 — Stage R P1-2/P1-3 static release audit

- Portability boundary: `smolvla_mlx/rmsnorm.py` currently imports
  `_rmsnorm_native` unconditionally and all four CPU compatibility operations
  dispatch directly to it. `setup.py` always registers the MLX CMake extension;
  `pyproject.toml` restricts installation to `>=3.12,<3.13`; neither the build
  metadata nor CMake declares `MACOSX_DEPLOYMENT_TARGET`. Consequently an absent
  or unloadable extension prevents even the production Metal path from
  importing, and the requested Python 3.11/3.13 wheel matrix is not yet
  representable by project metadata.
- Planned proof: force the absent-extension path in a subprocess and exercise
  pure-MLX RMSNorm, split-half RoPE, softmax, and SiLU; retain exact
  PyTorch-order CPU tests when the extension is available and skip them only
  with an explicit genuine-absence reason. Then build the sdist and each
  uv-available 3.11/3.12/3.13 wheel under repository-local caches, inspect its
  deployment tag, and run a fresh-environment import, isolation, and real CLI
  prediction smoke per artifact. No build or runtime test ran during T3B.
- Artifact baseline: the existing wheel is exactly
  `smolvla_mlx-0.0.1-cp312-cp312-macosx_26_0_arm64.whl`, and its embedded
  `WHEEL` metadata repeats that tag. The current native module has Mach-O
  minimum OS 26.0 because no target was declared. MLX 0.32.2 publishes
  macOS-14 arm64 wheels for CPython 3.11, 3.12, and 3.13 (plus a matching
  `mlx-metal` wheel), so 14.0 is the evidence-backed target rather than an
  invented compatibility claim. All three requested interpreters are
  available; matrix builds will still use `UV_PYTHON_INSTALL_DIR` under this
  repository and deliberately select the macOS-14 dependency wheels so the
  host's newer wheel cannot raise the linked binary minimum.
- Dependency matrix: every pinned base dependency declares Python 3.11–3.13
  support, and MLX has a macOS-14 wheel for each ABI. LeRobot 0.6.1 itself
  requires Python 3.12, so the Python-3.11 artifact smoke must not pretend the
  optional dataset bridge can install there. The public `predict` command will
  gain a dependency-free saved-observation input alongside `--dataset`; every
  base wheel can then perform a real installed prediction, while dataset-backed
  prediction remains an explicitly Python-3.12+ reference-extra smoke.
- Documentation boundary: the current README contains the basic API, CLI,
  strict-golden evidence, and the original Metal table, but it still states
  Python 3.12 only, identity-only postprocessing, and inference-only scope. It
  lacks the required stats-active evidence, cache inventory/cleanup contract,
  strict-parity versus default-production distinction, exact LeRobot GPU
  fine-tune handoff, async server path, and troubleshooting. Final README
  claims will be written only after their packages and evidence exist.
- GPU handoff audit: the installed pinned LeRobot 0.6.1 entry point accepts
  `--policy.path`, `--dataset.repo_id`, `--batch_size`, `--steps`, and
  `--output_dir`. Its SmolVLA defaults freeze the vision encoder and train the
  expert path, while `--policy.push_to_hub=false` and
  `--save_checkpoint_to_hub=false` keep the run local. It always saves the
  final step and maintains `checkpoints/last`; the loadable standard artifact
  is `checkpoints/last/pretrained_model`, suitable as this package's local
  `from_pretrained` path. These facts came from the exact installed 0.6.1
  sources; the documented command will be smoke-validated only after T3B.

## 2026-09-02 — T3B post-run chronology audit

- The legacy `scripts/check_lora_finetune.py` command performs MLX evaluation
  before it writes its outcome, so invoking it ahead of the T3B floor would
  violate the operator's floor-first rule even though the fixed gates are
  logically reported before the derived gate. The runbook now requires the
  comparison producer to be finalized and tested without loading T3B, followed
  by the PyTorch-only floor and its recorded hash/timestamps, then the one-shot
  comparison marker, then the first MLX evaluation and the bound evaluator.
- No T3B MLX-versus-PyTorch comparison has been run. No implementation file was
  changed during this audit because the active trainer revalidates its source
  bytes before export.

## 2026-09-02 — T3B training completion and export audit

- The original PID 69355 completed all **3,000** updates without a resume or
  checkpoint/metrics recovery and exited after the protected export. Final
  run status is `trained_and_exported`; retained checkpoints are exactly
  `step-002800`, `step-002900`, and `step-003000`.
- Training recorded **3,545.863376834008 seconds**, **2,899,690,676 bytes**
  peak MLX memory, final loss **0.10034868866205215**, and final smoothed loss
  **0.17088623540128797**. The metrics file contains one header plus all 3,000
  updates and hashes to
  `33f00adc5316cbc295e6f3fa1e153963b64fadec59a7db5401074794245f6278`.
- The independent artifact audit reconstructed the fixed-step expert-only run
  digest `09895b216aff79ea3e26294aa4ef0484e5d316ee88eef7733782f95a9da62350`,
  verified 24,000 samples and 24,000 flow draws, and bound the final adapter
  SHA-256
  `cce4eed18a7311594950f6d4da33a44dd337f66fbc29162d686c5338ec044826`
  to the final checkpoint model.
- The merged fp32 export passed its complete inventory, manifest, file-hash,
  dtype, tensor, and scalar audits: **500 tensors**, **450,046,176
  parameters**, model SHA-256
  `858704fa572501d9e5a048076f8da692693b90c463feda29201a72f3f0b18883`.
- The audit initially exposed that the outcome evaluator only reconstructed
  the legacy adaptive/full-scope T3 schema. Regression tests now cover the
  T3B fixed-step budget, expert-only adapter scope, and scope-bound support
  file hashes. The focused evaluator tests pass. No MLX-versus-PyTorch
  comparison of T3B has run, and no floor has yet been computed.

## 2026-09-02 — T3B comparison producer finalized before floor

- Added a floor-first producer that reconstructs and rehashes the prospective
  floor bundle, verifies the one-shot marker's actual SHA-256 and `mtime_ns`,
  validates every floor input group, and refuses an existing outcome or
  comparison before it can invoke model evaluation.
- The producer emits the legacy fixed-gate outcome for audit continuity and a
  separate no-clobber `smolvla-trained-checkpoint-mlx-comparison` artifact with
  all 56 MLX records, 56 Torch records, 56 stats-active parity records, exact
  floor/marker bindings, concrete floor-input locations, and semantic native
  conversion evidence.
- Focused verification: **132 passed in 17.06s** across the complete
  self-consistency, trained-parity, producer, and training-evaluation groups.
  The producer-specific tests prove exact schema assembly, validation of a real
  raw floor bundle and marker, failure before model execution for an invalid
  floor/marker, and a model-free CLI path.
- Full repository verification on the finalized pre-floor tree: **544 passed
  in 482.22s**. This run was functional verification, not a latency benchmark;
  no training or self-consistency worker was active.
- No T3B floor, marker, MLX inference, or Torch/MLX comparison existed when
  these implementation bytes and tests were finalized.

## 2026-09-02 — T3B prospective self-consistency floor frozen

- Machine-state declaration: the 3,000-step trainer had exited, the 544-test
  verification run had exited, and process inspection showed no training,
  self-consistency, benchmark, or pytest process before the floor began. No
  timing measurement, benchmark, training job, or MLX comparison ran while the
  floor workers were active.
- Command: `uv run --extra reference python
  scripts/compute_self_consistency_floor.py --checkpoint
  .cache/training/t3b/export --evaluation-dir
  .cache/training/t3-evaluation --cache-dir .cache/hf --work-dir
  .cache/training/t3b/self-consistency --output
  .cache/training/t3b/floor.json --purpose prospective_gate --max-threads 18`.
  All nine raw variants completed and the report was installed atomically.
- Floor path:
  `/Users/dan/Desktop/workshop/robotics-mlx-contrib/.cache/training/t3b/floor.json`.
  SHA-256:
  `28d83926a70e507671bfd694e032f81b71093d475075aad627b3c24c5b334efc`.
  Embedded creation: `2026-09-02T00:38:00.730626+00:00`,
  `created_at_ns=1788309480730626000`; actual
  `mtime_ns=1788309480735640183`.
- Reconstructed 18-file raw bundle SHA-256:
  `31ce3db6619294432742b38214132267cfecf735dc0ce1d98199bbd223e8a889`.
  Combined input SHA-256:
  `3688cdad4f40724fa82765bb1c2ba89369aed056e29cecb8b1c074d6069939bb`;
  implementation tree:
  `2e279c3feb3c2f72db4338a30fa16640e0f6dcd01c5efe75580b548aea5cf214`.
- Checkpoint tree SHA-256:
  `67da4777b1fc24ad8a7b186feaa781c4a80d745aee64e00fe9a8ea50543e5354`;
  merged model SHA-256:
  `858704fa572501d9e5a048076f8da692693b90c463feda29201a72f3f0b18883`.
  Run/config bindings remain
  `run.json=2af527bea4691862e89eb6daa674e6d99309668ac45f57397786976aab3c301e`,
  `run_config=09895b216aff79ea3e26294aa4ef0484e5d316ee88eef7733782f95a9da62350`,
  and `launch.json=95f765137fe0f70c034561cf96e7f845f18ab31285f2051c945264f3ccdbea81`.
- Result: `F(C)=2.467632293701172e-05` and
  `F64(C)=2.2446572480461224e-05`. Therefore `3F(C)` is
  `7.402896881103516e-05`, and the prospectively fixed normalized-action gate
  is the unchanged fallback `max(0.005, 3F)=0.005`.
- This entry was written before the start marker and before any T3B MLX
  inference or MLX-versus-PyTorch comparison. At this point neither
  `comparison-start.json`, `outcome.json`, nor `comparison.json` exists.

## 2026-09-02 — T3B bound comparison and statistical-alpha verdict

- After the floor record above was committed and pushed, the one-shot start
  marker was installed at `2026-09-02T00:39:06.152740+00:00`
  (`created_at_ns=1788309546152740000`, actual
  `mtime_ns=1788309546153727610`) with SHA-256
  `0e1121728fc30eb911e6f596d32ec5f7de97faa0d44e883b52367d7ac7dcd202`.
  It binds the prospective floor, raw floor bundle, checkpoint, and intended
  comparison output.
- Only after that marker did the comparison producer perform the first T3B
  MLX-versus-PyTorch evaluation. It installed `outcome.json` with SHA-256
  `75dff4b750dd1e8c8bc4d8426fe9af297bedb33ea1bfa7523dcc49d51460b33f`
  and `comparison.json` with SHA-256
  `6aa8e3771bbbd81ecd9599ec9605a4e1efb804fa9ec66c4f82d2d6aea3eb00c6`.
  The comparison creation time is `2026-09-02T00:42:52.253017+00:00`
  (`created_at_ns=1788309772253017000`, actual
  `mtime_ns=1788309772258209788`), strictly newer than both floor and marker.
- Fixed outcomes all pass: image preprocessing max
  `3.5762786865234375e-7`, state preprocessing max `0.0`, fine/base held-out
  MAE ratio `0.486008430646319`, and Torch/MLX MAE ratio
  `0.9999447391574267`. Fine-tuned MLX and Torch physical MAE are
  `2.2550044155546596` and `2.2548798021106493`, respectively.
- The timestamp- and hash-enforcing evaluator installed
  `parity-evaluation.json` with SHA-256
  `1e337f0bb87aa66a4270c526dd918bd18807aa6aa5291a59b119780080ea9eca`.
  Normalized action max is `0.013038858771324158` versus the prospectively
  derived `0.005`, so the derived deterministic gate alone fails. Raw and
  standardized physical maxima are `0.13149452209472656` and
  `0.013038855977356434`.
- A diagnostic trace on the frozen worst case (ordinal 24, episode 28, frame
  87, absolute index 6307) reproduced the final normalized maximum exactly.
  It starts at velocity/state differences
  `0.0035195350646972656` / `0.0003519505262374878` and ends after Euler step
  9 at `0.035959720611572266` / `0.013038858771324158`. Compared with T3,
  expert-only adaptation reduces final normalized divergence by **13.62x**
  and raw physical divergence by **50.44x**.
- `LORA_SCOPE_COMPARISON.md` records the complete T3/T3B table and both Euler
  curves. `FAILURE_LORA_FINETUNE_B.md` applies the required three-hypothesis
  discipline. The original `FAILURE_LORA_FINETUNE.md` remains unchanged at
  SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`;
  no tolerance or original gate was changed or reinterpreted.
- Milestone: **TRAINING ALPHA (STATISTICAL)**. T4 and T5 are unblocked because
  all fixed outcome gates passed; strict deterministic parity remains an
  explicit documented limitation.
- Post-verdict verification: the corrected focused floor/parity/producer/
  evaluation suite passes **132/132 in 17.89 seconds**. The complete repository
  passes **544/544 in 482.16 seconds**. An initial focused invocation named a
  nonexistent stale test path and collected no tests; it made no repository
  change and was immediately replaced by the correct target list.

## 2026-09-02 — Stage R P0-2 stats-active checkpoint generality complete

- The dependency-light public preprocessor now reads exact checkpoint
  `observation.state.mean/std` and `action.mean/std` tensors, validates their
  shapes and finite values, applies LeRobot's `1e-8` mean/std math, and checks
  that saved pre/post action statistics agree. The base checkpoint remains
  effective identity because its robot-prefixed stats do not match the runtime
  keys. The training loader detects the new core behavior and no longer
  double-normalizes exported checkpoints.
- Matching local directories and arbitrary Hub identifiers use the same strict
  loader. A complete public-API test resolves a non-hardcoded Hub identifier
  through `SmolVLAMLX.from_pretrained` and consumes all 500 tensors. Unsupported
  architecture and missing observation errors now name every checkpoint camera
  key/shape plus `observation.state` shape.
- Reference source inspection and a Torch conformance test establish exact
  camera behavior: all configured streams present are used; absent configured
  streams are skipped; only `empty_cameras` adds false-masked synthetic images.
  The base's three configured keys and `empty_cameras=0` therefore yield three,
  two, or one real streams based on input, with no implicit third-camera pad.
- The pinned base-plus-dataset-stats artifact uses base model SHA-256
  `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`
  and dataset `lerobot/svla_so101_pickplace` revision
  `f641879e22172be7e8161d5e6c1503c2d2feb657`. Its no-clobber/reusable artifact
  manifest hashes to
  `74b281cb476b4d5d8f76d02bbed0def7d9e00a22901d413da6c281a10f9d938d`;
  its eight-case golden manifest/metadata hashes are
  `b7e841821704fc338d075290b0e92a31d8249dc3b00f9992d57aa67d18627805`
  / `529c4dce5b9a09c6d7fab259c8449d582d93ccf7201d306abfeea035e08a4ed0`.
- Stats-active base deterministic maxima: image preprocessing
  `3.5762786865234375e-7`, state `0.0`, normalized actions fp32
  `2.8759241104125977e-6`, and normalized actions bf16
  `0.0048642754554748535`; all pass unchanged `1e-5`, `1e-6`, `0.005`, and
  `0.05` limits. Exact reference-action unnormalization differs by `0.0`.
- Its 50-frame statistical record SHA-256 is
  `96df60698ced61b227c8ce88cc90249130068de5d0be163fbfc71f76c6afdc38`.
  Torch fp32 MAE is `2.889570787350337`; MLX fp32/bf16 MAE values are
  `2.889570882717768` / `2.8853507562478384`, giving ratios
  `1.0000000330040129` / `0.9985395647267157`, both below `1.05`.
- The previously selected public target and all **1,197,819,154 model bytes**
  plus its **471,239,739-byte** dataset were downloaded at their pinned
  revisions. The public target's eight-case golden manifest/metadata hashes
  are `98f593a35357203081e787f3900d3544bbdeeeca20881c532ed262c033a10d38`
  / `5c46febfeb95f4140021333421e65ff12aef1a3068af766e29c89dea08f15e08`.
  It exercises `wrist_camera` and `top_camera` at 480x640 with active stats.
- Public fine-tune deterministic maxima: image `2.384185791015625e-7`, state
  `0.0`, normalized fp32 `0.000034362077713012695`, normalized bf16
  `0.004036039113998413`, and exact-reference unnormalization `0.0`; all fixed
  gates pass. Its 50-frame statistical record SHA-256 is
  `6ab112f49a84c98c7bd0bf93487f0132a3095d9e44c7027bc2097f4638577315`:
  Torch MAE `0.4209008048971494`, MLX fp32/bf16 MAE
  `0.4209010468920072` / `0.419238363802433`, ratios
  `1.0000005749451057` / `0.996050278176297`.
- `make goldens` now regenerates the base, stats-active, and public-fine-tune
  reference sets from pinned inputs. Generated model/golden artifacts remain
  ignored. The focused cross-lane suite passes **69/69 in 67.95 seconds**; the
  first complete post-package tree passed **566/566 in 518.67 seconds**. The
  added full Hub-ID public-API test passes **9/9** with its public-target group;
  the exact final P0-2 tree passes **567/567 in 519.76 seconds**.

## 2026-09-02 — Stage R P0-3 cache hygiene complete

- Cleanup began only after the T3B trainer and floor workers had exited. The
  earlier read-only baseline was 77,380,960 KiB while T3B was active; P0-2
  conversion and public-checkpoint work brought the immediate pre-cleanup
  `.cache/smolvla_mlx` allocation to **91,447,880 KiB** (87.21 GiB).
- The frozen dry run selected exactly **24** top-level directories: 23 names
  matching `debug-*` and exact `benchmark-debug`. Their logical size was
  **39,611,270,560 bytes**. No source snapshot, converted production cache,
  probe, training artifact, golden corpus, file, nested path, or symlink was
  eligible.
- `make clean-cache` reduced `.cache/smolvla_mlx` to **52,764,872 KiB** (50.32
  GiB), reclaiming **38,683,008 KiB** (36.89 GiB) of allocated space. The
  post-cleanup inventory accounts for **54,031,014,140 logical bytes** and
  marks every remaining top-level native-cache entry as retained.
- `make cache-inventory` reports each entry's type, logical byte size,
  regenerability, retention decision, and reason. `make clean-cache-dry-run`
  exposes the exact deletion set. The implementation refuses repository-root,
  training, outside, traversal, symlinked-cache, allowlisted-symlink, and
  allowlisted-file targets; it also requires Python's symlink-attack-resistant
  recursive remover.
- Before/after metadata fingerprints were identical for `.cache/training`,
  `policy-float32`, and all three golden corpora. `.cache/hf` remained exactly
  **3,498,900 KiB** with **420 files**; only six existing zero-byte dataset lock
  mtimes changed during reference tests, establishing that no model or dataset
  was downloaded. No deleted debug directory was regenerated, and the final
  dry run reports zero candidates.
- Focused cleanup verification passes **10/10**. The required post-cleanup
  complete repository suite passes **576/576 in 522.53 seconds**. The original
  T3 failure record remains unchanged at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.

## 2026-09-02 — Stage R P1-2 packaging portability complete

- The base metadata now supports `>=3.11,<3.14`. Because LeRobot 0.6.1 itself
  declares Python 3.12+, only `lerobot[dataset,smolvla]` and Torch in the
  optional `reference` extra carry a `python_version >= "3.12"` marker. The
  repository lock resolves all 103 packages across the widened range.
- `_rmsnorm_native` is optional at import and build time. Exact native CPU
  RMSNorm/RoPE/softmax/SiLU remains the default when installed; setting
  `SMOLVLA_MLX_BUILD_NATIVE=0` produces a binary-free wheel whose isolated
  import reports `pure-mlx-fallback`. A stale in-tree `.so` initially exposed
  a package-data contamination risk; conditional exclusion now proves the
  extension-free artifact contains no `.so` or `.dylib`.
- Saved-observation `predict` is now part of the dependency-light CLI, with a
  required mutually exclusive dataset/saved source and sample-name-to-task
  validation. A real invocation against `sample_000` completed successfully.
- Focused distribution/RMSNorm/CLI verification passes **17/17**. The exact
  pre-artifact source commit `3e30212604985dbaf2ad1360b1e4fc1023303cf6`
  passes the complete suite: **584/584 in 523.20 seconds**.
- `UV_PYTHON_INSTALL_DIR="$PWD/.cache/uv-pythons" uv python install 3.11
  3.12 3.13 --no-bin` provisioned CPython 3.11.15, 3.12.13, and 3.13.14.
  Builds used the clean detached source commit, repository-local `.cache/uv`,
  and `MACOSX_DEPLOYMENT_TARGET=14.0`. `dist/` contains one clean sdist and
  three wheels tagged `cp311`, `cp312`, and `cp313`, each
  `macosx_14_0_arm64`; hashes and byte sizes are in `DIST_MANIFEST.md`.
- Each wheel, plus an sdist build on 3.12, was installed with base dependencies
  into a separate fresh venv. All four imports resolved from their venvs,
  reported `native-reference`, proved Torch/LeRobot/Transformers absent, and
  completed an offline real saved-observation `predict` yielding a finite
  six-component action. Nothing was uploaded.
- All project wheel extensions declare Mach-O `minos 14.0`. The pinned
  `mlx==0.32.2` dependency's `libmlx.dylib` independently declares `minos
  26.2` in every smoke environment, matching the link warning. This upstream
  binary floor is recorded explicitly: project tagging is fixed, but actual
  macOS-14 execution is not claimed.

## 2026-09-02 — Stage R P1-3 release documentation complete

- `README.md` now opens with a single-paragraph native-MLX pitch and covers the
  verified Python 3.11–3.13 install boundary, conversion/cache behavior, an
  exactly ten-line API example, saved-observation and optional dataset CLI
  paths, cache inventory/cleanup, limitations, troubleshooting, and
  Apache-2.0/NOTICE attribution. Every local Markdown link resolves.
- The benchmark table keeps the canonical `BENCHMARK.md` measurements: a
  50-action chunk represents **1.67 seconds** at 30 fps; fp32's **111.34 ms**
  median is **15.0x** motion-duration/compute and bf16's **131.12 ms** is
  **12.7x**. The prose explicitly excludes capture, transport, and actuation
  and records bf16's measured 0.50 GiB memory saving and slower latency.
- The correctness section names the pinned LeRobot/PyTorch golden source,
  immutable deterministic/statistical gates, strict CPU arithmetic boundary,
  default MLX/Metal execution behavior, and the unresolved strict Metal module
  caveat. Its stats-active and public-fine-tune numbers are copied from the
  hashed P0-2 records, including ratios `1.0000000330040129` /
  `0.9985395647267157` and `1.0000005749451057` / `0.996050278176297`.
- The standard GPU handoff is pinned to `lerobot[training,smolvla]==0.6.1` and
  uses the upstream `lerobot-train` flags for the base policy, dataset, CUDA,
  200,000 steps, 20,000-step checkpoints, and local output. Both
  `--policy.push_to_hub=false` and `--save_checkpoint_to_hub=false` are
  explicit; WandB is disabled. Installed-source inspection confirms the
  resulting path `outputs/smolvla-finetune/checkpoints/last/pretrained_model`,
  which the README loads directly or by a separately published Hub ID. This
  PyTorch/LeRobot path is clearly separated from experimental native-MLX
  training.
- Command validation: all five documented native CLI forms parse; the ten-line
  API example executed offline and returned a finite `(6,)` action; offline
  conversion and saved-observation prediction succeeded; LeRobot's dynamic
  SmolVLA help accepted every documented training flag without starting a
  training run. The reference environment's TorchCodec/FFmpeg warning and
  PyAV fallback are reflected in troubleshooting. Artifact install smokes,
  golden generation, and cache cleanup commands were already exercised and
  recorded by P1-2, P0-2, and P0-3 respectively.
- Timing-command smoke began only after a process-table check at
  `2026-09-02T02:40:25Z` found no training or floor worker. The documented
  50-run fp32 benchmark command completed on `Device(gpu, 0)` at 110.93 ms
  median / 111.86 ms p95 and 3,157,868,570 peak bytes. This command-validation
  sample does not replace the canonical benchmark or the still-required P1-1
  production evidence table.
- Final P1-3 verification passes **584/584 tests in 529.75 seconds**. No
  hardware, serial port, robot directory, or upload was used. The original T3
  failure remains byte-identical at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.

## 2026-09-02 — Stage R P1-1 production-path evidence complete

- Test-first mode selection makes the public contract explicit. A policy loaded
  with default `execution_mode="production"` owns `Device(gpu, 0)` for every
  public inference call even inside an outer CPU context; `strict` analogously
  owns `Device(cpu, 0)` even inside an outer GPU context. The CLI exposes the
  same frozen choice as `--execution-mode`. Invalid modes fail before checkpoint
  resolution, and strict parity/training loaders now opt into strict mode
  rather than relying on ambient global state.
- The source and focused tests were committed and pushed before authoritative
  timing as `4824db9d289bec1c148a43509f41407c1458ef24`. This lets every evidence
  artifact and the benchmark name a clean source commit. The focused public,
  active-stats, CLI, benchmark, production-evidence, training-loader, and import
  isolation suite passes **94/94 in 62.89 seconds**.
- `.cache/production-deterministic.json` (SHA-256
  `3268f88be5ea854ff5162373146d1b2fd23cdbcc26bacbc060f0f8fa5b850398`)
  binds the pinned source model SHA-256, golden manifest/metadata SHA-256, MLX
  0.32.2, clean source commit, explicit production mode, and every per-case
  result. Across the same eight goldens, Metal fp32 maximum normalized absolute
  error is **0.04730653762817383** versus fixed **0.005** (fail); Metal bf16 is
  **0.044106483459472656** versus fixed **0.05** (pass). Both worst cases are
  `sample_004`. The parser independently freezes and recomputes the maxima,
  worst cases, and outcomes.
- Explicit strict and production 50-frame records hash to
  `b292736e3ec82b3eae8702c065c7c642326d226f06d35aa2214ac83fa1c23db5`
  and `c506ddcfdde50297e97b9905a299d55f117680a93b257e0af335ae6c9ad5fe07`.
  Strict fp32/bf16 ratios reproduce `0.9999999969253671` /
  `1.0000097740913103`; production Metal ratios are
  **1.0000127999805857** / **1.0000216963394593**. All four remain below the
  immutable `1.05` statistical ceiling. Ratio validation now recomputes each
  value from its recorded MAEs.
- At `2026-09-02T02:59:23Z`, immediately before timing, the process-table query
  returned only its own shell/search invocation and no pre-existing training,
  floor, test, evidence, benchmark, or policy-server worker. The machine was
  idle for the normative 5 warmups + 50 measured runs per dtype; no training or
  floor computation overlapped. On the M5 Pro / 48 GiB / macOS 26.6.2 / Python
  3.12.13 / MLX 0.32.2 production path, fp32 median/p95 is **110.54/111.41 ms**
  with **2.94 GiB** peak, and bf16 is **130.44/131.25 ms** with **2.44 GiB**
  peak. `BENCHMARK.md` hashes to
  `97b6298908da98e6f5b5e134610536446a23dcdc5de71c4b95e482eaf45b6ded`.
- `BENCHMARK.md`, `README.md`, and `ARCHITECTURE.md` now keep strict CPU and
  default production Metal claims separate. The production fp32 deterministic
  failure is a bounded documented limitation; no fixed threshold changed.
  `make production-evidence` reproduces all three ignored JSON inputs.
- Final P1-1 verification passes **593/593 tests in 530.24 seconds**. No upload,
  hardware, serial port, or robot directory was used. The original T3 failure
  remains byte-identical at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.

## 2026-09-02 — Stage R P1-4 software-only async serving complete

- The installed LeRobot 0.6.1 async service was audited before implementation.
  Its proto3 descriptor SHA-256 is
  `e116fbf44dd1fc65b67ff255c04857000c28e69055211af5ef3df85ac8d81f8d`;
  `ARCHITECTURE.md` records the exact four RPCs, fields, transfer-state values,
  pickle payload classes, 2 MiB chunking, 4 MiB message ceiling, timed-action
  schedule, and seven installed-source hashes.
- `smolvla-mlx serve` now implements that schema with the reference protobuf
  classes and client chunk transport. A one-item newest-observation queue,
  `must_go` and duplicate/similar filtering, `Ready` episode reset, requested
  action slicing, empty-queue timeout, and `1 / fps` timestamp/timestep schedule
  match the pinned server. MLX inference is serialized across concurrent RPC
  workers; queue/lock/latency waits observe cancellation, and invalid setup,
  stream, feature, checkpoint, or inference inputs return explicit gRPC status
  errors.
- The optional dependency boundary is `lerobot[async]==0.6.1` under the
  Python-3.12+ `serve` extra. The CLI imports it only inside the `serve`
  handler. The clean import-isolation subprocess still loads none of gRPC,
  protobuf's `google` package, Torch, LeRobot, or Transformers through the base
  runtime. The wheel-content regression asserts `smolvla_mlx/server.py` ships.
- Security is fail-closed: `127.0.0.1:8080` is the default, any non-loopback
  bind requires `--allow-remote`, observations are capped at 64 MiB, and the
  README states that LeRobot's unauthenticated/unencrypted pickle protocol is
  trusted-peer only. The server contains no robot, serial, camera-capture, or
  motion code. Import-time LeRobot log files are ignored as generated debris.
- The reference `AsyncInferenceStub`, `send_bytes_in_chunks`,
  `RemotePolicyConfig`, and `TimedObservation` drove an ephemeral localhost
  server using the pinned real base checkpoint and recorded
  `tests/golden/sample_000` cameras/state/task. With the opt-in worker-local
  seed `20260831`, the served `[3, 6]` fp32 CPU-Torch chunk has SHA-256
  `46a4b2809975a6f14925db404d55b7595d45df9ef578f1f2cd1ae760ce137981`
  and is exactly equal to three direct `select_action` results: 18/18 elements,
  maximum absolute difference `0.0`, no tolerance.
- The fast and real focused server/CLI/distribution/isolation suite passes
  **22/22 in 14.00 seconds**. The protected complete tree passes **601/601 in
  521.24 seconds** before the final wheel-content assertion was added; that
  assertion is exercised again in the final release full-suite run.
- README commands were checked against the installed LeRobot client's actual
  `--help` surface without constructing a robot. Hardware-in-the-loop remains
  explicitly pending. No hardware, robot directory, serial port, or upload was
  accessed. The original T3 failure remains byte-identical at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.

## 2026-09-02 — Stage R final artifact refresh and release closure

- P1-4 source, tests, protocol audit, README, and initial status were committed
  and pushed as `a50cd3b5720a061262a978130600215a30fb8fbd`. A clean detached worktree
  was advanced to that exact commit before any final artifact build. The four
  earlier P1-2 artifacts were moved intact to the ignored recoverable backup
  `.cache/release-build-backup/pre-a50cd3b`; none was deleted or uploaded.
- With repository-local uv/interpreter caches and
  `MACOSX_DEPLOYMENT_TARGET=14.0`, the refresh produced one sdist and CPython
  3.11/3.12/3.13 arm64 wheels. Their SHA-256 values are respectively
  `f778711e...61b35`, `2a9e1490...9014e`, `4412160a...12b1`, and
  `af93be38...54fd`; exact sizes and full hashes are in `DIST_MANIFEST.md`.
  Every wheel contains `smolvla_mlx/server.py`, declares the guarded
  `lerobot[async]==0.6.1` serve extra, retains its
  `macosx_14_0_arm64` tag, and embeds a project extension whose Mach-O minimum
  is 14.0. The pinned MLX dependency still declares 26.2 and remains the honest
  upstream deployment limitation.
- Four new base environments installed the sdist or matching wheel on CPython
  3.11.15, 3.12.13, and 3.13.14. From outside the checkout, all imported from
  their environment, used `native-reference`, kept gRPC/protobuf/Torch/LeRobot/
  Transformers outside the base import graph, and emitted a finite six-value
  action from the retained real observation with both Hub offline flags set.
- A fifth fresh CPython 3.12 environment installed the wheel with `.[serve]`,
  imported the packaged server, reproduced descriptor SHA-256
  `e116fbf4...1f8d`, bound an ephemeral loopback port, completed the reference
  `Ready` RPC, stopped cleanly, and rendered the installed serve help. No
  checkpoint, external service, hardware, robot directory, or serial port was
  used by that smoke.
- The Stage R closing `make test` run passes **601/601 tests in 537.54
  seconds**. All P0 and P1 acceptance criteria are green. `STATUS_RELEASE.md`
  records the package matrix, cache reduction, limitations, safety boundary,
  and no open human tasks, ending in the required `RELEASE READY` milestone.
  Nothing was uploaded.

## 2026-09-02 — Stage T4 native training UX and exact resume complete

- Added a lazy optional `smolvla-mlx train` command with a required `--lora` or
  `--full` mode, dataset repo ID or local path, steps, effective batch size,
  learning rate, output, checkpoint cadence, and resume controls. The `train`
  extra now pins LeRobot 0.6.1 and Torch 2.11.0 on Python 3.12+; base import
  isolation remains unchanged.
- T4 lives in a separate `training/ux.py` layer so the completed T3/T3B runner,
  launch schema, gates, and failure records stay untouched. It reuses the
  audited MLX loss/gradient/optimizer/checkpoint/export primitives and extends
  the bridge only with explicit dataset identity/root/revision inputs. Dataset
  statistics now support every sorted LeRobot data shard rather than assuming
  one Parquet file.
- Full mode applies LeRobot's reference trainable policy—state projection plus
  complete action expert—and proves 155 fp32 master tensors / 99,880,992
  scalars. Expert-only LoRA proves 112 adapters / 224 fp32 tensors / 1,708,032
  scalars. AdamW state contains exactly two moments per trainable in both modes.
- The first full preflight reached export and exposed JSON tuple/list metadata
  identity; canonical JSON metadata fixed it and a fresh preflight exported all
  500 tensors and emitted a finite six-value action. The first full resume
  attempt then failed closed because AdamW had promoted bf16 live trainables to
  fp32 while reconstruction began bf16. Full mode now creates explicit fp32
  master parameters before optimizer initialization. Both failed attempts are
  preserved under ignored `.cache/training` paths.
- The fixed 100-versus-50+resume harness passed for real expert-only LoRA and
  real full mode. Both report parameter max absolute `0.0`, per-step loss max
  absolute `0.0`, all-numerical-metric max absolute `0.0`, exact optimizer
  tensors, exact serialized draw chains, exact sampler state, and exact
  canonical step state. The immutable `1e-6` parameter and `1e-7` loss gates
  were not changed.
- LoRA's first-ten/last-ten mean loss is
  `0.9695772379636765`/`0.615369763597846`; full mode's is
  `1.8326249837875366`/`0.5984157636761666`. Both direct and resumed runs retain
  exactly steps 50/75/100, export 500 fp32 tensors, reload through the public
  MLX policy, and emit identical finite action hashes per mode. Smoke peak
  observations are 3,818,770,536 bytes for LoRA and 4,603,826,668 bytes for
  full; they are explicitly non-benchmark evidence because snapshot copying,
  export, and action validation are included.
- Evidence hashes are
  `44325aa73c012d5b9dfb5499a549eeb689b90c64ebd07b137ee024cefa797b57`
  (LoRA) and
  `2c46c621a08b59584701b1bc2171690cfc03c7a116e41d4e4fff35f217699748`
  (full). Focused CLI/training/distribution verification passes **22/22**;
  compatibility runs pass **152/152** and **137/137** across the complete
  legacy T3B, dataset, and T4 surfaces. `TRAINING_UX.md` records commands,
  semantics, gates, hashes, limitations, and results. The closing complete
  repository suite passes **608/608 tests in 533.65 seconds**.
- No upload, hardware, robot directory, or serial port was used. The original
  T3 failure remains byte-identical at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.

## 2026-09-02 — Stage T5 native training documentation and benchmark complete

- Froze the native training benchmark implementation and four-cell protocol in
  clean commit `0d897449b06d114d536756f2ed6850b52fd5bda4`: expert-only LoRA/full
  reference trainables crossed with bf16/fp32 base storage, effective batch
  eight, 3 excluded warmups, 10 measured updates, 3,000-step scheduler horizon,
  and learning rate `1e-4`. The pre-measurement check at
  `2026-09-02T04:44:39.438886+00:00` found no trainer, floor worker, test suite,
  or competing benchmark.
- The idle Metal medians are 1.145481625/1.093664292 seconds for LoRA
  bf16/fp32 and 1.195801812/1.167098813 seconds for full bf16/fp32. These equal
  0.873/0.914 and 0.836/0.857 updates/s. Peak MLX memory is 2.27/3.24 GiB and
  3.55/4.32 GiB, respectively. The 30,000-update LoRA bf16 optimizer-work
  projection is 9.55 hours; serialization, export, evaluation, and recording
  overheads remain explicitly excluded.
- Added `TRAINING_BENCHMARK.json` (SHA-256
  `bca3ad9d0c2285fa70f4083885a6a6708e8c9d98b6c999d4cabd87b061cef07a`),
  a tracked, path-sanitized record whose values and derivations are checked
  against the complete ignored source artifact. The latter is
  `.cache/training/t5-benchmark.json`, SHA-256
  `7112806471e55e55d98ae101bc2af8172c2cc18f01b3e0c2c0646446adba9423`.
- `BENCHMARK.md` publishes the full matrix and methodology. `README.md` now
  gives an exact local record → overnight native train/resume → morning predict
  workflow, honest M5 Pro budgets, the full-mode alternative, and the measured
  Torch round-trip proof. Tests mechanically trace every documented benchmark
  number to the tracked JSON record.
- The complete exact T5 repository suite passes **613/613 tests in 520.69
  seconds**. No upload, hardware, robot directory, or serial port was used. The
  original T3 failure remains byte-identical at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.

## 2026-09-02 — Stage Q P2-1 comparative protocol frozen before timing

- Added an initially red five-test contract for the first quality package,
  then implemented the minimum protocol and coordinator needed to make it
  green. Focused comparison/reference/benchmark/import-isolation coverage now
  passes **12/12 in 17.75 seconds**.
- The immutable comparison is the pinned base checkpoint and saved
  `sample_000` observation/noise, fp32 on both engines, 5 excluded warmups and
  50 synchronized measured chunks. Both standalone workers measure the same
  preprocessing-through-normalized-action-chunk boundary. The PyTorch worker
  clears inherited MPS switches and enables only fallback before importing
  Torch or LeRobot.
- The coordinator refuses a dirty tracked tree, less than 40 GiB free, a
  competing trainer/floor/test/benchmark process, output overwrite, incomplete
  input, or an output outside the repository. The machine-readable result will
  retain every raw timing, recompute medians/p95/rates, bind input and source
  hashes, and record environment and idle evidence. This source checkpoint is
  intentionally committed before running `make inference-comparison`.

## 2026-09-02 — Stage Q P2-1 MLX/PyTorch-MPS comparison complete

- From clean, pushed protocol commit
  `e210f7b76ae8657390a8101b76ee5815df1b15ab`, the idle check at
  `2026-09-02T05:16:10.169655+00:00` found no trainer, floor worker, test suite,
  or competing benchmark. Both engines then ran sequentially in fresh worker
  processes on the same M5 Pro.
- Native MLX fp32 measured 110.75147850351641 ms median, 111.19723780211643 ms
  p95, 9.02922483304139 chunks/s, and 3,157,857,042 bytes of reported peak/
  active framework memory. Pinned LeRobot/PyTorch-MPS fp32 measured
  204.5789789990522 ms median, 206.45897294743918 ms p95,
  4.888087744365138 chunks/s, and 2,321,498,112 bytes of maximum sampled MPS
  driver allocation. The bounded median speedup is 1.847189597496495×.
- `INFERENCE_COMPARISON.json` preserves all 100 raw timings, environment, fixed
  observation/noise hashes, complete source hashes, and the clean source
  commit. Its SHA-256 is
  `115ad58c0c618b65a6275018614f3ee6cf17dd02a9d4ad9c94aaf7e5a9842e48`.
  `BENCHMARK.md` records the common boundary, fallback environment, exclusions,
  single-case limitation, and non-equivalent allocator semantics.
- Publication coverage passes **14/14 in 17.68 seconds**. The package-closing
  complete repository suite passes **620/620 tests in 522.71 seconds**. No
  tolerance, upload, hardware, robot directory, or serial port was touched;
  `FAILURE_LORA_FINETUNE.md` remains byte-identical at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.

## 2026-09-02 — Stage Q P2-2 bf16 profile protocol frozen before timing

- Added a four-test red contract, then implemented the fixed component matrix;
  the profile/benchmark/import-isolation focus now passes **7/7 in 0.24
  seconds**. No model timing was run while the protocol tree was dirty.
- The profile uses isolated fp32 and bf16 production-Metal workers on the same
  pinned `sample_000` observation and noise, with 5 excluded warmups and 50
  measured iterations per dtype. Every iteration synchronizes six boundaries:
  preprocessing, vision encoder, connector, prefix, ten-step expert loop, and
  end-to-end total.
- The validator retains all 600 raw durations, recomputes every median/p95,
  derives the total slowdown and per-component deltas/shares, and binds the
  exact input, source, clean commit, environment, idle declaration, and memory
  counters. The coordinator inherits the proven P2-1 fail-closed disk,
  worktree, process, path, and no-clobber checks. This protocol is committed
  before `make profile-bf16` produces any measurements.

- The first post-commit attempt ran both isolated workers but published no
  artifact: canonical `sort_keys=True` worker JSON reordered the stage mapping,
  while the validator mistakenly required insertion order as well as exact
  membership. A new red JSON-round-trip regression reproduced the issue. The
  validator now enforces the same exact six-key set independent of irrelevant
  object-key order; counts, values, summaries, dtype order, and every other
  protocol field remain unchanged. A corrected clean commit is required before
  rerunning timings.

## 2026-09-02 — Stage Q P2-2 bf16 latency diagnosis complete

- The corrected clean protocol commit
  `adf40e62a7b652262fc08d7ed6449b4c60a0773d` produced the non-overwriting
  profile at `2026-09-02T05:33:56.285086+00:00`; the process preflight found no
  trainer, floor worker, pytest, or competing benchmark. `BF16_PROFILE.json`
  has SHA-256
  `74da9f937cb8bfeba4066d5518187490ff96a1447e4a2ad2253e2493245be1cf`.
- The synchronized total medians are 110.7135835045483 ms fp32 and
  130.2171044953866 ms bf16: +19.503520990838297 ms / +17.616195207010946%.
  Component deltas are +14.51791700674221 ms expert (74.4374% of the total),
  +2.90929050242994 ms prefix (14.9167%), +1.7123540019383654 ms vision
  (8.7797%), +0.3240629957872443 ms connector (1.6616%), and
  +0.013917502656113356 ms preprocessing (0.0714%). Component medians explain
  99.87% of the total delta. Profile memory is 2.9275 GiB fp32 versus 2.4373
  GiB bf16.
- A direct dtype trace found bf16 checkpoint weights but fp32 preprocessed
  pixels/state/noise and fp32 vision, connector, prefix/cache, and velocity
  outputs. The evidence therefore identifies mixed-dtype projection-heavy
  execution as the boundary of the slowdown and is consistent with MLX 0.32.2
  conversion/kernel behavior; it does not prove a particular private Metal
  kernel. Casting activations would change the protected arithmetic path, and
  upcasting weights would erase the memory benefit, so no unproven optimization
  or default change was made. The exact profile is designed to be rerun after
  an MLX upgrade.
- Artifact/publication coverage passes **10/10 in 0.24 seconds**. The complete
  P2-2 repository suite passes **627/627 tests in 533.26 seconds**. No tolerance,
  upload, hardware, robot directory, or serial port was touched; the original
  T3 failure remains byte-identical at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
