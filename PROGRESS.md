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
