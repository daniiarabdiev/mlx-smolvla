# Evidence index

The evidence tree separates measured results and negative findings from the
five-minute README. Fixed gates are not changed by moving or indexing a file.
Some immutable records retain the project's pre-release package name inside
their hashed content. The corresponding format-v1 `artifact_type` values and
domain-separation seeds are compatibility identifiers, not distribution or
CLI branding: producers and validators retain them so existing checkpoints,
resume chains, and hash-bound evidence remain readable. They must change only
with an explicit schema-version bump and migration, never as an in-place text
rename. Current installation instructions use only `mlx-smolvla` and
`mlx_smolvla`.

## Inference correctness

- [`INFERENCE_COMPARISON.json`](INFERENCE_COMPARISON.json): frozen MLX Metal
  versus PyTorch-MPS latency record with raw runs, input/source hashes, and idle
  declaration.
- [`BF16_PROFILE.json`](BF16_PROFILE.json): synchronized fp32/bf16 component
  timings and dtype trace.
- [`QUANTIZATION_EXPERIMENT.json`](QUANTIZATION_EXPERIMENT.json): 50-case
  correctness plus latency/memory evidence for dense bf16, VLM 8-bit, and VLM
  4-bit.
- [`FAILURE_vision.md`](FAILURE_vision.md),
  [`FAILURE_connector.md`](FAILURE_connector.md), and
  [`FAILURE_language.md`](FAILURE_language.md): diagnosed module-level negative
  results retained without tolerance changes.
- [Benchmark narrative](../BENCHMARK.md): strict/production mode tables,
  deterministic/statistical methodology, and scoped performance conclusions.

The base, statistics-active, and pinned public-fine-tune checkpoints are also
covered by the full suite's real-observation deterministic and 50-frame
statistical tests. Their immutable repository/model/dataset revisions and exact
outcomes are summarized in the benchmark and cumulative historical status.

## Compatibility and distribution

- [`MLX_COMPATIBILITY.md`](MLX_COMPATIBILITY.md) and
  [`mlx-compatibility.json`](mlx-compatibility.json): official-wheel/dylib
  inspection and identical gates on MLX 0.32.0, 0.32.1, and 0.32.2.
- [`DIST_MANIFEST.md`](DIST_MANIFEST.md): artifact inventories, hashes, tags,
  and clean-install smoke outcomes; refreshed for every release candidate.
- [`CI.md`](CI.md): why the full hosted macOS workflow is disabled, the local
  fast/full lanes, and exact self-hosted activation requirements.
- [`DOCTOR.txt`](DOCTOR.txt): captured from the fresh cp312 wheel plus `serve`
  extra; records the installed environment, extras, Metal default, empty probe
  cache, and verified compatibility verdict.

## Training

- [`TRAINING_FEASIBILITY.md`](TRAINING_FEASIBILITY.md): op-level MLX audit and
  frozen scope.
- [`GRADIENT_PARITY.md`](GRADIENT_PARITY.md): real step-zero loss and 155-gradient
  comparison.
- [`OPTIMIZER_LOCKSTEP.md`](OPTIMIZER_LOCKSTEP.md): 25-update loss/parameter
  lockstep evidence.
- [`SELF_CONSISTENCY_T3.md`](SELF_CONSISTENCY_T3.md) and
  [`PARITY_PROCEDURE_TRAINED.md`](PARITY_PROCEDURE_TRAINED.md): prospective
  PyTorch floor and chronology-enforced trained-checkpoint evaluator.
- [`FAILURE_LORA_FINETUNE.md`](FAILURE_LORA_FINETUNE.md): unchanged first LoRA
  result and original gates.
- [`LORA_SCOPE_COMPARISON.md`](LORA_SCOPE_COMPARISON.md) and
  [`FAILURE_LORA_FINETUNE_B.md`](FAILURE_LORA_FINETUNE_B.md): second-attempt
  expert-only outcome—fixed gates pass, derived deterministic gate fails.
- [`TRAINING_UX.md`](TRAINING_UX.md): LoRA/full smoke, exact resume, standard
  export, and reload evidence.
- [`TRAINING_BENCHMARK.json`](TRAINING_BENCHMARK.json): frozen four-cell native
  MLX update timings, memory, environment, and source hashes.

## Process-block evidence

- [`FAILURE_RELEASE_SPEC.md`](FAILURE_RELEASE_SPEC.md): retained stop record,
  later resolved when the operator supplied the normative Stage R brief.
