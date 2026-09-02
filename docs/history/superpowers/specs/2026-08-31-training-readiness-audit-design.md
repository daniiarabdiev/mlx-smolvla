# Stage T0 Training-Readiness Audit Design

## Goal

Prove, without changing runtime behavior, whether the audited SmolVLA MLX
architecture can execute a realistic full forward/backward step on this M5 Pro;
record finite-gradient, latency, memory, disk, RMSNorm, data-path, and T1
artifact-design evidence in `TRAINING_FEASIBILITY.md`.

## Scope

T0 is an audit, not a training implementation. It adds a training-only optional
extra, small isolated utilities under `training/`, a deterministic random-weight
smoke script, tests for parameter selection and differentiable primitives, and
the feasibility report. It does not load robot code, alter inference kernels,
generate T1 reference gradients, add an optimizer, or train a checkpoint.

## Execution design

The smoke path instantiates the existing vision, connector, truncated language,
state projection, and action expert modules with deterministic random weights.
It executes the visual/prefix/action loss path on Metal using two 512×512 camera
images and a 50×32 action chunk. The initial realistic microbatch is one policy
observation because the current v0.1 prefix API is intentionally single-sample;
the report records whether batch expansion is required before T1/T3.

The differentiated loss is one flow-matching training evaluation, not the
ten-step inference sampler: supplied actions, noise, and a timestep produce
`x_t = t * noise + (1 - t) * actions`, target velocity
`u_t = noise - actions`, predicted velocity from the full prefix/expert path,
and mean squared error over the six physical action dimensions.

## Differentiability policy

`ReferenceRMSNorm` and the native reference softmax/RoPE/SiLU extension exist to
reproduce PyTorch CPU inference arithmetic and do not expose a training VJP.
T0 excludes those native calls from autodiff. The smoke path uses pure MLX
equivalents and checks their gradients directly. The v0.1 inference code and
strict golden route remain unchanged.

The audited default trainable set is the action expert plus `state_proj`; the
vision and VLM parameters remain frozen but participate in the forward path.
The finite-gradient gate requires every selected tensor to have a present,
shape-matching, finite gradient. It reports tensor count, scalar count, zero-
norm count, and worst gradient magnitudes. A genuinely unused selected tensor
is a failure, not silently filtered.

## Resource measurement

The script synchronizes MLX before and after the step, excludes module
construction from latency, resets peak memory immediately before the measured
step, and records:

- device, Python, macOS, and MLX versions;
- microbatch, cameras, image/action shapes, dtype, and seed;
- trainable/frozen tensor and scalar counts;
- forward/backward/evaluation wall time;
- active and peak MLX memory;
- disk free before and after plus cache sizes.

The results are emitted as deterministic-schema JSON under `.cache/` and copied
as exact numbers into `TRAINING_FEASIBILITY.md`.

## Data and T1 artifact design

The report inventories the existing reference child-process/dataset path and
defines the next artifact as a manifest-backed directory containing NumPy or
safetensors values for images, masks, language IDs/masks, state, actions,
action-padding mask, noise, timesteps, scalar loss, gradients, sample identity,
normalization metadata, dtype/shape/hash, and exact source revisions.

The dataset bridge remains reference-only. `training/` consumes plain arrays
and never imports LeRobot or Torch at module import time.

## Tests and gates

- A failing parameter-selection test precedes the selector implementation and
  proves exactly the expert/state projection set is selected.
- Failing differentiable-primitive tests precede the training-only RMSNorm and
  activation implementation and prove finite input/weight gradients.
- A smoke-schema test precedes the audit runner and validates all required
  resource/gradient fields without mocking model components.
- The real full-architecture smoke must finish with every selected gradient
  finite and resource numbers recorded.
- The protected 179-test inference suite and import-isolation test must remain
  green before the package is committed and pushed.
