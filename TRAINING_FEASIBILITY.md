# SmolVLA MLX Training Feasibility

## Decision

**Stage T0 gate met for the audited reference-default training path.** The full
native vision → connector → 16-layer VLM prefix → state projection → 16-layer
action-expert architecture completed a synchronized random-weight forward and
backward step on Metal. Every parameter selected by the reference defaults
(`state_proj` plus the action expert) had a present, shape-matching, finite,
nonzero gradient.

This is a feasibility result, not training-parity or fine-tuning evidence. T1
must still compare real-checkpoint loss and gradients against the pinned
PyTorch reference.

## Reproducible command and artifact

```bash
export HF_HOME="$PWD/.cache/hf"
export UV_CACHE_DIR="$PWD/.cache/uv"
export SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx"
make training-audit
```

The command writes `.cache/training/t0-audit.json`. The recorded Stage T0
artifact has SHA-256:

```text
88dacde30996c2d9cbad90681204e2583c92f6d51f0f3747c6e37e57b709fd51
```

## Measured configuration

| Field | Value |
| --- | --- |
| Machine | Apple M5 Pro, 48 GiB unified memory |
| Device | `Device(gpu, 0)` |
| macOS | 26.5.2 |
| Python | 3.12.13 |
| MLX | 0.32.2 |
| Parameter storage | bf16 |
| Loss arithmetic | fp32 |
| Seed | 0 |
| Microbatch | 1 observation |
| Cameras | 2 × `3×512×512` |
| Language | 48 tokens |
| State | 6 physical dimensions, padded to 32 |
| Actions | `1×50×32`; loss over physical dimensions `0:6` |
| Flow draw | fixed `t=0.5`, deterministic random actions/noise |

The audit uses random model weights and a deterministic synthetic batch. All
real SmolVLA modules execute; no model component is mocked.

## Gradient result

| Metric | Result |
| --- | ---: |
| Scalar loss | 2.7361748218536377 |
| Trainable tensors | 155 |
| Trainable scalars | 99,880,992 |
| Gradient tensors | 155 |
| Missing or shape-mismatched gradients | 0 |
| Non-finite gradients | 0 |
| Zero-norm gradients | 0 |
| Maximum absolute gradient | 1.4765625 |

The selected names are flattened from the MLX container and mapped one-to-one
to the existing checkpoint convention. Action input/output/time projections
remain checkpoint-root `action_*` names; expert block tensors remain under
`expert.*`; state projection stays under `state_proj.*`. Duplicate canonical
names are fatal.

## Resource result

| Metric | Result |
| --- | ---: |
| Synchronized forward | 119.918 ms |
| Synchronized forward + backward | 196.799 ms |
| Active MLX memory after gradient evaluation | 1,108,300,302 bytes (1.032 GiB) |
| Peak MLX memory | 2,509,594,126 bytes (2.337 GiB) |
| Disk free before | 591,311,265,792 bytes (550.702 GiB) |
| Disk free after | 591,311,073,280 bytes (550.701 GiB) |

The run remains far above the mandatory 40 GiB free-space floor. These numbers
measure the reference-default trainable subset at microbatch 1; optimizer state,
LoRA/full-fine-tune checkpoint retention, real data decoding, and batch 8 are
not included. T3 must remeasure after those allocations exist.

## Native RMSNorm decision

The `_rmsnorm_native` extension and the CPU reference softmax/RoPE/SiLU helpers
exist to reproduce PyTorch CPU inference arithmetic. They expose no MLX VJP and
are therefore **excluded from the training autodiff path**.

- The v0.1 CPU inference compatibility path remains unchanged.
- T0 trains on Metal, where the existing modules select differentiable MLX
  kernels (`mx.fast.rms_norm`, MLX softmax/RoPE arithmetic, and MLX SiLU).
- `training.differentiable.differentiable_rms_norm` records the pure-MLX fp32
  RMSNorm formula and has tested finite input/weight gradients.
- T1 must use pure MLX differentiable operations for its CPU comparison lane;
  it must not route gradients through the native extension or weaken an
  inference tolerance.

Providing a custom native VJP is unnecessary for the first training milestone:
the additive pure-MLX lane preserves runtime isolation and avoids coupling
training correctness to an inference-only arithmetic shim.

## Data-path inventory

The existing repository has two pinned reference-only loading paths:

1. `reference.policy.load_dataset_observation` constructs LeRobot's
   `LeRobotDataset` for `lerobot/svla_so101_pickplace` at revision
   `f641879e22172be7e8161d5e6c1503c2d2feb657`, selects whole episodes, maps the
   `side` and `up` cameras, state, task, and action, and returns CPU tensors.
2. `smolvla_mlx.cli._dataset_observation` keeps runtime imports isolated by
   launching a child process that loads one LeRobot frame and crosses back as
   NumPy `.npz` arrays plus JSON task text.

T1 starts with the same bridge model: optional/reference code owns LeRobot and
Torch, while `training/` consumes plain contiguous arrays. The training batch
contract adds actions, `action_is_pad`, episode/frame identity, processed image
and language masks, normalization metadata, and caller-supplied flow draws.
Stage T3 will split by whole episode with at least 15% held out; adjacent frames
from one episode may never cross the train/evaluation boundary.

The current v0.1 prefix assembly accepts one observation at a time. T1 can use
microbatch 1 for parity. Before T3, the training layer must either add native
batched prefix assembly or use gradient accumulation to reach the effective
batch budget without altering the v0.1 API.

## T1 gradient-parity artifact design

The ignored artifact root will be `.cache/training/gradient_goldens/`. One
manifest-backed capture contains:

- source IDs/revisions, LeRobot/PyTorch/MLX versions, CPU/fp32 device contract,
  seed, episode/frame identity, and normalization behavior;
- processed camera tensors, pixel masks, token IDs, text masks, state, padded
  actions, and `action_is_pad`;
- the exact serialized beta-sampled timestep and Gaussian noise consumed by
  both frameworks;
- noisy actions, target velocity, predicted velocity, physical-action loss,
  and scalar loss;
- every trainable parameter and gradient under the canonical names proven in
  T0;
- for each tensor: dtype, shape, byte count, and SHA-256, plus a manifest hash.

The reference capture runs the actual LeRobot training forward on CPU/fp32.
The MLX check consumes the serialized arrays without resampling. It reports the
worst five tensors and enforces loss relative difference `≤ 1e-4`, per-tensor
gradient relative L2 `≤ 1e-2`, and cosine similarity `≥ 0.999` without changing
those gates.

## T0 conclusion and next gate

MLX autodiff and memory are not blockers for SmolVLA's default expert/state
training path on this machine. Stage T1 is the next unblocked work: real-batch
step-zero loss/gradient parity against the pinned PyTorch reference. Stage R
remains independently blocked until the exact `BRIEF_RELEASE.md` is supplied.
