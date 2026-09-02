# Stage T1 Gradient-Parity Design

## Purpose and authority

This design implements Stage T1 of `BRIEF_FULL.md` after the completed Stage T0
training-readiness audit. The immutable gates are:

- CPU/fp32 reference and MLX execution;
- scalar loss relative difference no greater than `1e-4`;
- every trainable tensor at gradient relative L2 no greater than `1e-2`; and
- every trainable tensor at cosine similarity at least `0.999`.

The protected inference behavior, its exact CPU primitives, and the
Torch/LeRobot/Transformers import-isolation boundary remain unchanged.

## Fixed reference case

The golden capture uses the pinned public inputs already established by the
repository:

| Field | Fixed value |
| --- | --- |
| Checkpoint | `lerobot/smolvla_base` at `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| Base VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Dataset | `lerobot/svla_so101_pickplace` at `f641879e22172be7e8161d5e6c1503c2d2feb657` |
| Episode / frame / absolute index | `0 / 100 / 100` |
| Draw seed | `20260831` |
| Device / dtype | Torch CPU / fp32 |

Frame 100 is far enough from both episode boundaries that all 50 action targets
are real and `action_is_pad` is false throughout. The temporal mask is still
serialized and the MLX objective is separately tested with padded timesteps so
masking behavior cannot be bypassed by the selected case.

The batch must reproduce the actual LeRobot training path, not the inference
helper:

1. construct `LeRobotDataset` with the policy's observation delta `[0]` and
   action deltas `[0, ..., 49]`, `return_uint8=True`, and the pinned revision;
2. use the normal PyTorch batch collation for microbatch one;
3. divide uint8 camera tensors by 255 exactly where `lerobot_train.py` does;
4. rename `observation.images.side` to `observation.images.camera1` and
   `observation.images.up` to `observation.images.camera2`;
5. load the checkpoint processor with CPU/tokenizer overrides and the public
   dataset statistics override used for non-resume fine-tuning; and
6. run the actual `SmolVLAPolicy.forward` training loss after calling
   `policy.train()`.

The dataset-stat override is significant. The checkpoint's saved inference
processor has only robot-prefixed action statistics and is effectively identity
for this unprefixed runtime, whereas a fresh LeRobot fine-tune injects the
dataset's matching `observation.state` and `action` statistics.

## Golden artifact

The ignored artifact root is `.cache/training/gradient_goldens/`. Capture is
atomic at the file level and uses the v0.1 manifest discipline: every contiguous
NumPy array has a logical name, relative path, shape, dtype, byte count, and
SHA-256. `metadata.json` records the SHA-256 of the finalized manifest.

The artifact contains:

- source IDs and revisions, package versions, device/dtype contract, seed,
  episode/frame identity, camera rename map, and dataset-stat provenance;
- model-ready 512px camera tensors and masks, token IDs and attention mask,
  padded state, padded actions, physical action width, and `action_is_pad`;
- the exact Torch-sampled beta timestep and Gaussian noise;
- noisy actions, target velocity, predicted velocity, per-entry squared error,
  and scalar masked loss;
- every selected parameter and its gradient under the existing canonical MLX
  names; and
- the complete reference-name to canonical-name bijection.

The reference selection must contain exactly the checkpoint-default trainable
set: `state_proj`, the 16-layer action expert, and the four root action/time
projections. The audited count is 155 tensors and 99,880,992 scalars. Missing,
extra, duplicate, non-finite, zero-norm, or shape-mismatched gradients abort
capture.

`scripts/make_training_goldens.py` owns reference generation. It imports Torch
and LeRobot only through the optional training/reference environment. The MLX
checker never resamples and treats the manifest as the sole batch/draw source.

## MLX model and differentiable CPU lane

`SmolVLATrainingModel.from_pretrained` reuses the existing strict fp32
`SmolVLAMLX.from_pretrained` conversion/loader, then composes its already-loaded
vision, connector, language, state projection, and expert modules under the
training container. This preserves one parameter tree and one checkpoint
conversion contract. Before differentiation, every selected MLX parameter must
match its serialized reference parameter exactly.

The inference CPU helpers for RMSNorm, RoPE, softmax, and SiLU intentionally use
a native extension to reproduce PyTorch's inference arithmetic, but that
extension has no VJP. `training.differentiable` therefore provides pure-MLX fp32
forms of those four operations plus an exception-safe, single-process scoped
adapter that redirects only the already-imported language/expert primitive
aliases while the T1 graph is built. The adapter restores every original
callable on exit and refuses non-CPU use.

This is deliberately limited to the isolated parity process:

- no `smolvla_mlx/` file or default behavior changes;
- `training/__init__.py` remains side-effect-free;
- inference outside the scope continues to use the exact native CPU helpers;
- T3 Metal training continues to use the existing differentiable Metal kernels;
  and
- tests prove restoration even when the scoped body raises.

A pre-implementation probe on the fixed case validated this design without
changing repository files: loss relative difference was
`9.074298999059449e-07`, worst gradient relative L2 was
`8.673578115837066e-06`, and minimum cosine was
`0.9999999999623879` across all 155 tensors.

## Exact loss and mask semantics

Both sides consume identical `actions`, `noise`, and `timesteps` and construct:

```text
x_t = t * noise + (1 - t) * actions
u_t = noise - actions
```

The predicted velocity remains 32-wide. Squared error is first cropped to the
six physical action dimensions. Invalid temporal positions from
`action_is_pad` are zeroed, and the scalar denominator is exactly
`valid_timestep_count * physical_action_dim`, clamped to one as in LeRobot. No
padded action dimension or padded timestep contributes to loss or gradients.

## Comparison and report

For reference gradient `r` and MLX gradient `m`, metrics are accumulated in
NumPy float64 without changing the tensors under test:

```text
relative_l2 = ||m - r||_2 / ||r||_2
cosine      = dot(m, r) / (||m||_2 * ||r||_2)
```

The capture rejects zero-norm reference gradients, so neither metric needs a
tolerance-hiding denominator. Loss uses absolute difference divided by the
absolute nonzero reference loss.

`scripts/check_gradient_parity.py` writes
`.cache/training/t1-parity.json`, exits nonzero on any immutable-gate failure,
and always includes the five largest relative-L2 errors plus the five lowest
cosines. `PROGRESS.md` records the official loss, all gate extrema, worst five
tensors, elapsed time, disk before/after, artifact size, and artifact/report
hashes.

## Test-first packages

Implementation is split into independently passing packages:

1. manifest-backed artifact IO and tamper detection;
2. exact temporal/physical masking and metric definitions;
3. scoped differentiable CPU primitives and restoration;
4. real reference batch/capture and canonical gradient bijection;
5. strict fp32 checkpoint composition and MLX parity runner; and
6. official artifact generation, immutable gate, protected full suite,
   documentation, commit, and push.

Each behavioral package begins with a failing test. No tolerance, inference
golden, runtime dependency, or base-package import is changed.
