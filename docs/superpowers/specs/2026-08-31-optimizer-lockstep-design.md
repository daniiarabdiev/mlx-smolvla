# Stage T2 Optimizer Lockstep Design

## Goal

Prove that native MLX reproduces the pinned LeRobot SmolVLA optimizer path for
the first 25 updates of a normal training run: the same real processed batch,
the same serialized flow draws, the same global gradient clipping, AdamW
semantics, and the same cosine-with-warmup learning-rate schedule.

The immutable gates are:

- every one of 25 pre-update losses has relative difference `<= 1e-3`; and
- every one of 155 final selected parameter tensors has relative-L2 drift
  `<= 5e-3`.

Tolerances are never adjusted in response to a result.

## Audited upstream behavior

The installed LeRobot 0.6.1 code and pinned checkpoint config establish:

| Semantic | Reference behavior |
| --- | --- |
| Optimizer | `torch.optim.AdamW` |
| Peak learning rate | `1e-4` |
| Betas | `(0.9, 0.95)` |
| Epsilon | `1e-8`, added after bias-corrected `sqrt(v)` |
| Weight decay | decoupled, multiplicative `parameter *= 1 - lr * 1e-10` |
| Bias correction | enabled for both moments, step count starts at one |
| Gradient clipping | global L2 norm over all present gradients, maximum `10`, denominator epsilon `1e-6` |
| Update order | backward, clip, optimizer step, zero gradients, scheduler step |
| Scheduler | LeRobot `CosineDecayWithWarmupSchedulerConfig` |
| Warmup / decay | 1,000 / 30,000 steps |
| Decay floor | `2.5e-6` |
| Default training horizon | 100,000 updates |

The step-zero T1 gradients have a PyTorch global norm of
`43.20928955078125`, so clipping is active and therefore part of the real gate.

## Scheduler horizon decision

The 25 steps are an observation window into the default 100,000-step training
configuration, not a replacement `TrainPipelineConfig.steps=25` run. This
preserves the actual 1,000-step warmup and records learning rates from
`9.990009990009991e-8` at update 0 through
`2.4975024975025017e-6` at update 24.

Passing `num_training_steps=25` to LeRobot would activate a special scheduler
auto-scaling branch, truncate warmup to zero (`int(1000 * 25 / 30000)`), and
start immediately at `1e-4`. That would test a short-run edge case rather than
the requested SmolVLA fine-tuning schedule. Both behaviors were measured
before this choice was fixed.

## Data and draw contract

T2 reuses the exact T1 real processed batch at episode/frame/absolute index
`0/100/100`. Repeating one non-padded real batch intentionally isolates
optimizer evolution from data-loader variation; weights and stochastic flow
inputs still change at every step.

The PyTorch capture seeds once with `20260831`, then serializes one Gaussian
noise tensor and one beta-sampled timestep tensor for every step. MLX consumes
those 25 pairs without sampling. The T2 metadata binds the T1 manifest hash,
and capture fails if the model-ready batch or initial selected parameters differ
from T1.

## Artifact layout

`.cache/training/optimizer_goldens/` uses the existing atomic, hash-verified
training artifact format. It stores:

- 25 `draws/<step>/noise` and `draws/<step>/timesteps` pairs;
- per-step reference loss, LR used, next LR, unclipped global norm, and clip
  coefficient; and
- all 155 final selected fp32 parameters under canonical checkpoint names.

Metadata records all source pins, Python/library versions, optimizer and
scheduler fields, the 100,000-step scheduler horizon, 25-step observation
window, exact update order, T1 manifest binding, scalar counts, disk, and
timing. The expected payload count is 330.

## MLX implementation

Training-only `training/optimizer.py` owns:

- the audited immutable optimizer/scheduler configuration;
- an exact host-visible learning-rate function matching LeRobot;
- global-norm clipping over the MLX gradient tree; and
- an MLX AdamW wrapper using decoupled pre-update weight decay and enabled
  moment bias correction.

The implementation may reuse `mlx.optimizers.AdamW` only after small real
Torch-vs-MLX tests prove step numbering, decay ordering, epsilon placement,
bias correction, schedule ordering, and clipping. The base inference package
does not import or expose optimizer code.

`training/lockstep.py` strictly loads both artifacts and the fp32 converted
checkpoint, validates initial parameter identity, performs 25 synchronized CPU
updates inside the scoped pure-MLX autodiff primitive context, and emits every
loss and parameter comparison. Name, dtype, shape, finite-value, count,
artifact-integrity, source-pin, device, and initial-value mismatches are hard
failures before threshold evaluation.

## Failure protocol

If a gate fails, do not change a threshold. Test three concrete hypotheses in
order:

1. schedule/update ordering or step-number mismatch;
2. global-norm reduction or clipping mismatch; and
3. AdamW scalar ordering/epsilon/bias-correction mismatch.

Fix a demonstrated implementation defect and regenerate evidence, or write
`FAILURE_OPTIMIZER_LOCKSTEP.md` with the measurements. T3 remains eligible
because its declared dependency is T1.

## Safety and resource policy

All caches and artifacts remain under the repository. At least 40 GiB free is
required before and after capture/check. No upload, credential, robot tree,
vendor fork, serial port, or hardware access is permitted. The artifact and
report remain ignored; source, tests, design, and evidence reports are
committed and pushed package by package.
