# Stage T3 LoRA Fine-Tune Design

## Goal and immutable evidence

Train a real SmolVLA adapter with native MLX on this M5 Pro, merge it into a
standard LeRobot checkpoint, and prove that the result is both useful and
portable. T3 succeeds only when all three brief-defined gates pass:

1. MLX held-out physical-action MAE is at most `0.9 *` the base checkpoint MAE
   on the same deterministic set of at least 50 samples from unseen episodes.
2. The exported checkpoint, loaded by the pinned Torch/LeRobot reference,
   scores within `[0.95, 1.05]` of the MLX MAE on those exact cases and draws.
3. The exported checkpoint passes the existing inference parity ladder with
   normalization statistics active and no relaxed tolerance.

Tolerances, split membership, evaluation cases, and flow noise are fixed before
training. A missed gate is documented; it is never repaired by weakening the
test.

## Split, statistics, and sample order

The pinned `lerobot/svla_so101_pickplace` revision contains 50 episodes and
11,939 frames. NumPy's `default_rng(20260901)` selects eight whole held-out
episodes (16%):

```text
2, 7, 21, 28, 31, 34, 35, 41
```

The other 42 episodes are the only training source. State and action mean,
standard deviation, minimum, maximum, and count are recomputed from their raw
Parquet rows; held-out rows never influence the processors. Visual features
retain identity normalization. The resulting statistics and split are hashed
into a local run manifest.

The bridge uses LeRobot's pinned dataset, delta-timestamp rules, episode-aware
sampler, default collator, and processor pipeline on CPU. It emits owned NumPy
arrays and then drops all Torch objects at the MLX boundary. A deterministic
sampler seed and persisted sampler position make every selected frame
auditable. The native prefix path currently supports one observation, so one
optimizer update accumulates eight sequential microbatch gradients. That is
the requested effective batch size of eight, not eight repeated copies.

## LoRA topology and precision

`training/lora.py` supplies a Torch-free `LoRALinear` around an ordinary MLX
`nn.Linear`. The frozen base calculation is unchanged and the additive branch
is

```text
scale * ((x @ A) @ B), where scale = alpha / rank
```

`A` has shape `[input, rank]`, `B` has `[rank, output]`, `A` uses the standard
uniform `+-1/sqrt(input)` initialization, and `B` starts at zero. The defaults
are rank 8, alpha 16, and no dropout. Base checkpoint tensors are bfloat16 on
Metal while adapter parameters and optimizer moments remain fp32. Operations
that the audited inference implementation deliberately evaluates in fp32 stay
fp32.

The insertion set is explicit and count-checked:

- all q/k/v/o attention and gate/up/down MLP linears in VLM layers 0..15
  (112 adapters);
- the same seven linears in all 16 action-expert layers plus its four
  action/time projections (116 adapters); and
- the state projection (one adapter).

The total is exactly 229 adapters. Vision, connector, embeddings,
normalizations, and the language head stay frozen. Only 458 LoRA tensors may
appear in `trainable_parameters()`.

Merging computes `weight + scale * B.T @ A.T` in fp32, replaces every wrapper
with a plain `nn.Linear`, preserves bias, and verifies that no adapter tensor
survives. A zero-initialized install must be bit-identical to the base output;
merge must reproduce the unmerged output before any long run is allowed.

## Optimizer, RNG, and budget

T3 reuses the T2-proven AdamW/global-clip/scheduler implementation with the
audited SmolVLA defaults: peak LR `1e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`,
decay `1e-10`, global clip 10, warmup 1,000/decay 30,000/floor `2.5e-6` scaled
by the actual run horizon exactly as LeRobot does. Gradients are accumulated as
an fp32 tree, divided by eight, then clipped once and updated once.

MLX's RNG is seeded with `20260901`. Training draws free-running Gaussian
noise and beta-distributed timesteps, as permitted by the brief. Metrics append
one durable CSV row per update: step, loss, smoothed loss, LR, gradient norm,
clip coefficient, elapsed seconds, updates/second, and peak MLX memory.

The nominal budget is 3,000 updates. Before the real run, 3 warm-up plus 10
measured effective-batch updates determine median update time. The selected
count is `min(3000, floor(6900 / median_seconds))`, reserving five minutes for
export/evaluation inside the approximate two-hour ceiling. The measured value
and any reduction are written before training starts. It is never silently
extended after seeing held-out results.

## Export and stats-aware loading

T3 chooses a merged export. `training/export.py` writes atomically into a local
directory:

- all 500 canonical model tensors mapped back to their exact LeRobot names in
  fp32 `model.safetensors` (the patch convolution alone changes OHWI to OIHW);
- the pinned policy `config.json`;
- native LeRobot pre/postprocessor JSON plus safetensors generated through the
  installed processor API using the train-only statistics; and
- an audit manifest containing source revisions, split/stats/run/evaluation
  hashes, tensor counts, LoRA settings, and file checksums.

The MLX training-only loader composes the strict native model with a
stats-aware processor. It reads exact `observation.state.mean/std` and
`action.mean/std` tensors from the standard processor files and performs the
same `(x - mean)/(std + 1e-8)` and `x * std + mean` operations as LeRobot. The
dependency-isolated v0.1 package remains unchanged.

Before outcome scoring, the export must load strictly through both:

- `SmolVLAMLX.from_pretrained(export_dir)` plus the training stats wrapper; and
- pinned `SmolVLAPolicy.from_pretrained(export_dir)` plus processors loaded
  from that directory.

## Evaluation and inference parity

The evaluation manifest fixes 56 cases: seven frames spread through each of
the eight held-out episodes. Each case stores episode/frame/absolute identity,
task, raw target action, and a deterministic 50x32 Gaussian flow-noise array.
Base and fine-tuned MLX evaluations use the same prepared observations, target,
and noise. The metric is mean absolute error over the first predicted physical
action versus the dataset's first target action, in physical units after the
train-only action unnormalizer.

Torch round-trip evaluation uses the exported processors and exact same raw
observations/noise. It does not compare to a separately converted intermediate.

The stats-active parity target uses deterministic held-out cases and compares
processor outputs, normalized action chunks, and unnormalized actions between
the exported Torch and MLX policies. Existing fp32 end-to-end maximum-absolute
tolerance `5e-3` is unchanged. Intermediate P0/P1/P2 boundaries are also
checked where the current golden machinery exposes them; the absent release
brief cannot be used to weaken or skip the T3 export gate.

## Failure and safety policy

If training diverges or a gate fails, first test data/stat identity, adapter
merge/name/layout, then framework preprocessing/noise identity. Fix only a
demonstrated implementation defect. Otherwise write `FAILURE_LORA_FINETUNE.md`
with the frozen result.

All datasets, metrics, checkpoints, and reports stay under `.cache/training/`.
Require at least 40 GiB free before benchmark, run, export, and evaluation. No
upload, credential, robot tree, serial port, vendor fork, or hardware action is
part of T3.
