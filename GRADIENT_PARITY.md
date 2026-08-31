# SmolVLA Step-Zero Gradient Parity

## Decision

**Stage T1 passes.** On the fixed real LeRobot training example, the native
MLX CPU/fp32 training path matches the pinned PyTorch CPU/fp32 reference loss
and every one of its 155 selected gradients within the immutable gates.

This result covers the actual checkpoint-backed forward and backward step. It
does not yet prove optimizer-state evolution; Stage T2 owns that lockstep gate.

## Reproduce

```bash
export HF_HOME="$PWD/.cache/hf"
export UV_CACHE_DIR="$PWD/.cache/uv"
export SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx"
make training-goldens
make training-parity
```

The first command uses the optional PyTorch/LeRobot reference dependencies and
writes `.cache/training/gradient_goldens/`. The second command uses native MLX
and writes `.cache/training/t1-parity.json`. Both locations are repository-local
and ignored by Git.

## Fixed reference case

| Field | Value |
| --- | --- |
| Checkpoint | `lerobot/smolvla_base` @ `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| Base VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Dataset | `lerobot/svla_so101_pickplace` @ `f641879e22172be7e8161d5e6c1503c2d2feb657` |
| Episode / frame / absolute index | `0 / 100 / 100` |
| Draw seed | `20260831` |
| Device and arithmetic | CPU / fp32 in both frameworks |
| Model-ready batch | 2 cameras, 48 tokens, 32-wide state, `50×32` actions |
| Physical loss width | 6 action dimensions; all 50 timesteps valid |
| Trainable selection | 155 tensors; 99,880,992 scalars |

The reference capture mirrors the installed LeRobot 0.6.1 path: default batch
collation, uint8 camera conversion through `/255`, camera renaming, public
dataset normalization statistics, tokenizer processing, and the actual
SmolVLA flow-matching forward. Noise and beta-sampled timestep are serialized;
MLX never resamples them.

## Artifact integrity and structural gates

The golden artifact contains 324 independently hashed NumPy payloads and is
805,554,153 bytes. Two complete captures produced the same manifest SHA-256:

```text
b029a0ed66312e785cb8aa3f1db0affb16c9502ad7b5d0fe0feea3177bf8c145
```

Before autodiff, the MLX runner requires exact equality for every selected
checkpoint parameter. All 155 names form the same canonical bijection, all
shapes and dtypes match, and all fp32 values are exactly equal. Missing, extra,
renamed, non-fp32, shape-mismatched, or value-mismatched tensors abort before
the numerical parity thresholds are evaluated.

## Immutable gate result

| Metric | Result | Required gate |
| --- | ---: | ---: |
| PyTorch loss | 2.101923942565918 | reference |
| MLX loss | 2.101925849914551 | candidate |
| Loss relative difference | 9.074298999059449e-7 | ≤ 1e-4 |
| Gradient tensors passing | 155 / 155 | 155 / 155 |
| Maximum gradient relative L2 | 8.673578115837066e-6 | ≤ 1e-2 per tensor |
| Minimum gradient cosine | 0.9999999999623879 | ≥ 0.999 per tensor |

All metric accumulation is float64 on the host. Non-finite candidate or
reference values, zero-norm reference gradients, and shape mismatches are hard
errors.

## Five largest relative-L2 differences

| Tensor | Relative L2 | Cosine | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| `action_time_mlp_in.weight` | 8.673578115837066e-6 | 0.9999999999623879 | 3.1888484954833984e-6 |
| `expert.layers.1.input_layernorm.weight` | 6.656345209857855e-6 | 0.9999999999789950 | 4.912726581096649e-8 |
| `expert.layers.1.self_attn.q_proj.weight` | 6.3123810635703525e-6 | 0.9999999999808424 | 2.3562461137771606e-7 |
| `expert.layers.9.input_layernorm.weight` | 5.851465222097282e-6 | 0.9999999999864958 | 6.286427378654480e-8 |
| `expert.layers.1.self_attn.k_proj.weight` | 5.559739433938906e-6 | 0.9999999999855081 | 5.997717380523682e-7 |

The complete ordered set of 155 comparisons, plus a separate lowest-cosine
view, is stored in `.cache/training/t1-parity.json`. Its recorded SHA-256 is:

```text
f4da0c16771a462e45bd615728bc02a059633db19eb77883342203426cb4d634
```

## Resource evidence

| Metric | Result |
| --- | ---: |
| MLX synchronized forward + backward | 1.161 seconds |
| Total load, integrity, identity, and parity validation | 2.169 seconds |
| Active MLX memory | 1,981,369,020 bytes (1.845 GiB) |
| Peak MLX memory | 2,173,315,618 bytes (2.024 GiB) |
| Disk free before | 582,902,222,848 bytes (542.870 GiB) |
| Disk free after | 582,902,194,176 bytes (542.870 GiB) |

The disk floor remains 40 GiB. No PyPI or Hugging Face upload, credential,
robot environment, vendor fork, serial port, or hardware access was involved.

## Runtime isolation

The inference package is unchanged. CPU parity temporarily redirects only the
runtime aliases for RMSNorm, split-half RoPE, softmax, and SiLU to pure MLX
formulas with VJPs; an exception-safe locked context restores every original
callable after the step. `training/__init__.py` remains side-effect-free, and
importing the base runtime plus `training` loads none of Torch, LeRobot, or
Transformers.

## Conclusion

The checkpoint conversion, real preprocessing path, flow-matching objective,
and complete step-zero derivative agree. Stage T2 optimizer lockstep and Stage
T3 outcome-based LoRA fine-tuning are now independently eligible.
