# Language decoder precision boundary

## Resolution — 2026-08-31

This boundary is resolved. The native CPU compatibility primitives now match
the pinned PyTorch arithmetic for RMSNorm, the fixed 177-token RoPE prefix,
masked attention softmax, and SwiGLU SiLU. The original reproduction command
now passes all 50 prefix/language cases under the unchanged Section 6
tolerances; the wider focused decoder/isolation suite passes 57/57. No test was
skipped, xfailed, or relaxed.

The diagnosis below is retained as historical context for why the native CPU
primitives exist. It is no longer an active block on the action-expert work.

## Historical status

The native MLX language decoder is functionally implemented and dependency
isolated, but its fp32 raw residual outputs do not meet the immutable Section 6
maximum-absolute tolerance for four of eight golden samples. No tolerance has
been changed and no test is skipped or xfailed.

## Reproduction

```bash
HF_HOME=$PWD/.cache/hf UV_CACHE_DIR=$PWD/.cache/uv \
SMOLVLA_MLX_CACHE=$PWD/.cache/smolvla_mlx \
uv run pytest tests/test_prefix.py tests/test_language.py -q
```

Observed result: 46 passed, 4 failed.

| Sample | Maximum fp32 raw-layer absolute error | Required maximum |
| --- | ---: | ---: |
| `sample_004` | `1.4648438e-03` | `1e-3` |
| `sample_005` | `3.2958984e-03` | `1e-3` |
| `sample_006` | `2.3193359e-03` | `1e-3` |
| `sample_007` | `1.0986328e-03` | `1e-3` |

All four pass the separate fp32 relative-L2 bound. All 25 bf16 focused cases
pass their fixed relative-L2 bound of `3e-2`.

## Boundary trace

- Prefix embeddings, pad mask, attention flags, full 2-D attention mask, and
  position IDs are exact against all goldens.
- fp32 post-RoPE cache differences stay below `4.351139e-05` for keys and
  `1.180172e-05` for values. The final normalized prefix output stays below
  `2.336502e-05` maximum absolute difference.
- The first material raw-residual divergence appears after an MLP, not at a
  mask, RoPE, cache, or weight boundary. In `sample_005`, layer 03, token 134,
  feature 87, the native MLP emits `-1584.9042` while PyTorch emits
  `-1584.9069`, a `2.6855469e-03` difference.
- Source hooks prove that, when supplied the exact PyTorch normalized input,
  native MLX gate/up projections match the source at that layer. The incoming
  normalization differs by only about `5e-7` to `1e-6`; SwiGLU then amplifies
  that perturbation through the unusually large activation.

## Hypotheses tested

1. **Architecture or dataflow mismatch.** Ruled out: the source safetensors
   header contains exactly 16 language layers; strict loading succeeds, prefix
   construction is exact, and cache K/V boundaries are near `1e-5`.
2. **Deferred-evaluation or direct-reference-form RMS expression.** Ruled out:
   materializing every layer and replacing `mx.fast.rms_norm` with
   `mean(x*x)` followed by `rsqrt` produced the same residual errors.
3. **Higher-precision normalization.** Partially improved several samples, but
   fp64 RMS still reached `1.953125e-03` on sample 005 and cannot satisfy the
   fixed `1e-3` maximum.
4. **Alternative numerical formulations.** Ruled out: a GEMM-based RMS
   reduction, fp64 attention, and fp64 SwiGLU projections did not pass every
   sample and in some cases increased the maximum error.
5. **Supported MLX precision control or upgrade.** Ruled out: MLX `0.32.2` is
   the current release, and its documented CPU RMSNorm fallback is the same
   fp32 `mean(square(x))`/`rsqrt` expression with no public reduction-precision
   control. See the upstream
   [RMSNorm source](https://github.com/ml-explore/mlx/blob/main/mlx/fast.cpp)
   and [release record](https://github.com/ml-explore/mlx/releases).

## Historical decision and next step

The implementation retains the source-semantic MLX fp32 model rather than a
sample-specific numerical adjustment. A future resolution needs an MLX CPU
reduction-precision control or a justified native kernel reproducing PyTorch's
RMSNorm arithmetic. Once available, rerun the reproduction command above before
using the prefix cache in expert or end-to-end integration.
