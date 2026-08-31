# Vision Metal precision limitation

## Status

The native `VisionEncoder` is mathematically validated against the immutable
CPU/fp32 reference goldens on MLX's CPU backend. Its fp32 test is below the
Section 6 limits (sample 000: relative L2 `6.996e-06`, max absolute
`3.338e-04`); all eight golden samples run in both converted fp32 and compact
bf16-weight modes with the original thresholds unchanged.

The default MLX Metal backend does not currently meet the same CPU-reference
thresholds on this M5 Pro. This is a documented limitation, not a relaxed test
or a change to `BRIEF.md`.

## Evidence and ruled-out hypotheses

1. **Conversion/layout error — ruled out.** Direct comparison of converted
   patch-convolution, position, attention, and normalization tensors to the
   reference model gave zero elementwise weight difference. Patch embeddings
   themselves are within relative L2 `4.266e-07` and max absolute
   `2.846e-06`.
2. **Attention implementation — ruled out.** At vision layer 0,
   `mx.fast.scaled_dot_product_attention` was closer to the reference than an
   explicit MLX `matmul`/`softmax` implementation, but both exceeded the fixed
   limit after the first residual layer.
3. **Linear reduction and layer-normalization formula — ruled out.** Splitting
   each 768-wide linear reduction into blocks from 384 down to 8 produced the
   same output envelope. MLX's built-in LayerNorm, centered variance, raw
   second-moment variance, and `mx.var` all produced layer-0 relative L2 about
   `7.03e-07` against CPU PyTorch; this small reduction-order difference is
   amplified by the following projections.
4. **Device and bf16 isolation — confirmed.** MLX CPU fp32 meets the contract.
   Metal fp32 yields vision relative L2 `8.44e-03` to `9.73e-03` and max
   absolute `0.219` to `0.421` across the eight samples. Pure bf16 activations
   add expected quantization error (CPU relative L2 up to `4.91e-02`), so the
   validated compact mode stores bf16 parameters but uses fp32 activations,
   matching the reference's own bf16-checkpoint-to-fp32-golden treatment.

## Consequence and next step

The strict golden suite explicitly selects MLX CPU, matching the CPU PyTorch
reference device. The runtime remains dependency-isolated MLX and can execute
on Metal, but Metal vision parity remains an open performance/correctness issue
for Phase 5. Do not loosen `BRIEF.md` Section 6. Revisit only with an MLX Metal
precision control or a custom kernel that can reproduce CPU reduction behavior.
