# Connector Metal precision limitation

## Status

The scale-4 pixel shuffle and projection pass all immutable Section 6 golden
limits on MLX CPU. With converted fp32 or compact bf16 weights and fp32
activations, CPU output is elementwise identical to the saved fp32 connector
goldens for the sampled inputs.

## Evidence

1. **Pixel-shuffle layout — ruled out.** The native reshape/transpose sequence
   is the audited mlx-vlm Idefics3 sequence and CPU output is exactly equal to
   the reference at all eight golden boundaries.
2. **Weight conversion — ruled out.** The connector projection tensor is
   checksum-accounted, has exact shape `[960, 12288]`, and directly matches its
   source value after conversion.
3. **Metal linear accumulation — isolated.** Supplying the *reference* vision
   features directly to the GPU connector produces relative L2 up to
   `3.77e-04` and max absolute difference up to `4.50e-02` in fp32. Thus the
   discrepancy is independent of vision and is in the GPU projection kernel.

## Consequence and next step

The strict golden suite runs the native connector on MLX CPU to match the
CPU/fp32 reference. Metal connector execution is retained for later benchmark
and investigation, but cannot presently be described as Section 6 exact. Do
not loosen the tolerance; revisit during Phase 5 alongside the vision issue.
