# Benchmark

Native MLX inference measured after warmup on one real SO-101 golden observation.

## Environment

- Device: `Device(gpu, 0)`
- CPU: `Apple M5 Pro`
- Unified memory: `51539607552` bytes
- macOS: `26.5.2`
- Python: `3.12.13`
- MLX: `0.32.2`
- Commit: `72b1716d3cda72a26bced98721133fbd158a5919`
- Measured runs / excluded warmups: `50` / `5`

## Results

| Storage dtype | Total median ms | Total p95 ms | Preprocess median ms | Vision+connector median ms | Prefix median ms | Expert loop median ms | Peak MLX GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 111.34 | 111.80 | 4.45 | 46.12 | 8.17 | 52.62 | 2.94 |
| bfloat16 | 131.12 | 131.71 | 4.50 | 48.25 | 11.08 | 67.36 | 2.44 |

The original under-200-ms bf16 target remains a target, not a correctness gate.
