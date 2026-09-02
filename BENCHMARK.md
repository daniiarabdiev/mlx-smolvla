# Benchmark

Strict CPU correctness and default production-Metal correctness/performance are reported separately.

## Environment

- Device: `Device(gpu, 0)`
- Execution mode: `production`
- CPU: `Apple M5 Pro`
- Unified memory: `51539607552` bytes
- macOS: `26.6.2`
- Python: `3.12.13`
- MLX: `0.32.2`
- Commit: `4824db9d289bec1c148a43509f41407c1458ef24`
- Measured runs / excluded warmups: `50` / `5`

## Execution modes

`production` is the public default and owns an MLX Metal device context. `strict` owns an MLX CPU context and uses the compatibility arithmetic validated against the pinned PyTorch CPU goldens. Callers select the latter with `execution_mode="strict"` or CLI `--execution-mode strict`.

## Strict-parity correctness (CPU)

The immutable deterministic limits apply to eight fixed real observations; the statistical limit applies to first-action MAE on 50 pinned real frames.

| Storage dtype | Deterministic result | Fixed max-abs gate | 50-frame MLX/reference MAE ratio | Statistical result |
| --- | ---: | ---: | ---: | ---: |
| float32 | pass (8/8) | 0.005 | 0.9999999969 | pass (`<=1.05`) |
| bfloat16 | pass (8/8) | 0.05 | 1.0000097741 | pass (`<=1.05`) |

## Default-production correctness (Metal)

These are the same observations, noise tensors, reference actions, and unchanged gates, executed through the installed default production mode.

| Storage dtype | Deterministic max abs | Fixed gate | Deterministic result | 50-frame MLX/reference MAE ratio | Statistical result |
| --- | ---: | ---: | ---: | ---: | ---: |
| float32 | 0.0473065376 | 0.005 | fail | 1.0000128000 | pass (`<=1.05`) |
| bfloat16 | 0.0441064835 | 0.05 | pass | 1.0000216963 | pass (`<=1.05`) |

Metal fp32 does not satisfy the strict deterministic contract; this is a recorded negative result, not a tolerance change. Metal bf16 satisfies its fixed deterministic gate, and both production dtypes satisfy the fixed statistical gate.

## Default-production performance (Metal)

| Storage dtype | Total median ms | Total p95 ms | Preprocess median ms | Vision+connector median ms | Prefix median ms | Expert loop median ms | Peak MLX GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 110.54 | 111.41 | 4.41 | 45.87 | 8.11 | 52.21 | 2.94 |
| bfloat16 | 130.44 | 131.25 | 4.41 | 47.98 | 11.00 | 67.10 | 2.44 |

A 50-action chunk represents about 1.67 seconds of motion at 30 fps. The latency table is model-only and excludes capture, transport, and actuation. The original under-200-ms bf16 target remains a target, not a correctness gate.
