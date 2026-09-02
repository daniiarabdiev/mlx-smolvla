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

## Native training performance (Metal)

The Stage T5 protocol measures one optimizer update over an effective batch of
eight real samples. Model/dataset construction is excluded. Each cell uses
three excluded warmups followed by ten synchronized measured updates with the
same 3,000-step scheduler horizon and `1e-4` learning rate. The four cells ran
sequentially on an otherwise idle machine; the pre-measurement process check at
`2026-09-02T04:44:39.438886+00:00` found no trainer, floor worker, test suite,
or competing benchmark.

| Training mode | Base storage | Median update s | p95 update s | Steps/s | Minutes / 1k steps | Projected 3k minutes | Peak MLX GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expert-only LoRA | bf16 | 1.145 | 1.150 | 0.873 | 19.09 | 57.27 | 2.27 |
| Expert-only LoRA | fp32 | 1.094 | 1.098 | 0.914 | 18.23 | 54.68 | 3.24 |
| Full reference trainable set | bf16 | 1.196 | 1.204 | 0.836 | 19.93 | 59.79 | 3.55 |
| Full reference trainable set | fp32 | 1.167 | 1.174 | 0.857 | 19.45 | 58.35 | 4.32 |

Here “full” means the complete LeRobot-default trainable policy—state
projection plus action expert—not the frozen vision/language backbone. Both
modes use fp32 master trainables; “base storage” controls the remaining model
weights. On this machine fp32 was 2.4–4.7% faster, while bf16 saved 0.78 GiB in
full mode and 0.96 GiB in LoRA mode. Those are measured outcomes, not general
claims that fp32 is always faster.

The per-1k and 3k figures are median-update projections and exclude checkpoint
serialization, final 500-tensor export, evaluation, data recording, and robot
I/O. A 30,000-update overnight LoRA bf16 run projects to 9.55 hours of optimizer
work before those overheads.

Every value above is copied or mechanically derived from the committed
[TRAINING_BENCHMARK.json](TRAINING_BENCHMARK.json). Its full ignored source
artifact is `.cache/training/t5-benchmark.json`, SHA-256
`7112806471e55e55d98ae101bc2af8172c2cc18f01b3e0c2c0646446adba9423`;
it binds clean protocol commit `0d897449b06d114d536756f2ed6850b52fd5bda4`,
the individual synchronized durations, environment, idle declaration, and
eight implementation hashes.
