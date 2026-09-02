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

## MLX versus PyTorch-MPS

Stage Q P2-1 compares the public MLX production engine with the pinned
LeRobot/PyTorch policy on MPS. Both standalone worker processes use
`lerobot/smolvla_base` at revision
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`, fp32 weights, the exact saved
`sample_000` observation and noise, five excluded warmups, and 50 synchronized
measurements. Model loading is excluded. The common timed boundary begins with
observation preprocessing and ends when the normalized 50-action chunk is
materialized on the accelerator; action unnormalization, device-to-host copy,
transport, and robot I/O are excluded.

| Engine | Dtype | Median ms / chunk | p95 ms / chunk | Chunks/s | Observed framework memory GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Native MLX Metal | fp32 | 110.75 | 111.20 | 9.029 | 2.94 |
| LeRobot/PyTorch MPS | fp32 | 204.58 | 206.46 | 4.888 | 2.16 |

MLX's median latency is **1.847×** faster in this bounded case (45.86% lower).
This is not a cross-platform claim: it is one pinned model, observation, noise,
software stack, and Apple M5 Pro. The PyTorch worker cleared inherited MPS
switches and set `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing Torch or
LeRobot. Fallback was enabled as required; the run did not instrument whether
an individual operation actually fell back to CPU.

Memory is retained only as a directional diagnostic because the framework APIs
do not expose identical semantics: MLX reports the maximum of its peak and
active allocator counters, while PyTorch 2.11 has no MPS peak-reset API and the
worker records the maximum sampled MPS driver allocation after each measured
chunk. Do not read the 2.94/2.16 GiB values as a precise allocator-efficiency
comparison.

The pre-measurement process check at
`2026-09-02T05:16:10.169655+00:00` found no trainer, floor worker, test suite,
or competing benchmark. All raw durations, the idle declaration, environment,
input hashes, and source hashes are committed in
[INFERENCE_COMPARISON.json](INFERENCE_COMPARISON.json), SHA-256
`115ad58c0c618b65a6275018614f3ee6cf17dd02a9d4ad9c94aaf7e5a9842e48`.
Its timing implementation was frozen first in clean commit
`e210f7b76ae8657390a8101b76ee5815df1b15ab`.

## bf16 latency diagnosis

Stage Q P2-2 reproduces and localizes the storage-bf16 slowdown with the same
pinned model, `sample_000` observation/noise, 5 excluded warmups, and 50
measurements per dtype. Unlike the end-to-end table, this diagnostic inserts a
synchronization after every component, so its component values should be
compared only within this table.

| Synchronized component | fp32 median ms | bf16 median ms | bf16 delta ms | Share of total delta |
| --- | ---: | ---: | ---: | ---: |
| Preprocessing | 4.36 | 4.38 | +0.01 | 0.07% |
| Vision encoder | 45.49 | 47.20 | +1.71 | 8.78% |
| Connector | 0.60 | 0.93 | +0.32 | 1.66% |
| Prefix prefill | 8.10 | 11.01 | +2.91 | 14.92% |
| Ten-step expert loop | 52.19 | 66.71 | +14.52 | 74.44% |
| **End-to-end total** | **110.71** | **130.22** | **+19.50** | **100.00%** |

The bf16 total is **17.62%** slower. The component medians account for 99.87%
of the 19.50 ms delta; the remaining 0.03 ms (0.13%) is boundary/Python timing
overhead. The expert loop is the dominant source, followed by prefix prefill;
preprocessing is effectively neutral. Profile memory is 2.93 GiB for fp32 and
2.44 GiB for bf16, preserving bf16's compact-storage benefit.

An explicit dtype trace explains the relevant execution mode: bf16 selects
bf16 checkpoint weights, but preprocessed pixels, state, fixed flow noise,
vision output, connector output, prefix/cache, and final velocity remain fp32.
It is therefore mixed-dtype fp32 activation compute with compact bf16 weight
storage, not an all-bf16 compute path. The fact that every projection-heavy
component slows while preprocessing does not is consistent with conversion or
mixed-dtype kernel cost in **MLX 0.32.2**. The measurements do not identify a
specific private Metal kernel, so that narrower claim is intentionally not
made.

No inference behavior changed. Casting activations and the Euler state to bf16
would change the numerical pathway protected by the immutable parity ladder;
upcasting stored weights would discard the measured memory benefit. Neither is
a justified optimization from this evidence alone. Keep the current default,
and rerun this exact artifact after an MLX upgrade to see whether mixed-dtype
kernel behavior changes before considering a code path change.

The clean idle declaration is
`2026-09-02T05:33:56.285086+00:00`. All 600 raw durations, summaries, delta
attribution, inputs, environment, and source hashes are committed in
[BF16_PROFILE.json](BF16_PROFILE.json), SHA-256
`74da9f937cb8bfeba4066d5518187490ff96a1447e4a2ad2253e2493245be1cf`.
The successful profile binds clean source commit
`adf40e62a7b652262fc08d7ed6449b4c60a0773d`.
