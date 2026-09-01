# T3 PyTorch Self-Consistency Floor

**T3B-1 COMPLETE — DIAGNOSTIC ONLY**

This is the retrospective arithmetic self-consistency measurement required by
`BRIEF_T3B.md` Section 1. It uses the failed T3 merged fp32 export, the exact 56
frozen held-out cases, every stored `1×50×32` noise tensor, and the same saved
pre/postprocessor tensors. It does not change, reopen, or reinterpret the
original T3 verdict in `FAILURE_LORA_FINETUNE.md`.

## Authoritative v3 result

| Perturbation | Device / dtype | CPU threads | Maximum normalized action-chunk max-abs versus (a) | Worst case ordinal |
| --- | --- | ---: | ---: | ---: |
| (a) golden path | CPU / fp32 | default = 6 | `0.0` | 0 |
| (b) single thread | CPU / fp32 | 1 | `0.000024199485778808594` | 1 |
| (c) maximum threads | CPU / fp32 | 18 | `0.000023752450942993164` | 1 |
| (d1) fallback process 1 | MPS / fp32 | default = 6 | `0.000022813677787780762` | 44 |
| (d2) fallback process 2 | MPS / fp32 | default = 6 | `0.000022813677787780762` | 44 |
| (d3) fallback process 3 | MPS / fp32 | default = 6 | `0.000022813677787780762` | 44 |
| (d4) fallback process 4 | MPS / fp32 | default = 6 | `0.000022813677787780762` | 44 |
| (d5) fallback process 5 | MPS / fp32 | default = 6 | `0.000022813677787780762` | 44 |
| (e) double precision | CPU / fp64 | default = 6 | `0.00003549918286283038` | 40 |

- `F = 0.00003549918286283038`, the maximum across every case and all eight
  non-baseline worker slots.
- `F64 = 0.00003549918286283038`, the float64-only component.
- Original MLX-versus-(a) normalized maximum, shown only as required context:
  `0.17762404680252075`.

The float64 perturbation defines the v3 envelope. The five MPS processes wrote
byte-identical normalized action arrays in this run. They are separate fresh
processes but share the same driver, device, and machine state; they are not
statistically independent, and this fixed five-process empirical envelope is
not a probabilistic or absolute upper bound. The count was frozen before the
run and was not adapted to the results.

## Frozen procedure

Procedure `smolvla-pytorch-self-consistency-v3` launched nine fresh Python
workers: one baseline, two CPU-thread perturbations, five prospectively fixed
MPS fallback slots, and one CPU-float64 worker. Every worker strictly loaded the
standard LeRobot export, validated all 56 frozen cases, reset the policy for
each case, reused the stored noise byte-for-byte, and persisted its complete
`56×50×6` normalized result before aggregation. Differences were recomputed in
float64 from the persisted chunks without rounding.

Before NumPy or Torch import, each worker cleared all controlled MPS and CPU
thread environment variables. Only the five MPS workers set
`PYTORCH_ENABLE_MPS_FALLBACK=1`; every other documented MPS switch was absent.
Python, NumPy, Torch, and MPS used seed `20260901` before model construction.
Workers recorded ordinary deterministic-algorithm mode as disabled and
float32 matmul precision as `highest`.

For the CPU float64 experiment, the complete model, floating preprocessor
inputs, and noise were cast to double. LeRobot 0.6.1 has a literal fp32 cast
immediately before its action-output projection; this experiment replaces only
that literal with the projection weight dtype so a double projection can
execute. That compatibility path and its implementation are source-hashed.

No training or timing process was running at launch. The computation collected
no latency, throughput, wall-time, or benchmark measurements, and no new
MLX-versus-PyTorch comparison was run before this floor was written and hashed.

## Artifact, timestamps, and inputs

- Floor: `.cache/training/t3/floor.json`
- Floor SHA-256:
  `cba4a856f9c907d986ffc8703789673611e54bad983c2afd0a987830466f0585`
- Floor creation: `2026-09-01T13:40:46.946734+00:00`
- Embedded creation timestamp: `1788270046946734000` ns
- Persisted file mtime: `1788270046951823007` ns
- Combined input SHA-256:
  `d31a0835867116d7bfbe63f6cd23666eecfdc0a660aba620730cd09320295299`

The file mtime is strictly later than the embedded creation time. Independent
post-write validation reloaded all nine artifacts, recomputed every action
maximum, rehashed every current input, and confirmed the exact environment
maps and runtime invariants.

| Input group | Files hashed | Tree SHA-256 |
| --- | ---: | --- |
| T3 checkpoint export | 7 | `a4ea6446f36d347d04d0ebe4884afbdbf9a7a3afd79a032b697a2a7220c03812` |
| Frozen evaluation artifact | 282 | `8f8cdde5376ff16e24579803f68cc7e32f6f2e402591fa2aea592f10ef6bb73d` |
| Pinned dataset validation inputs | 5 | `2ff3a60742d28ca289c68465c56f901aff67042ff60b451230afacdff733b604` |
| Pinned tokenizer snapshot | 10 | `add6f0ab0bada0aff47d894a491fef8b1ee968ec838522932eaec2d95f655aaa` |
| Procedure implementation, runtime dependencies, distribution records, and lock | 33 | `1b73640f406b751cfd8bccb1061673d4f6888dbec69cb9ce7d8444175cceaad6` |

The fixed source identities are dataset
`lerobot/svla_so101_pickplace@f641879e22172be7e8161d5e6c1503c2d2feb657`,
tokenizer/model
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`,
evaluation tensor manifest `9cabca6c...d261a3`, evaluation metadata
`f49ee54a...107`, sample count 56, and noise seed `20260902`.

## Variant evidence

| Variant | Normalized chunks SHA-256 | Variant artifact SHA-256 |
| --- | --- | --- |
| CPU fp32 baseline | `d69561abb2cb3ec8530d682822f331eb409bfdae35986f7c11b9b0cbeb0b01da` | `36e3c61ffd2855c36e1cd4df21b61c9b5f459a718656514b26f86d9c00e3f2ef` |
| CPU fp32 / 1 thread | `bbc7587f17c3d120b0843919c74f73e7770dc73d69096b3c602e1d0a1afcc2e2` | `7ee5d11a05748a8e81ec06b61e4c7ef85557b667156c56ed6d0c1431df764016` |
| CPU fp32 / 18 threads | `b8c33be101193a691fd43584615b9aeefbdd42c96416e8abc681febf56301e29` | `096f50e25f7eac2cc312f5b3c1486bf2ce7f66d9199a3f4952520498a93a28fa` |
| MPS fp32 / fallback 1 | `065573d47fbfe17ff99ed6ce3fcd4c84ece91ef388db8dc8508c22cfc8d5e8d9` | `5f8ed7821bf7fed2cd0665dc886efb40918bd6e815b1f2c3c3e64bb5bee1fe6c` |
| MPS fp32 / fallback 2 | `065573d47fbfe17ff99ed6ce3fcd4c84ece91ef388db8dc8508c22cfc8d5e8d9` | `11fc6640794f079314673c4ab5f3f76528f688ac30c3dba58a0a1eea5d39c01e` |
| MPS fp32 / fallback 3 | `065573d47fbfe17ff99ed6ce3fcd4c84ece91ef388db8dc8508c22cfc8d5e8d9` | `553124c4de8bff1d0429f126829e08c6ea069c9f0e9c4ed6b32c330ef6ea96c8` |
| MPS fp32 / fallback 4 | `065573d47fbfe17ff99ed6ce3fcd4c84ece91ef388db8dc8508c22cfc8d5e8d9` | `ad8bfb198659652139fcd5a9d7b535f074f07100851e8f71404d6913a2567bcb` |
| MPS fp32 / fallback 5 | `065573d47fbfe17ff99ed6ce3fcd4c84ece91ef388db8dc8508c22cfc8d5e8d9` | `4e9e7075a26f9aed2ab460d217675b503d1627e6b17b8f6332a0a50a9d5d185d` |
| CPU float64 | `0b4017fb7f3df09ecbda4d5285d02d633a2b80025b5f79ad50d2f3f816a19c51` | `cc3547b62d6a7a283695164fcd3faa11a5018adbb44d120183413f4895f29994` |

## Historical MPS variability disclosure

Review and hardening exposed material process/backend run-state variability.
The observations are preserved locally and disclosed, but they are excluded
from v3 `F` because they were produced under different hashed procedure/input
contracts:

| Observation | MPS max-abs versus CPU baseline | Evidence |
| --- | ---: | --- |
| Initial pre-review v1, incompletely recorded environment | `0.35390492528676987` | floor `463cec23...c2a`; actions `e3098534...61b2` |
| Six clean v1 follow-up processes | `0.000022813677787780762` each | actions `065573d4...e8d9`; clean floor `ebedb8b5...435f` |
| Sanitized v2 single process | `1.7168622612953186` | floor `cf867a1a...9168`; actions `3f71aa4b...74b6` |
| Explicit nondefault fast-math diagnostic | `1.8492990136146545` | `PYTORCH_MPS_FAST_MATH=1`; actions `9bdacae0...9574` |
| Authoritative fixed v3 slots 1–5 | `0.000022813677787780762` each | current report `cba4a856...f0585` |

Among retained default/fallback procedure observations, the largest is the v2
single-process value `1.7168622612953186`; the separate, explicitly nondefault
fast-math diagnostic reached `1.8492990136146545`. Neither is folded into the
prospectively frozen v3 arithmetic, and none of these finite samples supports a
statistical-independence or upper-bound claim.

Preserved paths include `.cache/training/t3/floor-pre-review.json`,
`.cache/training/t3/floor-v1-clean.json`,
`.cache/training/t3/floor-v2-single.json`, and their corresponding
self-consistency directories. Environment: Python 3.12.13, Torch 2.11.0,
LeRobot 0.6.1, Transformers 5.5.4, macOS 26.6.2 arm64. MPS was built and
available.
