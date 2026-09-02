# Stage T3B expert-only LoRA deterministic-parity failure

## Status

The prospectively frozen expert-only LoRA run completed all 3,000 updates and
produced a strict merged fp32 LeRobot checkpoint. Every fixed gate passed:
preprocessing identity, held-out improvement, and Torch/MLX MAE round trip.
The separately derived deterministic gate failed at normalized max absolute
`0.013038858771324158` versus `max(0.005, 3F) = 0.005`.

Per `BRIEF_T3B.md`, this outcome earns `TRAINING ALPHA (STATISTICAL)`. T4 and
T5 may proceed because they depend on the fixed outcome gates. This document is
a new T3B record; it does not reopen, amend, or reinterpret
`FAILURE_LORA_FINETUNE.md` or its original gates.

No tolerance, population, stored noise, split, update count, checkpoint, or
output selection was changed after observing the result.

## Frozen run and artifact identity

- Updates: **3,000**, effective batch **8**, expert-only LoRA rank **8**, alpha
  **16**, dropout **0**.
- Trainable scope: **112 adapters / 224 tensors / 1,708,032 scalars**.
- Training time: **3,545.863376834008 seconds**; peak MLX memory:
  **2,899,690,676 bytes**.
- Final loss: **0.10034868866205215**; final smoothed loss:
  **0.17088623540128797**.
- Run-state SHA-256:
  `2af527bea4691862e89eb6daa674e6d99309668ac45f57397786976aab3c301e`.
- Metrics SHA-256:
  `33f00adc5316cbc295e6f3fa1e153963b64fadec59a7db5401074794245f6278`.
- Adapter SHA-256:
  `cce4eed18a7311594950f6d4da33a44dd337f66fbc29162d686c5338ec044826`.
- Merged fp32 model SHA-256:
  `858704fa572501d9e5a048076f8da692693b90c463feda29201a72f3f0b18883`.
- Outcome SHA-256:
  `75dff4b750dd1e8c8bc4d8426fe9af297bedb33ea1bfa7523dcc49d51460b33f`.
- Bound comparison SHA-256:
  `6aa8e3771bbbd81ecd9599ec9605a4e1efb804fa9ec66c4f82d2d6aea3eb00c6`.
- Authoritative evaluation SHA-256:
  `1e337f0bb87aa66a4270c526dd918bd18807aa6aa5291a59b119780080ea9eca`.
- Held-out manifest / full metadata SHA-256:
  `9cabca6cd21e8658a94e42980af3e91ecd8ff5ed5daca5f75eb7a1ebd1d261a3`
  / `f49ee54aead7ce3ede7b94d5638864afd2e12ef57ae2622eb6574333820cd107`.
- Export: **500 tensors / 450,046,176 fp32 parameters**.
- Retained checkpoints: exactly **2,800 / 2,900 / 3,000**.

The run completed without resume or recovery.

## Prospective chronology

The machine was recorded idle and no training, floor, benchmark, or timing
process overlapped the floor or comparison sequence.

1. The PyTorch-only floor was atomically installed at
   `2026-09-02T00:38:00.730626+00:00`, before any MLX-versus-PyTorch comparison
   of this checkpoint. Floor SHA-256:
   `28d83926a70e507671bfd694e032f81b71093d475075aad627b3c24c5b334efc`.
2. The one-shot comparison marker was created at
   `2026-09-02T00:39:06.152740+00:00`. Marker SHA-256:
   `0e1121728fc30eb911e6f596d32ec5f7de97faa0d44e883b52367d7ac7dcd202`.
3. The bound comparison was created at
   `2026-09-02T00:42:52.253017+00:00`, after both floor and marker.
4. The timestamp-enforcing evaluator rehashed the floor inputs, checkpoint,
   held-out tree, support code, marker, outcome, and comparison before
   installing its non-overwriting verdict.

The measured floor is `F = 0.00002467632293701172`; its fp64 sensitivity
component is `F64 = 0.000022446572480461224`. Therefore `3F` is
`0.00007402896881103516`, and the prospectively frozen `0.005` fallback—not a
post-result adjustment—sets the derived threshold.

## Gates

| Gate | Result | Required | Verdict |
| --- | ---: | ---: | --- |
| Image preprocessing max absolute | `3.5762786865234375e-7` | `<= 1e-5` | Pass |
| State preprocessing max absolute | `0.0` | `<= 1e-6` | Pass |
| Base MLX held-out physical MAE | `4.639846293521779` | Frozen baseline | — |
| Fine-tuned MLX held-out physical MAE | `2.2550044155546596` | `<= 4.175861664169601` | Pass |
| Fine/base MAE ratio | `0.486008430646319` | `<= 0.9` | Pass |
| Torch held-out physical MAE | `2.2548798021106493` | Same 56 cases/draws | — |
| Torch/MLX MAE ratio | `0.9999447391574267` | `[0.95, 1.05]` | Pass |
| Normalized action max absolute | `0.013038858771324158` | `<= 0.005` derived | **Fail** |
| Raw physical action max absolute | `0.13149452209472656` | Reported | — |
| Standardized physical max absolute | `0.013038855977356434` | Reported | — |

All **56** precommitted cases participate. Normalized error is worst at
ordinal 24, episode 28, frame 87, absolute index 6307. The fixed gates pass and
only the derived deterministic gate fails.

## Three tested hypotheses

### 1. Data, split, statistics, or preprocessing identity

Ruled out. The comparison binds the original held-out manifest and full
metadata hashes, all 56 case identities, dataset and tokenizer revisions,
stored noise, and the exact export. Token IDs and masks match exactly. Image
preprocessing differs by at most `3.5762786865234375e-7`, and state
preprocessing is exact at `0.0`; both are comfortably inside their immutable
limits.

### 2. Adapter, checkpoint, export, tensor name, or layout error

Ruled out. The evaluator validates the expert-only scope against its private
descriptor snapshots, requires exactly 112 adapters and 224 LoRA tensors,
reconstructs the merged export semantically, and binds every support file by
hash. The standard LeRobot export contains all 500 expected fp32 tensors and
450,046,176 parameters. It loads strictly in both frameworks, and the
Torch/MLX held-out MAE ratio is `0.9999447391574267`, far inside the fixed
`[0.95, 1.05]` round-trip gate.

### 3. Framework input/noise identity or a localized model defect

No localized defect was demonstrated. Both frameworks consume the same stored
`1x50x32` fp32 noise. On the worst case, the first velocity differs by only
`0.0035195350646972656`; ten Euler updates accumulate a final normalized state
difference of `0.013038858771324158`. The per-step trace is recorded in
`LORA_SCOPE_COMPARISON.md` and exactly reproduces the evaluator's final maximum.

Relative to T3, freezing the prefix decoder and adapting only expert attention
and MLP linears reduces final normalized divergence by **13.62x** and removes
the original run's sharp late-step amplification. The remaining gap is
consistent with distributed fp32 reduction-order drift being accumulated by a
trained, numerically sensitive velocity field. That is an inference from the
measured trajectory, not proof of a single wrong operator and not grounds to
change the fixed tolerance.

## Consequence

The fixed training pipeline gates pass, so the repository records
`TRAINING ALPHA (STATISTICAL)` and proceeds to T4 and T5 exactly as authorized
by `BRIEF_T3B.md`. Strict deterministic parity remains a documented limitation.
The threshold stays `0.005`, and the original
`FAILURE_LORA_FINETUNE.md` remains byte-for-byte unchanged.

