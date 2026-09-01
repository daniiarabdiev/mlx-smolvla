# Full-Scope Status

ACTIVE — T3B AND RELEASE EXECUTION

T3B-1 COMPLETE — SELF-CONSISTENCY FLOOR RECORDED

T3B-2 COMPLETE — PROSPECTIVE PARITY PROCEDURE FROZEN

T3B-3A PRE-LAUNCH VERIFIED — REAL UPDATE-1 GATE PASSED

The protected SmolVLA MLX v0.1 inference baseline is intact. At full-scope
kickoff on 2026-08-31, `make test` passed **179/179** in **158.71 seconds** on
the M5 Pro. The repository had **553 GiB** free, above the mandatory 40 GiB
floor.

Stage T0 added 14 training/readiness regression cases. The exact post-T0 tree
passed **193/193 tests in 158.14 seconds**. `TRAINING_FEASIBILITY.md` records
the complete measured artifact and training-path decision.

Stage T1 now proves the real checkpoint-backed step-zero derivative. The fixed
PyTorch and MLX losses differ by **9.074298999059449e-7**; all **155/155**
gradient tensors pass, with worst relative L2 **8.673578115837066e-6** and
minimum cosine **0.9999999999623879**. See `GRADIENT_PARITY.md`.

Stage T2 now proves 25 cumulative optimizer updates. All **25/25** losses pass
with maximum relative difference **1.3529624562582406e-6**, and all **155/155**
final selected tensors pass with maximum relative-L2 drift
**2.8499913470883435e-8**. See `OPTIMIZER_LOCKSTEP.md`.

The exact post-T1 source and documentation tree passed **224/224 tests in
172.58 seconds**. The dependency lock resolves all **103 packages**, import
isolation is green, and **543 GiB** remains free.

The exact post-T2 source and documentation tree passed **236/236 tests in
189.07 seconds**. The dependency lock still resolves all **103 packages**,
import isolation remains green, and **542 GiB** remains free.

The checkpoint-hardened Stage T3 tree passes **278/278 tests in 215.81
seconds**. A full 458-trainable-tensor rebuild/restore proof reproduced update
2 exactly, including loss, gradient norm, model bytes, and optimizer bytes.
The dependency lock still resolves all **103 packages**, and **580 GiB**
remains free.

The fixed Stage T3 run completed all **3,000** Metal updates in
**4,956.693033957996 seconds** and exported all **500** fp32 tensors. Held-out
MLX MAE improved from **4.639846293521779** to **2.1164464077779224**, and the
Torch/MLX MAE ratio was **0.9927553940871358**; both gates passed. The unchanged
all-56-case stats-active parity gate failed at raw physical max absolute
**6.632053375244141** (normalized max **0.17762404680252075**) versus **0.005**. See
`FAILURE_LORA_FINETUNE.md`. `TRAINING ALPHA` was not reached.

The outcome-evidence-hardened tree passes **308/308 tests in 205.10 seconds**.
The current repository has **573 GiB** free, above the
mandatory 40 GiB floor.

The operator supplied the normative release specification and the T3B
amendment, closing the prior stop condition. The protected kickoff baseline
passed **308/308 tests in 226.41 seconds**. T3B-1 measured the failed T3
checkpoint's PyTorch self-consistency envelope with nine fresh workers on all
56 frozen cases:
`F = 0.00003549918286283038`, with `F64 = 0.00003549918286283038`. This result is
diagnostic only and does not alter the original T3 failure. See
`SELF_CONSISTENCY_T3.md`. The final v3 T3B-1 tree passes **347/347 tests in
244.83 seconds**; its floor-focused suite passes **39/39**.

T3B-2 froze the trained-checkpoint decision before a T3B checkpoint or MLX
comparison existed. It retains every fixed tolerance and derives normalized
action parity only as `max(0.005, 3 * F(C))`. The file evaluator reconstructs
all nine raw floor workers, enforces the real-clock floor/marker/comparison
order, rehashes every floor-bound input and complete export/evaluation tree,
validates the canonical fp32 conversion from private descriptor snapshots, and
installs one non-overwriting result. The focused suite passes **52/52**, the
combined floor/distribution/evaluator suite passes **97/97**, and the complete
repository passes **402/402 tests in 230.01 seconds**. Independent review found
no remaining substantive issue. See `PARITY_PROCEDURE_TRAINED.md`.

T3B-3a's pre-launch implementation is now verified. The explicit expert-only
scope contains 112 adapters / 224 fp32 tensors / 1,708,032 scalars; the legacy
T3 scope remains unchanged. The launcher enters the repository interpreter
with `-I -S`, provenance is frozen only after a real disposable video decode
and preprocessing pass, and every retained checkpoint identity is bound across
run-state publication. The first real disposable run exposed and safely
rejected a late PyAV import before update 1. After the test-first fix, a fresh
isolated run completed 29 updates, atomically bound the step-1 checkpoint, and
was intentionally interrupted. Independent review validated its PID, run,
metrics, pointer, metadata, model, and optimizer evidence and found no source
blocker. The focused exact-byte suite passes **142/142 in 235.70 seconds**.
With no training or floor process active at `2026-09-01T22:44:56Z`, the full
repository passed **536/536 tests in 490.35 seconds**. The canonical launch
artifact has intentionally not been created yet: its digests will be recorded
only after these exact code bytes are committed and pushed.

## Stage state

| Stage | State | Evidence / next action |
| --- | --- | --- |
| R — Release | Ready; ordered after T3B launch | The operator supplied `BRIEF_RELEASE.md`; the historical missing-spec blocker is resolved. Non-timing packages begin while T3B-3 trains. |
| T0 — Training-readiness | Complete | 155/155 gradients finite and nonzero over 99,880,992 trainable scalars; 196.799 ms forward+backward and 2,509,594,126-byte peak MLX memory. See `TRAINING_FEASIBILITY.md`. |
| T1 — Gradient parity | Complete | Identical real batch/draws; loss and all 155 gradients pass immutable gates. See `GRADIENT_PARITY.md`. |
| T2 — Optimizer lockstep | Complete | 25/25 losses and 155/155 final tensors pass immutable gates. See `OPTIMIZER_LOCKSTEP.md`. |
| T3 — LoRA fine-tune | Failure-documented | The 3,000-update run, merge, held-out improvement, and Torch round trip completed. All 56 frozen cases were parity-checked; raw physical max was `6.632053375244141` versus the unchanged `0.005` gate; see `FAILURE_LORA_FINETUNE.md`. |
| T3B-1 — Reference floor | Complete | Nine PyTorch workers, including a fixed five-process MPS empirical envelope, evaluated 56 cases each; `F = F64 = 0.00003549918286283038`; report SHA-256 `cba4a856...f0585`; informational only, with no statistical-bound claim. |
| T3B-2 — Prospective evaluator | Complete | Fixed gates unchanged; derived `max(0.005, 3F)` gate, chronology, complete input provenance, semantic conversion validation, and no-clobber output are frozen and pass 52 focused tests. |
| T3B-3 — Expert-only LoRA | Ready to launch | 112 expert attention/MLP adapters, isolated runtime provenance, full retained-checkpoint binding, and a real update-1 checkpoint probe pass; the exact tree passes 536/536 tests. Commit/push, canonical configuration hashing, and background launch remain. |
| T4 — Training UX/full fine-tune | Pending T3B-3 fixed gates | Full fine-tune is code plus a 100-update smoke only. |
| T5 — Training docs/benchmark | Pending T3B-3 fixed gates | Run only after training/floor processes are absent and idle state is recorded. |
| Q — Quality extras | Pending Stage R | Normative package definitions are now available in `BRIEF_RELEASE.md`. |
| H — Hardware readiness | Pending Stage R P1-4 | Documents and loopback-safe tooling only; no hardware access is authorized. |

Neither `RELEASE READY` nor `TRAINING ALPHA` has been reached.

## Current local evidence

| Artifact / cache | Size | Integrity |
| --- | ---: | --- |
| `.cache/training/t0-audit.json` | JSON | SHA-256 `88dacde30996c2d9cbad90681204e2583c92f6d51f0f3747c6e37e57b709fd51` |
| `.cache/training/gradient_goldens` | 769 MiB | manifest SHA-256 `b029a0ed66312e785cb8aa3f1db0affb16c9502ad7b5d0fe0feea3177bf8c145` |
| `.cache/training/t1-parity.json` | 52 KiB | SHA-256 `f4da0c16771a462e45bd615728bc02a059633db19eb77883342203426cb4d634` |
| `.cache/training/optimizer_goldens` | 381.329 MiB | manifest SHA-256 `88c3febc7da3e553bcb7c26f261721369ed1f56efd457887b7d43d50a077807c` |
| `.cache/training/t2-lockstep.json` | 60 KiB | SHA-256 `da8cabf5eecf4379065771b3a74407c47290b8aee9c2d0a9756893b6dd87a6a4` |
| `.cache/training/t3-benchmark.json` | JSON | SHA-256 `3598214cecd083cd3d5d143edd3edbe614dc899d30622152573e87dd104fe442` |
| `.cache/training/t3-evaluation` | 100 MiB | Tensor-manifest SHA-256 `9cabca6cd21e8658a94e42980af3e91ecd8ff5ed5daca5f75eb7a1ebd1d261a3`; full metadata SHA-256 `f49ee54aead7ce3ede7b94d5638864afd2e12ef57ae2622eb6574333820cd107` |
| `.cache/training/t3-base-evaluation.json` | JSON | SHA-256 `211d6778b0530208ca2e81abe6f4002cc683e24d496a09ddbe39c100ebd4f7ce`; base MAE `4.639846293521779` |
| `.cache/training/t3` | 1.8 GiB | Completed run SHA-256 `c7c3b86361c0872e26f2088cbd33ada865cf450b6711a9b737ece933c1868c82`; adapter/final-model SHA-256 `814e6f4b2a78a46b609aa7b48a28b4509f709d3e851e588dcd9a4bd2ca1408dc`; final optimizer SHA-256 `c9440be75315e04c1812ba18da0e0daccd2990fb0f6fdb1841d40ef7b01ffb5a`; retained checkpoints 2,800/2,900/3,000; export audit manifest SHA-256 `55ad6834cbb3acb9dd565a57296a274d78e7cdc863aa81c3e6ef25da8b66ba03` |
| `.cache/training/t3-outcome.json` | JSON | SHA-256 `8b74faf8f9cc96341090f91cfa795ed874c838026416944e4b77a550ad91bc44`; 15 source digests include the validated native conversion; improvement and round-trip pass, all-56-case parity fails |
| `.cache/training/t3/floor.json` | JSON | SHA-256 `cba4a856f9c907d986ffc8703789673611e54bad983c2afd0a987830466f0585`; all nine worker artifacts and every input file hash; combined input SHA-256 `d31a0835867116d7bfbe63f6cd23666eecfdc0a660aba620730cd09320295299`; `F = F64 = 0.00003549918286283038`; retrospective diagnostic only |
| `.cache/hf` | 1.8 GiB | repository-local source cache |
| `.cache/smolvla_mlx` | 73 GiB | repository-local conversion/golden cache; T1 fp32 policy subset is 4.2 GiB |

Training evidence is intentionally ignored by Git and has not been uploaded.

## Safety and external state

- `origin`: `git@github.com:daniiarabdiev/smolvla_mlx.git`.
- No PyPI or Hugging Face uploads were made.
- No credentials, robot environment, vendor fork, serial ports, or hardware
  were accessed.
- The release brief and T3B amendment are committed. Execution is proceeding in
  the mandated order without additional operator input.
