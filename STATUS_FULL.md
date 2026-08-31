# Full-Scope Status

IN PROGRESS

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

## Stage state

| Stage | State | Evidence / next action |
| --- | --- | --- |
| R — Release | Blocked | `BRIEF_RELEASE.md` is absent after four recovery searches; see `FAILURE_RELEASE_SPEC.md`. The operator-supplied empty GitHub repository is now `origin`, and the verified v0.1 history was pushed. |
| T0 — Training-readiness | Complete | 155/155 gradients finite and nonzero over 99,880,992 trainable scalars; 196.799 ms forward+backward and 2,509,594,126-byte peak MLX memory. See `TRAINING_FEASIBILITY.md`. |
| T1 — Gradient parity | Complete | Identical real batch/draws; loss and all 155 gradients pass immutable gates. See `GRADIENT_PARITY.md`. |
| T2 — Optimizer lockstep | Complete | 25/25 losses and 155/155 final tensors pass immutable gates. See `OPTIMIZER_LOCKSTEP.md`. |
| T3 — LoRA fine-tune | Ready | T1 passed; eligible independently of T2's eventual result. |
| T4 — Training UX/full fine-tune | Pending | Depends on T3. |
| T5 — Training docs/benchmark | Pending | Depends on T3. |
| Q — Quality extras | Blocked | Depends on Stage R and the missing normative package definitions. |
| H — Hardware readiness | Blocked | Depends on Stage R P1-4; documents only unless the exact live-session confirmation is supplied. |

Neither `RELEASE READY` nor `TRAINING ALPHA` has been reached.

## Current local evidence

| Artifact / cache | Size | Integrity |
| --- | ---: | --- |
| `.cache/training/t0-audit.json` | JSON | SHA-256 `88dacde30996c2d9cbad90681204e2583c92f6d51f0f3747c6e37e57b709fd51` |
| `.cache/training/gradient_goldens` | 769 MiB | manifest SHA-256 `b029a0ed66312e785cb8aa3f1db0affb16c9502ad7b5d0fe0feea3177bf8c145` |
| `.cache/training/t1-parity.json` | 52 KiB | SHA-256 `f4da0c16771a462e45bd615728bc02a059633db19eb77883342203426cb4d634` |
| `.cache/training/optimizer_goldens` | 381.329 MiB | manifest SHA-256 `88c3febc7da3e553bcb7c26f261721369ed1f56efd457887b7d43d50a077807c` |
| `.cache/training/t2-lockstep.json` | 60 KiB | SHA-256 `da8cabf5eecf4379065771b3a74407c47290b8aee9c2d0a9756893b6dd87a6a4` |
| `.cache/hf` | 965 MiB | repository-local source cache |
| `.cache/smolvla_mlx` | 67 GiB | repository-local conversion/golden cache; T1 fp32 policy subset is 4.2 GiB |

Training evidence is intentionally ignored by Git and has not been uploaded.

## Safety and external state

- `origin`: `git@github.com:daniiarabdiev/smolvla_mlx.git`.
- No PyPI or Hugging Face uploads were made.
- No credentials, robot environment, vendor fork, serial ports, or hardware
  were accessed.
- The required release brief is the only open human-supplied input; independent
  training work continues.
