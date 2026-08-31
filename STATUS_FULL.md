# Full-Scope Status

IN PROGRESS

The protected SmolVLA MLX v0.1 inference baseline is intact. At full-scope
kickoff on 2026-08-31, `make test` passed **179/179** in **158.71 seconds** on
the M5 Pro. The repository had **553 GiB** free, above the mandatory 40 GiB
floor.

Stage T0 added 14 training/readiness regression cases. The exact post-T0 tree
passed **193/193 tests in 158.14 seconds**. `TRAINING_FEASIBILITY.md` records
the complete measured artifact and training-path decision.

## Stage state

| Stage | State | Evidence / next action |
| --- | --- | --- |
| R — Release | Blocked | `BRIEF_RELEASE.md` is absent after four recovery searches; see `FAILURE_RELEASE_SPEC.md`. The operator-supplied empty GitHub repository is now `origin`, and the verified v0.1 history was pushed. |
| T0 — Training-readiness | Complete | 155/155 gradients finite and nonzero over 99,880,992 trainable scalars; 196.799 ms forward+backward and 2,509,594,126-byte peak MLX memory. See `TRAINING_FEASIBILITY.md`. |
| T1 — Gradient parity | Ready | T0 passed; next capture a real fixed reference batch, draws, loss, and all gradients. |
| T2 — Optimizer lockstep | Pending | Depends on T1. |
| T3 — LoRA fine-tune | Pending | Depends on T1. |
| T4 — Training UX/full fine-tune | Pending | Depends on T3. |
| T5 — Training docs/benchmark | Pending | Depends on T3. |
| Q — Quality extras | Blocked | Depends on Stage R and the missing normative package definitions. |
| H — Hardware readiness | Blocked | Depends on Stage R P1-4; documents only unless the exact live-session confirmation is supplied. |

Neither `RELEASE READY` nor `TRAINING ALPHA` has been reached.

## Safety and external state

- `origin`: `git@github.com:daniiarabdiev/smolvla_mlx.git`.
- No PyPI or Hugging Face uploads were made.
- No credentials, robot environment, vendor fork, serial ports, or hardware
  were accessed.
- The required release brief is the only open human-supplied input; independent
  training work continues.
