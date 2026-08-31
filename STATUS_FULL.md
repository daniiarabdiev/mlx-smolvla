# Full-Scope Status

IN PROGRESS

The protected SmolVLA MLX v0.1 inference baseline is intact. At full-scope
kickoff on 2026-08-31, `make test` passed **179/179** in **158.71 seconds** on
the M5 Pro. The repository had **553 GiB** free, above the mandatory 40 GiB
floor.

## Stage state

| Stage | State | Evidence / next action |
| --- | --- | --- |
| R — Release | Blocked | `BRIEF_RELEASE.md` is absent after four recovery searches; see `FAILURE_RELEASE_SPEC.md`. The operator-supplied empty GitHub repository is now `origin`, and the verified v0.1 history was pushed. |
| T0 — Training-readiness | In progress | Execute the differentiability, memory, native-RMSNorm, data-path, and gradient-harness audit. |
| T1 — Gradient parity | Pending | Depends on T0. |
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
