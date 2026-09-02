# Development history

These files preserve how the implementation and its gates were specified and
executed. They are historical records, not current installation instructions;
legacy package names, paths, commands, machine context, and superseded scope are
kept so earlier results remain auditable.

## Correction recorded 2026-09-02

The original `BRIEF.md` claimed that nobody had shipped native Apple Silicon
inference for SmolVLA. That claim was made in error and corrected on 2026-09-02
after identifying the earlier independent
[`tokimoa/smolvla-mlx`](https://huggingface.co/tokimoa/smolvla-mlx) Hub port,
uploaded on 2026-07-29. The original brief remains unchanged below this index
for provenance; current public documents make no first-port claim.

## File index

| Path | Why it is retained |
| --- | --- |
| `AGENTS.operator.md` | Original machine/operator working rules before the public agent guide replaced them. |
| `BRIEF.md` | Initial inference-port specification and immutable numerical gates. |
| `BRIEF_FULL.md` | End-to-end release, training, and hardware-ready scope. |
| `BRIEF_RELEASE.md` | Normative Stage R packaging and release specification. |
| `BRIEF_T3B.md` | Second training-attempt amendment and prospective evaluation order. |
| `BRIEF_PUBLIC_RELEASE.md` | Hardware/public-release amendment before the canonical rename. |
| `BRIEF_RENAME.md` | Operator amendment selecting the final `mlx-smolvla` identity. |
| `PLAN.md` | Initial implementation plan. |
| `PLAN_FULL.md` | Full-scope continuation plan. |
| `PLAN_T3B.md` | Second-attempt training plan. |
| `PLAN_PUBLIC_RELEASE.md` | Public-release execution plan and hardware stop gate. |
| `STATUS.md` | Early implementation status snapshot. |
| `STATUS_RELEASE.md` | Stage R status snapshot. |
| `STATUS_FULL.md` | Cumulative milestone ledger through compatibility/rename work. |
| `STATUS_PUBLIC_RELEASE.md` | Closing software release-candidate status and five-blocker decision. |
| `HANDOFF_REPORT.md` | Earlier agent-to-agent handoff requested by the operator. |
| `PROGRESS.md` | Append-only implementation and verification ledger through public-release preparation. |
| `HUMAN_TASKS.md` | Resolved external gates plus the exact remaining hardware/publication operator actions. |
| `SETUP.md` | Superseded operator setup procedure. |
| `python-version.txt` | Original local Python selector, retained rather than published as a root policy. |
| `setup.py`, `CMakeLists.txt`, `MANIFEST.in` | Superseded root build files retained after migration to the PEP 517 backend and package-local CMake project. |
| `superpowers/` | Design notes and execution plans used during implementation. |

The public-release status records the completed software verification and the
remaining supervised hardware blocker.
