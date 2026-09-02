# AGENTS.md — smolvla-mlx

This repository ports SmolVLA inference (checkpoint `lerobot/smolvla_base`) to MLX with proven numerical parity against LeRobot's PyTorch implementation. The full specification is `BRIEF.md`. Read it completely at the start of every session before acting. This file holds the rules that apply at all times.

## Where you are running

- The operator's Apple Silicon Mac (M5 Pro, 48 GB unified memory). MLX runs on Metal here; this machine is the source of truth for correctness and for the benchmark.
- Sandbox: `workspace-write`. Only this repository is writable. Route every cache inside it: set `HF_HOME=$PWD/.cache/hf`, `UV_CACHE_DIR=$PWD/.cache/uv`, and `SMOLVLA_MLX_CACHE=$PWD/.cache/smolvla_mlx` (all gitignored) in the `Makefile` and in every command you run. If a tool insists on writing elsewhere, do not try to escalate; add a `HUMAN_TASKS.md` entry naming the exact path and continue.
- `~/robot/so101` is the operator's live robot-control environment (a vendor LeRobot fork). Never read it as a reference, never activate its venv, never touch serial ports or hardware. Use mainline LeRobot from PyPI or GitHub as the reference, installed in this repo's own venv.

## Hard rules

1. Tolerances in `BRIEF.md` Section 6 are immutable. Tighten if you like; never loosen. If you think one is wrong, argue it in `PROGRESS.md` with numbers and leave the value unchanged.
2. The runtime package `smolvla_mlx` never imports `torch`, `lerobot`, or `transformers`. A test enforces this; keep it passing.
3. No mocking of model components, no skipped tests, no `xfail` without a matching `FAILURE_<module>.md`.
4. No secrets in the repo. No uploads to the Hugging Face Hub. No pushes to any remote other than this repo's `origin`.
5. Commit after every passing test. If `origin` exists, push at least hourly. Message format: `phase-N: <what> (<test> passes)`.
6. Do not end your turn to ask questions. Do not wait for approval between phases. End a turn only at a stop condition or Definition of Done (`BRIEF.md` Sections 8 and 9).
7. Prefer small, boring, verifiable steps over clever ones.

## Files you maintain

- `PLAN.md` — written first, updated whenever the plan changes.
- `PROGRESS.md` — append an entry after every meaningful step: what, evidence (test names and numbers), decisions, open questions, next step.
- `HUMAN_TASKS.md` — your only channel to the operator. Use it for a writable path you need, a failed push, or a decision only they can make. Give exact commands. Continue with other work while waiting; check it at the start of each session.
- `STATUS.md` — written at every stop condition, and the line `DEFINITION OF DONE MET` when v0.1 is complete.
- `ARCHITECTURE.md`, `REUSE_DECISIONS.md`, `BENCHMARK.md`, `FAILURE_<module>.md` as specified in `BRIEF.md`.

## Session start checklist

- Read `BRIEF.md`, `PLAN.md`, the last three entries of `PROGRESS.md`, and `HUMAN_TASKS.md`.
- Run `make test` and confirm the state matches `PROGRESS.md`.
- Continue from the "next step" of the last `PROGRESS.md` entry.
