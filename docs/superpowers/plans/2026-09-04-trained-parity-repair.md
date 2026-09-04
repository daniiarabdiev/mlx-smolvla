# Trained Checkpoint Strict-Parity Repair Plan

> **For agentic workers:** Follow systematic-debugging, then test-driven-development and verification-before-completion. Execute inline in the existing operator-selected checkout, with durable checkpoints. Do not start hardware work.

**Goal:** Resolve the native-trained T3B checkpoint's strict deterministic inference gap without retraining, changing its weights, or changing an acceptance tolerance.

**Architecture:** Reuse the immutable 3,000-update expert-only export and preserve its pre-comparison PyTorch floor. The boundary trace isolated precision loss in the reference loader: initialize/cast its parameters before loading saved values. Prove exact stored-weight identity on CPU fp32, CPU fp64, and MPS fp32 before inference. New self-consistency measurements are informational only; a separate fixed-threshold repair evaluation does not replace or reinterpret the original prospective verdicts. Keep the default Metal runtime unchanged.

**Tech Stack:** MLX 0.32.2, pinned PyTorch/LeRobot reference extras, Python 3.12, pytest, optional native CPU compatibility primitives.

**Spec:** `AGENTS.md`, `docs/history/BRIEF_FULL.md`, `docs/history/BRIEF_T3B.md`, `docs/evidence/PARITY_PROCEDURE_TRAINED.md`, and the operator's 2026-09-04 software-only continuation request.

## Global Constraints

- No robot, cameras, serial ports, or `~/robot/so101`; hardware is disconnected.
- Preserve both historical T3 and T3B failure records. Original T3 failure SHA-256: `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
- Preserve checkpoint SHA-256 `858704fa572501d9e5a048076f8da692693b90c463feda29201a72f3f0b18883`, all 56 held-out cases, stored noise, and base report.
- Original prospective floor SHA-256 `28d83926a70e507671bfd694e032f81b71093d475075aad627b3c24c5b334efc` predates every comparison. Never replace it with a post-hoc floor. A corrected-loader informational envelope is not labeled prospective and never changes a tolerance or historical milestone.
- Fixed gates: image `<= 1e-5`, state `<= 1e-6`, fine/base MAE `<= 0.9`, Torch/MLX MAE `[0.95, 1.05]`; normalized action `<= max(0.005, 3F)`, currently `0.005`.
- New repair outputs and start markers are no-clobber; the repair harness enforces chronology and current input hashes. Historical support snapshots prove only the old floor, never the implementation executed by a new comparison. Leave the original trained-parity evaluator unchanged.
- Production remains the default Metal path; strict is explicit CPU. No Torch, Transformers, or LeRobot imports in runtime.
- No benchmarks while tests, training, or floor work compete. Maintain at least 40 GiB free disk. No uploads/publication; commit and push source/evidence only.

### Task 1: Baseline and prospective evidence audit

**Files:** Existing `docs/history/PROGRESS.md`; new `.cache/training/parity-repair-20260904/` diagnostic artifacts.

- [x] Run `make test`, retain `.cache/training/parity-repair-20260904/baseline-tests.log`, and record the exact result and starting commit `f41594a`: 773 passed in 746.12 seconds, no skip/xfail.
- [x] Rehash the original floor, nine raw worker bundles, export, case tree, and failure records. Use `validate_floor_bundle` and the existing comparison's `floor_input_evidence` to identify renamed support sources.
- [x] Recover any floor-bound historical support bytes into the ignored diagnostic directory using their exact git revision; never overwrite current or original evidence. Verify every recovered file against its recorded hash.
- [x] Before diagnostic inference, create a fresh one-shot start marker against the original floor with `scripts/start_trained_comparison.py`, binding a new diagnostic report path.

### Task 2: Reproduce and isolate the divergence

**Files:** Ignored `.cache/training/parity-repair-20260904/trace_case.py` and trace artifacts, new `docs/evidence/TRAINED_PARITY_REPAIR.md`.

- [x] Reproduce ordinal 24 (episode 28, frame 87, absolute index 6307) with original weights/noise and record all ten Euler velocity/action maxima. Exact reproduction: `0.013038858771324158`.
- [x] Record pixels, vision, connector, prefix, all 16 K/V pairs, and suffix/velocity/action trajectories. Diagnostic outputs explicitly are not a release verdict.
- [x] Teacher-force exact reference prefix/cache and per-layer operator inputs. The reference query projection matches bf16-rounded export weights exactly, but differs from original fp32 weights by `0.0045614540576934814`.
- [x] Write the supported root cause and regression-first loader fix design before changing implementation. See `docs/evidence/TRAINED_PARITY_REPAIR.md`.

### Task 3: Regression-first repair and canonical evaluation

**Files:** `training/reference_export.py`, new `tests/test_reference_export_precision.py`, and `docs/evidence/TRAINED_PARITY_REPAIR.md`. No MLX inference or training algorithm changes are indicated.

- [x] Add a real-checkpoint parameter-identity regression with independent fp32-only probe values. CPU fp32 fails with four rounded values; save `precision-red-2.log`.
- [x] Implement one bounded correction. Establish destination dtype before strict state loading; no checkpoint, MLX runtime, or tolerance changes.
- [x] Run the new regressions and existing export/floor/evaluator/import-isolation tests: 123 passed. `make test-fast`: 482 passed, 294 deselected (before the fourth new regression was added).
- [ ] Freeze the corrected reference source and run `scripts/compute_self_consistency_floor.py --checkpoint .cache/training/t3b/export --evaluation-dir .cache/training/t3-evaluation --work-dir .cache/training/parity-repair-20260904/self-consistency --output .cache/training/parity-repair-20260904/reference-envelope-v3.json --purpose retrospective_diagnostic`. Record actual T3B path/hashes and explain the v3 schema's legacy T3-only context fields.
- [ ] Write a fresh repair-start manifest after the informational envelope and before inference, binding current implementation/input hashes, original history, exact output paths, and all original fixed limits.
- [ ] Run the existing all-56 `run_finetune_outcome_evaluation` inside a newly reserved private directory. Rehash inputs/history and enforce envelope/start/outcome chronology before no-clobber installation of the repair report. Record every normalized, physical, preprocessing, improvement, and round-trip result. Do not claim original T3B derived acceptance or use the envelope to increase a limit.
- [ ] Commit the independently passing repair and its evidence. A lower diagnostic error alone is not completion.

### Task 4: Complete software verification and handoff

**Files:** `docs/evidence/TRAINED_PARITY_REPAIR.md`, `docs/history/PROGRESS.md`, `docs/history/STATUS_FULL.md`, training limitations in current docs, and `docs/evidence/DIST_MANIFEST.md` if runtime artifacts change.

- [ ] Run `make test` on final code and record exact totals. Preserve hardware limitations and original historical verdicts.
- [ ] If runtime changes, rebuild sdist and Python 3.11/3.12/3.13 wheels, repeat isolated base/serve/hardware-import/cache-shim smokes without hardware, and refresh the manifest with exact hashes.
- [ ] Review `git diff --check`, artifact import isolation, historical failure hashes, and the complete changed-file diff. Update current status only to milestones actually verified.
- [ ] Commit and push to `origin`; verify clean worktree and remote commit. Hand off the software result and retain physical validation for when the operator returns home.
