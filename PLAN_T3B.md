# T3B and Full-Scope Completion Implementation Plan

This plan executes `BRIEF_T3B.md` in the order mandated by its Section 4 while
preserving the fixed gates and historical conclusions in `BRIEF_FULL.md` and
`FAILURE_LORA_FINETUNE.md`. `BRIEF_RELEASE.md` is normative for Stage R.

## Goal

Complete a reproducible second LoRA attempt whose PyTorch self-consistency floor
is established before any MLX comparison, package the project for practical MLX
use on Apple silicon, add the specified training and quality work, and finish
with documentation-only hardware handoff. No hardware access or uploads are part
of this plan.

## Invariants

- Work only in `/Users/dan/Desktop/workshop/robotics-mlx-contrib` on `main`.
- Keep Hugging Face, uv, and model caches under the repository `.cache/` tree.
- Never access `~/robot/so101`, serial ports, credentials, or physical hardware.
- Never upload model artifacts or datasets. The only network publication is
  pushing source/history to the configured `origin` repository.
- Do not edit, reopen, or reinterpret `FAILURE_LORA_FINETUNE.md` or its original
  fixed gates.
- Do not change fixed tolerances: image `1e-5`, state `1e-6`, trained/base MAE
  ratio `0.9`, Torch/MLX MAE ratio `[0.95, 1.05]`, and deterministic fallback
  `0.005` with multiplier `3`.
- For every trained checkpoint, write and hash its PyTorch self-consistency floor
  before producing any MLX-versus-PyTorch comparison artifact. The evaluator
  must reject a comparison whose timestamp precedes the floor.
- Do not collect timing data while training or floor computation is active.
  Before each timing package, verify the relevant processes are absent and
  record that the machine was idle.
- Preserve the last three checkpoints and keep large run data ignored and local.
- Add behavioral tests first, observe the relevant failure, implement the minimum
  change, and rerun focused tests before the full suite.
- Commit every passing package with a `phase-N:` message and push completed
  packages to `origin`. Do not add agent attribution to GitHub metadata.

## Recorded kickoff state

- [x] Commit `BRIEF_RELEASE.md` and `BRIEF_T3B.md` before reading them.
- [x] Read all required specifications and failure/status records in full.
- [x] Mark the missing release specification blocker resolved without deleting
  its historical record, and close its `HUMAN_TASKS.md` entry.
- [x] Run the pre-change baseline: `308 passed` via `make test`; test command
  elapsed `231.59s` (`226.41s` pytest duration).
- [x] Record kickoff evidence and free disk in `PROGRESS.md`.

## Task 1 — T3B-1: failed-checkpoint self-consistency floor

**Status: complete.** The authoritative v3 report is written and hashed before
any new MLX comparison; `SELF_CONSISTENCY_T3.md` records the evidence and the
historical MPS variability disclosure.

**Files:**

- Add `training/self_consistency.py`.
- Add `scripts/compute_self_consistency_floor.py`.
- Add `tests/test_training_self_consistency.py` and CLI coverage.
- Generate `.cache/training/t3/floor.json`.
- Add `SELF_CONSISTENCY_T3.md`; update `PROGRESS.md` and `STATUS_FULL.md`.

**Implementation:**

1. Define a canonical, versioned floor schema containing checkpoint/export
   digests, evaluation-set and stored-noise digests, processor/model identifiers,
   perturbation descriptions, per-case normalized action-chunk maxima, `F`,
   `F64`, UTC creation time, and nanosecond creation timestamp.
2. Reuse the existing 56 held-out cases and stored noise exactly. Do not create a
   new draw or split.
3. Run each PyTorch perturbation in an isolated worker process so thread settings,
   MPS fallback state, and dtype cannot leak between runs:
   CPU fp32 golden path; CPU fp32 one thread; CPU fp32 maximum threads; MPS with
   fallback enabled; and CPU float64. Because repeated clean MPS launches showed
   materially different backend outcomes, execute perturbation (d) in a fixed,
   prospectively frozen five-process empirical envelope. All five slots are
   required, share one driver/device, and make no statistical-independence or
   upper-bound claim; the count does not change in response to results.
4. Compare every non-golden action chunk with the golden result using the
   existing normalized action metric. Define `F` as the maximum across cases and
   all perturbations (b)–(e), including all five (d) process slots, and also
   report the float64-only component as `F64`.
5. Before NumPy or Torch import, clear and record the documented MPS and CPU
   thread-control variable sets, enable only MPS fallback for (d), and
   seed Python, NumPy, Torch, and MPS with the fixed worker seed. Keep ordinary
   deterministic-algorithm mode disabled and record the matmul-precision mode.
6. Write the machine-readable artifact atomically, hash it, then write
   `SELF_CONSISTENCY_T3.md`. Include the original trained-checkpoint MLX
   discrepancy (`0.17762404680252075`) only as context and make no new verdict
   about the historical T3 outcome.

**Verification:**

- Unit tests reject changed case/noise/checkpoint digests and malformed or
  incomplete perturbation tables.
- CLI test proves deterministic aggregation and atomic output without loading the
  full model.
- Run the real floor command with no training or timing process active.
- Run focused tests and `make test`, then commit and push.

## Task 2 — T3B-2: prospective parity procedure and evaluator

**Status: complete.** The fixed and derived gate procedure, real-clock start
marker, concrete input-evidence contract, semantic conversion audit, and
no-clobber evaluator were frozen before any T3B checkpoint or comparison
existed. The focused evaluator suite passes 52/52, the related contract and
wheel suite passes 97/97, and the exact package tree passes 402/402.

**Files:**

- Add `smolvla_mlx/training/trained_parity.py`.
- Add `scripts/evaluate_trained_parity.py`.
- Add `tests/test_trained_parity.py` and CLI coverage.
- Add `PARITY_PROCEDURE_TRAINED.md`; update status/progress records.

**Implementation:**

1. Encode the fixed preprocessing, held-out improvement, and Torch/MLX MAE-ratio
   gates without modifying their constants.
2. Compute the deterministic trained-checkpoint threshold as
   `max(0.005, 3 * F(checkpoint))` and retain the fixed fallback and multiplier in
   the artifact schema.
3. Require the floor artifact to match the exact checkpoint/export, 56 cases,
   stored noise, processor, and procedure version used by the comparison.
4. Require the floor artifact's creation timestamp and file timestamp to precede
   the MLX comparison artifact. Reject missing, malformed, future, mismatched, or
   post-comparison floors before reading a comparison verdict.
5. Produce a single auditable result that distinguishes fixed-gate outcomes from
   the derived deterministic gate and never mutates the floor.

**Verification:**

- Tests cover equality boundaries and both sides of every fixed range.
- Tests prove timestamp-order, digest, checkpoint, case/noise, and schema
  rejection, including a floor timestamp one nanosecond later than comparison.
- Run focused tests and `make test`, then commit and push.

## Task 3 — T3B-3a: expert-only LoRA configuration and background launch

**Status:** implementation, isolated-launch hardening, a real disposable
update-1 checkpoint probe, independent review, pre-launch full-suite
verification, commit/push, canonical configuration generation, and the
background launch are complete. The 3,000-update run subsequently completed
without resume or recovery; Task 5 owns its completed floor/evaluation and
gates.

**Files:**

- Update `smolvla_mlx/training/lora.py`, `finetune.py`, configuration schemas,
  the isolated `scripts/finetune_lora` launcher, and its Python entrypoint.
- Add or update focused LoRA topology, configuration-hash, checkpoint, and resume
  tests.
- Update `ARCHITECTURE.md`, `PROGRESS.md`, and `STATUS_FULL.md`.

**Implementation:**

1. Verify the installed reference implementation's default freeze policy and
   record the inspected package version/source and result in `ARCHITECTURE.md`.
2. Add an explicit LoRA scope. Preserve the original full scope for historical
   T3 compatibility; make T3B's scope exactly the expert transformer's attention
   and MLP linear layers, excluding the vision tower, language-prefix decoder,
   state projection, and action input/output projections.
3. Freeze and hash a T3B run configuration for rank `8`, alpha `16`, dropout `0`,
   `3000` updates, effective batch `8`, the original seed/draw/split, and the
   exact checkpoint/export/evaluation inputs. Disable budget-selection timing so
   the precommitted update count cannot change.
4. Prove trainable tensor names/counts, optimizer coverage, adapter merge, last-
   three checkpoint retention, interrupted checkpoint recovery, and exact resume.
5. Start the real T3B run from the repository root with the exact committed
   launcher below. It starts Python with `-I -S` before Python startup can load
   site hooks, uses repository-local caches, binds the run-directory log and
   PID/identity metadata, and performs no budget-selection benchmark or timing.
   Required per-update, wall-clock, throughput, and peak-memory evidence remains
   enabled. Record the immutable configuration digest and launch evidence before
   proceeding.

   ```sh
   nohup scripts/finetune_lora --checkpoint-interval 100 \
     --cache-dir .cache/hf \
     --native-cache .cache/smolvla_mlx/policy-float32 \
     --output .cache/training/t3b \
     --lora-scope expert_only --budget-mode fixed_steps \
     --launch-config .cache/training/t3b/launch.json \
     --log-file .cache/training/t3b/training.log \
     </dev/null >/dev/null 2>&1 &
   ```

   Only after a `run.json` exists, restart an interrupted run with the same
   command plus `--resume` (equivalently, `make lora-finetune-resume`).

**Verification:**

- New topology/configuration tests fail before implementation and pass afterward.
- Full suite passes before launch.
- Inspect the live process and initial checkpoint/log evidence; do not wait for
  completion before starting the allowed non-timing release work.
- Commit and push code/configuration before or immediately after launch.

## Task 4 — Stage R non-timing packages while T3B trains

Only P0-1, P0-2, P0-3, P1-2, and P1-3 may run in this interval. Recheck the
training PID between packages. Do not run benchmarks, floor computation, P1-1,
P1-4, Stage Q, or any command that reports performance timing.

### Task 4.1 — P0-1: repository, license, and backup readiness

- Verify `origin`, upstream reconciliation evidence, ignored-artifact policy, and
  source history without modifying external repositories.
- Add the complete Apache-2.0 `LICENSE` and README/license metadata references.
- Add static tests for license/package metadata where useful.
- Run focused tests and `make test`; update status/progress; commit and push.

### Task 4.2 — P0-2: stats-active conversion and loading

**Status:** Complete. Dependency-light active mean/std loading, base identity
preservation, checkpoint-derived diagnostics, observed camera-slot behavior,
the synthetic stats-active target, and the pinned public fine-tune all pass the
unchanged deterministic and 50-frame statistical gates in fp32 and bf16.

- Place the true dataset statistics in the designated reference artifact tree and
  hash their provenance.
- Build the second eight-sample golden corpus with the fixed stored noise and run
  the unchanged fp32/bf16 parity ladder.
- Make arbitrary matching local paths and Hub identifiers load through
  `from_pretrained`; emit actionable errors for missing/mismatched observation
  keys and document observed one-/two-/three-camera behavior.
- Perform the capped public fine-tune search only within 90 minutes and 5 GiB;
  record a negative result without blocking if no compliant artifact exists.
- Add conversion/loading/error tests first, then run the full suite, update
  architecture/status/progress, commit, and push.

### Task 4.3 — P0-3: cache audit and safe cleanup

**Status:** Complete. After T3B exited, the frozen allowlist removed only 23
top-level `debug-*` trees and exact `benchmark-debug`, reducing the native
cache from 91,447,880 KiB to 52,764,872 KiB. Inventory, dry-run, traversal,
symlink, root, retained-evidence, and post-cleanup full-suite checks are green.

- Add an inventory command that reports repository cache categories, sizes,
  retention status, and whether each path is regenerable.
- Add a narrowly scoped cleanup target that refuses paths outside the repository,
  preserves retained checkpoints/evidence by default, and supports a dry run.
- Document cache layout and cleanup in README. Test traversal, symlink, root, and
  retained-evidence protections; run the full suite; commit and push.

### Task 4.4 — P1-2: portable normalization and distribution artifacts

**Status:** Complete. The base runtime supports Python 3.11–3.13; LeRobot's
reference extra is correctly gated to 3.12+; the native CPU-reference extension
has an extension-free pure-MLX fallback; and the sdist plus all three
`macosx_14_0_arm64` wheels pass fresh-install, import-isolation, native-backend,
and offline saved-observation prediction smokes. `DIST_MANIFEST.md` records the
artifact hashes and the pinned MLX dylib's separate 26.2 binary-floor caveat.

- Add an optional native MLX RMSNorm path with a numerically equivalent fallback,
  exercised in both modes.
- Declare the supported deployment target and build sdist plus wheels for Python
  3.11, 3.12, and 3.13 using only repository-local build caches.
- Install each artifact into a fresh environment and run import plus tiny-model
  smoke tests. Store build manifests/hashes, not environment caches, in history.
- Run the full suite, update status/progress, commit, and push.

### Task 4.5 — P1-3: release documentation and LeRobot GPU training path

**Status:** Complete. The release README now has the verified pitch, install
matrix, exactly ten-line executable API example, CLI quickstart, real-time
benchmark framing, immutable correctness methodology and active-statistics
evidence, strict-versus-production boundary, cache contract, limitations,
troubleshooting, and license/NOTICE links. The pinned LeRobot 0.6.1 GPU command
was schema-checked with both upload paths disabled, and its local output can be
loaded here by path or a deliberately published Hub ID. All non-hardware
commands are covered by direct smokes or their already-recorded package gates;
the exact tree passes 584/584 tests.

- Polish README installation, conversion, inference, cache, strict-parity versus
  production-mode, known-limitations, and troubleshooting sections.
- Document a reproducible LeRobot GPU fine-tune path and clearly separate it from
  this project's native MLX training path.
- Validate every documented command with non-hardware smoke tests; run the full
  suite; update status/progress; commit and push.

## Task 5 — T3B-3b: evaluate training and apply gates

**Status:** Complete. The fixed gates pass, only the derived deterministic
gate fails, and the result is recorded as `TRAINING ALPHA (STATISTICAL)` in
`LORA_SCOPE_COMPARISON.md` and `FAILURE_LORA_FINETUNE_B.md`.

1. Wait for the background run to terminate successfully and verify exactly 3000
   updates, retained checkpoint integrity, log completeness, and final adapter.
2. Merge/export fp32 to the exact LeRobot layout and repeat the existing export
   audit before any parity decision.
3. Finalize and test the comparison-artifact producer without loading or
   evaluating the T3B checkpoint. The legacy `check_lora_finetune.py` command
   cannot run here because it performs MLX inference before writing its outcome.
4. With no timing process active, compute the trained T3B PyTorch-only floor using
   the T3B-1 procedure. Atomically write it, compute its SHA-256, and record the
   path, digest, timestamp, checkpoint/config digests, and idle/non-timing state in
   `PROGRESS.md` before invoking the MLX comparison command.
5. Only then create the one-shot comparison marker and produce the single bound
   comparison artifact. Run fixed gates in their required logical order inside
   that producer: preprocessing, held-out improvement, and Torch/MLX roundtrip;
   then run the T3B-2 evaluator for the derived deterministic gate.
6. Write a side-by-side original-T3/T3B report containing held-out MAE, improvement
   ratio, `F`, MLX-versus-reference normalized maximum, amplification, wall time,
   and peak memory. Use already captured training resource data; do not rerun
   training for timing.
7. If all fixed and derived gates pass, record `TRAINING ALPHA`. If fixed gates
   pass and only the derived gate fails, write `FAILURE_LORA_FINETUNE_B.md` and
   record `TRAINING ALPHA (STATISTICAL)`. T4/T5 depend only on the fixed outcomes.
8. Run focused tests and `make test`; update status/progress; commit and push.

## Task 6 — Remaining Stage R

Before timing work, verify no training or floor worker is alive and append an idle
machine declaration with process-check evidence to `PROGRESS.md`.

### Task 6.1 — P1-1: production Metal execution and benchmark table

- Define strict-parity and production execution modes explicitly. Optimize the
  production path for Metal/unified memory without weakening strict parity.
- Collect the specified fp32 and bf16 latency/memory tables only on an idle
  machine, with fixed warmups/repetitions and environment/configuration hashes.
- Test mode selection and output provenance; run full suite; update benchmark,
  status, and progress documents; commit and push.

### Task 6.2 — P1-4: asynchronous gRPC serving

- Add the exact LeRobot 0.6.1-compatible asynchronous service schema, `serve`
  dependency extra, server CLI, and reference client.
- Cover serialization, validation, error propagation, cancellation, concurrency,
  and a localhost loopback inference smoke test. No hardware control is included.
- Document lifecycle/security boundaries; run full suite; update status/progress;
  commit and push.

## Task 7 — T4: full fine-tune code and smoke only

- Add full-parameter training mode with an explicit configuration distinct from
  LoRA; do not schedule a multi-thousand-step full run.
- Prove all intended parameters are trainable and optimizer-covered.
- Run the specified 100-update smoke, retain the last three checkpoints, export a
  loadable artifact, and validate finite loss/action outputs.
- For both LoRA and full modes, prove uninterrupted versus interrupted/resumed
  equivalence using configuration, optimizer, RNG, sampler, and step-state hashes.
- Record smoke resource observations as non-benchmark evidence. Run full suite,
  update status/progress, commit, and push.

## Task 8 — T5: training resource evidence

- Proceed only if T3B fixed gates pass.
- Verify no training/floor process is alive and record idle state before measuring.
- Run the normative LoRA training benchmark protocol and produce reproducible
  wall-time, throughput, and peak unified-memory evidence with configuration and
  environment hashes.
- Keep functional conclusions distinct from performance conclusions. Run full
  suite, update training/benchmark/status/progress documents, commit, and push.

## Task 9 — Stage Q

Each experiment gets a frozen config, machine-readable output, focused tests,
full-suite run, status/progress update, commit, and push. Timing experiments
require a new idle-process declaration.

1. **P2-1:** collect the prescribed PyTorch MPS comparison against the same cases,
   model, dtype, warmups, repetitions, and metric definitions.
2. **P2-2:** isolate the bf16 latency anomaly through a bounded component matrix;
   report evidence without changing parity tolerances.
3. **P2-3:** evaluate VLM-only 8-bit and 4-bit quantization while keeping vision
   and expert modules bf16; ship a preset only if all specified quality/parity
   gates pass, otherwise retain the negative experiment record.
4. **P2-4:** add a macOS 15 CI workflow when a capable runner is available, or a
   disabled, syntactically validated workflow with explicit activation and secret/
   cost requirements when it is not.

## Task 10 — Stage H: documentation-only hardware handoff

- Write `HARDWARE_RUNBOOK.md` for an operator, including prerequisites, exact
  non-destructive commands, rollback/abort boundaries, expected observations, and
  how to attach results.
- Add a serve-side latency logging script and tests using simulated/local loopback
  requests only.
- Do not open serial ports, execute actions, access robot directories, or claim a
  hardware result. Run the full suite; update status/progress; commit and push.

## Task 11 — Final verification and handoff

1. Confirm no background training/floor/serve test process remains.
2. Run formatting/static checks, packaging checks, documented smoke commands, and
   the complete `make test` suite from a clean-enough checkout.
3. Audit generated and tracked files for secrets, absolute private paths, large
   artifacts, upload destinations, placeholders, weakened gates, and accidental
   changes to the immutable T3 failure record.
4. Reconcile every `BRIEF_FULL.md`, `BRIEF_RELEASE.md`, and `BRIEF_T3B.md`
   deliverable against its artifact and evidence. Add milestone lines to
   `STATUS_FULL.md` only when their gates have actually been reached.
5. Commit the final verified state, push `main` to `origin`, verify the remote head
   matches local `HEAD`, and report exact commits, tests, artifacts, gates, and any
   bounded negative findings.
