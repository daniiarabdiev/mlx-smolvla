# Trained-checkpoint reference precision repair

## Status

**Strict-parity repair validated, 2026-09-04.** The unchanged retained T3B
checkpoint passes all 56 cases and every original fixed limit with the
corrected reference loader. This is a separate software-repair result, not a
retrospective reassignment of the original T3/T3B milestones. The complete
software suite passes **790/790 tests**, with no skip or expected failure.

The operator authorized software-only completion while the robot is powered
off. No hardware, camera, serial port, or vendor-tree access was involved.

## All-56-case result

| Check | Repaired result | Unchanged fixed limit | Result |
| --- | ---: | ---: | --- |
| Image preprocessing maximum | `3.5762786865234375e-7` | `1e-5` | Pass |
| State preprocessing maximum | `0.0` | `1e-6` | Pass |
| Normalized action maximum | `0.000021457672119140625` | `0.005` | Pass |
| Physical action maximum | `0.00042724609375` | `0.005` | Pass |
| Standardized physical maximum | `0.000021445659513119608` | `0.005` | Pass |
| Fine-tuned/base MLX MAE ratio | `0.486008430646319` | `<= 0.9` | Pass |
| Torch/MLX MAE ratio | `1.00000078541285` | `[0.95, 1.05]` | Pass |

The MLX held-out MAE is unchanged at `2.2550044155546596` versus frozen base
`4.639846293521779`. Corrected Torch MAE is `2.2550061866641045`. All 56
frozen cases, saved noise, train-only statistics, checkpoint tensors, native
conversion, and original training/evaluation source digests are unchanged.
No retraining, tolerance adjustment, or native inference arithmetic change was
needed. Strict inference here is the explicit CPU compatibility mode; this
does not change the separate default-Metal deterministic limitation.

The result validates one retained 3,000-update expert-only LoRA run. Native
training remains a research preview; full fine-tuning has code/smoke evidence,
not a long-run task-quality study. Physical robot task success is not claimed.

## Demonstrated cause

The old `TorchExportPolicy.load` called LeRobot's `from_pretrained` before
casting the constructed policy to the requested dtype. The nested expert is
initialized from a backbone configuration with bfloat16 storage. Safetensors'
strict loading verifies keys and shapes, but copies fp32 export values into
those bfloat16 parameters. A later `.to(float32)` cannot restore the discarded
bits. The MLX conversion audit was exact; the comparison was not evaluating
the same stored expert weights in both frameworks.

The original worst case (ordinal 24, episode 28, frame 87, absolute index
6307) reproduces exactly: normalized action maximum
`0.013038858771324158`. Supplying exact reference prefix K/V to the native
expert leaves `0.013046815991401672`, locating this case's error downstream
of the prefix.

For expert layer 0's query projection, with an identical saved reference input:

| Weight values used by Torch `linear` | Maximum difference from recorded reference projection |
| --- | ---: |
| Original fp32 export | `0.0045614540576934814` |
| Export rounded to bf16, then widened to fp32 | `0.0` |

This is a reproducible precision-loss defect, not evidence for changing a
tolerance or adding sample-specific runtime arithmetic.

A new real-checkpoint regression puts four independent fp32-only literals in
the expert query weight, loads through the production reference adapter, and
compares every stored parameter exactly. Before the fix it fails on all four
probe values, with maximum weight error `0.0006510317325592041`. It performs
no policy inference and uses no held-out data.

## Implemented repair

Source checkpoint `5180912` constructs the reference policy, casts its CPU
destination parameters to the requested dtype, strictly loads the saved
safetensors values, and only then transfers it to the requested device. All
500 tensors / 450,046,176 scalars in the retained T3B export now match their
stored values exactly. The old load order rounded 98,220,177 scalars across
112 adapted tensors through bf16.

Real-checkpoint regressions pass on CPU fp32, CPU fp64, and MPS fp32; missing
or unexpected state keys still fail. The existing focused export/floor/parity/
import suite passes 123 tests. The default Metal runtime, native MLX training
math, fixed tolerances, and checkpoint bytes are unchanged.

## Protected evidence and chronology

- Original T3 failure SHA-256:
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
- Original T3B failure SHA-256:
  `3bc6bb3cb7302797fa39c80309375c343a9348da18a18fbe2c08e504f4b37276`.
- Unchanged T3B export model SHA-256:
  `858704fa572501d9e5a048076f8da692693b90c463feda29201a72f3f0b18883`.
- Original floor SHA-256:
  `28d83926a70e507671bfd694e032f81b71093d475075aad627b3c24c5b334efc`;
  created `2026-09-02T00:38:00.730626+00:00`, actual mtime
  `1788309480735640183` ns. It precedes the original and diagnostic comparisons.
- Original nine-worker raw bundle SHA-256:
  `31ce3db6619294432742b38214132267cfecf735dc0ce1d98199bbd223e8a889`.
- New diagnostic start marker SHA-256:
  `60c847eb8590186d540a1494db5f697771c684fb86993aaecd5866b33d7afbe5`,
  created at `1788502483023418000` ns, before diagnostic model work.

All 337 floor-bound files validate: seven checkpoint files, 282 evaluation
files, five dataset files, ten tokenizer files, and 33 implementation files.
Four historical support files changed during subsequent development/rename;
their exact floor-bound bytes were recovered from commit `54a4e0b` into an
ignored snapshot, not substituted into the working source or original evidence.

The original floor measured the old reference loader. It and the original
failure verdicts remain historical facts and will not be overwritten. This
checkpoint has already been compared, so a new floor cannot retrospectively
qualify as its original prospective floor. Independent review caught that
distinction before any corrected-loader floor or full-model comparison ran.

Before the corrected-loader full-model comparison, rerun the same nine
PyTorch-only perturbations as **informational self-consistency**, not a new
derived gate. The legacy v3 diagnostic envelope includes original T3 context
(`retained-t3-merged-fp32-export` and `0.17762404680252075`); those are legacy
schema fields, not new T3B measurements. Its actual checkpoint path, file
hashes, worker outputs, `F`, and `F64` identify the current T3B experiment.

A separate no-clobber repair-start manifest binds the original history,
informational envelope, exact current implementation/input hashes, output
paths, and fixed thresholds before inference. The existing all-56 fixed
outcome evaluator measures the repair. Its original image/state/improvement/
round-trip limits and `0.005` normalized, physical, and standardized-physical
limits remain unchanged; no new floor can increase them. Rehash inputs and
check the envelope/start/outcome chronology before installing a separate repair
verdict. Do not combine archived floor implementation bytes with corrected
comparison evidence, modify the original trained-parity evaluator, or claim
original T3B derived acceptance from this post-fix validation.

The corrected-loader nine-worker envelope completed with
`F = 0.000025600194931030273` and `F64 = 0.000023670888653737343`.
Its SHA-256 is
`06e48641f235e74c2c4ddf8fc8e885867499fdaf1cb206fa50ba5b82c26af06f`,
creation time `1788504562305003000` ns, and actual mtime
`1788504562311518199` ns. It was written before repaired cross-framework
inference. No timing benchmark ran during its computation.

The separate repair evaluator is committed as `8b5c485`:
`scripts/check_trained_parity_repair.py`. Its 13 regressions cover chronology,
no-clobber outputs, input replacement, frozen-baseline integrity, complete
sample evidence, physical-action limits, and publication-time mutation. The
start marker binds all 337 floor inputs, current runtime/reference sources,
both original and corrected raw worker bundles, original failure/evaluation
records, and the actual native converted weights/name map (454 file bindings).
Every canonical output is installed without replacement; all original
thresholds are retained.

The actual repair chain is:

- Envelope creation/write: `1788504562305003000` /
  `1788504562311518199` ns.
- Start creation/write: `1788504791896025000` /
  `1788504791897735127` ns. Marker SHA-256:
  `c53864d301507f4e53c4c31d791874c4e613d90044198bf991a7988f4d01094d`.
- Outcome write: `1788505075900093547` ns. Full outcome SHA-256:
  `f98b6394454a6682e0f96fb54f9c5f977d6b6a7c3866cff48cf081ebdac9b07e`.
- Separate repair verdict SHA-256:
  `213e7d7a662a61f757328895e1aa0bcff2932689a6429acd663a6898a7383d85`.

The evaluator enforced this ordering and rehashed all bindings after inference
and again at final publication. Both protected failure-file hashes still
match. Independent review found no remaining issue in the final publication
and actual-conversion binding checks.

## Reproduction

Use the pinned reference environment and retained repository-local inputs.
The work/output directories must be new; the commands intentionally refuse to
overwrite evidence. Do not run training or performance benchmarks concurrently.

```bash
.venv/bin/python scripts/compute_self_consistency_floor.py \
  --checkpoint .cache/training/t3b/export \
  --evaluation-dir .cache/training/t3-evaluation --cache-dir .cache/hf \
  --work-dir .cache/training/parity-repair-20260904/self-consistency \
  --output .cache/training/parity-repair-20260904/reference-envelope-v3.json \
  --purpose retrospective_diagnostic

.venv/bin/python scripts/check_trained_parity_repair.py \
  --envelope .cache/training/parity-repair-20260904/reference-envelope-v3.json \
  --variants .cache/training/parity-repair-20260904/self-consistency/variants \
  --output-dir .cache/training/parity-repair-20260904/fixed-repair
```

For lossless optional PyTorch reference loading of an fp32 native export,
use `training.reference_export.TorchExportPolicy.load(export, cache_dir=...)`.
With the pinned LeRobot version, calling its `from_pretrained` first and then
casting to fp32 does not preserve every fp32 export value. This adapter is
reference tooling, not an import in the dependency-light MLX runtime.

## Local reproduction artifacts

Ignored run directory: `.cache/training/parity-repair-20260904/`.

- `baseline-tests.log`: unmodified-code full-suite run from `f41594a`.
- `trace_case.py`, `diagnostic-start.json`, `diagnostic-baseline.json`, and
  `reference-trace.npz`: read-only worst-case and teacher-forced diagnostics.
- `precision-red-2.log`: genuine regression failure before the fix.
- `floor-source/`: exact archived support bytes used to verify historical
  floor inputs. No generated weights or private observation data enter git.

## Closing verification

- Baseline at `f41594a`: `make test`, **773 passed in 746.12 seconds**.
- Focused precision/export/floor/evaluator/import checks: **123 passed**.
- Added evidence-publication guards: **13 passed**.
- Closing `make test`: **790 passed in 774.76 seconds**, no skip or xfail.
  These are correctness-run durations, not throughput benchmarks. No training
  or floor was active; non-inference builds/installations overlapped part of
  the closing suite, and installed model probes ran separately afterward.
- Final distribution, release, naming, repository-hygiene, import-isolation,
  and repair-guard checks after refreshing the documentation and artifacts:
  **47 passed in 14.38 seconds**.
- The lock resolves 122 packages. Actionlint passes with only the deliberately
  disabled hosted-CI condition excluded. Both original failure records, the
  original floor, and the retained export still match their protected hashes.
- Clean pushed source `8bb5c7e` produced a new sdist and CPython 3.11/3.12/3.13
  wheels. All seven fresh-install environments pass: four base predictions and
  import/doctor/backend checks, serving/quantization, hardware-import-only,
  cache compatibility, and reference/training exact-weight checks. The actual
  trained export has zero stored-weight mismatches on CPU fp32, CPU fp64, and
  MPS fp32 from the installed package. See [`DIST_MANIFEST.md`](DIST_MANIFEST.md)
  for artifact hashes and retained local logs. Previous distribution bytes
  are backed up, not deleted. No distributions or weights were uploaded.
