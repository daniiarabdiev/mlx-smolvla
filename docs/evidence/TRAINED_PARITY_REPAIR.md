# Trained-checkpoint reference precision repair

## Status

Investigation in progress, 2026-09-04. No acceptance verdict has changed yet.
The operator authorized software-only completion while the robot is powered
off. No hardware, camera, serial port, or vendor-tree access is involved.

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

## Local reproduction artifacts

Ignored run directory: `.cache/training/parity-repair-20260904/`.

- `baseline-tests.log`: unmodified-code full-suite run from `f41594a`.
- `trace_case.py`, `diagnostic-start.json`, `diagnostic-baseline.json`, and
  `reference-trace.npz`: read-only worst-case and teacher-forced diagnostics.
- `precision-red-2.log`: genuine regression failure before the fix.
- `floor-source/`: exact archived support bytes used to verify historical
  floor inputs. No generated weights or private observation data enter git.
