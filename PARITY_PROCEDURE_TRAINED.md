# Prospective Parity Procedure for Trained Checkpoints

**T3B-2 COMPLETE — PROSPECTIVELY FROZEN BEFORE T3B TRAINING**

This document defines the trained-checkpoint decision required by
`BRIEF_T3B.md` Section 2. The constants and evidence contract were fixed before
a T3B checkpoint existed. They must not be adjusted after observing that
checkpoint. The original T3 verdict and `FAILURE_LORA_FINETUNE.md` remain
unchanged.

## Fixed gates

| Gate | Inclusive pass boundary |
| --- | ---: |
| Image preprocessing max-abs | `≤ 0.00001` |
| State preprocessing max-abs | `≤ 0.000001` |
| Fine-tuned/base MLX held-out MAE ratio | `≤ 0.9` |
| Torch/fine-tuned-MLX held-out MAE ratio | `[0.95, 1.05]` |

All ratios use the unrounded stored values recomputed from the 56 per-case
records. Both MLX denominators must be finite and nonzero. Every per-case and
summary metric must be finite and nonnegative.

## Derived deterministic gate

For a trained checkpoint `C`, run the versioned PyTorch-only self-consistency
procedure on the exact merged fp32 export, 56 frozen cases, stored noise, saved
processors, pinned dataset, and pinned tokenizer. Let the reconstructed envelope
be `F(C)`. The normalized action-chunk decision is:

```text
MLX-versus-PyTorch normalized max-abs <= max(0.005, 3 * F(C))
```

The fallback `0.005` and multiplier `3` are fixed. MLX is one additional
reduction order among the measured reference orders, so it may differ by a
small multiple of the reference's observed spread, while the original `0.005`
remains the minimum tolerance for a well-conditioned model. This empirical
rule is not a probabilistic or absolute error bound.

## Mandatory one-way chronology

The floor is a prerequisite, never a post-hoc explanation. For every trained
checkpoint, execution must follow this order:

1. Finish and byte-audit the merged fp32 checkpoint export without running a
   new MLX comparison.
2. Run all nine fixed PyTorch floor workers and atomically write
   `.cache/training/<run>/floor.json` plus their raw `metadata.json` and
   `normalized_actions.npy` files.
3. Record the floor path, SHA-256, embedded creation time, actual `mtime_ns`,
   and raw-bundle SHA-256 in `PROGRESS.md`.
4. Immediately before the first MLX inference, create a one-shot marker with
   `scripts/start_trained_comparison.py`. The script reads the clock itself,
   validates the raw floor, binds one absolute comparison-artifact path, and
   refuses to overwrite an existing marker.
5. Produce the bound comparison path without overwriting an existing artifact.
6. Run the file evaluator. Only its installed and hashed output is the trained
   parity verdict.

The evaluator enforces:

```text
floor.created_at_ns <= floor.mtime_ns < marker.created_at_ns
marker.created_at_ns <= marker.mtime_ns <= comparison.created_at_ns
comparison.created_at_ns <= comparison.mtime_ns <= evaluation.created_at_ns
```

The strict floor/marker boundary is the required "floor first" check. Across
evaluator runs, the persisted SHA-256 and `mtime_ns` bindings protect the floor
and marker chronology. Within one evaluator run, descriptor snapshots also bind
device, inode, size, timestamp, and bytes through final revalidation. A marker
cannot be used for a different comparison path.

## Raw floor reconstruction

The evaluator does not trust the floor JSON's stored maxima. It descriptor-reads
all nine raw action arrays and worker metadata files, verifies each array digest,
dtype, shape, runtime contract, and perturbation identity, then calls the fixed
floor assembler again. Every per-case maximum, worst-case identity, `F`, and
`F64` must exactly equal the reconstruction. A coherent edit of the JSON table
therefore fails when the raw arrays do not support it.

The raw-bundle digest binds the names and SHA-256 digests of all 18 worker files
to both the start marker and comparison artifact.

## Complete comparison evidence

The comparison artifact uses type
`smolvla-trained-checkpoint-mlx-comparison` and procedure
`smolvla-trained-checkpoint-parity-v1`. It contains and binds:

- the exact floor file, raw bundle, procedure, input tree, checkpoint path,
  source identity, and 56 ordered case identities;
- the exact start-marker file, actual file timestamp, and one comparison path;
- the complete frozen base MLX report, whose bytes must hash to the pre-training
  digest `211d6778b0530208ca2e81abe6f4002cc683e24d496a09ddbe39c100ebd4f7ce`;
- all 56 fine-tuned MLX MAE records and all 56 Torch MAE records, each with six
  metric elements, plus their totals and MAEs;
- all 56 stats-active parity records, including image, state, preprocessing,
  normalized, physical, and standardized-physical maxima; and
- actual path/SHA-256 pairs for the frozen base report, native conversion model,
  native conversion name map, and comparison implementation;
- concrete evidence locations for every file in all five floor input groups:
  exact checkpoint, evaluation/noise, and tokenizer trees plus every named
  pinned-dataset and floor-implementation input; and
- the merged export's `training_manifest.json`, whose complete pre-manifest file
  map and tensor/scalar counts must equal the actual canonical fp32 conversion.

The evaluator recomputes both MAE totals and ratios, all six parity summary
maxima, every per-case preprocessing aggregate, and the comparison's scalar
summary. It opens and hashes every named file rather than accepting digest-shaped
strings as evidence. The checkpoint and evaluation trees must contain exactly
the floor-declared relative-file inventory: an extra, missing, symlinked, or
changed entry is rejected. Hugging Face tokenizer file symlinks are accepted only
when their resolved regular-file targets remain inside the explicitly recorded
cache root; both the link identity and target bytes are snapshotted and checked
again before result installation.

The conversion check is semantic rather than declarative. The evaluator loads
the floor-bound source `model.safetensors`, the recorded native fp32 file, and
the canonical name map; it verifies the complete tensor-name bijection, dtype,
shape/layout transforms, per-tensor values, tensor count, scalar count, and all
three actual file digests. Those counts must also equal the floor-bound training
manifest. Updating a converted-file digest cannot bless a changed tensor.

## Stable reads and no-clobber output

Every input is read through one no-follow regular-file descriptor and captured
as a snapshot containing device, inode, size, `mtime_ns`, SHA-256, and bytes.
Immediately before output installation, every path is read again and must match
that snapshot. The result is written and fsynced to a same-directory temporary
file, then installed with an atomic hard-link operation that cannot replace an
existing or concurrently created result.

The result embeds the complete four evidence documents, fixed thresholds,
recomputed ratios, and individual gate decisions. Its standalone validator
again recomputes the base-report body digest, every sample aggregate, threshold,
ratio, and Boolean.

## Commands

Create the marker immediately before the comparison producer begins MLX work:

```bash
uv run python scripts/start_trained_comparison.py \
  --floor .cache/training/t3b/floor.json \
  --variants .cache/training/t3b/self-consistency/variants \
  --comparison .cache/training/t3b/comparison.json \
  --output .cache/training/t3b/comparison-start.json
```

Produce the exact path bound by that marker. This command first reconstructs
the floor bundle, rechecks its marker and all five input-hash groups, and only
then begins MLX/Torch model evaluation:

```bash
uv run --extra reference python scripts/produce_trained_comparison.py \
  --floor .cache/training/t3b/floor.json \
  --variants .cache/training/t3b/self-consistency/variants \
  --start-marker .cache/training/t3b/comparison-start.json \
  --comparison .cache/training/t3b/comparison.json \
  --outcome .cache/training/t3b/outcome.json \
  --cache-dir .cache/hf \
  --native-cache .cache/smolvla_mlx/policy-float32 \
  --run-dir .cache/training/t3b \
  --evaluation-dir .cache/training/t3-evaluation \
  --base-report .cache/training/t3-base-evaluation.json
```

After the bound comparison artifact exists, install the authoritative verdict:

```bash
uv run python scripts/evaluate_trained_parity.py \
  --floor .cache/training/t3b/floor.json \
  --variants .cache/training/t3b/self-consistency/variants \
  --start-marker .cache/training/t3b/comparison-start.json \
  --comparison .cache/training/t3b/comparison.json \
  --evidence-root . \
  --output .cache/training/t3b/parity-evaluation.json
```

The evaluator exits zero only when every fixed and derived gate passes. A valid
failed decision is still installed and hashed before the CLI exits nonzero.

## Verification status

The procedure has 52 focused evaluator tests. They cover inclusive boundaries
and immediate failures beyond every gate; fallback and `3F` branches; strict
ordering and error precedence; raw-array reconstruction; exact floor, marker,
checkpoint, source, and population binding; complete 56-case evidence
recomputation; frozen base-body and actual source-file hashes; exact export and
evaluation inventories; all floor-bound input groups; contained tokenizer
symlinks; training-manifest and private-snapshot semantic-conversion validation;
non-finite derived values; output/input overlap rejection; input replacement
between validation and install; marker touching/path reuse; concurrent-writer
preservation; persisted-result tampering; and the complete model-free CLI.
Four additional producer tests cover the exact frozen comparison schema, real
floor-bundle/marker validation, failure-before-model-execution ordering, and the
model-free producer CLI. The floor-input suite separately verifies that every
hashed input has a concrete evaluator-readable location.

The distribution suite also performs a real wheel build, checks that the parity,
training-contract, and reference modules are present, installs the wheel into an
isolated target, and imports the parity module from that installed wheel.
