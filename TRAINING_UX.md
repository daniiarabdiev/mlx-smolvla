# Native MLX Training UX

## Outcome

Stage T4 is complete. `smolvla-mlx train` accepts a LeRobot dataset repo ID or
local path, a fixed step count, effective batch size, learning rate, output
directory, checkpoint cadence, `--resume`, and exactly one of `--lora` or
`--full`. Both modes use native MLX model, loss, gradient accumulation,
clipping, AdamW, sampler, checkpoint, and export code.

`--full` has a precise meaning: it trains every parameter enabled by LeRobot
0.6.1's default SmolVLA freeze policy—state projection plus the complete action
expert—without adapters. The vision encoder and language backbone remain
frozen, matching the audited reference policy. Full mode exposes **155 fp32
master tensors / 99,880,992 scalars**; LoRA's expert-only topology exposes
**112 adapters / 224 fp32 tensors / 1,708,032 scalars**. The optimizer owns
exactly two moment tensors per trainable: 310 for full and 448 for LoRA.

## Commands

```bash
uv sync --extra train

smolvla-mlx train owner/dataset \
  --lora --steps 100 --batch-size 1 --lr 1e-4 \
  --checkpoint-every 25 --output .cache/training/my-lora-run

smolvla-mlx train /path/to/lerobot-dataset \
  --full --steps 100 --batch-size 1 --lr 1e-4 \
  --checkpoint-every 25 --output .cache/training/my-full-run

smolvla-mlx train /path/to/lerobot-dataset \
  --full --steps 100 --batch-size 1 --lr 1e-4 \
  --checkpoint-every 25 --output .cache/training/my-full-run --resume
```

Fresh runs refuse an existing output. Resume reconstructs the same base and
trainable topology, validates the configuration digest, restores the newest
bound checkpoint, truncates any uncheckpointed metrics tail, restores AdamW
and sampler position, advances the MLX flow-draw stream, and continues. Each
run writes `metrics.csv`, `run.json`, `checkpoints/`, and a complete 500-tensor
LeRobot checkpoint under `export/`. Checkpoints are atomic and only the newest
three are retained.

For repo IDs, the dataset is materialized under the repository-local cache
before train-only statistics are recomputed. Local paths are used directly.
The current bridge requires a SmolVLA-compatible LeRobot dataset with the
expected state/action/language and camera features; incompatible schemas fail
before an optimizer update.

## Fixed resume proof

`scripts/check_training_resume.py` implements the T4 gate without adjustable
tolerances. It runs 100 updates continuously, snapshots the committed step-50
state, reconstructs a new process trajectory from that snapshot, runs updates
51–100, and then compares every output. The immutable gates remain:

- maximum per-tensor parameter absolute difference: **1e-6**;
- maximum per-step loss difference: **1e-7**;
- optimizer tensors, serialized flow-draw chain, sampler state, and canonical
  step state: **exact**.

Both real modes passed more strongly than required:

| Mode | Parameter max abs | Loss max abs | All metric max abs | Optimizer | Draws / sampler / step state |
| --- | ---: | ---: | ---: | --- | --- |
| Expert-only LoRA | 0.0 | 0.0 | 0.0 | exact | exact / exact / exact |
| Full reference trainable set | 0.0 | 0.0 | 0.0 | exact | exact / exact / exact |

The ignored machine-local reports are:

- `.cache/training/t4-resume-lora/evidence.json`, SHA-256
  `44325aa73c012d5b9dfb5499a549eeb689b90c64ebd07b137ee024cefa797b57`;
  exactness-file SHA-256
  `3e0325b71bc6c6c053598061e68bfd564a511a600825b9dc4c28fa50e64df157`.
- `.cache/training/t4-resume-full-v2/evidence.json`, SHA-256
  `2c46c621a08b59584701b1bc2171690cfc03c7a116e41d4e4fff35f217699748`;
  exactness-file SHA-256
  `be45552040b1837c094fee28101354554156b2726a8a064c58b1a586b799d296`.

## 100-update smoke outcomes

The loss-decrease check compares the mean of the first ten and last ten
updates. Both direct and resumed forms passed:

| Mode | First-10 mean | Last-10 mean | Final loss | Peak MLX bytes | Export model SHA-256 | Finite action SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| LoRA | 0.9695772379636765 | 0.6153697635978460 | 0.36160343885421753 | 3,818,770,536 | `44ccc8ea...17c755` | `15748dd9...2a39a` |
| Full | 1.8326249837875366 | 0.5984157636761666 | 0.7582890987396240 | 4,603,826,668 | `76526b04...6749f` | `165f5bf8...8956` |

Each export contains 500 fp32 tensors / 450,046,176 parameters, reloads through
the public MLX policy, and emits a finite `(6,)` action from a real training
observation. Both direct and resumed runs retain exactly steps 50, 75, and 100
plus `latest.json`.

The peak figures and runner wall-time fields are **functional smoke
observations, not benchmarks**. The direct harness deliberately copies a large
step-50 snapshot while paused and both paths include export/action validation;
Stage T5 separately measures idle-machine training throughput and memory under
its fixed protocol.

## Resolved implementation findings

Two preflight failures remain preserved in ignored cache directories. The
first exposed tuple-versus-JSON-list metadata identity at final export; export
metadata is now canonicalized before publication. The first full resume attempt
then proved that MLX AdamW promotes bf16 trainables when applying fp32
gradients. Full mode now establishes explicit fp32 master parameters before
optimizer initialization, so a fresh reconstruction and a saved checkpoint
have the same schema. Neither finding changed a gate, tolerance, dataset split,
or the completed T3/T3B records.

No upload, hardware, robot directory, or serial port was used. The original T3
failure document remains byte-identical at SHA-256
`d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
The complete repository verification passes **608/608 tests in 533.65
seconds**.
