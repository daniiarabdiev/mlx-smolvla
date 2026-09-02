# Coding-agent guide

`mlx-smolvla` is a native MLX implementation of SmolVLA inference, serving,
and training for Apple Silicon. This file gives any coding agent the contracts
needed to make a safe, reviewable contribution.

## Repository map

- `mlx_smolvla/`: dependency-light conversion and inference runtime, CLI,
  diagnostics, and serving adapter.
- `training/`: native MLX objectives, gradients, optimizers, LoRA/full
  fine-tuning, checkpointing, and export.
- `reference/`: pinned PyTorch/LeRobot evidence-generation lane; never import it
  from the base runtime.
- `tests/`: unit, parity, statistical, packaging, serving, and training tests.
- `scripts/`: reproducible evidence and benchmark entry points.
- `docs/`: architecture, operations, development guidance, and evidence.

## Validate changes

Run the dependency-light lane while iterating:

```bash
make test-fast
```

Before completing behavior, model, packaging, or release work, run:

```bash
make test
```

The full lane uses pinned local model/dataset artifacts. Never hide a failure
with a skip or expected-failure marker.

## Invariants

1. Published numerical tolerances are immutable: diagnose regressions and
   never loosen a gate after it has judged an implementation.
2. Importing `mlx_smolvla` must not import Torch, Transformers, or LeRobot.
   Keep reference, dataset, serving, and training dependencies behind their
   explicit optional paths.
3. `execution_mode="production"` remains the default Metal path.
   `execution_mode="strict"` remains an explicit CPU compatibility path for
   deterministic parity work.
4. Conversion must stay strict about configuration, tensor names, shapes, and
   complete consumption. Offline reload must reproduce the same converted
   artifact.
5. Never upload models, datasets, distributions, releases, or telemetry as an
   implicit development step. Never commit credentials, caches, or generated
   model weights.

## Common workflows

Inspect the environment:

```bash
mlx-smolvla doctor
```

Serve a checkpoint to a trusted LeRobot 0.6.1 async client over loopback:

```bash
mlx-smolvla serve --host 127.0.0.1 --port 8080
```

Run a native MLX training smoke on a local LeRobot dataset:

```bash
mlx-smolvla train /path/to/dataset --lora --steps 2 --batch-size 1 \
  --output .cache/training/smoke
```

## Adding a checkpoint target

Start with an immutable repository revision and a configuration/tensor
inventory. Reuse the closest native modules, add an explicit total weight-name
mapping, and write real-observation deterministic/statistical gates before
comparing outputs. Cover conversion, offline reload, import isolation, and a
fresh-installed artifact. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[evidence index](docs/evidence/README.md) for the complete review contract.
