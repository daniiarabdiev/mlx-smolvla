# SmolVLA MLX Full Continuation Design

## Purpose and authority

This design operationalizes the operator-approved `BRIEF_FULL.md` while
preserving the completed v0.1 inference implementation as a protected baseline.
It decomposes the extended work into independently shippable projects so a
blocked package cannot leave the repository incoherent.

`AGENTS.md`, the immutable gates in `BRIEF.md`, and `BRIEF_FULL.md` remain
authoritative. Stage R and Stage Q additionally depend on the missing
`BRIEF_RELEASE.md`; their requirements will not be guessed.

## Chosen architecture

Training is an additive compatibility layer under `training/`. The runtime
package continues to own checkpoint conversion, native inference modules,
policy behavior, and its strict no-Torch/LeRobot/Transformers import contract.
Training code may use the optional `reference` and `train` extras, but runtime
imports never transitively load it.

Canonical converted parameter names are the interface between inference,
training, export, and parity tests. Training adapters consume the existing MLX
modules where their operations are differentiable. Inference-only native CPU
compatibility primitives are replaced inside the training path by explicit MLX
operations with autodiff support; they are not changed or removed from v0.1.

Two alternatives were rejected:

- Adding optimizer, dataset, and loss concerns directly to `smolvla_mlx/`
  would weaken runtime isolation and increase regression risk.
- Maintaining a second complete model implementation under `training/` would
  duplicate 450M-parameter architecture logic and invite name/behavior drift.

The selected thin-adapter design keeps one architecture and two explicit
execution policies: exact inference compatibility and differentiable training.

## Project decomposition

### Release line — Stage R

The GitHub origin is configured and the verified history is published. All
remaining release packages wait for the exact `BRIEF_RELEASE.md`. Recovery and
continuation are documented in `FAILURE_RELEASE_SPEC.md`.

### Training foundation — Stages T0, T1, and T2

T0 proves the current full forward path can support finite autodiff over the
reference-trainable set, records realistic resource numbers, decides the native
RMSNorm policy, and fixes the file/data contracts for parity work.

T1 adds a deterministic reference artifact containing the real batch, sampled
noise/timesteps, loss, and all trainable gradients. The MLX loss consumes those
serialized draws so RNG differences cannot hide model differences. T2 layers
an explicitly matched AdamW and scheduler over the same deterministic stream.

### Outcome training — Stages T3, T4, and T5

T3 introduces LoRA adapters and a Metal training loop. Episode-level splitting
prevents adjacent frames from the same trajectory leaking into evaluation.
The export is a merged standard checkpoint rather than an adapter-only format:
this minimizes cross-framework loading logic and makes the required Torch/MLX
round-trip proof direct.

T4 adds resumable CLI UX and the full-fine-tune path. Checkpoints serialize
model/adapter parameters, optimizer state, scheduler step, RNG state or the
deterministic draw cursor, split identity, and configuration. T5 publishes only
numbers traceable to local committed metadata artifacts.

### Quality and hardware readiness — Stages Q and H

Stage Q remains gated by Stage R and the missing normative definitions. Stage H
produces a runbook and observation-to-chunk latency logger only. Its code has no
serial or actuator dependency and cannot satisfy the live-arm gate itself.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `training/differentiable.py` | Differentiable RMSNorm, RoPE, softmax, SiLU, and execution policy used only by training. |
| `training/model.py` | Batched prefix/action forward and exact flow-matching loss over caller-provided noise and timesteps. |
| `training/data.py` | Reference bridge loader, NumPy boundary, deterministic episode split, and serialized batch format. |
| `training/gradients.py` | Parameter selection, flattening, finite-gradient audit, and gradient comparison metrics. |
| `training/optim.py` | Matched AdamW, clipping, and cosine-with-warmup schedule. |
| `training/lora.py` | Adapter insertion, parameter filtering, merge, and configuration. |
| `training/checkpoint.py` | Last-three retention, exact resume state, and standard safetensors export. |
| `training/runner.py` | Train/evaluate loop, local CSV metrics, timing, memory, and disk guard. |
| `training/reference.py` | Torch-only reference loss/gradient/export bridge, imported only through the optional training environment. |
| `scripts/` | Reproducible artifact generation, audits, benchmarks, and hardware-safe server smoke entrypoints. |

Files are introduced only in the stage that needs them. The table fixes
responsibilities and avoids a monolithic trainer.

## Parameter and gradient policy

The default reference checkpoint has `freeze_vision_encoder=True`,
`train_expert_only=True`, and `train_state_proj=True`. “Every parameter” in the
T0/T1 gates therefore means every parameter marked trainable by those audited
settings, while the frozen VLM/vision path must still execute and feed the
expert. Parameter selection is name-based and tested against the reference;
unmatched or duplicate names are fatal.

LoRA deliberately expands the trainable set in T3 to the used VLM attention and
MLP linears plus expert linears. The exact insertion manifest is serialized
with each run and export.

## Data flow

1. The reference-only bridge loads public SO-101 samples and applies the pinned
   processor in a child process or optional-extra context.
2. The boundary emits plain NumPy arrays plus task text, action-padding masks,
   episode/frame identity, normalization metadata, and deterministic split ID.
3. Training code converts arrays to MLX, constructs the batched prefix, samples
   or consumes supplied flow draws, evaluates MSE over valid physical action
   dimensions, and differentiates the selected parameter tree.
4. Metrics are synchronized before timing and written to CSV under the run
   directory. No external metrics service is used.
5. Export merges adapters, writes standard safetensors/config/processor files,
   and is independently loaded in fresh MLX and Torch processes.

## Failure handling and safety

- All numerical thresholds are copied verbatim into tests and never relaxed.
- A stage gets three measured hypotheses before a `FAILURE_*.md`; dependent
  stages remain blocked, while independent stages continue.
- The disk guard checks for at least 40 GiB before each training stage, keeps
  only three intermediate checkpoints, and never deletes the v0.1 caches or
  golden artifacts.
- Runtime import isolation is tested after every training package.
- No Hub/PyPI uploads, credentials, vendor-fork imports, robot directories,
  serial devices, or physical motion occur.

## Verification strategy

Each behavioral change follows red-green-refactor with focused tests. Stage
gates then add real-model/reference evidence, the full v0.1 suite, package
import isolation, artifact manifests, disk/resource records, a commit, and a
push. Milestone strings enter `STATUS_FULL.md` only after their exact gates are
satisfied.
