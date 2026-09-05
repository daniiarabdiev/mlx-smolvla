# mlx-smolvla

Run SmolVLA checkpoint inference, LeRobot-protocol serving, and preview
fine-tuning natively with MLX on Apple Silicon—without Torch, Transformers, or
LeRobot in the base runtime.

On the pinned Apple M5 Pro test case, MLX fp32 produced a 50-action chunk in
**110.75 ms median** versus **204.58 ms** for PyTorch-MPS (**1.847× faster**),
while that chunk represents **1.67 s** at 30 fps (about **15.0× real-time
duration**); scope and raw timings are in the [benchmark evidence](docs/BENCHMARK.md#mlx-versus-pytorch-mps).

> **Hardware validation:** a final 60-second no-motion loop, one valid guarded
> action, and a two-chunk bounded-continuous run passed on a connected SO-101.
> A separate 20-chunk attempt disabled torque safely but failed exact return
> under the temporary 10% torque profile. This is bounded integration evidence,
> not reliable task success or sustained 20-chunk validation. See the
> [first-contact status](hardware/FIRST_CONTACT.md) and
> [media guidance](docs/media/README.md).

## Requirements

| Requirement | Supported release surface |
| --- | --- |
| Mac | Apple Silicon |
| macOS | 14 or newer |
| Python | 3.11-3.13 for inference; 3.12-3.13 for reference, serve, train, and hardware extras |
| MLX | 0.32.0, 0.32.1, or 0.32.2 |

All three MLX versions passed conversion, strict deterministic checks,
production fp32/bf16 statistical gates, installed offline prediction, `doctor`,
and loopback serving on macOS-14-compatible official wheels. See the
[compatibility matrix](docs/evidence/MLX_COMPATIBILITY.md) for exact wheel and
dylib inspection evidence.

## Install

Install v0.1.0 from PyPI:

```bash
python -m pip install mlx-smolvla
```

Or install from a checkout:

```bash
git clone https://github.com/daniiarabdiev/mlx-smolvla.git
cd mlx-smolvla
python -m pip install .
mlx-smolvla doctor
```

The first online load downloads and converts the checkpoint once. Converted
weights default to `~/.cache/mlx_smolvla`; set `MLX_SMOLVLA_CACHE` or pass
`cache_dir=` to move them. A populated cache supports subsequent
`HF_HUB_OFFLINE=1` loads.

## Run a checkpoint

The Python path is three lines once `observation` contains two CHW `uint8`
camera arrays, a six-value `float32` state, and a task string:

```python
from mlx_smolvla import SmolVLAMLX
policy = SmolVLAMLX.from_pretrained("lerobot/smolvla_base")
action = policy.select_action(observation)
```

Or predict from a saved observation directory:

```bash
mlx-smolvla predict --model lerobot/smolvla_base --observation /path/to/saved-observation
```

`select_action` returns one checkpoint-domain action and queues the remainder
of the 50-action horizon; call `policy.reset()` between episodes. It is a
physical-unit action only when the checkpoint contains effective statistics
for the exact robot state/action interface. The upstream base checkpoint's
saved stats do not bind to `observation.state` and `action`, so do not send its
raw output to a robot. Local LeRobot-style checkpoint directories and complete
Hub repository IDs use the same strict configuration/tensor loader. Quantized
`vlm-8bit` and `vlm-4bit` presets are explicit production-only opt-ins
documented in the [benchmark](docs/BENCHMARK.md#vlm-only-quantization).

```bash
mlx-smolvla predict --observation /path/to/saved-observation --quantization vlm-8bit
mlx-smolvla predict --observation /path/to/saved-observation --quantization vlm-4bit
```

Dense bf16 remains the default; neither quantized preset is selected implicitly.

## Serve for your robot

Install the optional protocol surface and start the trusted loopback server on
the Mac:

```bash
python -m pip install "mlx-smolvla[serve]"
mlx-smolvla serve --host 127.0.0.1 --port 8080 --dtype bfloat16
```

A mainline LeRobot 0.6.1 client can connect using its real, operator-reviewed
robot and camera configuration:

```bash
python -m lerobot.async_inference.robot_client \
  --policy_type=smolvla \
  --pretrained_name_or_path=<REVIEWED_LOCAL_CHECKPOINT_WITH_MATCHING_ROBOT_STATS> \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=<CALIBRATION_ID> \
  --robot.cameras='<CAMERA_CONFIG>' \
  --actions_per_chunk=10 \
  --task='pick up the object' \
  --server_address=127.0.0.1:8080 \
  --policy_device=cpu --client_device=cpu --fps=30
```

The server implements the audited four-RPC LeRobot protocol, but robot I/O and
safety remain the client's responsibility. The generic command above is a
protocol example, not authorization to actuate hardware. This repository now
ships a fail-closed Hiwonder SO-101 client under the optional `hardware` extra;
one guarded action and a short bounded-continuous run passed on the connected
follower. Review the [hardware runbook](docs/HARDWARE_RUNBOOK.md),
[current first-contact status](hardware/FIRST_CONTACT.md), and
[bring-your-own-robot guide](examples/bring_your_own_robot/README.md). Remote
serving is an explicit trusted-network-only mode because LeRobot 0.6.1 uses
unauthenticated pickle payloads.

## Run your own fine-tune

You can train with standard LeRobot on any supported accelerator, keep uploads
disabled, then transfer the complete checkpoint directory to the Mac:

```bash
python -m pip install "lerobot[training,smolvla]==0.6.1"
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=<USER>/<DATASET> \
  --batch_size=64 --steps=200000 \
  --output_dir=outputs/smolvla-finetune \
  --save_freq=20000 --save_checkpoint_to_hub=false \
  --wandb.enable=false
```

Load its `checkpoints/last/pretrained_model` directory directly, or use a Hub
ID only after deliberately publishing the complete checkpoint yourself:

```python
policy = SmolVLAMLX.from_pretrained("/path/to/checkpoints/last/pretrained_model")
# policy = SmolVLAMLX.from_pretrained("<USER>/<MODEL>")
```

The separately pinned public multitask fine-tune passed all eight deterministic
cases and its 50-frame fp32/bf16 MAE ratios were 1.0000005749 and 0.9960502782;
the [public-checkpoint evidence](docs/evidence/README.md#inference-correctness)
records the immutable revision and regeneration path.

## Fine-tune on your Mac (preview)

Native MLX training supports LoRA/full exports and exact resume. Step-zero
gradients and 25 optimizer updates pass the fixed lockstep gates. A repaired
PyTorch reference loader now preserves the native-trained fp32 weights: the
retained expert-only LoRA export passes all 56 fixed-limit cases, with normalized
maximum **0.0000214577**, physical maximum **0.0004272461** (both below **0.005**),
and Torch/MLX held-out MAE ratio **1.0000007854**. See the
[repair evidence](docs/evidence/TRAINED_PARITY_REPAIR.md). Training remains a
research preview: this validates one retained LoRA run, while full fine-tuning
has code/smoke coverage rather than a long-run task-quality study.

```bash
uv sync --extra train
mlx-smolvla train /path/to/lerobot-dataset \
  --lora --dtype bfloat16 --steps 30000 --batch-size 8 --lr 1e-4 \
  --checkpoint-every 100 --output .cache/training/overnight
```

Resume the exact output directory with the same arguments plus `--resume`.
Validate the exported checkpoint before serving it:

```bash
mlx-smolvla predict \
  --model .cache/training/overnight/export \
  --observation /path/to/saved-observation
```

On the measured host, effective-batch-8 LoRA bf16 ran at **0.873 updates/s**,
or **19.09 minutes per 1,000 updates**; 30,000 updates project to **9.55 hours**
of optimizer work before checkpoint/export/evaluation overhead, with **2.27 GiB**
peak MLX memory. The full four-cell protocol and raw timing source are in the
[training benchmark](docs/BENCHMARK.md#native-training-performance-metal).

## Execution modes

`execution_mode="production"` is the default MLX Metal path and is accepted by
the pinned 50-frame statistical gate. `execution_mode="strict"` selects the MLX
CPU compatibility path for bit-close deterministic comparison with PyTorch CPU.

## Limitations

- Connected SO-101 state/camera capture, one valid guarded action, and a two-chunk continuous run are validated. A separate 20-chunk attempt failed exact return under the temporary low-torque profile while still disabling torque; [first-contact evidence](hardware/FIRST_CONTACT.md) records the bounded result and limitation.
- Raw `lerobot/smolvla_base` output is not a physical-action interface because its saved state/action statistics do not bind to the generic keys. Motion clients must use a reviewed checkpoint with effective statistics matching the robot.
- Production Metal fp32 passes the statistical gate but fails the strict `0.005` deterministic maximum; use strict mode for that contract and see the [mode table](docs/BENCHMARK.md#default-production-correctness-metal).
- Native training is a research preview. The retained LoRA export passes the [post-repair fixed parity gates](docs/evidence/TRAINED_PARITY_REPAIR.md); this does not establish task success on a robot or generalize to every training run. The original [T3B verdict](docs/evidence/FAILURE_LORA_FINETUNE_B.md) is preserved as historical evidence.
- Checkpoints must match the audited SmolVLA/SmolVLM2 configuration and complete tensor inventory described in the [architecture](docs/ARCHITECTURE.md).
- The LeRobot serving protocol is suitable only for trusted peers; security boundaries are documented in the [architecture](docs/ARCHITECTURE.md) and [security policy](.github/SECURITY.md).

## Correctness methodology

The reference lane pins LeRobot 0.6.1, exact model/dataset revisions, fixed
noise, eight real observations, all 16 used prefix layers and K/V boundaries,
the action expert, every Euler step, normalized chunks, and physical actions.
The strict normalized-action maxima are fixed at `0.005` (fp32) and `0.05`
(bf16); an independent 50-frame gate requires MLX/reference first-action MAE
`<= 1.05`. Thresholds are never loosened after evaluation. The
[evidence index](docs/evidence/README.md#inference-correctness) links reports,
hashes, negative results, and reproduction commands.

## Contributing, citation, and license

Start with [CONTRIBUTING.md](CONTRIBUTING.md); `make test-fast` is the iteration
lane and `make test` is the complete gate. [AGENTS.md](AGENTS.md) gives coding
agents the repository map and immutable contracts for agent-assisted work.
Citation metadata is in [CITATION.cff](CITATION.cff). The project is licensed
under [Apache-2.0](LICENSE), with upstream attribution in [NOTICE](NOTICE).

## Acknowledgments

This work builds on [SmolVLA](https://huggingface.co/lerobot/smolvla_base),
[LeRobot](https://github.com/huggingface/lerobot), and
[MLX](https://github.com/ml-explore/mlx). The SmolVLM backbone work in
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) was also a useful architecture
reference; `mlx-smolvla` does not depend on it at runtime.

## Related projects

[`tokimoa/smolvla-mlx`](https://huggingface.co/tokimoa/smolvla-mlx), uploaded
to the Hugging Face Hub on **2026-07-29**, is an earlier, independent inference
port of SmolVLA to MLX. This project differs in scope by adding verified parity gates,
a base runtime without Torch or Transformers at runtime,
LeRobot-protocol serving, and training; no comparative performance claim is
made against that project.
