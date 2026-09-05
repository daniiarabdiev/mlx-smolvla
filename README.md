# mlx-smolvla

Run SmolVLA checkpoint inference, LeRobot-protocol serving, and preview
fine-tuning natively with MLX on Apple Silicon—without Torch, Transformers, or
LeRobot in the base runtime.

In the Apple M5 Pro benchmark setup, MLX fp32 produced a 50-action chunk in
**110.75 ms median** versus **204.58 ms** for PyTorch-MPS (**1.847× faster**),
while that chunk represents **1.67 s** at 30 fps (about **15.0× real-time
duration**); scope and raw timings are in the [benchmark evidence](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/BENCHMARK.md#mlx-versus-pytorch-mps).

## Install

| Requirement | Supported |
| --- | --- |
| Mac | Apple Silicon |
| macOS | 14 or newer |
| Python | 3.11-3.13 for inference; 3.12-3.13 for reference, serve, train, and hardware extras |
| MLX | `>=0.32.0,<0.32.3`; 0.32.0, 0.32.1, and 0.32.2 verified |

Install from PyPI:

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

The Python API takes three lines once `observation` contains two CHW `uint8`
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
documented in the [benchmark](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/BENCHMARK.md#vlm-only-quantization).

```bash
mlx-smolvla predict --observation /path/to/saved-observation --quantization vlm-8bit
mlx-smolvla predict --observation /path/to/saved-observation --quantization vlm-4bit
```

Dense bf16 remains the default; neither quantized preset is selected implicitly.

## Serve for your robot

Install the serving extra and start the loopback server on your Mac:

```bash
python -m pip install "mlx-smolvla[serve]"
mlx-smolvla serve --host 127.0.0.1 --port 8080 --dtype bfloat16
```

A mainline LeRobot 0.6.1 client can connect using your own reviewed robot and
camera configuration:

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

The server implements LeRobot's four-RPC protocol; your client handles robot
I/O, calibration, motion limits, and shutdown. The example shows how to connect
the protocol. For the Hiwonder SO-101, this repository includes a client that
checks those conditions and stops when they fail, under the optional `hardware`
extra. Review the [hardware runbook](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/HARDWARE_RUNBOOK.md) and
[bring-your-own-robot guide](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/examples/bring_your_own_robot/README.md) before use.
Remote serving requires an explicit trusted-network setting because LeRobot
0.6.1 uses unauthenticated pickle payloads.

## Fine-tune with LeRobot, run on the Mac

You can train with standard LeRobot on any supported accelerator, keep uploads
disabled, then transfer the complete checkpoint directory to the Mac. This
20,000-step example follows the [LeRobot v0.6.1 tutorial](https://github.com/huggingface/lerobot/blob/v0.6.1/docs/source/smolvla.mdx):

```bash
python -m pip install "lerobot[training,smolvla]==0.6.1"
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=<USER>/<DATASET> \
  --batch_size=64 --steps=20000 \
  --output_dir=outputs/smolvla-finetune \
  --save_freq=2000 --save_checkpoint_to_hub=false \
  --wandb.enable=false
```

Load its `checkpoints/last/pretrained_model` directory directly, or use a Hub
ID only after deliberately publishing the complete checkpoint yourself:

```python
policy = SmolVLAMLX.from_pretrained("/path/to/checkpoints/last/pretrained_model")
# policy = SmolVLAMLX.from_pretrained("<USER>/<MODEL>")
```

## Fine-tune on your Mac (preview)

Native MLX training supports LoRA/full exports and exact resume. Step-zero
gradients and 25 optimizer updates pass the fixed lockstep checks. Training is
a research preview: one retained LoRA run passes export parity validation,
while full fine-tuning has code/smoke coverage rather than a long-run
task-quality study.

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

On an M5 Pro, effective-batch-8 LoRA bf16 ran at **0.873 updates/s**,
or **19.09 minutes per 1,000 updates**; 30,000 updates project to **9.55 hours**
of optimizer work before checkpoint/export/evaluation overhead, with **2.27 GiB**
peak MLX memory. The full four-cell protocol and raw timing source are in the
[training benchmark](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/BENCHMARK.md#native-training-performance-metal).

## Execution modes

`execution_mode="production"` is the default MLX Metal path and is accepted by
the fixed 50-frame statistical check. `execution_mode="strict"` selects the MLX
CPU compatibility path for bit-close deterministic comparison with PyTorch CPU.
Strict mode supports both the native reference kernels and the verified
pure-MLX fallback.

## Status and evidence

Hardware integration passed on a connected Hiwonder SO-101: a final 60-second
no-motion loop with live state and camera capture, one valid guarded action,
and a two-chunk continuous run. The powered runs returned exactly and verified
torque-off shutdown under the temporary 10% torque profile. These are limited
integration results; reliable task completion and sustained operation remain
unvalidated. The [first-contact record](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/hardware/FIRST_CONTACT.md) gives the
full history. No demo media is published; see the [media guidance](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/media/README.md).

The separately pinned public multitask fine-tune passed all eight deterministic
cases and its 50-frame fp32/bf16 MAE ratios were 1.0000005749 and 0.9960502782;
the [public-checkpoint evidence](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/evidence/README.md#inference-correctness)
records the immutable revision and regeneration path.

The repaired PyTorch reference loader preserves native-trained fp32 weights.
The retained expert-only LoRA export passes all 56 fixed-limit cases, with
normalized maximum **0.0000214577**, physical maximum **0.0004272461** (both
below **0.005**), and Torch/MLX held-out MAE ratio **1.0000007854**. See the
[repair evidence](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/evidence/TRAINED_PARITY_REPAIR.md). This validates one
retained run; it does not establish task success or generalize to every run.

All three MLX versions passed conversion, strict deterministic checks,
production fp32/bf16 statistical gates, installed offline prediction, `doctor`,
and loopback serving on macOS-14-compatible official wheels. See the
[compatibility matrix](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/evidence/MLX_COMPATIBILITY.md) for exact wheel and
dylib inspection evidence.

## Limitations

- Raw `lerobot/smolvla_base` output is not a physical-action interface because its saved state/action statistics do not bind to the generic keys. Motion clients must use a reviewed checkpoint with effective statistics matching the robot.
- Production Metal fp32 passes the statistical check but fails the strict `0.005` deterministic maximum. The [architecture](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/ARCHITECTURE.md#runtime-execution-modes) records Vision and Connector reduction differences on Metal; use the CPU strict mode for that deterministic contract. See the [mode table](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/BENCHMARK.md#default-production-correctness-metal).
- Native training remains a research preview. The original [T3B verdict](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/evidence/FAILURE_LORA_FINETUNE_B.md) is preserved alongside the repaired result above.
- Checkpoints must match the audited SmolVLA/SmolVLM2 configuration and complete tensor inventory described in the [architecture](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/ARCHITECTURE.md).
- The LeRobot serving protocol is suitable only for trusted peers; security boundaries are documented in the [architecture](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/ARCHITECTURE.md) and [security policy](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/.github/SECURITY.md).

## Correctness methodology

The reference lane pins LeRobot 0.6.1, exact model/dataset revisions, fixed
noise, eight real observations, all 16 used prefix layers and K/V boundaries,
the action expert, every Euler step, normalized chunks, and physical actions.
The strict normalized-action maxima are fixed at `0.005` (fp32) and `0.05`
(bf16); an independent 50-frame gate requires MLX/reference first-action MAE
`<= 1.05`. Thresholds are never loosened after evaluation. The
[evidence index](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/docs/evidence/README.md#inference-correctness) links reports,
hashes, negative results, and reproduction commands.

## Contributing, citation, and license

Start with [CONTRIBUTING.md](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/CONTRIBUTING.md); `make test-fast` is the iteration
lane and `make test` is the complete gate. [AGENTS.md](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/AGENTS.md) gives coding
agents the repository map and immutable contracts for agent-assisted work.
Citation metadata is in [CITATION.cff](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/CITATION.cff). The project is licensed
under [Apache-2.0](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/LICENSE), with upstream attribution in [NOTICE](https://github.com/daniiarabdiev/mlx-smolvla/blob/v0.1.2/NOTICE).

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
