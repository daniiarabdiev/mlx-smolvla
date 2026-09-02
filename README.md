# SmolVLA MLX

SmolVLA MLX is a native MLX inference port of LeRobot's SmolVLA policy for
Apple Silicon. It converts compatible SmolVLA checkpoints into MLX
safetensors, runs vision, the truncated SmolVLM prefix decoder, the action
expert, and flow matching through MLX, and keeps PyTorch, LeRobot, and
Transformers out of the base runtime so inference can use the Mac GPU and
unified memory directly.

## Install

Use an Apple Silicon Mac and Python 3.11, 3.12, or 3.13:

```bash
python -m pip install .
```

For repository development and the pinned PyTorch reference lane, use Python
3.12 or 3.13 because LeRobot 0.6.1 requires Python 3.12 or newer:

```bash
uv sync --extra reference
```

The first load downloads the checkpoint and tokenizer, then converts all 500
model tensors once. The default cache is `~/.cache/smolvla_mlx`; set
`SMOLVLA_MLX_CACHE` or pass `cache_dir=` to put it elsewhere. A complete cached
checkpoint, tokenizer, and conversion can be loaded again with
`HF_HUB_OFFLINE=1`.

The native compatibility extension is optional. A source build can force the
tested pure-MLX fallback with `SMOLVLA_MLX_BUILD_NATIVE=0`. See
[DIST_MANIFEST.md](DIST_MANIFEST.md) for the locally built CPython 3.11–3.13
artifacts, checksums, and deployment caveat.

## Python API

```python
import numpy as np
from smolvla_mlx import SmolVLAMLX
policy = SmolVLAMLX.from_pretrained("lerobot/smolvla_base")
observation = {
    "observation.images.camera1": np.zeros((3, 480, 640), np.uint8),
    "observation.images.camera2": np.zeros((3, 480, 640), np.uint8),
    "observation.state": np.zeros(6, np.float32),
    "task": "pick up the object",
}
action = policy.select_action(observation)
```

`select_action` returns one postprocessed physical action and keeps the rest of
the checkpoint's action horizon in a FIFO. Call `policy.reset()` between
episodes. Use `policy.predict_action_chunk(observation)` when the normalized
`[1, 50, action_dim]` chunk is the desired interface. Passing a local
checkpoint directory instead of a Hub ID uses the same strict loader. The
default `execution_mode="production"` owns an MLX Metal context; pass
`execution_mode="strict"` to own the CPU compatibility context used by the
golden parity ladder.

## CLI quickstart

From a repository checkout with the generated golden observation present:

```bash
smolvla-mlx convert --model lerobot/smolvla_base --dtype float32
smolvla-mlx predict --observation tests/golden/sample_000
smolvla-mlx predict --observation tests/golden/sample_000 --execution-mode strict
smolvla-mlx bench --dtype float32 --runs 50 --warmups 5
smolvla-mlx test
```

`predict --observation` and `bench` use the task and sample mapping in
`tests/golden/metadata.json`; run `make goldens` if the ignored local goldens
are absent. Dataset-backed prediction is optional and isolated in a child
process so the base runtime stays dependency-light:

```bash
python -m pip install ".[reference]"
smolvla-mlx predict --dataset lerobot/svla_so101_pickplace --episode 0 --index 0
```

That bridge may also require a working FFmpeg/TorchCodec or PyAV setup for the
dataset's video encoding. It does not change the base saved-observation path.

## Performance

One SmolVLA chunk contains 50 actions, or about 1.67 seconds of commanded
motion at 30 fps. On the measured Apple M5 Pro with 48 GiB unified memory,
macOS 26.6.2, and MLX 0.32.2, the installed default production path produced
the chunk in 110.54 ms at fp32—about **15.1× the chunk's motion duration per
unit of compute**.

| Storage dtype | Median chunk | P95 | Peak MLX memory | Motion/compute |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 110.54 ms | 111.41 ms | 2.94 GiB | 15.1× |
| bf16 | 130.44 ms | 131.25 ms | 2.44 GiB | 12.8× |

These are model-only local timings, not an end-to-end robot control-rate
claim; camera capture, transport, and actuation are excluded. On this measured
stack bf16 saved about 0.50 GiB of peak MLX memory but was slower than fp32.
The stage breakdown, warmup policy, and measured source commit are in
[BENCHMARK.md](BENCHMARK.md).

## Correctness evidence

The golden source is LeRobot 0.6.1 with PyTorch 2.11.0 on CPU fp32. The base
policy is pinned to revision
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`, and its SmolVLM2-500M backbone to
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Fixed noise and eight real
observations compare preprocessing, all 16 prefix layers and K/V boundaries,
the action-expert blocks, all ten Euler steps, normalized chunks, and
postprocessed actions. The acceptance limits were written before evaluation
and have not been loosened.

The deterministic normalized-action limits are `0.005` maximum absolute error
for fp32 and `0.05` for bf16. The independent statistical gate compares first
action MAE on 50 pinned real frames and requires MLX/reference `<= 1.05`:

| Checkpoint evidence | fp32 normalized max | bf16 normalized max | fp32 MAE ratio | bf16 MAE ratio |
| --- | ---: | ---: | ---: | ---: |
| Base checkpoint | pass (`<=0.005`) | pass (`<=0.05`) | 0.9999999969 | 1.0000097741 |
| Base + active dataset statistics | 0.0000028759 | 0.0048642755 | 1.0000000330 | 0.9985395647 |
| Public fine-tune + active statistics | 0.0000343621 | 0.0040360391 | 1.0000005749 | 0.9960502782 |

The public fine-tune is
`soonweihong0857/swhfypv3_smolvla_multitask_model` at pinned revision
`5e2491c809ec892427f54db1eb23bf8c4bbbf770`; it exercises non-base camera
names plus saved state/action mean and standard deviation. Exact values,
artifact hashes, and regeneration commands are recorded in
[PROGRESS.md](PROGRESS.md).

Strict module parity and production inference are separate claims. `strict`
owns an MLX CPU context and reproduces the pinned PyTorch operations;
`production` is the public default, owns an MLX Metal context, and is the path
measured above.

The default production path was also run against the same eight deterministic
goldens and 50-frame statistical corpus:

| Production dtype | Deterministic max | Fixed gate | Deterministic | 50-frame MAE ratio | Statistical |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 0.0473065376 | 0.005 | fail | 1.0000128000 | pass (`<=1.05`) |
| bf16 | 0.0441064835 | 0.05 | pass | 1.0000216963 | pass (`<=1.05`) |

Metal fp32 therefore does not inherit the strict deterministic guarantee;
select `strict` when that contract is required. Metal bf16 passes its unchanged
deterministic gate, and both production storage modes pass the unchanged
statistical gate. The known Vision and Connector module discrepancies remain
documented without changing their thresholds; full methodology and both mode
tables are in [BENCHMARK.md](BENCHMARK.md).

## Run your own fine-tune

Fine-tune with standard LeRobot 0.6.1 on a GPU machine, then copy the saved
checkpoint directory to the Mac. This is LeRobot/PyTorch training; it is
separate from this repository's experimental native-MLX training research.
For an NVIDIA CUDA host:

```bash
python -m pip install "lerobot[training,smolvla]==0.6.1"
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=<USER>/<DATASET> \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/smolvla-finetune \
  --save_freq=20000 \
  --save_checkpoint_to_hub=false \
  --wandb.enable=false
```

The two explicit Hub flags keep the run local. LeRobot maintains the latest
saved policy at
`outputs/smolvla-finetune/checkpoints/last/pretrained_model`; load that
directory directly after transferring it:

```python
from smolvla_mlx import SmolVLAMLX

policy = SmolVLAMLX.from_pretrained(
    "/path/to/smolvla-finetune/checkpoints/last/pretrained_model"
)

# Or, after you deliberately publish the complete checkpoint yourself:
policy = SmolVLAMLX.from_pretrained("<USER>/<MODEL>")
```

The checkpoint must include `config.json`, `model.safetensors`, both processor
JSON files, and any normalization safetensors required by its config.

## Serve for your robot

The optional server speaks the exact async-inference gRPC protocol shipped by
LeRobot 0.6.1 while running the policy with native MLX. On the Apple Silicon
Mac, use Python 3.12 or 3.13, install the isolated extra, and start the safe
loopback default:

```bash
python -m pip install ".[serve]"
smolvla-mlx serve \
  --host 127.0.0.1 \
  --port 8080 \
  --dtype bfloat16 \
  --execution-mode production
```

On that same Mac, a LeRobot user runs its standard client with their existing
calibration and camera values substituted for the placeholders:

```bash
python -m lerobot.async_inference.robot_client \
  --policy_type=smolvla \
  --pretrained_name_or_path=lerobot/smolvla_base \
  --robot.type=so101_follower \
  --robot.port=<YOUR_FOLLOWER_PORT> \
  --robot.id=<YOUR_CALIBRATION_ID> \
  --robot.cameras='<YOUR_LEROBOT_CAMERA_CONFIG>' \
  --actions_per_chunk=10 \
  --task='pick up the object' \
  --server_address=127.0.0.1:8080 \
  --policy_device=cpu \
  --client_device=cpu \
  --fps=30
```

The client's `policy_device` field is retained for wire compatibility; the
server's `--execution-mode` owns the actual MLX device. A different-machine
client requires a trusted private network plus an explicit server bind such as
`--host 0.0.0.0 --allow-remote` and the Mac's private address in
`--server_address`. LeRobot 0.6.1 uses unauthenticated, unencrypted gRPC and
Python pickle payloads, so never expose this port to an untrusted peer or the
public internet.

`Ready` starts a fresh episode, the one-item observation queue keeps the latest
frame, inference is serialized, and each returned CPU Torch `TimedAction`
inherits the observation timestep and advances its timestamp by `1 / --fps`.
Ctrl-C stops the server. Schema, validation, cancellation, concurrency, and a
localhost real-checkpoint test are covered in `tests/test_server.py`; the
recorded three-action chunk is exactly equal to three direct
`select_action` calls. Hardware-in-the-loop validation is still pending and
must be performed in a separate supervised operator session.

## Cache layout and cleanup

Repository commands route caches under `.cache/`: Hugging Face snapshots and
datasets in `.cache/hf`, MLX conversions and parity caches in
`.cache/smolvla_mlx`, and protected training evidence in `.cache/training`.
Golden evidence is under the ignored `tests/golden*` trees.

Inspect the exact candidates before deleting the narrowly allowed debug
entries:

```bash
make cache-inventory
make clean-cache-dry-run
make clean-cache
```

Cleanup refuses symlinked or out-of-repository targets and removes only
top-level `debug-*` directories plus exact `benchmark-debug`. It never enters
model sources, converted weights, golden outputs, or training evidence.

## Scope, limitations, and troubleshooting

- The base installed surface is dependency-light inference and offline
  software evaluation. The `serve` extra adds a software-only policy server;
  robot I/O remains in LeRobot's client, and hardware-in-the-loop validation
  is not claimed.
- Checkpoints must match the audited SmolVLA/SmolVLM2 architecture. Camera
  names and state/action shapes come from each checkpoint's config; at least
  one configured camera must be present, and errors print the expected input
  contract. Configured streams that are absent are skipped unless the config
  explicitly requests `empty_cameras` padding.
- `predict --dataset` and `make goldens` need the Python 3.12+ reference extra;
  ordinary imports, conversion, and saved-observation inference do not import
  PyTorch, LeRobot, or Transformers.
- `serve` needs Python 3.12+ and `.[serve]`. Its pickle wire format is safe only
  with trusted peers; loopback is enforced unless `--allow-remote` is explicit.
- The project wheels and native extension target `macosx_14_0_arm64`, but the
  pinned MLX 0.32.2 wheel's own `libmlx.dylib` declares macOS 26.2. A working
  end-to-end macOS 14 installation therefore depends on a lower-target MLX
  build; see [DIST_MANIFEST.md](DIST_MANIFEST.md).
- If strict CPU arithmetic reports `pure-mlx-fallback`, the optional extension
  was not loaded. Reinstall from a native wheel for the exact extension-backed
  path, or retain the tested fallback when portability matters more.
- If an offline load fails, first complete one online load into the same
  `SMOLVLA_MLX_CACHE`; both the policy and its tokenizer must be present.

The project is licensed under the [Apache License 2.0](LICENSE). Upstream code
and model attribution is collected in [NOTICE](NOTICE).
