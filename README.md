# SmolVLA MLX

Native MLX inference for the `lerobot/smolvla_base` policy on Apple Silicon.
The runtime contains no PyTorch, LeRobot, or Transformers imports: it uses MLX
for vision, the truncated SmolVLM prefix decoder, the action expert, and the
flow-matching sampler.

## Install

On Apple Silicon with a supported Python 3.12 environment:

```bash
pip install .
```

For repository development and the optional PyTorch reference lane:

```bash
uv sync --extra reference
```

`predict --dataset` uses the optional LeRobot dataset bridge in a child
process. Install that extra when you need the dataset-backed CLI command:

```bash
pip install ".[reference]"
```

The first `from_pretrained` call downloads the checkpoint and tokenizer, then
converts the 500 tensors into MLX safetensors. By default this uses
`~/.cache/smolvla_mlx`; set `SMOLVLA_MLX_CACHE` or pass `cache_dir` to control
the location. Once the checkpoint and converted safetensors are present, the
same call works offline.

## Python API

```python
import numpy as np
from smolvla_mlx import SmolVLAMLX

policy = SmolVLAMLX.from_pretrained("lerobot/smolvla_base")
observation = {
    "observation.images.camera1": np.zeros((3, 480, 640), dtype=np.uint8),
    "observation.images.camera2": np.zeros((3, 480, 640), dtype=np.uint8),
    "observation.state": np.zeros(6, dtype=np.float32),
    "task": "pick up the object",
}

# A [1, 50, 6] normalized action chunk.
chunk = policy.predict_action_chunk(observation)

# One postprocessed action at a time; the 50-action queue refills only when empty.
action = policy.select_action(observation)
assert action.shape == (6,)
policy.reset()
```

`predict_action_chunk` mirrors the reference policy's normalized chunk API.
`select_action` applies the checkpoint postprocessor before putting actions in
the FIFO. For this checkpoint, the saved postprocessor has no matching action
statistics, so its effective transform is identity.

## CLI

```bash
smolvla-mlx convert --model lerobot/smolvla_base --dtype bfloat16
smolvla-mlx test
smolvla-mlx bench --runs 50 --warmups 5
smolvla-mlx predict --dataset lerobot/svla_so101_pickplace --episode 0 --index 0
```

`bench` uses a saved real golden observation by default; run `make goldens`
first if the local golden files are absent. `predict --dataset` extracts a
dataset frame through the optional child-process LeRobot bridge, so the
core-only installation remains dependency-isolated.

## Correctness

The reference is LeRobot 0.6.1/PyTorch 2.11.0 on CPU fp32, checkpoint
`lerobot/smolvla_base` revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`,
and base VLM revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.

- All 16 VLM prefix layers, K/V cache boundaries, action-expert block outputs,
  Euler states, and velocity outputs pass the fixed per-module tolerances.
- Deterministic end-to-end action chunks pass all eight real goldens at fp32
  maximum absolute error ≤ `5e-3` and bf16 ≤ `5e-2`.
- The 50-frame action-MAE gate recorded fp32 MLX/reference ratio
  `0.9999999969` and bf16 ratio `1.0000097741`, both below the fixed `1.05`
  limit.

## Performance

On an Apple M5 Pro with 48 GiB unified memory, macOS 26.5.2, and MLX 0.32.2,
the 50-run Metal benchmark measured:

| Storage dtype | Median chunk latency | P95 | Peak MLX memory |
| --- | ---: | ---: | ---: |
| fp32 | 111.34 ms | 111.80 ms | 2.94 GiB |
| bf16 | 131.12 ms | 131.71 ms | 2.44 GiB |

See [BENCHMARK.md](BENCHMARK.md) for the vision, prefix, and expert-loop stage
breakdown and the precise measured commit.

## Scope and limitations

- v0.1 targets the audited SmolVLA checkpoint and two camera inputs.
- CPU paths use focused compatibility primitives to match the PyTorch golden
  arithmetic exactly; Metal uses native MLX kernels and is the performance path.
- Robot I/O, serial ports, training, and quantization are deliberately out of
  scope.
- License and source-attribution details are in [NOTICE](NOTICE).
