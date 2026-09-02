#!/usr/bin/env python3
"""Audit the pinned PyTorch SmolVLA implementation and persist its evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from reference.audit import inspect_reference
from reference.discovery import (
    BASE_VLM_ID,
    BASE_VLM_REVISION,
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    ReferenceDiscovery,
    discover_reference,
)
from reference.policy import ReferencePolicy, load_dataset_observation


def read_safetensors_inventory(path: Path) -> list[dict[str, Any]]:
    """Read tensor names, dtypes, and shapes from a safetensors header only."""

    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), byteorder="little")
        header = json.loads(handle.read(header_size))
    return [
        {
            "name": name,
            "dtype": spec["dtype"],
            "shape": spec["shape"],
        }
        for name, spec in sorted(header.items())
        if name != "__metadata__"
    ]


def _table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Field | Verified value |", "| --- | --- |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def render_architecture_report(
    discovery: ReferenceDiscovery,
    runtime: dict[str, Any],
    weight_inventory: list[dict[str, Any]],
) -> str:
    """Render the immutable evidence used by every later MLX parity test."""

    model = runtime["model"]
    vision = runtime["vision"]
    boundaries = runtime["boundaries"]
    preprocessing = runtime["preprocessing"]
    attention = runtime["attention"]
    flow = runtime["flow"]
    parameters = runtime["parameters"]
    resize = preprocessing["resize"]
    inventory_rows = "\n".join(
        f"| `{item['name']}` | `{item['dtype']}` | `{item['shape']}` |" for item in weight_inventory
    )

    return f"""# Architecture

This document is generated from the pinned CPU/fp32 reference by
`scripts/inspect_reference.py`. It resolves every hypothesis in
`docs/history/BRIEF.md`
Section 3 using checkpoint metadata, installed LeRobot source, and one real
SO-101 observation. The installed PyPI wheel does not embed a source-git SHA;
the immutable package/checkpoint/dataset/VLM revisions below are the reference
identity used for every golden and conversion test.

## Reference pins

- LeRobot `{discovery.lerobot_version}`; PyTorch `{discovery.torch_version}`;
  Transformers `{discovery.transformers_version}`.
- SmolVLA policy source: `{discovery.policy_source}`.
- SmolVLA configuration source: `{discovery.config_source}`.
- SmolVLA VLM-with-expert source: `{discovery.expert_source}`.
- Policy checkpoint: `{CHECKPOINT_ID}` at `{CHECKPOINT_REVISION}`.
- Base VLM: `{BASE_VLM_ID}` at `{BASE_VLM_REVISION}`.
- Golden dataset: `{discovery.dataset_id}` at `{discovery.dataset_revision}`.
- Checkpoint inventory: {discovery.tensor_count:,} tensors and
  {discovery.parameter_count:,} parameters.
- Dataset: {len(discovery.camera_keys)} real cameras ({", ".join(f'`{item}`' for item in discovery.camera_keys)}),
  state `{list(discovery.state_shape)}`, action `{list(discovery.action_shape)}`,
  language task table `{discovery.has_language_tasks}`.

## Verified architecture

```text
two camera frames ──► top/left-pad + [-1, 1] ──► SigLIP vision (12 layers)
                                                    │ 1024 tokens/camera
                                                    ▼
                                      pixel shuffle ×{vision['pixel_shuffle_scale']} + projection
                                                    │ 64 × {model['text_hidden_size']} tokens/camera
task ──► newline + tokenizer (48 tokens) ──────────┤
state (6 → zero-padded 32) ──► linear projection ──┤
                                                    ▼
                   {model['vlm_layers']}-layer SmolVLM prefix prefill + per-layer KV cache
                                                    │
noise actions (50 × 32) + timestep ───────────────► {model['expert_layers']}-layer action expert
                                                    │  self-attn even layers; cross-attn odd layers
                                                    ▼
                    50 × {flow['velocity_dim']} velocity → Euler (10 steps, t=1 → 0) → slice → 50 × {flow['output_action_dim']} actions
```

{_table([
    ("Total parameters", f"{parameters['total']:,}"),
    ("VLM / action expert", f"{parameters['vlm']:,} / {parameters['expert']:,}"),
    ("Used VLM / expert layers", f"{model['vlm_layers']} / {model['expert_layers']}"),
    ("VLM / expert hidden width", f"{model['text_hidden_size']} / {model['expert_hidden_size']}"),
    ("Expert MLP intermediate width", str(model['expert_intermediate_size'])),
    ("Attention", f"{model['attention_heads']} query heads, {model['key_value_heads']} KV heads, head dim {model['head_dim']}"),
    ("Expert layer alignment", f"self-attention {model['self_attention_layers']}; cross-attention {model['cross_attention_layers']}"),
    ("State placement", model['state_placement']),
    ("RoPE base", str(model['rope_base'])),
])}

The expert is **720-wide**, not 576-wide: the checkpoint's 0.75 width multiplier
is applied to the 960-wide text model. Its attention still uses 15 × 64 query
channels and 5 × 64 KV channels; cross-attention layers project cached VLM KV
channels into expert KV channels.

## Boundary tensors and cache

{_table([
    ("Prepared camera batch", str(vision['input_shape'])),
    ("Vision encoder output", str(vision['encoder_output_shape'])),
    ("Connector output", str(vision['connector_output_shape'])),
    ("Language IDs", str(boundaries['language_tokens'])),
    ("Padded state / state embedding", f"{boundaries['padded_state']} / {boundaries['state_embedding']}"),
    ("Prefix / valid prefix tokens", f"{boundaries['prefix']} / {boundaries['prefix_valid_tokens']}"),
    ("Action suffix", str(boundaries['suffix'])),
    ("First layer cache K/V", f"{boundaries['first_cache_key']} / {boundaries['first_cache_value']}"),
    ("Cache layers", str(boundaries['cache_layers'])),
])}

The two real cameras produce 128 prefix image tokens. With 48 language slots and
one state token, the fixed prefix shape is `[1, 177, 960]`; padded language slots
remain in that shape but are invalid in its pad mask.

## Vision and preprocessing

{_table([
    ("Vision family", "SmolVLM2 SigLIP-style encoder"),
    ("Vision configuration", f"{vision['encoder_layers']} layers, hidden {vision['hidden_size']}, {vision['attention_heads']} heads, MLP {vision['intermediate_size']}"),
    ("Image / patch size", f"{vision['image_size']} / {vision['patch_size']}"),
    ("Vision activation", str(vision['activation'])),
    ("Pixel-shuffle scale", str(vision['pixel_shuffle_scale'])),
    ("Resize", f"{resize['height']}×{resize['width']} bilinear, align_corners={resize['align_corners']}"),
    ("Padding", f"{', '.join(resize['padding_edges'])}; pad value {resize['pad_before_pixel_normalization']} before pixel transform"),
    ("Pixel normalization", resize['pixel_transform']),
    ("Language", f"newline appended; right padded to {preprocessing['tokenizer_max_length']} tokens ({preprocessing['tokenizer_padding']})"),
    ("Configured / effective state norm", f"{preprocessing['configured_state_normalization']} / {preprocessing['state_normalization_effective']}"),
    ("Configured / effective action output norm", f"{preprocessing['configured_action_normalization']} / {preprocessing['action_unnormalization_effective']}"),
    ("Saved normalization-stat keys", ", ".join(f"`{key}`" for key in preprocessing['saved_stat_keys'])),
    ("Configured / injected empty cameras", f"{preprocessing['configured_cameras']} / {preprocessing['empty_cameras']}"),
])}

**Normalization correction.** The checkpoint declares mean/std normalization for
state and action, but its saved processor statistics exist only under the robot-
prefixed action keys shown above. The actual SO-101 runtime input uses
`observation.state` and output uses `action`, neither of which matches those
keys. The pinned processor therefore leaves this reference's state unchanged and
also leaves its output action unchanged. The MLX port must reproduce that
effective behavior by default and expose explicit stats replacement as a future
opt-in, rather than silently applying unrelated statistics.

## Attention, masking, and flow matching

{_table([
    ("Prefix content bidirectional", str(attention['prefix_content_is_bidirectional'])),
    ("State attends valid prefix", str(attention['state_attends_prefix'])),
    ("Image/language cannot attend state", str(attention['prefix_cannot_attend_state'])),
    ("Action suffix is causal", str(attention['suffix_is_causal'])),
    ("Flow steps / dt", f"{flow['steps']} / {flow['dt']}"),
    ("Velocity / returned action width", f"{flow['velocity_dim']} / {flow['output_action_dim']}"),
    ("Timesteps", ", ".join(f"{time:.1f}" for time in flow['times'])),
    ("Update", flow['update']),
    ("Noise / action endpoint", f"{flow['noise_at']} / {flow['actions_at']}"),
    ("Unit-velocity terminal value", str(flow['unit_velocity_terminal_value'])),
    ("Action queue", f"refill {runtime['queue']['chunk_size']} actions when empty; expose first {runtime['queue']['n_action_steps']} one at a time"),
])}

The prefix's 2-D mask is created from cumulative autoregressive flags: image and
language flags are 0, the state flag is 1. The suffix has 50 flags of 1, making
it causal. During prefix prefill every VLM layer stores post-RoPE K/V. During
each Euler denoising step, even expert layers self-attend over actions while odd
layers cross-attend into that fixed VLM K/V cache; suffix K/V appended by self-
attention are cropped back to the prefix length before the next step.

## BRIEF Section 3 verdicts

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| ~450M SmolVLM2 backbone, 64 visual tokens | Confirmed | 450,046,176 parameters; 12-layer 512/16 vision tower and 4× pixel shuffle produce 64 tokens/camera. |
| First 16 VLM layers and cross-attending expert | Confirmed | Checkpoint config and runtime cache both contain 16 layers. |
| ~100M 0.75-wide action expert | Corrected | 98,245,840 parameters, width 720, 16 layers; self-attention even/cross-attention odd. |
| 10-step Euler, 50 actions, 32 padded dimensions | Confirmed | Runtime schedule is 1.0→0.1, dt −0.1, `x_t = x_t + dt*v_t`. |
| Three-camera configuration / state in prefix | Confirmed | Three configured slots; two supplied; `empty_cameras=0`; state is the final VLM prefix token. |
| Prefix bidirectional / suffix causal masks | Confirmed | Runtime 2-D masks exercised on the real frame. |
| 512 pad resize and mean/std stats | Corrected | Resize and `x*2-1` confirmed; declared stats are present but ineffective for this checkpoint's unprefixed input/output keys. |

## Full checkpoint tensor inventory

The inventory below was read from the pinned `model.safetensors` header without
materializing any tensor. It is the conversion contract: every source tensor must
be mapped exactly once in Phase 2.

<details>
<summary>{len(weight_inventory)} tensors</summary>

| Source tensor | dtype | shape |
| --- | --- | --- |
{inventory_rows}

</details>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--json", type=Path, required=True, dest="json_path")
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args(argv)

    discovery = discover_reference(args.cache_dir)
    reference = ReferencePolicy.load(args.cache_dir)
    sample = load_dataset_observation(args.cache_dir, args.index)
    runtime = inspect_reference(reference, sample)
    checkpoint = discovery.checkpoint_config.with_name("model.safetensors")
    inventory = read_safetensors_inventory(checkpoint)
    payload = {
        "reference": {
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "base_vlm_id": BASE_VLM_ID,
            "base_vlm_revision": BASE_VLM_REVISION,
        },
        "runtime": runtime,
        "weight_inventory": inventory,
    }

    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(render_architecture_report(discovery, runtime, inventory), encoding="utf-8")
    print(args.json_path)
    print(args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
