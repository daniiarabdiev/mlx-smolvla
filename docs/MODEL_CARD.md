---
license: apache-2.0
library_name: mlx
pipeline_tag: robotics
tags:
  - apple-silicon
  - lerobot
  - mlx
  - smolvla
  - vision-language-action
---

# mlx-smolvla

`mlx-smolvla` is a native MLX runtime, serving adapter, and preview training
surface for compatible SmolVLA checkpoints on Apple Silicon. This repository
does not redistribute the upstream model weights; loading
`lerobot/smolvla_base` downloads its pinned files from the original Hub
repository and converts them into a local cache.

## Sources

- Policy: [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base),
  revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
- Backbone: SmolVLM2-500M family, pinned revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Reference implementation: LeRobot 0.6.1 with PyTorch 2.11.0 CPU/fp32 for
  deterministic goldens.

## Intended use

Use the package for local research inference, software-only LeRobot-protocol
serving, parity evaluation, or experimental native MLX training on compatible
Apple Silicon Macs. The base runtime imports neither Torch, Transformers, nor
LeRobot. Robot actuation requires a separately reviewed client and safety
system.

## Validation

Eight pinned real observations cover preprocessing, vision, connector,
16-layer language prefix/KV export, action expert, ten Euler steps, and
postprocessing. A separate 50-frame population evaluates first-action MAE.
Thresholds were fixed before evaluation and are indexed in
[`evidence/README.md`](evidence/README.md).

The retained native-trained expert-only LoRA checkpoint also passes a separate
56-case, fixed-limit repair validation after correcting reference-loader
precision loss: normalized maximum `0.000021457672119140625`, physical maximum
`0.00042724609375`, each below the unchanged `0.005` limit. Its original
prospective verdict is not overwritten or relabeled. See
[`TRAINED_PARITY_REPAIR.md`](evidence/TRAINED_PARITY_REPAIR.md).

## Limitations

- Connected Hiwonder SO-101 state/camera capture, one valid guarded action, and
  a two-chunk continuous stage passed. A separate 20-chunk attempt failed exact
  return under the temporary low-torque profile, so sustained operation and
  reliable task completion are not claimed.
- Raw `lerobot/smolvla_base` output does not have effective physical
  state/action statistics for the generic keys. A motion client must use a
  reviewed checkpoint whose statistics match the target robot.
- Production Metal fp32 passes the statistical gate but not the strict
  deterministic `0.005` maximum; strict CPU mode owns that contract.
- Native training is a research preview. The retained LoRA checkpoint passes
  post-repair fixed parity, but full fine-tuning has code/smoke coverage only;
  neither result establishes real-robot task success or all-run convergence.
- Only the audited SmolVLA/SmolVLM2 configuration and complete compatible
  checkpoint inventories are accepted.

## Related work

[`tokimoa/smolvla-mlx`](https://huggingface.co/tokimoa/smolvla-mlx), uploaded
2026-07-29, is an earlier independent SmolVLA-to-MLX inference port. This
project's scope additionally includes fixed parity gates, a base runtime
without Torch/Transformers, LeRobot-protocol serving, and training; no
comparative performance claim is made against that project.
