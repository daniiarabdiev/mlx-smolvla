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

## Limitations

- Physical SO-101 operation has not completed the gated hardware protocol and
  is not claimed.
- Production Metal fp32 passes the statistical gate but not the strict
  deterministic `0.005` maximum; strict CPU mode owns that contract.
- Native training is statistical alpha: standard export and Torch/MLX
  round-trip gates pass, while the trained checkpoint's derived deterministic
  gate remains a documented failure.
- Only the audited SmolVLA/SmolVLM2 configuration and complete compatible
  checkpoint inventories are accepted.

## Related work

[`tokimoa/smolvla-mlx`](https://huggingface.co/tokimoa/smolvla-mlx), uploaded
2026-07-29, is an earlier independent SmolVLA-to-MLX inference port. This
project's scope additionally includes fixed parity gates, a base runtime
without Torch/Transformers, LeRobot-protocol serving, and training; no
comparative performance claim is made against that project.
