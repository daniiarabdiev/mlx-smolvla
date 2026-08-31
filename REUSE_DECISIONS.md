# MLX reuse decisions

This is the implementation boundary for the native runtime. The package under
`smolvla_mlx/` must never import PyTorch, LeRobot, or Transformers. The
reference-only `reference/`, `scripts/`, and test lanes may do so.

## Decision summary

| Component | Decision | Why |
| --- | --- | --- |
| Checkpoint access and conversion | Reimplement | The checkpoint needs a one-to-one PyTorch-to-MLX name and layout map, plus source/target checksum accounting. Neither upstream package knows the SmolVLA expert names. |
| Image resize, padding, pixel normalization | Reimplement | SmolVLA pads on the left and top before `x * 2 - 1`; the general `mlx-vlm` processing stack is Transformers-coupled and does not provide this policy-specific path. |
| Tokenization | Reimplement around `tokenizers` | `tokenizer.json` can be loaded directly with the allowed runtime dependency. This avoids importing Transformers while retaining the checkpoint tokenizer vocabulary and special-token behavior. |
| Vision encoder | Vendor and adapt focused `mlx-vlm` Idefics3 code | Its MLX Conv2D, position embedding, attention, and layout match the required SigLIP-style tower. A focused copy is needed because the checkpoint requires `gelu_pytorch_tanh`, while the available implementation constructs precise GELU. |
| Pixel-shuffle connector | Vendor and adapt focused `mlx-vlm` Idefics3 code | The upstream pixel-shuffle layout is the right 1024 → 64-token mechanism when configured with scale 4. SmolVLA only needs the projection path, not generic image-token insertion. |
| VLM text layers and KV cache | Vendor and substantially adapt focused `mlx-vlm` Llama/Idefics3 primitives | We require exactly the first 16 layers, an explicit non-causal prefix mask, the reference's split-half RoPE with base 10,000, eager fp32 attention math, exported post-RoPE K/V, and cache crop/reset. The generic `LanguageModel` does not expose this execution boundary. |
| Action expert | Reimplement from the verified LeRobot behavior | Its interleaved self/cross-attention and unusual 720-wide hidden state with 960-wide attention channels are policy-specific. A clean MLX implementation is less risky than attempting to fit it into a generic VLM abstraction. |
| Prefix assembly, masks, timestep embedding, Euler loop, action queue | Reimplement | These are compact policy semantics whose exact reference behavior is recorded in `ARCHITECTURE.md`; they do not benefit from importing a broad framework. |

## Why the runtime will not import `mlx_vlm`

`mlx-vlm` is a useful source implementation, but its top-level package imports
generation and processing utilities, and its processor modules import
Transformers. Importing it would violate the hard dependency-isolation rule even
if only its model classes were intended for use. The port will retain only the
small MLX-only portions needed for parity as local vendored modules.

## Required adaptations

- Vision MLP activation must use the PyTorch tanh approximation of GELU, not
  the precise GELU constructed by `mlx_vlm.models.idefics3.vision.MLP`.
- The connector's scale factor is **4**, yielding 64 tokens from the 32×32
  patch grid. The generic default is 2 and must not be used.
- Text attention needs the reference's 2-D boolean masks and fp32 QK/softmax
  path. The prefix is bidirectional among valid image/language tokens, while
  the state and action suffix flags introduce causal boundaries.
- Text RoPE is applied to split halves at base 10,000. The MLX implementation
  will prove this against goldens rather than inherit a generic model default.
- Expert odd layers cross-attend to cached base-VLM K/V; even layers append
  action K/V temporarily and the cache is cropped back after every Euler step.

## Licensing and attribution plan

`mlx-vlm` 0.6.4 is MIT licensed (copyright © 2025 Prince Canuma). Any copied
or adapted source file will retain its MIT header, identify its upstream source
path and version in a file comment, and be enumerated in `NOTICE`.

LeRobot 0.6.1 is Apache-2.0 (copyright 2024 The Hugging Face team), with the
SmolVLA files carrying 2025 Hugging Face headers. Any code derived closely from
its policy semantics will retain the Apache-2.0 header and be enumerated in
`NOTICE`. Pure clean-room implementations are still documented here as derived
from the reference's externally verified behavior.

The following focused MIT adaptations are now vendored, retaining upstream
source/version headers in the files and a full license notice in `NOTICE`:

| Local file | Upstream source | Adaptation |
| --- | --- | --- |
| `smolvla_mlx/vision.py` | `mlx_vlm/models/idefics3/vision.py` at mlx-vlm 0.6.4 | Fixed the audited 512px SigLIP dimensions, accepts NCHW policy input, omits unused variable-resolution mask machinery, and replaces precise GELU with `gelu_pytorch_tanh`. |
| `smolvla_mlx/connector.py` | `mlx_vlm/models/idefics3/idefics3.py` at mlx-vlm 0.6.4 | Retains the pixel-shuffle layout while fixing the checkpoint's scale factor to 4 and exposing only the required modality projection. |
