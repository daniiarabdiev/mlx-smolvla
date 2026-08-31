# Status

The native correctness path is complete through Phase 4. It includes strict
500-tensor conversion, preprocessing, vision, connector, truncated VLM prefix
cache, 16-layer action expert, Euler flow sampling, and the queue-backed public
policy API. Runtime imports remain free of `torch`, `lerobot`, and
`transformers`.

The 50-frame statistical gate is green: fp32 MLX MAE is
`55.78303926587105` versus PyTorch `55.783039437383415` (ratio
`0.9999999969253671`); bf16 MLX MAE is `55.78358466590444` (ratio
`1.0000097740913103`). Both are below the immutable 1.05 ratio limit.

Phase 5 benchmark evidence is now present. Remaining v0.1 work is Phase 6
CLI/documentation and fresh-install proof, followed by the final repository
audit. No human task is currently required.
