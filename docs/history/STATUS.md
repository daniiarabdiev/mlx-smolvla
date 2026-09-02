# Status

DEFINITION OF DONE MET

SmolVLA v0.1 is a native MLX inference implementation for the audited
`lerobot/smolvla_base` checkpoint. It strictly converts and loads all 500
checkpoint tensors, performs preprocessing, vision, truncated VLM prefix
caching, the 16-layer action expert, flow-matching Euler sampling, and exposes
the queue-backed `SmolVLAMLX` policy API.

Final verification on this M5 Pro:

- `make test`: **179 passed in 147.26 seconds** on the exact final tree.
- Eight deterministic real-frame goldens pass in fp32 and bf16 at the fixed
  `5e-3` and `5e-2` end-to-end maximum-absolute thresholds.
- The 50-frame statistical gate is green: fp32 MLX/PyTorch MAE ratio
  `0.9999999969253671`; bf16 ratio `1.0000097740913103`, both below `1.05`.
- `make goldens` regenerated all 2,160 tensors twice with identical manifest
  SHA-256 `8531e61b98506d5a43e0b1235de7aece578bf4efa66b007fa13f6cecd1ceb215`.
- `BENCHMARK.md` records M5 Pro Metal chunk latency: fp32 111.34 ms median;
  bf16 131.12 ms median.
- Clean Python 3.12 environments successfully completed `pip install .` and
  wheel installation; the installed CLI converted the real checkpoint and
  predicted a real public SO-101 dataset frame. Runtime import isolation stays
  free of `torch`, `lerobot`, and `transformers`.

The native CLI provides `convert`, `test`, `bench`, and `predict`. The core
package has six pinned native dependencies; `.[reference]` remains an explicit
optional extra only for the dataset-backed CLI bridge and reference tests.

No human task is open. Robot I/O, training, and quantization remain deliberately
out of scope for v0.1.
