# Status

Implementation is blocked at Phase 3 decoder parity.

Completed and committed work includes the deterministic CPU/fp32 reference
harness, eight real golden cases, architecture audit, strict 500-tensor weight
conversion, preprocessing, vision encoder, and connector. The native language
decoder, prefix builder, and KV-cache implementation are present and strictly
load the checkpoint's actual 16-layer language subtree.

Focused language evidence: 46 of 50 prefix/decoder checks pass. Exact
prefix/mask/cutoff/cache tests, all 25 bf16 cases, import isolation, and the
final normalized fp32 prefix output pass. Four fp32 raw decoder-residual checks
fail their immutable `1e-3` max-absolute bound by `9.86328e-05` to
`2.29590e-03`; relative-L2 requirements pass. See `FAILURE_language.md` for
the traced RMS-reduction/SwiGLU amplification analysis and ruled-out remedies.

The action expert, Euler loop, end-to-end API, statistics, benchmarking, and
packaging remain unstarted because they depend on a Section-6-passing prefix
decoder. The exact next step is to obtain an MLX CPU reduction-precision control
or implement a justified native kernel that reproduces the PyTorch RMSNorm
arithmetic, then rerun `uv run pytest tests/test_prefix.py tests/test_language.py -q`.
