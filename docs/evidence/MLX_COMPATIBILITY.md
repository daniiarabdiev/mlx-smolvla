# MLX and macOS compatibility evidence

`mlx-smolvla` supports Apple Silicon Macs running macOS 14 or newer with
`mlx>=0.32.0,<0.32.3`. MLX 0.32.0, 0.32.1, and 0.32.2 each passed the same
conversion, strict parity, production statistical, installed prediction,
diagnostic, and loopback-serving gates. The complete machine-readable record
is [`mlx-compatibility.json`](mlx-compatibility.json).

## Why an MLX 0.32.2 install can look like macOS 26 only

PyPI publishes separate official wheel families for current MLX releases. On
this macOS 26.6.2 host, the installer selected the specialized macOS 26 wheel.
Direct `vtool -show-build` inspection reports `minos 26.2` for its core
extension, `libjaccl.dylib`, and `libmlx.dylib`; their SHA-256 values exactly
match the corresponding files unpacked from the official wheel pair.

That selected wheel does not establish the package's oldest supported macOS.
The same MLX releases also publish macOS 14 arm64 wheels. The official
[MLX](https://pypi.org/project/mlx/) and
[mlx-metal](https://pypi.org/project/mlx-metal/) release metadata provided the
candidate filenames and expected hashes; every local download matched.

| MLX | CPython 3.12 core wheel SHA-256 | mlx-metal wheel SHA-256 | core / jaccl / mlx `minos` |
| --- | --- | --- | --- |
| 0.32.0 | `ea5a594355c89c0095eaba413fd39d4caa8642fa13432dfb0c9354d141046467` | `5b64b20ac24b0c401f489de01e8209edc4d372125201f19314e6f39e385322aa` | 14.0 / 14.0 / 14.0 |
| 0.32.1 | `f0b6a28089caeacdc27b89e18ff786ee956b8253cf8819778575fd90d2af8caa` | `3fbf7d3de783680e771818189bc734877c9b129ae91312338b5ede420cde44c9` | 14.0 / 14.0 / 14.0 |
| 0.32.2 | `77217798a2b036bae9f213b851d4cde4581893787c9964458b7d471f86036bd6` | `3825fff379dbc107dd3413e564a06caeaa24819910ec49c0439e454c06a1b9b8` | 14.0 / 14.0 / 14.0 |

The exact primary metadata endpoints are recorded in the JSON companion.

## Correctness and package checks

Each version ran in its own CPython 3.12.13 environment with the exact macOS
14 wheel pair. For every version:

- six conversion-contract tests and all 16 strict end-to-end golden cases
  passed across fp32 and bf16;
- the unchanged 50-frame production gate passed in both dtypes: fp32 MAE ratio
  `1.0000000009803565` and bf16 MAE ratio `1.0000097741322698`, both below the
  fixed `1.05` maximum;
- a canonical wheel installed outside the source checkout, exposed only the
  `mlx_smolvla` package, and produced a finite six-component action with Hub
  access forced offline;
- `mlx-smolvla doctor` reported the exact MLX version, the
  `macosx_14_0_arm64` dependency wheel tag, Metal as the default device, and a
  `verified` compatibility verdict; and
- an actual gRPC server bound an ephemeral loopback port, answered the pinned
  LeRobot `Ready` RPC, and stopped cleanly.

MLX 0.32.0 and 0.32.1 use the tested pure-MLX RMSNorm compatibility fallback;
0.32.2 uses the native reference extension. This avoids loading a native
extension built against a different MLX C++ ABI while preserving all fixed
numerical gates.

## Packaging correction found during the audit

The first post-rename trial wheel reused a stale generated `build/` directory.
It consequently contained both the retired and canonical packages and was
incorrectly tagged for macOS 26. That wheel was rejected and retained only in
the ignored diagnostic cache. A clean rebuild produced
`mlx_smolvla-0.0.1-cp312-cp312-macosx_14_0_arm64.whl` with SHA-256
`d69f71a06c96ead38f54b6b127e14487cfc7b8e4ea3d62bc9435b66e3b8399e3`,
26 canonical package entries, zero legacy package entries, and a project
extension whose Mach-O minimum is 14.0. A distribution test now rejects any
wheel containing the retired package directory. Final 0.1.0 artifacts are
rebuilt from a clean committed tree later in the release sequence.

The linker can still warn on this newer host that the locally selected MLX
dylib targets macOS 26.2. That host-build warning is expected: installation on
macOS 14 resolves the official macOS 14 dependency wheel whose dylibs were
directly inspected above.
