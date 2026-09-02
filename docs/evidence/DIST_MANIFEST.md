# Distribution artifact manifest

This manifest records the `mlx-smolvla` 0.1.0 release-candidate artifacts
built locally from clean, pushed source commit
`5efb48aa5d47e30438d6a4aaafc180a2983b1235` on 2026-09-02. Nothing was
uploaded. The hardware gate remains open, so no release tag was created.

## Build environment

- Host: Apple Silicon (`arm64`), macOS 26.6.2 build 25G83.
- Build frontend: `uv 0.11.25`.
- Compiler: Apple clang 21.0.0 (`clang-2100.1.1.101`).
- Deployment target: `MACOSX_DEPLOYMENT_TARGET=14.0`.
- Interpreters: repository-local CPython 3.11.15, 3.12.13, and 3.13.14.
- Build/download cache: repository-local `.cache/uv`.
- Source: clean detached worktree `.cache/release-mlx-smolvla-5efb48a` at
  the commit above.
- Output: ignored directory
  `.cache/release-mlx-smolvla-5efb48a-artifacts`.

The sdist was built with `uv build --sdist`; each native wheel used
`uv build --wheel --python <repository-local-interpreter>`. All four commands
set the deployment target and repository-local uv/Python cache roots. The
sdist contains 145 entries, including the PEP 517 backend, package-local CMake
project, native sources, canonical runtime, reference tools, training tools,
and tests; it contains no prebuilt `.so` or `.dylib`.

## Artifacts

| Artifact | Bytes | SHA-256 | Wheel tag / contents | Project extension `minos` |
| --- | ---: | --- | --- | ---: |
| `mlx_smolvla-0.1.0.tar.gz` | 413,439 | `01efceb57a5bc22679f486160a6746894849cb577f1772aa597b22ee3e71180e` | Source distribution; native extension built during installation | 14.0 in the installation smoke |
| `mlx_smolvla-0.1.0-cp311-cp311-macosx_14_0_arm64.whl` | 364,651 | `7cdecaf15a638ab11f2770341118f563fcef09f79ffbf989e027b9854aa92ae4` | `cp311-cp311-macosx_14_0_arm64`; 71 entries | 14.0 |
| `mlx_smolvla-0.1.0-cp312-cp312-macosx_14_0_arm64.whl` | 363,654 | `fd574f417f942ed4f4c0fd142f3cd254e7a9ed943fa940831100bc5304b53088` | `cp312-cp312-macosx_14_0_arm64`; 71 entries | 14.0 |
| `mlx_smolvla-0.1.0-cp313-cp313-macosx_14_0_arm64.whl` | 363,690 | `b9dc2a9e77d63d86706abedc4e8fdae6dca3e78128006629e3539dc47b330583` | `cp313-cp313-macosx_14_0_arm64`; 71 entries | 14.0 |

Archive inspection found only the canonical `mlx_smolvla/` package path.
Each wheel declares distribution `mlx-smolvla`, version `0.1.0`,
`Requires-Python: >=3.11,<3.14`, and console entry point
`mlx-smolvla = mlx_smolvla.cli:main`. The server, training, quantization, and
native-extension surfaces are present. No wheel contains the retired
`smolvla_mlx/` package path. `vtool -show-build` independently reports
`platform MACOS` and `minos 14.0` for every packaged native extension.

`twine check` passed the sdist and all three wheels without warnings.

## Fresh-environment smoke matrix

Each artifact was installed with dependencies into a new virtual environment.
Every probe ran from `/private/tmp/mlx-smolvla-smoke-5efb48a`, outside both the
main checkout and detached build worktree, and asserted that the imported
module path was inside the virtual environment. Hub and dataset access were
forced offline for prediction.

| Installed artifact | Interpreter | Canonical import and isolation | `doctor` | Native backend | Offline finite prediction |
| --- | --- | --- | --- | --- | --- |
| sdist | CPython 3.12.13 | Pass | Pass | `native-reference` | Pass, six components |
| cp311 wheel | CPython 3.11.15 | Pass | Pass | `native-reference` | Pass, six components |
| cp312 wheel | CPython 3.12.13 | Pass | Pass | `native-reference` | Pass, six components |
| cp313 wheel | CPython 3.13.14 | Pass | Pass | `native-reference` | Pass, six components |

For all four base installs:

- distribution version was exactly 0.1.0 and `smolvla_mlx` was not importable;
- importing `mlx_smolvla` neither imported nor made available Torch,
  Transformers, LeRobot, or gRPC;
- `mlx-smolvla doctor` reported Metal as the default device, MLX 0.32.2, and
  the `verified` compatibility verdict; and
- `mlx-smolvla predict --observation ...` used the retained real
  `sample_000` observation and emitted six finite values with
  `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.

A fifth clean CPython 3.12 environment installed the cp312 wheel with its
`serve` extra. It imported the server from that environment, reproduced the
pinned protobuf descriptor SHA-256
`e116fbf44dd1fc65b67ff255c04857000c28e69055211af5ef3df85ac8d81f8d`,
bound an ephemeral `127.0.0.1` gRPC port, completed the reference `Ready` RPC,
and shut down cleanly. Installed CLI help exposed `serve`, `train`, `doctor`,
both VLM quantization presets, LoRA/full/resume controls, the remote-bind guard,
and the no-clobber latency-log option. Installed `vlm-8bit` and `vlm-4bit`
predictions each emitted six finite values offline. No robot, serial port,
camera, model host, or other external service was contacted.

The installed cp312 wheel also verified the one-release cache compatibility
shim: `SMOLVLA_MLX_CACHE` alone is used with a `FutureWarning`;
`MLX_SMOLVLA_CACHE` wins and the warning says the legacy value was ignored
when both are set; and an explicit cache path wins while the warning says the
legacy value was ignored because an explicit directory was supplied.

## Verified MLX range and macOS floor

The current host selected MLX 0.32.2's specialized macOS 26 wheel, which is
why the captured doctor report names that dependency tag. Separate official
macOS 14 arm64 wheel pairs were hash-verified and binary-inspected for each
supported MLX release:

| MLX | Official core wheel SHA-256 | Official `mlx-metal` SHA-256 | Dependency binary `minos` | Fixed gates / installed smoke |
| --- | --- | --- | --- | --- |
| 0.32.0 | `ea5a594355c89c0095eaba413fd39d4caa8642fa13432dfb0c9354d141046467` | `5b64b20ac24b0c401f489de01e8209edc4d372125201f19314e6f39e385322aa` | 14.0 | Pass; pure-MLX compatibility backend |
| 0.32.1 | `f0b6a28089caeacdc27b89e18ff786ee956b8253cf8819778575fd90d2af8caa` | `3fbf7d3de783680e771818189bc734877c9b129ae91312338b5ede420cde44c9` | 14.0 | Pass; pure-MLX compatibility backend |
| 0.32.2 | `77217798a2b036bae9f213b851d4cde4581893787c9964458b7d471f86036bd6` | `3825fff379dbc107dd3413e564a06caeaa24819910ec49c0439e454c06a1b9b8` | 14.0 | Pass; native reference backend |

The exact primary-source URLs, wheel hashes, binary hashes, conversion and
golden results, unchanged 50-frame statistical results, installed predictions,
doctor reports, and loopback results are in
[`MLX_COMPATIBILITY.md`](MLX_COMPATIBILITY.md) and
[`mlx-compatibility.json`](mlx-compatibility.json).

## Publication status

These files are local release-candidate artifacts, not published releases.
They remain under the ignored `.cache/` tree. After the supervised hardware
gate clears, the final tag must be created and a fresh artifact set must be
built from that tag, re-smoked, and recorded in a successor manifest. The
operator must recheck the PyPI name and match every uploaded byte to that
tag-built manifest before publication; this untagged set must not be uploaded.
