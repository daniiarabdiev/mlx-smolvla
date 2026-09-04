# Distribution artifact manifest

This manifest records the `mlx-smolvla` 0.1.0 release-candidate artifacts
built locally from clean, pushed source commit
`8bb5c7eca16062d120478956824a3ed79759f21e` on 2026-09-04. Nothing was
uploaded. The hardware gate remains open, so no release tag was created.

## Build environment

- Host: Apple Silicon (`arm64`), macOS 26.6.2 build 25G83.
- Build frontend: `uv 0.11.25`.
- Compiler: Apple clang 21.0.0 (`clang-2100.1.1.101`).
- Deployment target: `MACOSX_DEPLOYMENT_TARGET=14.0`.
- Interpreters: repository-local CPython 3.11.15, 3.12.13, and 3.13.14.
- Build/download cache: repository-local `.cache/uv`.
- Source: clean `git archive` of the commit above, materialized in
  `.cache/release-mlx-smolvla-8bb5c7e-KNkxdO` before building.
- Output: ignored directory
  `.cache/release-mlx-smolvla-8bb5c7e-artifacts`.
- The four verified bytes were mirrored into ignored `dist/` for local use.
  The preceding 0.1.0 candidate bytes were moved intact to
  `.cache/dist-pre-reference-precision-20260904` rather than deleted. The older
  `.cache/dist-pre-stale-goal-20260904` backup is also retained.

The sdist was built with `uv build --sdist`; each native wheel used
`uv build --wheel --python <repository-local-interpreter>`. All four commands
set the deployment target and repository-local uv/Python cache roots. The
sdist contains 150 entries, including the PEP 517 backend, package-local CMake
project, native sources, canonical runtime, reference tools, training tools,
hardware safety/client modules, and tests; it contains no prebuilt `.so` or
`.dylib`.

## Artifacts

| Artifact | Bytes | SHA-256 | Wheel tag / contents | Project extension `minos` |
| --- | ---: | --- | --- | ---: |
| `mlx_smolvla-0.1.0.tar.gz` | 441,745 | `8e69411bbb5d525c096de416869e21a02806a397eb370f2431e49caf95925268` | Source distribution; native extension built during installation | 14.0 in the installation smoke |
| `mlx_smolvla-0.1.0-cp311-cp311-macosx_14_0_arm64.whl` | 379,081 | `ca7a981c60aae2dd3d6be9edc7e093d7e1658280bf5e3e15b913756400b78ce3` | `cp311-cp311-macosx_14_0_arm64`; 73 entries | 14.0 |
| `mlx_smolvla-0.1.0-cp312-cp312-macosx_14_0_arm64.whl` | 378,081 | `cf651589c99b45e5197dec2a4fde310039ca9db27506333375dae07ee65ffab4` | `cp312-cp312-macosx_14_0_arm64`; 73 entries | 14.0 |
| `mlx_smolvla-0.1.0-cp313-cp313-macosx_14_0_arm64.whl` | 378,120 | `e4db345cc4395277eb29d4bc1adf5914a2b93438886d672dd26d94986c609552` | `cp313-cp313-macosx_14_0_arm64`; 73 entries | 14.0 |

Archive inspection found only the canonical `mlx_smolvla/` package path.
Each wheel declares distribution `mlx-smolvla`, version `0.1.0`,
`Requires-Python: >=3.11,<3.14`, and console entry point
`mlx-smolvla = mlx_smolvla.cli:main`. The server, training, quantization,
hardware-safety/client, and native-extension surfaces are present. Every wheel
declares the Python-3.12+ pinned `hardware` extra. No wheel contains the retired
`smolvla_mlx/` package path. `vtool -show-build` independently reports
`platform MACOS` and `minos 14.0` for every packaged native extension.
Archive inspection also confirms that packaged hardware clients preload raw
present positions as goals while torque is off, require exact goal/fresh-
present equality before enable, and return through bounded one-public-unit
steps. Numeric camera indices remain session-local and must be assigned by
visual preflight; public camera keys and checkpoint mapping are unchanged.
Both startup orders passed with the two intended UVC cameras, and the earlier
apparent order failure involved the built-in Mac camera under the wrong role.
These bytes remain a local, untagged candidate and are not hardware-motion
release artifacts.

Every archive also contains the repaired `training/reference_export.py`
byte-exactly, SHA-256
`48af935ea41355d25a0a342a8ccbb6aaa4d87c48fb3034243540010983557801`.
This changes optional reference loading, not native inference or training math.

`twine check` passed the sdist and all three wheels without warnings.

## Fresh-environment smoke matrix

Each artifact was installed with dependencies into a new virtual environment.
Every probe ran from `/private/tmp/mlx-smolvla-smoke-8bb5c7e-Fh7DuG`, outside
both the main checkout and clean build source, and asserted that the imported
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

A sixth clean CPython 3.12 environment installed the cp312 wheel with its
`hardware` extra. Dependency checking passed; gRPC, LeRobot 0.6.1, serial, and
camera modules were present while PyAV remained absent. The installed
`hardware_safety` and `hiwonder_client` modules imported from that environment,
validated the stats-active six-axis checkpoint, and exposed the expected
two-camera mapping. Packaged-source inspection found the exact torque-off
stale-goal preload/readback guard and gradual bounded-return implementation.
The repository example rendered all three graduated modes and the hardware-
profile option with this installed interpreter. It did not open a device,
camera, vendor checkout, or network connection.

The installed cp312 wheel also verified the one-release cache compatibility
shim: `SMOLVLA_MLX_CACHE` alone is used with a `FutureWarning`;
`MLX_SMOLVLA_CACHE` wins and the warning says the legacy value was ignored
when both are set; and an explicit cache path wins while the warning says the
legacy value was ignored because an explicit directory was supplied.

A seventh clean CPython 3.12 environment installed the cp312 wheel with both
`reference` and `train` extras. Its installed reference loader matched the
archive/source hash and preserved all **500 tensors / 450,046,176 scalars** of
the actual retained T3B checkpoint with **zero mismatches** on CPU fp32,
CPU fp64, and MPS fp32. Checkpoint SHA-256 remained
`858704fa572501d9e5a048076f8da692693b90c463feda29201a72f3f0b18883`.
No model inference, training, or hardware access was needed for this identity
probe. Reference/data extras retain their known PyAV/OpenCV import warnings;
the serving and hardware-only environments still contain no PyAV.

All seven environments passed dependency integrity checks. Reproduction helpers,
archive audit, and smoke outputs are retained under
`.cache/release-smoke-8bb5c7e/`; environments use `.cache/smoke-8bb5c7e-*`.
The complete project suite passed **790/790 tests**, with no skip or xfail.

## Verified MLX range and macOS floor

The current host selected MLX 0.32.2's specialized macOS 26 wheel, which is
why the captured doctor report names that dependency tag. Separate official
macOS 14 arm64 wheel pairs were hash-verified and binary-inspected for each
supported MLX release:

The range evidence below is retained from the prior compatibility audit; this
rebuild's seven fresh environments use MLX 0.32.2. No new cross-version timing
or throughput measurement was taken, and native runtime sources are unchanged.

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
They remain in ignored `.cache/` and `dist/` directories. After the supervised
hardware gate clears, the final tag must be created and a fresh artifact set
must be built from that tag, re-smoked, and recorded in a successor manifest.
The operator must recheck the PyPI name and match every uploaded byte to that
tag-built manifest before publication; this untagged set must not be uploaded.
