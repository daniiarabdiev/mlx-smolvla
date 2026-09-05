# Distribution artifact manifest

This manifest records the final local `mlx-smolvla` 0.1.0 artifacts built from
the pushed annotated tag `v0.1.0`. The tag resolves to verified source commit
`9b28dc216e24aa86d121d9b805c1fc1733afbf9d`. Published on 2026-09-05 with explicit operator authorization. The repository
is public. PyPI and GitHub Release each contain all four artifacts; their
SHA-256 digests exactly match this manifest. The source tag is unchanged.
No converted weights were uploaded to a model Hub.

- [PyPI 0.1.0](https://pypi.org/project/mlx-smolvla/0.1.0/)
- [GitHub Release](https://github.com/daniiarabdiev/mlx-smolvla/releases/tag/v0.1.0)

## Build environment

- Host: Apple Silicon (`arm64`), macOS 26.6.2 build 25G83.
- Build frontend: `uv 0.11.25`.
- Compiler: Apple clang 21.0.0 (`clang-2100.1.1.101`).
- Deployment target: `MACOSX_DEPLOYMENT_TARGET=14.0`.
- Interpreters: repository-local CPython 3.11.15, 3.12.13, and 3.13.14.
- Build/download cache: repository-local `.cache/uv`.
- Source: clean detached worktree at `.cache/release-v0.1.0-source`, checked
  out from `v0.1.0` and verified at the commit above.
- Output: ignored `.cache/release-v0.1.0-artifacts`.
- The four verified bytes are mirrored in ignored `dist/`. The preceding
  untagged candidate remains intact at
  `.cache/dist-pre-tag-v0.1.0-20260904`; earlier candidate backups are also
  preserved.

The sdist used `uv build --sdist`; each wheel used
`uv build --wheel --python <repository-local-interpreter>`. Every command set
the deployment target and repository-local uv/Python cache roots. The sdist
contains 150 entries, including the PEP 517 backend, native sources, canonical
runtime, reference/training tools, hardware safety/client modules, and tests;
it contains no prebuilt `.so` or `.dylib`.

## Artifacts

| Artifact | Bytes | SHA-256 | Wheel tag / contents | Project extension `minos` |
| --- | ---: | --- | --- | ---: |
| `mlx_smolvla-0.1.0.tar.gz` | 444,185 | `7c55afe86aa47e59e200a6297584ee3f4f8216153ce8544d0a5bc010cf88b7ce` | Source distribution; native extension built during installation | 14.0 in the installation smoke |
| `mlx_smolvla-0.1.0-cp311-cp311-macosx_14_0_arm64.whl` | 379,722 | `14ca4c82e55b32203dde420f217fbb064d30d5ea6796852e964205c450ba1944` | `cp311-cp311-macosx_14_0_arm64`; 73 entries | 14.0 |
| `mlx_smolvla-0.1.0-cp312-cp312-macosx_14_0_arm64.whl` | 378,725 | `41ba7c798fa4f7553aaa68e1b22582eff620b1096005cbb895e48d274cf69542` | `cp312-cp312-macosx_14_0_arm64`; 73 entries | 14.0 |
| `mlx_smolvla-0.1.0-cp313-cp313-macosx_14_0_arm64.whl` | 378,761 | `9a0549d96a70a0b608895829c0148a442334d8741fad1258b70884c272bbada0` | `cp313-cp313-macosx_14_0_arm64`; 73 entries | 14.0 |

Archive inspection found only the canonical `mlx_smolvla/` package path. Each
wheel declares distribution `mlx-smolvla`, version `0.1.0`,
`Requires-Python: >=3.11,<3.14`, and console entry point
`mlx-smolvla = mlx_smolvla.cli:main`. Server, training, quantization, hardware
safety/client, and native-extension surfaces are present; every wheel declares
the Python-3.12+ pinned `hardware` extra. No wheel contains the retired
`smolvla_mlx/` path. `vtool -show-build` independently reports `platform MACOS`
and `minos 14.0` for all three extensions.

Packaged-source inspection also proves the final hardware changes are present:
raw present positions are checked against controller minimum/maximum limits
before the first goal write; explicit-torque and goal-write arming modes require
exact torque readback; post-write failures enter torque-off cleanup; and
single-action mode waits through rejected holds before stopping after its first
valid action. All archives contain the repaired `training/reference_export.py`
byte-exactly at SHA-256
`48af935ea41355d25a0a342a8ccbb6aaa4d87c48fb3034243540010983557801`.

`twine check` passed all four artifacts without warning.

## Fresh-environment smoke matrix

Each artifact was installed with dependencies into a new virtual environment.
Every runtime probe ran from
`/private/tmp/mlx-smolvla-smoke-v0.1.0-WmPSkXML`, outside the main and detached
source checkouts, and asserted that imports came from its virtual environment.
Hub and dataset access were forced offline for prediction.

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
- offline prediction on retained real observation `sample_000` emitted six
  finite values.

A fifth clean CPython 3.12 environment installed the cp312 wheel with `serve`.
It reproduced protobuf descriptor SHA-256
`e116fbf44dd1fc65b67ff255c04857000c28e69055211af5ef3df85ac8d81f8d`,
bound an ephemeral loopback gRPC port, completed `Ready`, and shut down. CLI
help exposed serving/training/doctor/predict, remote-bind and latency-log guards,
quantization, and LoRA/full/resume controls. Installed `vlm-8bit` and `vlm-4bit`
each emitted six finite offline values. PyAV was absent.

A sixth clean CPython 3.12 environment installed the cp312 wheel with
`hardware`. It found gRPC, LeRobot 0.6.1, serial, and camera modules while PyAV
remained absent. Installed modules validated the stats-active six-axis
checkpoint, two-camera mapping, raw controller-limit guard, both arming modes,
bounded single-valid-action rule, stale-goal guard, and gradual return. The
tagged example rendered all three modes plus the hardware-profile, chunk-limit,
and arming-mode options. No device, camera, vendor checkout, or network peer
was opened.

The installed cp312 wheel also passed all cache-shim precedence/warning cases:
legacy-only, current variable wins, and explicit path wins.

A seventh clean CPython 3.12 environment installed the cp312 wheel with both
`reference` and `train`. Its installed reference loader matched the archive
hash and preserved all **500 tensors / 450,046,176 scalars** in the retained
T3B checkpoint with **zero mismatches** on CPU fp32, CPU fp64, and MPS fp32.
Checkpoint SHA-256 remained
`858704fa572501d9e5a048076f8da692693b90c463feda29201a72f3f0b18883`.
Reference/data extras emitted their known PyAV/OpenCV warning; serving and
hardware-only environments remained PyAV-free. No inference, training, or
hardware access was needed for this identity probe.

All seven environments passed dependency integrity checks: 22 packages in each
base environment, 56 in serve, 65 in hardware, and 80 in reference/train.
Reproduction helpers, build/audit output, and smoke logs are retained under
`.cache/release-smoke-v0.1.0/`; environments use
`.cache/smoke-v0.1.0-*`. The final tagged source passed **803/803 tests**, with
no skip or xfail; the unchanged fast lane passed 502 selected tests under two
minutes.

| Verification record | SHA-256 |
| --- | --- |
| Archive audit | `e0120423163cdb16e9f4c1024d037c134e62674f9a772320c86c2d753bb64ca3` |
| Twine check | `97cab7fcaba700b434d53596e1f29a7dd27cd8d29b87ac4b57b7840d716ee62a` |
| Native-extension `vtool` inspection | `415007276ec4ba0f1e7dc204a145f47378a2fbb61491132f05ae4f17ae796370` |
| Sdist base smoke | `c144fb42500d9406088a298fab853956cc2c84af6e50af326662208214663f87` |
| CPython 3.11 wheel base smoke | `3270eb73aa67ad22d53e30992159c920a6d520340ff8d5778d9df704bdaef3ac` |
| CPython 3.12 wheel base smoke | `4039437f9b35c3c6ea68375bce5428a951bc2cb5b1781547d952ea6686036977` |
| CPython 3.13 wheel base smoke | `62f084c2769632ea8866109354a05b3d8d9ba7583146e95b85220124f21cb9a0` |
| Serve/quantization smoke | `88f8048a7d0d09a6be05cca50fed3f876652670ead9146a385574c937e22fd37` |
| Hardware no-device smoke | `4be8e1e3f1861e4e539c0d1549565855c5ec31b23b27523f8a391f0ef56f88eb` |
| Cache-shim smoke | `50fa88577b689cc778e5c947cdbb0e60d3e52aefb20ad771f799fc58b5cd72ba` |
| Reference/training identity smoke | `5859a65e7689b1a962ce1fbb316d0c3afbd9815bf30687cab38121d8c0f8d031` |

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

Published on 2026-09-05 with explicit operator authorization. The repository
is public. PyPI and GitHub Release each contain all four artifacts; their
SHA-256 digests exactly match this manifest. The source tag is unchanged.
No converted weights were uploaded to a model Hub.

- [PyPI 0.1.0](https://pypi.org/project/mlx-smolvla/0.1.0/)
- [GitHub Release](https://github.com/daniiarabdiev/mlx-smolvla/releases/tag/v0.1.0)
