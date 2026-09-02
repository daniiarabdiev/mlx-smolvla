# Distribution artifact manifest

This manifest records the Stage R P1-2 artifacts built locally from source
commit `a50cd3b5720a061262a978130600215a30fb8fbd` on
`2026-09-02T03:44:23Z`. Nothing was uploaded. The earlier P1-2 artifacts were
moved intact to the ignored `.cache/release-build-backup/pre-a50cd3b` before
this final P1-4 refresh.

## Build environment

- Host: Apple Silicon (`arm64`), macOS 26.6.2 build 25G83.
- Build frontend: `uv 0.11.25`.
- Compiler: Apple clang 21.0.0 (`clang-2100.1.1.101`).
- Deployment target: `MACOSX_DEPLOYMENT_TARGET=14.0`.
- Interpreter root: repository-local `.cache/uv-pythons`.
- Build/download cache: repository-local `.cache/uv`.
- Source: clean detached worktree `.cache/release-source` at the commit above.

The interpreters were provisioned with:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" \
UV_PYTHON_INSTALL_DIR="$PWD/.cache/uv-pythons" \
uv python install 3.11 3.12 3.13 --no-bin
```

The sdist was built with `uv build --sdist --clear`; each wheel used
`uv build --wheel --python <repository-local interpreter>`. Every build set
the three environment values above and wrote to the top-level `dist/`.

## Artifacts

| Artifact | Bytes | SHA-256 | Wheel tag / contents | Native `minos` |
| --- | ---: | --- | --- | ---: |
| `smolvla_mlx-0.0.1.tar.gz` | 366,428 | `f778711e1cadcdf6251b4a249857281feb8c114941d1d37b93afc4739af61b35` | No `.so`/`.dylib`; includes CMake, server, tests, and all native sources | Built smoke: 14.0 |
| `smolvla_mlx-0.0.1-cp311-cp311-macosx_14_0_arm64.whl` | 321,285 | `2a9e149025a0433829e2f7b3a150807409cf2838555b33da7ae357ce70a9014e` | `cp311-cp311-macosx_14_0_arm64`; server module included | 14.0 |
| `smolvla_mlx-0.0.1-cp312-cp312-macosx_14_0_arm64.whl` | 320,279 | `4412160a2b6f613f237d881347d4a93d27e2c96d82c82bf3991f59e4efa612b1` | `cp312-cp312-macosx_14_0_arm64`; server module included | 14.0 |
| `smolvla_mlx-0.0.1-cp313-cp313-macosx_14_0_arm64.whl` | 320,320 | `af93be38cd18bb3a45f94b039cd5f52a4a410fc8665d3aeb7eed3a5686a954fd` | `cp313-cp313-macosx_14_0_arm64`; server module included | 14.0 |

Each wheel declares `Requires-Python: >=3.11,<3.14`. Its optional LeRobot and
Torch reference requirements are guarded by `python_version >= "3.12"`; the
separate `serve` extra declares exactly
`lerobot[async]==0.6.1; python_version >= "3.12"`.

## Fresh-environment smoke matrix

Each artifact was installed with dependencies into a newly created venv. The
check ran from a directory outside the source checkout and asserted that the
import path was inside that venv, the backend was `native-reference`, and
gRPC, Torch, LeRobot, and Transformers were both unimported and unavailable;
protobuf's `google` package was also absent from the loaded module set. With
`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`, the installed
`smolvla-mlx predict --observation ...` command then emitted one finite
six-component action from the retained real golden observation.

| Installed artifact | Interpreter | Import isolation | Native backend | Offline `predict` |
| --- | --- | --- | --- | --- |
| sdist | CPython 3.12.13 | Pass | Pass | Pass |
| cp311 wheel | CPython 3.11.15 | Pass | Pass | Pass |
| cp312 wheel | CPython 3.12.13 | Pass | Pass | Pass |
| cp313 wheel | CPython 3.13.14 | Pass | Pass | Pass |

A fifth fresh CPython 3.12 environment installed the CPython 3.12 wheel with
its `serve` extra. From outside the source checkout it imported
`smolvla_mlx.server` from that environment, reproduced protobuf descriptor
SHA-256 `e116fbf44dd1fc65b67ff255c04857000c28e69055211af5ef3df85ac8d81f8d`,
bound an ephemeral loopback gRPC port, completed the reference `Ready` RPC,
stopped cleanly, and rendered the installed `smolvla-mlx serve --help` surface.
No model, hardware, serial port, or external service was contacted.

The source tree also builds a genuinely extension-free `py3-none-any` wheel
under `SMOLVLA_MLX_BUILD_NATIVE=0`; an isolated test proves that wheel contains
no binary and reports `pure-mlx-fallback`. The exact native and forced-fallback
paths run in the protected full suite.

## Upstream binary floor caveat

The project wheels are no longer macOS-26-tagged, and each packaged extension's
Mach-O load command independently says `minos 14.0`. The pinned `mlx==0.32.2`
wheel installed into all four smoke environments contains
`mlx/lib/libmlx.dylib` with `minos 26.2`; the linker reported the same fact at
build time. Consequently, these records prove this project's tag and binary
target, but do **not** prove an end-to-end runtime on macOS 14. The dependency's
binary is the effective tested limitation until MLX supplies (or is rebuilt as)
a lower-minimum binary.
