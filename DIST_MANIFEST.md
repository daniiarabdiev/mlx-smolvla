# Distribution artifact manifest

This manifest records the Stage R P1-2 artifacts built locally from source
commit `3e30212604985dbaf2ad1360b1e4fc1023303cf6` on
`2026-09-02T02:19:42Z`. Nothing was uploaded.

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
| `smolvla_mlx-0.0.1.tar.gz` | 346,849 | `f689537bca083cf63a771e44b904e52b107d147a3a06df67517de64e6e2db81c` | No `.so`/`.dylib`; includes CMake and all native sources | Built smoke: 14.0 |
| `smolvla_mlx-0.0.1-cp311-cp311-macosx_14_0_arm64.whl` | 308,734 | `9e814355869151d17711045e14f4e0a6a323084228febcf2c7e55ea7448f4d76` | `cp311-cp311-macosx_14_0_arm64` | 14.0 |
| `smolvla_mlx-0.0.1-cp312-cp312-macosx_14_0_arm64.whl` | 307,730 | `f1ffafddeca298f4b7e96c71c2fc97cc0b7ce5cfcd1128dc1592598458ff2212` | `cp312-cp312-macosx_14_0_arm64` | 14.0 |
| `smolvla_mlx-0.0.1-cp313-cp313-macosx_14_0_arm64.whl` | 307,767 | `56d8718c04dbd59f7f028a234fc5a5b1222156fff67078d695de4121ab590595` | `cp313-cp313-macosx_14_0_arm64` | 14.0 |

Each wheel declares `Requires-Python: >=3.11,<3.14`. Its optional LeRobot and
Torch reference requirements are guarded by `python_version >= "3.12"`.

## Fresh-environment smoke matrix

Each artifact was installed with dependencies into a newly created venv. The
check ran from a directory outside the source checkout and asserted that the
import path was inside that venv, the backend was `native-reference`, and
Torch, LeRobot, and Transformers were both unimported and unavailable. With
`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`, the installed
`smolvla-mlx predict --observation ...` command then emitted one finite
six-component action from the retained real golden observation.

| Installed artifact | Interpreter | Import isolation | Native backend | Offline `predict` |
| --- | --- | --- | --- | --- |
| sdist | CPython 3.12.13 | Pass | Pass | Pass |
| cp311 wheel | CPython 3.11.15 | Pass | Pass | Pass |
| cp312 wheel | CPython 3.12.13 | Pass | Pass | Pass |
| cp313 wheel | CPython 3.13.14 | Pass | Pass | Pass |

The source tree also builds a genuinely extension-free `py3-none-any` wheel
under `SMOLVLA_MLX_BUILD_NATIVE=0`; an isolated test proves that wheel contains
no binary and reports `pure-mlx-fallback`. The exact native and forced-fallback
paths both ran in the 584-test full suite.

## Upstream binary floor caveat

The project wheels are no longer macOS-26-tagged, and each packaged extension's
Mach-O load command independently says `minos 14.0`. The pinned `mlx==0.32.2`
wheel installed into all four smoke environments contains
`mlx/lib/libmlx.dylib` with `minos 26.2`; the linker reported the same fact at
build time. Consequently, these records prove this project's tag and binary
target, but do **not** prove an end-to-end runtime on macOS 14. The dependency's
binary is the effective tested limitation until MLX supplies (or is rebuilt as)
a lower-minimum binary.
