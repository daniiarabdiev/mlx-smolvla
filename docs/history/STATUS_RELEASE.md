# Release Status

Stage R is complete against the normative `BRIEF_RELEASE.md` package
definitions and the ordering amendments in `BRIEF_FULL.md` and
`BRIEF_T3B.md`. Nothing was uploaded to PyPI, the Hugging Face Hub, or any
container registry.

## Baseline and closing verification

- Full-scope kickoff baseline: **308/308 tests passed in 226.41 seconds**.
- P1-4 protected source run: **601/601 passed in 521.24 seconds**.
- Final post-artifact Stage R run: **601/601 passed in 537.54 seconds**.
- Original T3 failure record SHA-256 remains exactly
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
  No fixed or derived tolerance was changed.

## Package outcomes

| Package | Outcome | Primary evidence |
| --- | --- | --- |
| P0-1 — remote/license | Pass | `origin/main` mirrors canonical history; Apache-2.0 `LICENSE`, `NOTICE`, and reconciliation evidence are present. |
| P0-2 — checkpoint generality | Pass | Base identity stats, stats-active base, and pinned public fine-tune all pass deterministic and 50-frame statistical gates in fp32/bf16; see `ARCHITECTURE.md` and `PROGRESS.md`. |
| P0-3 — cache hygiene | Pass | Frozen cleanup reduced `.cache/smolvla_mlx` from 91,447,880 KiB to 52,764,872 KiB without changing protected fingerprints or redownloading model data. |
| P1-1 — production Metal | Pass with bounded limitation | Default production is explicit; bf16 passes its fixed deterministic gate, fp32 does not, and both pass the unchanged statistical gate. Exact timing and memory are in `BENCHMARK.md`. |
| P1-2 — portability | Pass with upstream limitation | Optional native extension/fallback are tested; final sdist and CPython 3.11–3.13 arm64 wheels pass fresh base installs and offline real prediction. Project binaries target macOS 14; pinned MLX's dylib still targets 26.2. |
| P1-3 — release documentation | Pass | Install, ten-line API, CLI, measured performance, correctness, GPU fine-tune handoff, cache safety, troubleshooting, license, and attribution are documented and command-checked. |
| P1-4 — async server | Pass in software | Exact LeRobot 0.6.1 schema/transport is audited; validation, errors, cancellation, concurrency, lifecycle, and security are tested. A recorded localhost chunk equals direct `select_action` exactly. Hardware validation remains pending. |

## Final distribution artifacts

Built from clean pushed source
`a50cd3b5720a061262a978130600215a30fb8fbd` with no upload:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `smolvla_mlx-0.0.1.tar.gz` | 366,428 | `f778711e1cadcdf6251b4a249857281feb8c114941d1d37b93afc4739af61b35` |
| `smolvla_mlx-0.0.1-cp311-cp311-macosx_14_0_arm64.whl` | 321,285 | `2a9e149025a0433829e2f7b3a150807409cf2838555b33da7ae357ce70a9014e` |
| `smolvla_mlx-0.0.1-cp312-cp312-macosx_14_0_arm64.whl` | 320,279 | `4412160a2b6f613f237d881347d4a93d27e2c96d82c82bf3991f59e4efa612b1` |
| `smolvla_mlx-0.0.1-cp313-cp313-macosx_14_0_arm64.whl` | 320,320 | `af93be38cd18bb3a45f94b039cd5f52a4a410fc8665d3aeb7eed3a5686a954fd` |

Four fresh base environments pass import isolation, native-backend loading,
and offline saved-observation prediction. A fifth CPython 3.12 environment
installed `.[serve]`, reproduced the pinned protobuf descriptor, bound an
ephemeral loopback port, completed `Ready`, and shut down cleanly. Full build
and smoke details are in `DIST_MANIFEST.md`.

## Open work and safety boundary

`HUMAN_TASKS.md` has no open item; its only entry is closed. Stage T4, T5, Q,
and the document-only Stage H remain in the full-scope plan and do not block
this release milestone. Hardware-in-the-loop validation is explicitly pending.
No robot directory, vendor fork, serial port, physical hardware, credential,
or upload was accessed during Stage R.

RELEASE READY
