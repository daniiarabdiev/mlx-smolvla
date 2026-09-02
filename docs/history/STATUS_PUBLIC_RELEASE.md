# Public-release status

Date: 2026-09-02

`mlx-smolvla` is a 0.1.0 software release candidate on the canonical renamed
GitHub repository. The connected-follower read path and two bounded no-motion
RPC loops now pass, but physical motion has not run. The version tag, uploads,
visibility change, release creation, announcement, and hardware-motion claim
remain withheld.

## Stage outcomes

| Stage | Outcome | Evidence |
| --- | --- | --- |
| A — real SO-101 | **Partial: no-motion complete; motion blocked** | The live `ARM SESSION CONFIRMED` gate was supplied. Follower-only calibration/state reads, two cameras, and two 60-second native-MLX no-motion loops completed with all six torque bits off and zero motor writes. Camera framing, neutral start pose, low-limit attestation, and the physical checklist block single-action and continuous modes. See [`FIRST_CONTACT.md`](../../hardware/FIRST_CONTACT.md) and [`PREFLIGHT.md`](../../hardware/PREFLIGHT.md). |
| B — macOS / MLX floor | **Complete** | Official macOS 14 arm64 wheels for MLX 0.32.0, 0.32.1, and 0.32.2 were hash- and Mach-O-verified, then passed the unchanged correctness and installed-runtime gates. See [`MLX_COMPATIBILITY.md`](../evidence/MLX_COMPATIBILITY.md). |
| C — public-release preparation | **Complete** | Canonical distribution/import/CLI/cache/GitHub identities, prior-project acknowledgment, community metadata, hobbyist-first README, agent guide, and root hygiene are committed. Hardware wording is restricted to the observed no-motion result. |
| D — software verification | **Complete; publication held** | Fresh canonical artifacts include the optional hardware surface and pass archive, Twine, native-binary, fresh-install, offline prediction, doctor, quantization, serving, and hardware-extra checks. The clean fast lane passes 479/479 and the complete suite passes 770/770. See [`DIST_MANIFEST.md`](../evidence/DIST_MANIFEST.md). |

## Hardware continuation evidence

- The raw-base no-motion run processed 295 observations/chunks in 60.027 s at
  4.914 sampled camera FPS. Observation-to-chunk latency was 149.313 ms median
  and 151.139 ms p95; 281 chunks were held outside the public action domain,
  with zero timeouts and zero writes.
- The same weights plus effective SO-101 state/action statistics processed 295
  observations and 294 chunks in 60.022 s at 4.915 sampled camera FPS. Latency
  was 149.746 ms median and 152.502 ms p95; 12 chunks were held outside the
  public action domain, with zero timeouts and zero writes.
- The repository-owned client is fail-closed around exact follower identity,
  calibration, torque-off readback, camera validity, checkpoint statistics,
  controller-limit profile, inset start pose, action shape/domain, rate limit,
  watchdog, duration/chunk caps, rollback, and torque-off shutdown.
- Fresh ignored Python 3.12 `serve` and `hardware` environments were installed
  separately. PyAV is absent from the server environment; its CLI import and
  loopback-only startup/shutdown completed without the duplicate
  AVFoundation-class warning seen in the all-extras environment.
- Focused hardware, server, import-isolation, and distribution regressions pass
  **115/115**.
- The leader was never opened. The vendor checkout was neither installed nor
  modified; its 696-file tracked-content composite remained
  `d280efa881ab9e412cd071bbef38d8d9ec5050e484b49b5a2397df2a39bdb764`.

## Closing software verification

- Clean, pushed source `9c549557f2e3a355bf5c0206e6c86fa54ad191bf`
  produced the sdist and CPython 3.11/3.12/3.13 native arm64 wheels. All four
  passed Twine; archive names and metadata are canonical; all project native
  extensions report macOS `minos 14.0`.
- Four fresh base installations outside either checkout resolved only the
  installed package, retained Torch/Transformers/LeRobot/gRPC isolation,
  reported `native-reference`, passed `doctor`, and emitted finite six-value
  actions offline. The serve-extra install passed descriptor identity,
  ephemeral loopback `Ready`, and both quantization presets. The hardware-
  extra install passed dependency integrity, installed-module origin,
  stats-active checkpoint validation, camera mapping, and example CLI help
  without opening hardware.
- The exact verified artifacts are mirrored under ignored `dist/`; the retired
  0.0.1 directory is recoverable under ignored `.cache/`. Exact sizes and
  SHA-256 values are in the distribution manifest.
- With no trainer, floor worker, pytest, server, or hardware client active,
  `make test-fast` passed **479/479** selected tests with 291 slow tests
  deselected in **104.95 test seconds / 108.16 seconds wall**. The complete
  `make test` then passed **770/770** in **703.76 test seconds / 707.22 seconds
  wall**. Neither run reported a skip or xfail.
- `uv lock --check` resolves 122 packages, the active 100-package environment
  passes dependency checking, and `actionlint` reports only the documented
  constant-false condition that keeps hosted macOS CI disabled. Current-tree
  scans find no exact device serial, private home path, credential pattern,
  explicit skip/xfail, or tracked file over 10 MiB. The protected original
  LoRA failure remains byte-identical at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
- Nothing was uploaded. No tag, release, visibility change, training run,
  floor computation, tolerance change, single action, or continuous hardware
  session occurred.

## Public-sharing blocker table

| Blocker | Status | Reason |
| --- | --- | --- |
| Serve untested on hardware | **Partial / blocked for motion** | The connected no-motion stage passes. Both camera views are operationally misframed, lift/elbow are outside the 10%-inset start envelope, no operator-attested low controller profile exists, and the workspace/base/power checklist is not recorded. |
| macOS / MLX floor | **Clear** | MLX 0.32.0–0.32.2 and their official macOS 14 wheel family passed the fixed software gates. |
| Claims exceed evidence | **Clear** | Public language distinguishes live no-motion RPC evidence from unperformed physical motion; no claim says the model drives the arm. |
| Operator material in tree | **Clear** | The current tracked tree contains no exact device serial or private path. Telemetry, camera frames, and the bystander-containing image remain ignored and untracked; public evidence is redacted. |
| First-page friction | **Clear** | The README leads with requirements/install/run/serve/train paths and links the hardware evidence; the rebuilt artifact matrix and installed `doctor` checks pass. |

## Exact next gate

The physical prerequisites and exact new statement are maintained in
[`HUMAN_TASKS.md`](HUMAN_TASKS.md). Power the follower off before manually
reframing cameras or moving the arm. Motion may resume only after both cameras
show the complete workspace, lift/elbow are manually inside their inset ranges,
an operator-known procedure has produced an exact low-limit JSON profile, and
the workspace/base/power checklist is true. The required new in-session phrase
is:

```text
MOTION PREREQUISITES CONFIRMED: cameras framed, arm neutral, low limits profiled, workspace clear, base secure, hand on power.
```

Until single-action and bounded-continuous results are reviewed and committed,
do not tag, publish, make the repository public, create a GitHub Release,
announce hardware motion support, or share the release as complete.
