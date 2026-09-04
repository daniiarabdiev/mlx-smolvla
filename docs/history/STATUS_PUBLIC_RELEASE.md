# Public-release status

Date: 2026-09-04

`mlx-smolvla` is a 0.1.0 software release candidate on the canonical renamed
GitHub repository. The connected-follower read path and four bounded no-motion
RPC loops passed. Camera identity/framing and the mechanically supported inset
pose were subsequently verified; they are not unresolved failures. Physical
motion has not run because the exact low-controller-limit profile and final
physical attestation remain open. The operator is now away and the hardware is
powered off, so the next session must freshly verify the connected setup.
The version tag, uploads, visibility change, release creation, announcement,
and hardware-motion claim remain withheld.

## Stage outcomes

| Stage | Outcome | Evidence |
| --- | --- | --- |
| A — real SO-101 | **Partial: no-motion complete; motion deferred** | Four 60-second native-MLX no-motion loops completed with all six torque bits off and zero motor writes. The corrected fixed/wrist camera mapping and supported inset pose passed in the last connected session. The low-controller-limit profile, final physical attestation, single action, and bounded-continuous result remain open. Recheck camera identity and pose after reconnecting; prior confirmation is not current motion clearance. See [`FIRST_CONTACT.md`](../../hardware/FIRST_CONTACT.md) and [`HUMAN_TASKS.md`](HUMAN_TASKS.md). |
| B — macOS / MLX floor | **Complete** | Official macOS 14 arm64 wheels for MLX 0.32.0, 0.32.1, and 0.32.2 were hash- and Mach-O-verified, then passed the unchanged correctness and installed-runtime gates. See [`MLX_COMPATIBILITY.md`](../evidence/MLX_COMPATIBILITY.md). |
| C — public-release preparation | **Complete** | Canonical distribution/import/CLI/cache/GitHub identities, prior-project acknowledgment, community metadata, hobbyist-first README, agent guide, and root hygiene are committed. Hardware wording is restricted to the observed no-motion result. |
| D — software verification | **Complete; publication held** | This closeout's full suite passed 790/790; its fast lane passed 489/489 in 99.71 test seconds / 103.03 seconds wall, below two minutes. The reference repair passed all 56 trained-checkpoint fixed gates. The current sdist and three wheels from `8bb5c7e` passed seven fresh-install environments, including exact trained-weight loading. See [`TRAINED_PARITY_REPAIR.md`](../evidence/TRAINED_PARITY_REPAIR.md) and [`DIST_MANIFEST.md`](../evidence/DIST_MANIFEST.md). |

## Hardware continuation evidence

- The raw-base no-motion run processed 295 observations/chunks in 60.027 s at
  4.914 sampled camera FPS. Observation-to-chunk latency was 149.313 ms median
  and 151.139 ms p95; 281 chunks were held outside the public action domain,
  with zero timeouts and zero writes.
- The same weights plus effective SO-101 state/action statistics processed 295
  observations and 294 chunks in 60.022 s at 4.915 sampled camera FPS. Latency
  was 149.746 ms median and 152.502 ms p95; 12 chunks were held outside the
  public action domain, with zero timeouts and zero writes.
- After isolating and fixing fixed-versus-wrist UVC startup order, a third
  stats-active run processed 294 observations and 293 chunks in 60.064 s at
  4.895 sampled camera FPS. Observation-to-chunk latency was 152.080 ms median
  and 154.139 ms p95; one chunk was held invalid, with zero timeouts and zero
  writes. Both streams opened together at the requested format, but visual
  framing still failed the motion gate at that point; the purported fixed
  view was subsequently identified as the built-in Mac camera.
- The camera-identity correction established fixed index 0 and wrist index 1,
  excluding built-in index 2. Both intended UVC streams passed both startup
  orders at 640x480/30. The fourth 60-second no-motion loop recorded 293
  observations, 292 chunks, 4.876 sampled FPS, 148.907/150.512 ms
  observation-to-chunk median/p95, zero timeouts, and all six torque bits off.
  The operator confirmed the fixed workspace view and unobstructed, soft-focus
  wrist view. Numeric camera indices are session-local, not permanent IDs.
- The later mechanically supported, torque-free pose read measured lift/elbow
  at -20.396/62.989 degrees. Every joint passed the 10%-inset start envelope.
  This resolves the earlier pose finding for that session; freshly support and
  re-read the pose before any future arming attempt.
- Stale-goal hardening now preloads and verifies raw present positions as goals
  while torque is off; outward and return motion are gradual and bounded.
  These software guards have not yet been exercised with physical actuation.
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

## Latest software verification — 2026-09-04

- The reference-loader repair preserves the retained trained fp32 weights.
  All 56 fixed-limit cases passed: normalized maximum
  `0.000021457672119140625`, physical maximum `0.00042724609375`, and
  Torch/MLX held-out MAE ratio `1.00000078541285`. Original T3/T3B failures and
  their historical verdicts are unchanged; training remains a research preview.
- Clean pushed source `8bb5c7eca16062d120478956824a3ed79759f21e` produced the
  current sdist and Python 3.11/3.12/3.13 wheels. All seven isolated installed
  environments passed their checks. Exact artifact hashes and smoke details
  are in [`DIST_MANIFEST.md`](../evidence/DIST_MANIFEST.md); prior package
  bytes remain backed up rather than deleted.
- The repair's complete suite recorded **790/790 in 774.76 seconds**, and
  subsequent documentation/artifact checks recorded **47/47 in 14.38 seconds**,
  with no skip or xfail. Those are correctness-run durations, not model
  throughput benchmarks.
- The repair's intermediate fast run passed all 482 selected tests but took
  **144.09 seconds**, above the unchanged two-minute target. It is retained as
  evidence, not replaced by an earlier faster run. The fresh unmodified
  `make test-fast` recheck passed **489/489**, with 301 existing slow tests
  deselected, in **99.71 test seconds / 103.03 seconds complete-command wall**.
  It reported no skip or xfail and meets the unchanged **120-second** limit.
  No extra test filter, plugin override, marker change, or tolerance change
  was used; the fast lane now includes the seven added repair guards.
- At the `2026-09-04T07:50:04Z` preflight, CPU was **93.23% idle**, with no
  active repository training, floor, test, build, policy-server, or hardware
  client. macOS reported no recorded thermal/performance warning. Unrelated
  resident services were left untouched. No other project compute was launched
  alongside the timed run; only this documentation update was in progress.
  This establishes a passing idle recheck, not the cause of the prior slower
  observation or a promise that every future host run takes under two minutes.
  Local preflight and raw timing log are under
  `.cache/release-status-closeout-20260904-uwICdV/`.
  Fast-log SHA-256:
  `3829d0f07645ea936de21781a891a30da9681d4f82f1c201ed8228dbd0ad0850`.
- This closeout's final `make test` passed **790/790 in 721.59 test seconds /
  724.74 seconds complete-command wall**, with no skip or xfail. A separate
  preflight found no competing project jobs and 92.82% idle CPU. No other
  model, training, floor, test, or build job ran alongside the full suite;
  lightweight documentation review continued. The raw log is
  `.cache/release-status-closeout-20260904-uwICdV/full-tests.log`, SHA-256:
  `44634d517d4c053dabb7215ba49f1ef1c531dbe6bd29725d7424301a6de0ca35`.
- Final release-document, repository-hygiene, and distribution checks after
  recording the full-suite result passed **25/25 in 13.90 seconds**.
- Only documentation changed. No runtime, test selection, marker, tolerance,
  hardware configuration, or distribution bytes changed. The four existing
  package files and both original training failure records still match their
  recorded hashes. Hardware remained powered off and untouched.

## Historical verification — 2026-09-03 (superseded by the evidence above)

- Clean, pushed source `79a97e734afd49981ad09eb08d4877d82c707eea`
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
- The exact verified artifacts are mirrored under ignored `dist/`; the
  immediately preceding set is recoverable at
  `.cache/dist-pre-camera-startup-20260903`. Exact sizes and SHA-256 values are
  in the distribution manifest.
- With no trainer, floor worker, pytest, server, or hardware client active,
  `make test-fast` passed **479/479** selected tests with 291 slow tests
  deselected in **97.22 test seconds / 100.85 seconds wall**. The complete
  `make test` then passed **770/770** in **639.36 test seconds / 642.38 seconds
  wall**. Neither run reported a skip or xfail.
- `uv lock --check` resolves 122 packages, the active 100-package environment
  passes dependency checking, and `actionlint` reports only the documented
  constant-false condition that keeps hosted macOS CI disabled. Current-tree
  scans find no exact device serial, public-facing private home path, credential
  pattern, unconditional skip/xfail, or tracked file over 10 MiB; the sole
  optional-native `skipif` was exercised, so the full run skipped nothing. The
  protected original LoRA failure remains byte-identical at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
- Nothing was uploaded. No tag, release, visibility change, training run,
  floor computation, tolerance change, single action, or continuous hardware
  session occurred.

## Public-sharing blocker table

| Blocker | Status | Reason |
| --- | --- | --- |
| Serve untested on hardware | **Partial / motion deferred** | The four no-motion runs, corrected camera roles/framing, and supported inset pose passed in the recorded sessions. The exact low-controller-limit profile and physical checklist are still missing, and neither motion stage has run. Fresh camera/pose checks are required after reconnection, not because the earlier fixes failed. |
| macOS / MLX floor | **Clear** | MLX 0.32.0–0.32.2 and their official macOS 14 wheel family passed the fixed software gates. |
| Claims exceed evidence | **Clear** | Public language distinguishes live no-motion RPC evidence from unperformed physical motion; no claim says the model drives the arm. |
| Operator material in tree | **Clear** | The current tracked tree contains no exact device serial or private path. Telemetry, camera frames, and the bystander-containing image remain ignored and untracked; public evidence is redacted. |
| First-page friction | **Clear** | The README, rebuilt artifact matrix, and installed `doctor` checks are verified. The unchanged `make test-fast` target passed all 489 selected tests in 103.03 seconds complete-command wall time, below two minutes. |

## Exact next gate

The physical prerequisites and exact new statement are maintained in
[`HUMAN_TASKS.md`](HUMAN_TASKS.md). Begin a new supervised session after the
operator returns and reconnects the hardware. Visually re-identify the fixed
and wrist cameras and freshly verify the supported inset pose; the previous
passes remain valid historical evidence, not a persistent clearance. Power
off before any manual adjustment and support the torque-free arm safely.
Motion may resume only after an operator-known procedure has produced the
exact low-limit JSON profile and the workspace/base/power checklist is true.
The required new in-session phrase is:

```text
MOTION PREREQUISITES CONFIRMED: cameras framed, arm neutral, low limits profiled, workspace clear, base secure, hand on power.
```

Until single-action and bounded-continuous results are reviewed and committed,
do not tag, publish, make the repository public, create a GitHub Release,
announce hardware motion support, or share the release as complete.
