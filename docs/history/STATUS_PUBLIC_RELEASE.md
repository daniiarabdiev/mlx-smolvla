# Public-release status

Date: 2026-09-04

`mlx-smolvla` 0.1.0 now has a verified local release package on the canonical
renamed GitHub repository. The final connected-hardware session passed a fresh
60-second no-motion run, one valid guarded single action, and a two-chunk
bounded-continuous run with exact return and verified torque-off shutdown. A
separate attempt at the 20-chunk ceiling exited nonzero because several
gravity-loaded joints did not return exactly within the cleanup cap under the
temporary 10% torque profile; torque-off cleanup and the independent safe-pose
check still passed. Claims are limited to the bounded result.

The controller was observed to auto-enable all six torque bits on the first raw
goal preload. The client now checks raw controller position limits before that
write, supports explicit controller arming modes, never double-enables the
observed unit, and disables torque after any post-write failure. Single-action
mode waits through rejected holds within its 20-chunk cap and stops after its
first valid action. Focused hardware/readiness verification passes 109 tests.
The unchanged final fast lane passes 502 selected tests in 106.44 pytest
seconds / 109.48 wall
seconds, and the complete lane passes all 803 tests in 752.18 pytest seconds /
755.65 wall seconds, with no skips or expected failures. Pushed annotated tag
`v0.1.0` resolves to the verified source; its sdist and three native wheels pass
archive, Twine, macOS-floor, and seven-environment installed smoke checks. The
repository remains private and publication actions remain withheld.

## Stage outcomes

| Stage | Outcome | Evidence |
| --- | --- | --- |
| A — real SO-101 | **Bounded integration complete; sustained return limited** | A fresh 60-second no-motion run, one valid single action, and a two-chunk continuous run passed with exact return and all-six torque off. The separate 20-chunk attempt failed exact return under torque limit 100 but still disabled torque and stopped in a verified safe pose. See [`FIRST_CONTACT.md`](../../hardware/FIRST_CONTACT.md) and [`PREFLIGHT.md`](../../hardware/PREFLIGHT.md). |
| B — macOS / MLX floor | **Complete** | Official macOS 14 arm64 wheels for MLX 0.32.0, 0.32.1, and 0.32.2 were hash- and Mach-O-verified, then passed the unchanged correctness and installed-runtime gates. See [`MLX_COMPATIBILITY.md`](../evidence/MLX_COMPATIBILITY.md). |
| C — public-release preparation | **Complete** | Canonical identity and community metadata are committed. Hardware wording now states the bounded passing scope and the failed sustained-return result. |
| D — software and tagged artifacts | **Complete** | The current changes pass 109 focused, 502 fast-lane, and 803 full-suite tests. The pushed `v0.1.0` tag produced one sdist and three wheels that pass canonical archive/Twine/`minos 14.0` checks and all seven clean-install smoke environments. Exact bytes are in [`DIST_MANIFEST.md`](../evidence/DIST_MANIFEST.md). |

## Hardware continuation evidence

Final update — 2026-09-04: all physical preflight gates passed. The final
no-motion run completed 295 observations/chunks in 60.002 seconds with zero
timeouts and writes. The successful single-action run held one rejected chunk,
executed one valid one-unit action, returned exactly, and disabled torque. The
accepted two-chunk continuous stage likewise held once, executed one valid
action, returned exactly, and ended with an independent zero-drift,
profile/calibration/status, all-six torque-off check. The server was stopped.

The earlier 20-chunk continuous attempt moved gradually but failed exact return
within its 20-step cleanup cap at torque limit 100. It exited nonzero, disabled
torque, and stopped inside every inset bound with zero drift. No controller
limit was raised. This is a sustained-operation limitation, not a failure of
the accepted bounded two-chunk gate.

Earlier same-day checkpoint (historical): the operator delegated setup and client execution,
and the runbook now reflects that amendment. Two new 60-second runs used the
reviewed stats-active checkpoint; the warm repeat met the zero-timeout gate.
At 16:52 UTC, calibration matched and all six controllers reported model 777,
firmware 3.13, position mode 0, no alarms, and torque off. The elbow read
86.110 degrees, above its 77.187-degree upper inset margin. Fresh images show
the folded follower near the monitor and a cable near the gripper; physical
setup still needs correction.

Hiwonder's own pinned register map and manual support the three temporary SRAM
settings now staged: acceleration 1, speed 56 (about 4.9 degrees/s), and torque
limit 100 (10% of its scale). Other profile values, startup force 32, mode 0,
and raw present/goal positions were unchanged, and torque remained off. The
private session profile matches the exact readbacks. These settings are not
a universal safe preset or evidence of gravity-hold/task success. Re-establish
and verify them after power cycling; never raise them automatically.

The controller guard now rechecks all nine profile registers after goal preload,
requires integer position mode, and rejects excessive or changed startup force.
Focused hardware/readiness checks pass 104/104; the final fast lane passes
497/497 in 96.91 test seconds / 99.57 seconds wall. The full suite passes all
798 tests in 723.25 test seconds / 726.60 seconds wall. All 696 vendor tracked files and
32 operator-wrapper files remain unchanged. No physical action has run. The
four earlier successful results below remain historical evidence. See the
latest [`PREFLIGHT.md`](../../hardware/PREFLIGHT.md) entry.

At the operator's request, a later 17:31 UTC read-only recheck found lift/elbow
at -54.330/33.187 degrees and every joint within the inset envelope, with no
measured two-second drift. Calibration, the exact controller profile, mode 0,
startup force 32, status 0, and torque-off still match. This resolves the
16:52 pose failure. The fixed view contains the raised follower and tabletop;
the wrist now faces the operator/ceiling, so its mount needs adjustment before
the final no-motion and motion checks. No write was attempted; all handles closed.

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

## Offline continuation baseline — 2026-09-04

- Starting `main` was clean at `7fdf2fc2f35f41a776250f71c613039d0c41b6f3`;
  the canonical SSH remote returned the identical HEAD. GitHub was private and
  had no `v0.1.0` tag. The operator confirmed the hardware is not connected.
- The idle preflight at `2026-09-04T11:00:39Z` found no competing project
  compute or inherited test overrides, 92.42% idle CPU, and no recorded
  thermal/performance warning. Unmodified `make test-fast` passed **489/489**
  selected tests with 301 slow tests deselected in **101.38 test seconds /
  106.30 complete-command wall seconds**. No skip or xfail was reported; the
  unchanged 120-second gate passes.
- All four candidate artifacts match the manifest's hashes and sizes. All
  60 packaged Python source files in each wheel match the checkout. Both
  original failure records, the original T3B floor, and the retained export
  match their protected hashes. Existing artifacts remain an untagged candidate.
- Fresh logs are retained under
  `.cache/hardware-release-continuation-20260904-k9uzrei5/`. The fast-log
  SHA-256 is `00c7aa67ec0fa144b19be61d277c8390676629cb774aa5b40d499c4723b6de05`.
- A separate `2026-09-04T11:04:12Z` full-suite preflight found no competing
  project jobs or test overrides and 92.40% idle CPU. Unmodified `make test`
  passed **790/790 in 723.49 test seconds / 726.60 complete-command wall
  seconds**, with no skip or xfail. No other model, training, floor, test, or
  build job ran concurrently; documentation review and dependency-metadata
  checks continued. Full-log SHA-256:
  `6be04ec38c609210e70f9d67115da8308fe3f239a7d225da4898bc8e7e0125a2`.
- The lock resolves 122 packages. Dependency integrity passes for the
  development environment (100 packages), existing server environment (56),
  and existing client environment (65). Both separated environments retain
  LeRobot 0.6.1 and no PyAV. Raw actionlint output contains only the documented
  constant-false hosted-CI guard; the check passes when only that diagnostic
  is excluded. No environment or CI configuration was changed.
- Final release-document, repository-hygiene/link, and distribution checks
  passed **25/25 in 15.63 seconds**. The post-suite artifact and protected-file
  hashes still match; all seven changed documents have resolving local links
  and no newly added private home path or credential pattern. The focused-log
  SHA-256 is `e3f71ffd5a3b16c7cc5b2b0082c0ab5bfd1488837c4c2206d1745fc3f9e1e905`.
- The [continuation plan](PLAN_HARDWARE_RELEASE_CONTINUATION.md) preserves
  both fresh live confirmations and operator-run client authority. The release
  checklist now explicitly requires fresh tag-built artifacts and repeated
  installed checks; the current candidate must not be uploaded. No runtime,
  tests, limits, tolerances, hardware records, or distribution bytes changed.

## Prior software verification — 2026-09-04

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
| Serve untested on hardware | **Clear at bounded scope** | No-motion, one valid action, and two continuous chunks passed. The failed 20-chunk exact return remains a disclosed limitation; sustained operation is not claimed. |
| macOS / MLX floor | **Clear** | MLX 0.32.0–0.32.2 and their official macOS 14 wheel family passed the fixed software gates. |
| Claims exceed evidence | **Clear** | Public language limits hardware evidence to one valid action and two continuous chunks and preserves the failed 20-chunk return. |
| Operator material in tree | **Clear** | The current tracked tree contains no exact device serial or private path. Telemetry, camera frames, and the bystander-containing image remain ignored and untracked; public evidence is redacted. |
| First-page friction | **Clear** | The README and final artifact matrix are verified. The unchanged fast lane passed all 502 selected tests under two minutes, the full suite passed 803, and all seven tag-built clean-install environments passed. |

## Exact next gate

The technical preparation is complete. Making the repository public, uploading
to PyPI or a model Hub, creating the GitHub Release, and announcing the release
require separate explicit publication authorization. Immediately before any
authorized PyPI upload, recheck name availability and match every uploaded byte
to `DIST_MANIFEST.md`. The failed 20-chunk return must remain disclosed in all
hardware claims.
