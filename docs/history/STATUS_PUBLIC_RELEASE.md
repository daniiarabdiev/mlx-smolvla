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
| D — software verification | **Refresh in progress; publication held** | The earlier canonical artifacts passed archive, Twine, fresh-install, prediction, doctor, quantization, and serving checks. The new optional hardware package surface and client require a fresh artifact matrix and closing full-suite result before this row can return to complete. See [`DIST_MANIFEST.md`](../evidence/DIST_MANIFEST.md). |

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
  **115/115**. Final fast/full suite and artifact refresh are still pending in
  this checkpoint.
- The leader was never opened. The vendor checkout was neither installed nor
  modified; its 696-file tracked-content composite remained
  `d280efa881ab9e412cd071bbef38d8d9ec5050e484b49b5a2397df2a39bdb764`.

## Public-sharing blocker table

| Blocker | Status | Reason |
| --- | --- | --- |
| Serve untested on hardware | **Partial / blocked for motion** | The connected no-motion stage passes. Both camera views are operationally misframed, lift/elbow are outside the 10%-inset start envelope, no operator-attested low controller profile exists, and the workspace/base/power checklist is not recorded. |
| macOS / MLX floor | **Clear** | MLX 0.32.0–0.32.2 and their official macOS 14 wheel family passed the fixed software gates. |
| Claims exceed evidence | **Clear** | Public language distinguishes live no-motion RPC evidence from unperformed physical motion; no claim says the model drives the arm. |
| Operator material in tree | **Clear** | Exact device serials, private paths, telemetry, camera frames, and the bystander-containing image remain ignored and untracked. Public evidence is redacted. |
| First-page friction | **Clear, pending artifact refresh** | The README leads with requirements/install/run/serve/train paths and links the new hardware evidence; fresh-install checks will be repeated for the rebuilt artifacts. |

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
