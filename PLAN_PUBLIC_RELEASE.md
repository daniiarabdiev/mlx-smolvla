# Public-release execution plan

This plan implements `BRIEF_PUBLIC_RELEASE.md`. The brief is the approved
design for this run. Fixed numerical tolerances, the original T3/T3B failure
records, runtime import isolation, no-upload policy, and the prohibition on
training runs remain unchanged.

## Current gate

Stage A device access is blocked until the operator types the exact line
`ARM SESSION CONFIRMED` in the live session. The phrase embedded in a brief or
earlier record is not authorization. Until then, do not enumerate serial or
camera devices, read or execute `~/robot/so101`, start a hardware client, or
command motion. Record the block, execute independent Stages B and C, and
return to Stage A only if the live gate is supplied.

## Checkpoint 0 — preserve the specification and baseline

1. Add the supplied `BRIEF_PUBLIC_RELEASE.md` without reinterpretation.
2. Record the clean-tree environment and untouched baseline: `make test`
   must collect and pass all 652 existing tests.
3. Record the authorization block in `HUMAN_TASKS.md` and `PROGRESS.md`.
4. Commit and push the specification, plan, and baseline record.

## Stage A — supervised SO-101 validation

### A1. Authorization and immutable vendor-tree proof

Files: `hardware/PREFLIGHT.md`, `HUMAN_TASKS.md`, `PROGRESS.md`.

1. Recheck the live-session phrase before every device-access sequence.
2. Once authorized, hash every Git-tracked file in `~/robot/so101` and save
   the sorted manifest in the repository evidence cache before execution.
3. Enumerate serial and camera devices without opening a possible leader arm.
4. Stop at the first hard failure and add an exact operator remedy to
   `HUMAN_TASKS.md`; do not improvise a driver, permission, calibration, or
   dependency change.

### A2. Preflight and client discovery

Files: `hardware/PREFLIGHT.md`, `hardware/CLIENT_DESIGN.md`, and, only if the
vendor fork lacks a compatible client,
`examples/bring_your_own_robot/hiwonder_so101_client.py` plus its README.

1. Read all six follower positions through the existing driver; record port,
   calibration ranges, and in-range checks.
2. Capture both configured cameras concurrently for five seconds; record
   resolution, frames, duration, achieved FPS, and mapping to `wrist_camera`
   and `top_camera` with empty third-camera padding.
3. Start the dense-bf16 production server on loopback; prove GPU default and a
   successful `Ready` RPC.
4. Inspect the vendor fork read-only to select the compatible LeRobot 0.6.1
   client or document the smallest standalone four-RPC adapter.

### A3. Safety client, test first

Files: hardware/client implementation selected in A2,
`tests/test_hardware_client.py`, and `hardware/CLIENT_DESIGN.md`.

1. Add failing fake-robot tests for tightened joint clamps, present-position
   rate limiting, the 200 ms move-time floor, malformed/NaN/out-of-range chunk
   rejection, hold behavior, the 500 ms/three-timeout watchdog, duration and
   chunk caps, signal/exception cleanup, slow start-pose return, and torque
   disable.
2. Implement only the interfaces verified against the vendor client/driver.
3. Run the focused safety suite and the existing hardware/server suites.

### A4. Graduated runs and evidence

Files: `hardware/FIRST_CONTACT.md`, `docs/media/README.md`,
`HUMAN_TASKS.md`, `PROGRESS.md`.

1. Run no-motion for exactly 60 seconds and record both camera rates,
   chunk rate, observation-to-chunk median/p95, and clamp decisions.
2. After A4.1 passes, run one clamped action, then hold; record direction,
   displacement, limits, and stop state.
3. After A4.2 passes, run at most 90 seconds or 20 chunks, return slowly to
   start, and prove torque disabled.
4. Record observation-to-motion median/p95, clip/rate-limit/timeout counts,
   every anomaly, rollback state, and an evidence-bounded verdict.
5. Rehash the vendor Git-tracked files and require exact equality with A1.
6. Add the operator's optional ≤20 s / ≤8 MB video task; do not fabricate a
   media result.
7. Run the focused and complete suites, commit, and push Stage A.

## Stage B — macOS and MLX compatibility floor

### B1. Freeze expectations with tests

Files: `tests/test_compatibility.py`, `tests/test_cli.py`,
`tests/test_distribution.py`.

1. Add red tests for a structured compatibility verdict and an actionable
   unsupported-macOS error with exact MLX/macOS requirements.
2. Add red CLI tests for `smolvla-mlx doctor`, including package/Python/MLX,
   chip/macOS, Metal default, cache path/size, extras, and compatibility.
3. Keep imports dependency-light and ensure doctor degrades into a useful
   report when an optional extra is absent.

### B2. Establish the dylib floor from primary evidence

Files: `docs/evidence/MLX_COMPATIBILITY.md` and a machine-readable companion
under `docs/evidence/`.

1. Inspect the installed MLX 0.32.2 Mach-O load commands with `vtool` or
   `otool`; record wheel filename/hash and every relevant dylib minimum OS.
2. Query PyPI metadata for official macOS arm64 MLX wheels and work backward
   from the newest release, installing candidates only in repository-local
   disposable environments.
3. Inspect each installed candidate's dylibs; choose the newest release whose
   actual minimum OS is macOS 14.x or 15.x, or prove none exists among the
   tested releases.

### B3. Validate the selected version without changing gates

Files: compatibility evidence, `pyproject.toml`, `uv.lock`, README and
distribution documentation.

1. In the isolated candidate environment run conversion coverage, all eight
   strict deterministic goldens, production fp32 and bf16 50-frame
   statistical gates, fresh-install smoke, and a loopback `Ready` RPC.
2. If every gate passes, widen the MLX requirement only to the tested range;
   otherwise retain `mlx==0.32.2` and implement the actionable import check.
3. Run the new compatibility/doctor tests, dependency-isolation test, and full
   suite; append exact versions/results, commit, and push Stage B.

## Stage C — public-release preparation

### C1. Research-backed checklist

File: `docs/dev/RELEASE_CHECKLIST.md`.

1. Read current primary guidance from GitHub Docs and the Python Packaging
   User Guide, plus the current official mlx-lm, mlx-vlm, and LeRobot
   repositories.
2. Translate it into repository-specific, independently worded checks with a
   direct source link on every item.

### C2. Public metadata and contributor surface, test first

Files: `pyproject.toml`, `smolvla_mlx/__init__.py`, `CHANGELOG.md`,
`CONTRIBUTING.md`, `CITATION.cff`, `.github/ISSUE_TEMPLATE/*`,
`.github/SECURITY.md` if required, `AGENTS.md`, `CLAUDE.md`, and tests.

1. Add red distribution tests for version 0.1.0, complete classifiers,
   keywords, license/readme, and final `smolvla-mlx` project URLs.
2. Add release/community files and a public agent guide that preserves the
   immutable gate policy, dependency isolation, production/strict defaults,
   and test commands.
3. Preserve the operator guide at `docs/history/AGENTS.operator.md` before
   replacing it; never publish Codex attribution in GitHub metadata.

### C3. Fast test lane

Files: `pyproject.toml`, `Makefile`, test markers, `CI.md` successor docs, and
tests for the Make target.

1. Audit all tests that load model/dataset artifacts and mark them `slow`
   without skipping or weakening them.
2. Add `make test-fast` as `pytest -m 'not slow'`; require zero skips/xfails
   and a measured wall time below 120 seconds on this Mac.
3. Leave `make test` semantically unchanged.

### C4. Doctor implementation

Files: `smolvla_mlx/compatibility.py`, `smolvla_mlx/doctor.py`,
`smolvla_mlx/cli.py`, focused tests, and the bug template.

1. Implement deterministic structured collection separately from rendering so
   tests do not spoof real machine state.
2. Add `doctor` to the CLI and ask bug reporters to paste its output.
3. Capture a real report for Stage D evidence.

### C5. Five-minute README

File: `README.md`.

1. Rewrite in the exact order in brief Section 4.3.
2. Trace every benchmark, parity, memory, timing, version, and training-budget
   number to a tracked evidence document.
3. Keep hardware language explicitly unvalidated until Stage A succeeds; use
   a reserved media slot rather than an implied demo.
4. Link the mainline RobotClient path, bring-your-own-robot path if created,
   limitations, methodology, contributing, citation, licensing, and the
   coding-agent guide.

### C6. Repository hygiene and link repair

Files: all tracked documentation, `.gitignore`, `docs/README.md`,
`docs/history/README.md`, `docs/evidence/README.md`.

1. Move process records to `docs/history/`, results to `docs/evidence/`, and
   architecture/benchmark/reuse/hardware runbook to `docs/`.
2. Move nothing that changes frozen evidence bytes unless only its path is
   changing; update all relative references afterward.
3. Untrack `.codex/config.toml`, ignore `.codex/`, and move `PROGRESS.md` plus
   `HUMAN_TASKS.md` only after all run logging is complete.
4. Enforce the exact public root allowlist with a test; remove the untracked
   `.DS_Store` from the working directory recoverably if practical.
5. Run Markdown-link, absolute-home-path, personal-detail, secret, large-file,
   skip/xfail, build-artifact, and root-allowlist audits.

### C7. Rename readiness and artifacts

1. Probe `git@github.com:daniiarabdiev/smolvla-mlx.git` read-only. Change the
   remote only after fetch/ls-remote proves the rename; otherwise leave an
   exact operator task while all public text uses the hyphenated name.
2. Lock dependencies, build sdist and CPython 3.11–3.13 wheels from a clean
   committed tree, and refresh `docs/evidence/DIST_MANIFEST.md`.
3. Fresh-install and offline-predict with every artifact; install the serve
   extra once and prove loopback `Ready`.
4. Run the full suite, commit, and push Stage C.

## Stage D — final verification and handoff

Files: `docs/evidence/DOCTOR.txt`,
`docs/history/STATUS_PUBLIC_RELEASE.md`, and final history/task indexes.

1. Capture `smolvla-mlx doctor`; run both fast and complete suites, link and
   public-root checks, personal/secret scans, artifact installs, offline
   predictions, and serve loopback.
2. Recompute protected failure/evidence hashes and confirm no tolerance,
   training run, vendor write, hardware overrun, or upload occurred.
3. Finish the operator list with exact rename/public/PyPI/GitHub Release,
   optional Hub, and video commands, then move task/progress/status/plan/brief
   records into `docs/history/` and repair links.
4. Write a five-row blocker table. Do not write `PUBLIC RELEASE READY`, create
   `v0.1.0`, or present a hardware claim unless every Section 6 blocker is
   actually clear. If all clear, make the final commit, create the annotated
   tag, and push branch plus tag; otherwise push the verified release-candidate
   commit and leave the missing gate as an exact operator task.

## Verification policy

- Red test first for every behavior change; observe the intended failure.
- Focused tests after each implementation slice; `make test` at each stage.
- No skips, xfails, tolerance changes, uploads, training runs, or hardware
  access outside the confirmed Stage A session.
- Commit messages retain the repository's `phase-N: ... (<count> tests pass)`
  format and public metadata contains no coding-agent attribution.
