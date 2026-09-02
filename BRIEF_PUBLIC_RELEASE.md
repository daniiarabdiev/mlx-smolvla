# BRIEF_PUBLIC_RELEASE.md — hardware validation, blockers, and public-release preparation

Amends `BRIEF_FULL.md` / `BRIEF_T3B.md`. `AGENTS.md` still governs, with the
narrow, explicit exceptions in Section 1. Immutable tolerances, import
isolation, red-test-first, no uploads. Nothing here reopens any `FAILURE_*.md`.

Goal of this run: make the repository ready to be renamed, flipped public, and
announced — with every claim backed by evidence, including "drives a real
SO-101 from a MacBook", and with a first page a hobbyist can act on in
five minutes.

---

## 0. Kickoff message (the operator pastes this as the first prompt)

> ARM SESSION CONFIRMED. The operator is present in the room, the follower arm
> and both cameras are connected, and the operator can cut power at any time.
> Read `AGENTS.md`, `BRIEF_PUBLIC_RELEASE.md`, `HARDWARE_RUNBOOK.md`,
> `STATUS_FULL.md`, and the last entries of `PROGRESS.md` before acting.
> This session runs with device access enabled, so the operating system no
> longer protects `~/robot/so101`: the Section 1 rules for that directory are
> now your responsibility — read and execute only, never write.
> Then: (1) run `make test` and record the baseline; (2) write
> `PLAN_PUBLIC_RELEASE.md`; (3) execute Stage A (hardware) first while the
> operator is present, then B, C, D in order, appending to `PROGRESS.md` with
> numbers, committing after every passing test, pushing after every stage.
> Anything only the operator can do goes to `HUMAN_TASKS.md` with exact
> commands; keep working around it. Finish with a full-suite run, a commit, a
> push, `STATUS_PUBLIC_RELEASE.md`, and the line `PUBLIC RELEASE READY` if
> every blocker in Section 6 is cleared.

---

## 1. What is different about this session

- **Device access.** The workspace-write sandbox is expected to block serial
  ports and cameras. The operator launches this session with full access
  (`codex --sandbox danger-full-access`, or the equivalent permission
  profile). Consequently: write nothing outside this repository except the
  runtime files the vendor stack itself writes (calibration cache, logs) in
  their normal locations; never install, upgrade, or edit anything in
  `~/robot/so101`; never modify its virtual environment.
- **Narrow authorization for the vendor stack.** You may read `~/robot/so101`
  to discover how the operator's `robot` CLI instantiates the follower arm and
  cameras (ports, camera indices, calibration paths, motor driver semantics),
  and you may execute scripts inside that venv. Read and execute only.
- **The leader arm is not used.** Policy serving needs the follower and the
  cameras. If a leader arm is detected on a serial port, ignore it and never
  open it.
- **Operator present, not babysitting.** The safety story is by construction
  (Section 2.4), not by supervision. Design so that a loop left running with
  nobody watching is still bounded, clamped, and self-terminating.

---

## 2. Stage A — Serve on the real SO-101 (run first)

### 2.1 Preflight checks (write `hardware/PREFLIGHT.md`)
Run each and record the result; stop at the first hard failure and write the
fix to `HUMAN_TASKS.md`:
1. Serial: enumerate `/dev/tty.*` and `/dev/cu.*`; identify the follower's
   BusLinker port by opening it with the vendor driver and reading present
   positions for all six joints. Confirm the sandbox allows it.
2. Cameras: enumerate video devices; grab one frame from each at the resolution
   the operator's teleop config uses; measure sustained capture rate over 5
   seconds for both concurrently. Note that macOS may prompt for camera
   permission on first use — if capture fails with a permission error, write
   the exact System Settings path to `HUMAN_TASKS.md`.
3. Calibration: locate the follower calibration file, print joint ranges, and
   confirm present positions lie inside them.
4. Server: start `smolvla-mlx serve` on loopback with `lerobot/smolvla_base`,
   confirm the `Ready` RPC and `mx.default_device()` is the GPU.
5. Camera-to-policy mapping: read the base checkpoint's expected image keys via
   the P0-2 loading ergonomics and decide the mapping from the operator's two
   physical cameras (wrist on gripper, fixed overhead/side 1080p) to the
   checkpoint's slots; the third slot is empty-camera padding. Record the
   mapping and the reason.

### 2.2 Discovery of the client path
The vendor fork may or may not contain LeRobot's async-inference client.
Determine which of these applies and record it in `hardware/CLIENT_DESIGN.md`:
- (a) The fork ships a compatible RobotClient speaking the 0.6.1 protocol:
  use it, configured from the operator's existing teleop settings.
- (b) It does not: write `hardware/hiwonder_so101_client.py`, a standalone
  client that runs inside the `~/robot/so101` venv, uses the fork's robot and
  camera classes for I/O only, and speaks the audited four-RPC protocol with
  `grpcio` plus the stubs you already audited in P1-4. No LeRobot import on
  the client side beyond the fork's robot class. If `grpcio` is missing from
  that venv, do not install it there: run the client from this repository's
  environment and import the fork's robot class by path only if that works
  without modification; otherwise write `HUMAN_TASKS.md` asking the operator
  to `pip install grpcio protobuf` in the fork venv and continue with
  Stage B meanwhile.

### 2.3 Graduated protocol (each step gated on the previous passing)
1. **No-motion run** (`--no-motion`): full loop — read state, capture both
   cameras, send observations, receive chunks, log — with motor writes
   suppressed. 60 seconds. Validates protocol, camera cadence, latency, and
   the clamps' decisions on real chunks with zero motion risk.
2. **Single action**: one chunk, one clamped action applied, slow move time,
   then hold. Verify the joint moved where commanded and stayed within limits.
3. **Bounded continuous**: the loop runs with all Section 2.4 limits active
   for a capped duration (default 90 seconds) or capped chunk count (default
   20), whichever first, then returns to the start pose and disables torque.
4. Optional, if 1–3 are clean: a second bounded run with the public multitask
   fine-tune from P0-2 to exercise a checkpoint with real camera keys. Motion
   remains meaningless; that is expected and stated.

### 2.4 Safety by construction (all implemented in the client, all tested)
- Joint clamps: calibrated min/max tightened by 10% on each side; any action
  outside is clipped, and the clip event is logged.
- Rate limit: maximum per-step joint delta relative to the *present* position
  read back from the servos (default 2% of joint range per 33 ms step);
  commands exceeding it are scaled down.
- Move-time floor: if the vendor driver supports a move duration parameter,
  never command faster than the floor (default 200 ms).
- Chunk sanity: reject any chunk containing NaN, wrong shape, or values outside
  the normalized range; hold position on rejection.
- Watchdog: no fresh chunk within 500 ms → hold position; three consecutive
  timeouts → end session, return to start pose, disable torque.
- Session caps: duration and chunk-count caps from 2.3, non-overridable
  without an explicit flag that prints a warning.
- Exit path: on any exception or signal, hold, then disable torque; never exit
  with torque enabled mid-motion.
- Start pose: read on entry; every session returns to it slowly on exit.
- Unit tests with a fake robot class cover every rule above; the fake must
  expose the same methods the client uses on the real class.

### 2.5 Telemetry and findings
- Use the P1-4 latency logger plus client-side timestamps: camera capture,
  observation send, chunk receive, first motor write. Report median and p95
  for observation-to-chunk and observation-to-motion, chunk rate, camera fps,
  clip and rate-limit event counts.
- Write `hardware/FIRST_CONTACT.md`: what was connected, mapping, every step's
  result, numbers, anomalies, and a plain-language verdict on whether "drives
  a real SO-101 from a MacBook" is now an evidenced claim. Link it from the
  README's limitations section.
- Video: reserve a README slot and a `docs/media/` path. Write the exact
  filename and length/size guidance (≤ 20 s, ≤ 8 MB) to `HUMAN_TASKS.md`;
  when the operator drops the file in, convert to an embeddable GIF or WebP
  if needed and wire the README.

### 2.6 Ship what you learned
- If the standalone client was written, keep it as
  `examples/bring_your_own_robot/` with a README explaining how to adapt it
  to any non-mainline robot class; mainline users are pointed at LeRobot's own
  RobotClient with the exact command.
- Acceptance: steps 2.3.1–2.3.3 completed with numbers; safety tests green;
  `FIRST_CONTACT.md` and `PREFLIGHT.md` committed; nothing in `~/robot/so101`
  changed (record a before/after hash of its tracked files).

---

## 3. Stage B — macOS / MLX compatibility floor

- Verify the claim that pinned MLX 0.32.2 requires macOS 26.2 by inspecting the
  installed dylib's minimum OS (`otool -l` / `vtool`), not by reading release
  notes.
- Identify, from PyPI wheel metadata and dylib inspection, the newest MLX
  release whose macOS minimum is 14.x or 15.x.
- In a separate venv, install that MLX with this package and run: conversion
  test, the eight deterministic goldens in strict mode, the 50-frame
  statistical gate in production fp32 and bf16, the fresh-install smoke, and
  one serve loopback. Unchanged tolerances.
- Outcome A (gates hold): widen the pin to a tested range, add a
  "verified MLX versions" table to the README and `DIST_MANIFEST.md`, and
  rebuild artifacts.
- Outcome B (gates fail or no such release exists): keep the exact pin, add an
  import-time check that fails with a one-sentence, actionable message naming
  the required macOS and MLX versions, and state the requirement in the
  README requirements box and `pyproject` metadata.
- Either way: `smolvla-mlx doctor` (Section 4.5) reports the resolved status.
- Acceptance: the outcome documented with the exact versions tested; no user
  can hit an unexplained stack trace from an unsupported macOS.

---

## 4. Stage C — Public-release preparation

### 4.1 Research first, then act
Spend a bounded hour reading current best-practice sources with the network:
GitHub's community-standards checklist, the Python Packaging User Guide's
release guidance, and the READMEs of three well-regarded MLX-ecosystem repos
(e.g., mlx-lm, mlx-vlm) plus LeRobot's. Write
`docs/dev/RELEASE_CHECKLIST.md` — a concrete checklist with a source link per
item — and then execute it. Do not copy prose from any source.

### 4.2 Repository hygiene
- Root contains only: `README.md`, `LICENSE`, `NOTICE`, `CHANGELOG.md`,
  `CONTRIBUTING.md`, `CITATION.cff`, `AGENTS.md` (the new public one, 4.4),
  `pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore`, `.github/`, the
  package, `training/`, `reference/`, `scripts/`, `tests/`, `examples/`,
  `hardware/`, `docs/`.
- Everything else moves, nothing is deleted: briefs, plans, status files,
  failure records, parity and lockstep reports, self-consistency and procedure
  docs, LoRA comparison, training feasibility/UX/benchmark docs, DIST and CI
  notes → `docs/history/` (process) or `docs/evidence/` (results), with an
  index page explaining what each is and why it exists. `ARCHITECTURE.md`,
  `REUSE_DECISIONS.md`, `BENCHMARK.md`, `HARDWARE_RUNBOOK.md` → `docs/` top
  level. Tracked evidence JSON → `docs/evidence/`. Update every relative link;
  the link check must pass.
- Operator-specific material leaves the tracked tree: the current `AGENTS.md`
  becomes `docs/history/AGENTS.operator.md` for provenance, `.codex/` is added
  to `.gitignore` and untracked, `HUMAN_TASKS.md` and `PROGRESS.md` move to
  `docs/history/` at the end of this run (keep using them until then).
- Scan the tracked tree for the operator's home path, machine identifiers,
  and any personal detail; the final audit's checks are rerun.

### 4.3 README, hobbyist-first
Target length: readable in five minutes, everything else linked.
1. One-sentence pitch and one benchmark line (MLX vs PyTorch-MPS, real-time
   multiple), with the video slot directly beneath.
2. Requirements box: macOS and MLX versions from Stage B, Python versions,
   Apple Silicon.
3. Install: `pip install smolvla-mlx` (with a note that it lands on PyPI at
   release; `pip install .` until then).
4. Run a checkpoint: three lines of Python; one CLI predict line.
5. Serve for your robot: server command on the Mac, mainline LeRobot client
   command, link to bring-your-own-robot example and `FIRST_CONTACT.md`.
6. Run your own fine-tune: standard LeRobot fine-tune anywhere, then load by
   repo id or path; the public-checkpoint parity evidence in one sentence.
7. Train on your Mac (preview): honest one-paragraph framing — lockstep-exact,
   round-trip-validated, deterministic-parity gap on MLX-trained checkpoints
   under investigation — exact commands, budget numbers.
8. Execution modes in two sentences: production (Metal, statistically
   identical) versus strict (CPU, bit-close).
9. Limitations, each one sentence with a link to its evidence.
10. Correctness methodology in one paragraph with a link to `docs/evidence/`.
11. Contributing, citation, license, acknowledgments (LeRobot, SmolVLA
    authors, MLX).
Every number in the README traces to a committed artifact. No superlatives
the evidence doesn't back.

### 4.4 Public `AGENTS.md` for adopters' coding agents
Replace the operator file with a generic one written for anyone's agent: repo
map, how to run the fast test subset and the full suite, the numerical-gate
policy (tolerances immutable; failures documented, never loosened), how to add
a checkpoint target, how to run serve and training, what must never change
(import isolation, execution-mode defaults). Add a one-line `CLAUDE.md`
pointing to it. Mention in the README that this file exists to kickstart
agent-assisted work.

### 4.5 `smolvla-mlx doctor`
A CLI command that prints: macOS version, chip, Python, MLX version and
whether Metal is the default device, package version, cache location and size,
whether the serve and train extras are installed, and the Stage B
compatibility verdict. Bug-report template asks for its output.

### 4.6 Packaging and versioning
- Version `0.1.0`; `CHANGELOG.md` with a real entry summarizing v0.1 and its
  evidence.
- `pyproject` metadata complete: description, readme, license, classifiers
  (macOS, Python versions), keywords, project URLs pointing at the final repo
  name `smolvla-mlx`.
- `CONTRIBUTING.md`: fast tests, full tests, gate policy, how to propose a new
  checkpoint target. `CITATION.cff`. Issue templates (bug with doctor output,
  feature). A short `SECURITY.md` only if the checklist calls for it.
- Tests: add a `slow` marker; `make test-fast` runs the subset under two
  minutes; CI note updated; `make test` unchanged.
- Rebuild artifacts from the final tree; refresh `DIST_MANIFEST.md`.

### 4.7 Rename readiness
The operator renames the GitHub repository to `smolvla-mlx` before or during
this run. Run `git remote set-url origin git@github.com:daniiarabdiev/smolvla-mlx.git`
once the rename is confirmed (a fetch succeeds); update every URL in the tree.
If the rename has not happened yet, write it to `HUMAN_TASKS.md` and keep
going with the new name in all text.

---

## 5. Stage D — Final verification and handoff

- Full suite, link check, personal-detail scan, fresh-install smokes for all
  artifacts, `smolvla-mlx doctor` output captured into `docs/evidence/`.
- Create and push annotated tag `v0.1.0` at the final commit.
- `HUMAN_TASKS.md` ends with the operator's manual list, each with the exact
  command: flip the repository to public; `uv publish` (PyPI token; check the
  name is free); create the GitHub Release from the tag with the CHANGELOG
  entry; optionally upload pre-converted weights to the Hub; add the video.
- Write `STATUS_PUBLIC_RELEASE.md` with per-stage outcomes and the blocker
  table from Section 6.

---

## 6. Blockers that must be cleared before the link is shared

| Blocker | Cleared when |
| --- | --- |
| Serve untested on hardware | `FIRST_CONTACT.md` shows 2.3.1–2.3.3 completed with numbers |
| macOS / MLX floor | Stage B outcome A or B fully implemented |
| Claims exceed evidence | README audit: every claim links to evidence; training marked preview |
| Operator material in tree | Scan clean; `.codex/` untracked; public `AGENTS.md` in place |
| First-page friction | README meets 4.3; `doctor` works; `test-fast` under two minutes |

`PUBLIC RELEASE READY` is written only when all five are cleared.

---

## 7. Out of scope for this run

Any training run. The op-level numerical audit and the PyTorch-trained twin
(next brief). Uploads of any kind. Loosening any tolerance. Modifying
`~/robot/so101`. Motion beyond the graduated protocol's caps without the
explicit-override flag, which is not to be used in this run.
