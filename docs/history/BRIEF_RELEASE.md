# BRIEF_RELEASE.md — smolvla-mlx v0.1 → release-ready (overnight scope)

This brief is the operator's explicit authorization and specification for a
follow-on scope, as required by the v0.1 handoff report. `AGENTS.md` still
governs at all times: same sandbox, same immutable tolerances, same
import-isolation contract, caches stay inside the repo, red test first for any
new behavior. This brief extends `BRIEF.md`; it does not replace it.

The v0.1 baseline (commit `f1ae9d6`, `STATUS.md`: DEFINITION OF DONE MET) is
the protected asset. Nothing in this brief may regress it. Run the full suite
before starting and after finishing; both must pass.

---

## 0. Kickoff message (the operator pastes this as the first prompt)

> Read `AGENTS.md`, `BRIEF_RELEASE.md`, the handoff report, `STATUS.md`, and
> the last entries of `PROGRESS.md` before doing anything.
> Then: (1) run `make test` and record the baseline result in `PROGRESS.md`;
> (2) write `PLAN_RELEASE.md` ordering the work packages in Section 2 by their
> priority; (3) execute them in order, appending to `PROGRESS.md` with numbers
> after every step.
> Work packages are independently shippable: commit after each one and never
> let a lower-priority package block or destabilize a completed higher one.
> Never loosen a tolerance, never touch `~/robot/so101` or hardware, never
> upload anything to PyPI or the Hugging Face Hub. Pushing to this repo's
> `origin` is required if it exists.
> Work until all P0 and P1 packages meet their acceptance criteria (or are
> documented as blocked in a `FAILURE_*.md`), then write `STATUS_RELEASE.md`
> ending with the line `RELEASE READY` and stop. P2 packages are attempted only
> after that line is written, each followed by the full suite.

---

## 1. Why this scope

v0.1 proves parity for exactly one configuration: `lerobot/smolvla_base` with
identity-resolved normalization, two cameras, one dataset. The operator's real
use is loading **fine-tuned** SmolVLA checkpoints — which carry active
normalization statistics, their own camera keys, and their own tasks — and the
repository currently has no remote, a 65 GiB private cache, a wheel that only
installs on macOS 26 / CPython 3.12, and no license file. "Release-ready"
means: the port works for the checkpoints people will actually load, the
evidence says so, and the repository can be made public without embarrassment.

---

## 2. Work packages

### P0-1 — Remote, backup, and license
The operator created `github.com/daniiarabdiev/smolvla_mlx` (SSH remote). The
GitHub quick-setup snippet (`echo "# smolvla_mlx" >> README.md`, `git init`,
"first commit") is for empty directories — never run it in this repository; it
would pollute the existing history and README. From the canonical repo instead:

- `git remote add origin git@github.com:daniiarabdiev/smolvla_mlx.git`, or
  `git remote set-url origin …` if an `origin` already exists.
- `git fetch origin`. If `origin/main` exists with an unrelated history (a
  stray "first commit"), verify it contains nothing beyond a trivial
  auto-generated README, record that verification in `PROGRESS.md`, then
  `git push -u origin main --force-with-lease` so the canonical history wins.
  If the remote contains anything else, do not force; write `HUMAN_TASKS.md`
  and continue with other packages.
- Otherwise, plain `git push -u origin main`.
- If a stray `# smolvla_mlx` line was appended to the local `README.md` by the
  quick-setup snippet, remove it as part of the P1-3 README work.
- If the push is refused for authentication, write the exact fix to
  `HUMAN_TASKS.md` (SSH key registration or `gh auth login` + HTTPS remote)
  and retry the push after every completed package.
- Add an Apache-2.0 `LICENSE` file consistent with `NOTICE` (vendored LeRobot
  code is Apache-2.0, mlx-vlm is MIT). Reference both in `README.md`.
- Acceptance: `origin/main` mirrors the canonical local history, with the
  reconciliation evidence in `PROGRESS.md`, or a `HUMAN_TASKS.md` entry says
  precisely what the operator must do; `LICENSE` present and referenced.

### P0-2 — Checkpoint generality: active normalization (the core package)
The v0.1 identity-normalization pin is correct parity behavior for the base
checkpoint, and it stays. But fine-tuned checkpoints carry real statistics, and
the port must be proven against that case.

- **Stats-active reference target.** In `reference/`, construct a variant of
  the base policy whose normalization buffers hold the true
  `lerobot/svla_so101_pickplace` statistics, using the reference stack's own
  mechanism for attaching dataset stats at load time; if the 0.6.1 API resists,
  set the buffers by explicit state-dict surgery and document it. Save it as a
  local checkpoint under `reference/artifacts/` so it is a first-class
  conversion input.
- Extend the golden harness to produce a second golden set for this checkpoint
  (same 8 deterministic samples, same fixed noise, same manifest discipline).
- Convert it with the existing converter (which must carry normalization
  buffers), and pass the full parity ladder: preprocessing (now including real
  mean/std on state, and un-normalization on actions), end-to-end
  deterministic, and the 50-sample statistical gate — fp32 and bf16, at the
  unchanged Section 6 tolerances of `BRIEF.md`.
- **Public fine-tune, if one exists.** Time-box 90 minutes: search the Hub for
  a public SmolVLA fine-tune on SO-100/SO-101 data whose architecture matches
  the audited config and whose evaluation dataset is downloadable within about
  5 GB. If found, repeat the parity ladder against it as a third golden set,
  including whatever camera-key mapping it declares. If nothing suitable loads,
  record the search queries and candidates in `PROGRESS.md` and move on; the
  stats-active target above already exercises the machinery.
- **Three-camera behavior.** Determine what the reference actually does with
  the third camera slot (real third stream vs `empty_cameras` padding) and
  add a parity test matching that observed behavior. Document it in
  `ARCHITECTURE.md`; matching observed reference behavior is the goal, not
  inventing a nicer behavior.
- **Loading ergonomics.** `SmolVLAMLX.from_pretrained` must accept any Hub
  repo id or local checkpoint directory of matching architecture. On an
  architecture mismatch, or when the caller's observation dict is missing keys
  the checkpoint expects, it must raise an error that names the expected
  camera/state keys and shapes read from the checkpoint config. Both behaviors
  tested.
- Acceptance: new tests in the suite, green at unchanged tolerances; statistical
  ratios for the stats-active checkpoint recorded in `PROGRESS.md`;
  `ARCHITECTURE.md` updated; the v0.1 base-checkpoint tests still green;
  loading-ergonomics tests green.

### P0-3 — Cache audit and hygiene
- Account for the 65 GiB in `.cache/smolvla_mlx`: list what is stored and why.
  Keep converted weights and anything expensive to regenerate; delete
  regenerable debris. Add a `make clean-cache` target that removes only safe
  items and a short cache-layout section in `README.md`.
- Acceptance: cache size reported before and after in `PROGRESS.md`;
  `make test` fully green after cleanup with no re-downloads beyond what the
  cleanup intentionally removed.

### P1-1 — Production-path evidence on Metal
The handoff notes Metal kernels fail the strict per-module fp32 comparisons for
Vision and Connector (that stays documented in the `FAILURE_*.md` files and is
not to be relitigated by tolerance edits). What must be unambiguous is the
guarantee attached to what `select_action` actually runs.

- State explicitly, in code and `README.md`, which engine the public API uses
  by default (Metal kernels vs CPU-compatibility primitives) and how to select
  the other, if selectable.
- Record end-to-end deterministic and 50-sample statistical results **on the
  default production path**, fp32 and bf16, as their own labeled table in
  `BENCHMARK.md` — separate from the strict-parity mode results — so the README
  claim "what you install is what was measured" is literally true.
- Acceptance: the table exists with numbers; README's correctness section
  distinguishes strict-parity mode from production mode in two sentences.

### P1-2 — Packaging portability
- Make `_rmsnorm_native` optional: pure-MLX fallback when the extension is
  absent, with both paths tested (the strict-parity tests may require the
  extension; mark them to skip-with-reason when it is genuinely absent — this
  is an environment guard, not an `xfail`, and must not reduce coverage on the
  build machine).
- Set a sane `MACOSX_DEPLOYMENT_TARGET` so wheels are not tagged
  macOS-26-only; rebuild and verify the tag.
- Build the sdist plus wheels for CPython 3.11, 3.12, and 3.13 where `uv` can
  supply interpreters, routing interpreter installs inside the repo cache
  (`UV_PYTHON_INSTALL_DIR="$PWD/.cache/uv-pythons"`). Fresh-venv install smoke
  test per artifact: import, dependency-isolation check, one CLI `predict`.
- Acceptance: `dist/` contains the artifacts; each passed its smoke test; the
  build command and tags are recorded in `PROGRESS.md`.

### P1-3 — README and release polish
- README covers: one-paragraph pitch; install; 10-line API example;
  quickstart CLI; benchmark table with the real-time framing (a 50-action
  chunk is ~1.7 s of motion at 30 fps, computed in ~111 ms on this machine —
  state the multiple); correctness methodology (goldens, immutable tolerances,
  statistical gate, now including the stats-active checkpoint evidence);
  limitations (Metal strict-module caveat, audited-input contract, inference
  only); license and NOTICE pointers.
- Include a "Run your own fine-tune" section: fine-tune SmolVLA with standard
  LeRobot on any GPU, then load the resulting checkpoint here by repo id or
  path — with the exact commands.
- Keep it honest and specific; no superlatives the tables don't back.
- Acceptance: README renders cleanly and every number in it traces to
  `BENCHMARK.md` or `PROGRESS.md`.

### P1-4 — LeRobot async-inference policy server (software only)
The adoption path for real robots: a LeRobot user points mainline LeRobot's
async-inference RobotClient at a Mac running this package, and their arm is
driven by MLX inference. The server half is pure software and lands tonight;
on-robot validation is a separate operator session and is out of scope.

- Implement `smolvla-mlx serve`, speaking the exact gRPC protocol of the
  pinned reference stack's async-inference PolicyServer. The proto and
  transport code ship inside the installed `lerobot` 0.6.1 package; read them
  there, match message schemas, observation encoding, chunking and timing
  semantics exactly, and record the audit in `ARCHITECTURE.md`. Do not invent
  a nicer protocol.
- The gRPC/server dependencies must not leak into the base runtime contract:
  put them in a `serve` extra, and keep the import-isolation test green for
  the base install.
- Tests, software only: schema-conformance tests, plus a loopback integration
  test that drives the server with the reference package's own client-side
  transport machinery where importable (reference extra is allowed in tests),
  feeding recorded dataset observations and asserting the served action chunks
  equal direct `select_action` output on the same inputs. No serial ports, no
  hardware, no `~/robot/so101`.
- README: a "Serve for your robot" section showing the server command on the
  Mac and the client-side command a LeRobot user runs, with an explicit note
  that hardware-in-the-loop validation is pending.
- Acceptance: loopback test green; served-equals-direct equality demonstrated;
  docs section present; base-install import isolation unaffected.

### P2-1 — Comparative benchmark: PyTorch-MPS on the same machine
- Using the `reference` extra only (never as a golden source), measure the
  reference policy's per-chunk latency on MPS on this machine, with fallback
  enabled if an op is unsupported, and add an MLX-vs-PyTorch-MPS row to
  `BENCHMARK.md` with methodology notes.
- Acceptance: reproducible script committed; numbers and caveats recorded.

### P2-2 — bf16 latency anomaly
- bf16 (131 ms) is slower than fp32 (111 ms). Profile where the time goes;
  apply only changes that keep every gate green; record findings even if the
  outcome is "MLX 0.32.2 kernel behavior, revisit on upgrade."
- Acceptance: a written explanation with measurements; any optimization
  followed by the full suite.

### P2-3 — Quantization experiment (now explicitly authorized)
- 8-bit and 4-bit on VLM linear layers only; vision encoder and expert stay
  bf16. Rerun the statistical gate per variant; add an
  accuracy-delta / latency / memory table to `BENCHMARK.md`. Ship as an opt-in
  flag only if the statistical gate passes; otherwise report and do not ship.
- Acceptance: table present; default behavior unchanged.

### P2-4 — CI, only if `origin` is a GitHub repository
- A `macos-15` workflow running the suite with goldens regenerated in-job from
  the pinned reference. If runner limits (memory, time) make this infeasible,
  commit the workflow disabled with a comment explaining exactly why and what
  it needs.
- Acceptance: workflow file committed; its status honestly documented.

### P2-5 — Training-readiness audit (audit only, no training code)
The prospective v0.3 goal is fine-tuning SmolVLA in MLX on Apple Silicon,
LoRA first. Tonight's job is to de-risk it on paper and with one smoke test,
changing zero runtime behavior.

- Differentiability smoke test: instantiate the full architecture with random
  weights at reduced sequence length, compute the flow-matching training loss,
  take `mx.grad` end to end, and assert every parameter receives a finite
  gradient. Record step time and peak memory at a realistic batch size as the
  first throughput datapoint.
- Name every op that blocks or complicates training: in particular, decide and
  document whether the native RMSNorm extension is excluded from any future
  training path or needs a custom VJP.
- Inventory the data path: what LeRobot dataset loading needs (video decode,
  augmentation, batching), the pragmatic bridge (reference torch dataloader
  feeding numpy into an MLX trainer) versus a pure path, with the bridge as
  the recommended v0.3 starting point.
- Design, do not implement, the step-0 gradient-parity harness: golden
  gradients from the torch CPU reference on a fixed batch, with proposed
  tolerances clearly marked as proposals for the future training brief.
- Deliverable: `TRAINING_FEASIBILITY.md` plus the committed smoke-test script
  and its recorded numbers.
- Acceptance: document and numbers exist; `make test` unchanged and green.

---

## 3. Explicitly out of scope tonight

Physical hardware of any kind: `~/robot/so101`, serial ports, cameras, the
arm — including on-robot validation of the P1-4 server, which is a daytime
session with the operator. Training implementation (P2-5 is an audit, not an
implementation). Uploads of any kind — PyPI, Hugging Face, container
registries; publishing to PyPI is the operator's manual two-minute step after
`RELEASE READY`. Repository or package renames. Editing tolerances or the
existing `FAILURE_*.md` conclusions. Deleting anything outside this
repository.

---

## 4. Stop and handoff

On completing P0+P1 (or documenting blockers), write `STATUS_RELEASE.md`:
baseline result, per-package outcomes with evidence pointers, cache size
before/after, artifact list, open `HUMAN_TASKS.md` items, and the final line
`RELEASE READY`. Then attempt P2 packages in order, appending results. Final
act regardless of how far P2 got: run `make test`, record the result, commit,
push if `origin` exists.
