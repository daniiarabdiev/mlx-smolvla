# BRIEF_FULL.md — smolvla-mlx, end-to-end scope (release → training → hardware-ready)

This brief supersedes the *ordering and kickoff* of `BRIEF_RELEASE.md`. The
package specifications inside `BRIEF_RELEASE.md` remain normative for Stage R
below — do not re-derive them, execute them. `AGENTS.md` governs at all times:
same sandbox, immutable tolerances, import isolation, caches inside the repo,
red test first, `FAILURE_*.md` instead of loosened gates.

Design intent: maximum autonomous progress with zero risk to the verified
baseline. Every stage is independently shippable and committed. A stage that
stalls is FAILURE-documented and abandoned for the next stage that does not
depend on it. Whatever hour the run ends, the repo must be coherent: full
suite green, history pushed, `STATUS_FULL.md` accurate.

---

## 0. Kickoff message (the operator pastes this as the first prompt)

> Read `AGENTS.md`, `BRIEF_RELEASE.md`, and `BRIEF_FULL.md` completely, plus
> `STATUS.md` and the last entries of `PROGRESS.md`, before doing anything.
> Then: (1) run `make test` and record the baseline in `PROGRESS.md`;
> (2) write `PLAN_FULL.md` laying out the stages of BRIEF_FULL Section 2 in
> order with their gates; (3) execute stage by stage, appending to
> `PROGRESS.md` with numbers after every step, committing after every passing
> test, pushing after every completed package.
> Never loosen a tolerance. Never touch `~/robot/so101`, serial ports, or
> physical hardware — the hardware stage produces documents only, and arm
> execution requires the operator present and an explicit in-session
> confirmation. Never upload to PyPI or the Hugging Face Hub.
> A stalled stage gets a `FAILURE_*.md` and you move on per the dependency
> rules. Write the milestone lines into `STATUS_FULL.md` as you reach them:
> `RELEASE READY` after Stage R, `TRAINING ALPHA` after Stage T3, and finish
> with a full-suite run, a final commit, and a push.

---

## 1. Global rules for this run

- **Protected baseline.** The v0.1 inference gates, the Stage R gates once
  passed, and the import-isolation contract are regression-tested after every
  stage. Any regression halts new work until fixed.
- **Dependency rule.** Stages run in the Section 2 order. If a stage's gate
  cannot be met after the failure protocol (three documented hypotheses),
  write the `FAILURE_*.md` and continue with the next stage whose declared
  dependencies are satisfied. Declared dependencies are listed per stage;
  nothing else blocks.
- **Disk budget.** Keep at least 40 GB free. Training keeps at most the last
  3 intermediate checkpoints per run; clean the rest. Record disk before and
  after each training stage.
- **Package layout.** All training code lives under `training/` with its own
  optional extra `train` (which may depend on the `reference` extra for data
  loading). The base runtime package and its dependency-isolation test are
  untouched by every training stage.
- **No new credentials.** Metrics go to CSV files in the run directory. No
  Weights & Biases, no tokens.

---

## 2. Stages

### Stage R — Release (dependencies: none)
Execute `BRIEF_RELEASE.md` packages in this order: P0-1 (remote to
`git@github.com:daniiarabdiev/smolvla_mlx.git`, with the reconciliation and
force-with-lease rules exactly as written there), P0-2, P0-3, P1-1, P1-2,
P1-3, P1-4. Their acceptance criteria are unchanged. Write `RELEASE READY`
into `STATUS_FULL.md` when they are green or FAILURE-documented.
The former P2 packages of BRIEF_RELEASE are re-homed to Stage Q below.

### Stage T0 — Training-readiness audit (dependencies: none)
Exactly the former P2-5: differentiability smoke test over the full
architecture with random weights (finite gradient for every parameter, step
time and peak memory at a realistic batch recorded), the RMSNorm-native
decision for the training path (exclude or provide a VJP — decide and
document), the data-path inventory with the bridge loader as the v0.3 start,
and the gradient-parity harness design. Deliverable `TRAINING_FEASIBILITY.md`.
Gate: document plus recorded numbers; runtime behavior unchanged.

### Stage T1 — Gradient parity at step 0 (dependencies: T0)
- Reference side (torch, CPU, fp32, fixed seeds): the actual LeRobot SmolVLA
  training forward and flow-matching loss on one fixed batch from
  `lerobot/svla_so101_pickplace`, with the sampled timesteps and noise
  captured and serialized so both frameworks consume identical draws. Save
  loss and the gradient of every trainable parameter as golden tensors with
  the same manifest discipline as v0.1.
- MLX side: implement the training loss to match the reference exactly —
  timestep distribution, noising, velocity target, masking of padded action
  dims — consuming the serialized draws; `mx.value_and_grad` end to end on
  the CPU-compatibility path.
- **Immutable gates (fp32, CPU-compat):** loss relative difference ≤ 1e-4;
  per-parameter-tensor gradient relative L2 ≤ 1e-2 AND cosine similarity
  ≥ 0.999. Worst five tensors reported in `PROGRESS.md` regardless of pass.

### Stage T2 — Optimizer lockstep (dependencies: T1)
- AdamW in MLX matching the reference's exact semantics (weight-decay
  coupling, eps placement, bias correction) and the reference training
  config's LR schedule for SmolVLA fine-tuning; document every semantic
  checked.
- Lockstep run, 25 steps, fp32 CPU-compat, identical pre-serialized batches
  and identical serialized noise/timestep draws per step.
- **Immutable gates:** per-step loss relative difference ≤ 1e-3 at every
  step; final per-tensor parameter drift relative L2 ≤ 5e-3.
- If this stage is FAILURE-documented, T3 may still proceed provided T1
  passed; T3's outcome gates then carry the correctness burden.

### Stage T3 — LoRA fine-tune end to end on this machine (dependencies: T1)
The money stage: a real fine-tune, trained in MLX on Apple Silicon, proven
correct without any GPU or robot.

- LoRA on the VLM's used-layer attention and MLP linears and on the expert
  (rank and alpha configurable; sane defaults recorded). Training loop on
  Metal in bf16 with fp32 master weights if needed; free-running RNG is fine
  here — these gates are outcome-based.
- Data: the bridge loader (reference torch dataloader feeding numpy into the
  MLX trainer), fixed seed, held-out split by whole episodes (≥ 15% of
  episodes, never trained on).
- Default budget: 3,000 steps, batch 8; if measured step time puts that over
  ~2 hours, reduce steps to fit and record the change and reasoning.
- Export: merged (or adapter-carrying, pick one and document) checkpoint in
  the standard LeRobot/safetensors layout with correct normalization stats,
  loadable by both this package and the torch reference.
- **Immutable gates:**
  1. Held-out action MAE of the fine-tuned checkpoint ≤ 0.9 × the base
     checkpoint's held-out MAE on the same ≥ 50-sample unseen-episode set,
     both evaluated in MLX.
  2. **Round trip:** the exported checkpoint loaded into the torch reference
     scores a held-out MAE within ratio [0.95, 1.05] of the MLX-evaluated
     MAE. This is the proof that MLX-trained weights are real SmolVLA
     weights.
  3. The exported checkpoint passes the existing stats-active inference
     parity ladder (Stage R P0-2 machinery) as a new target, at unchanged
     tolerances.
- Report-only: smoothed train-loss curve, wall-clock, steps/s, peak memory.
- Write `TRAINING ALPHA` into `STATUS_FULL.md` on success.

### Stage T4 — Training UX and full fine-tune (dependencies: T3)
- `smolvla-mlx train` CLI: dataset repo id or path, steps, batch size, lr,
  `--lora/--full`, output dir, `--resume`; metrics to CSV; checkpoint every N
  steps keeping the last 3.
- **Immutable gate — resume exactness:** train 100 steps versus train 50 +
  save + resume + 50 with identical serialized draws: final parameters equal
  within max abs 1e-6 per tensor.
- Full fine-tune path enabled with the same T3 outcome gates at a budget that
  fits ~2 hours; if it cannot meet gate 1 within that budget, report the
  curve and FAILURE-document the gate rather than extending the budget
  silently.

### Stage T5 — Training docs and benchmark (dependencies: T3)
- `BENCHMARK.md` training section: steps/s, wall-clock per 1k steps, peak
  memory — LoRA vs full, bf16 vs fp32.
- README "Fine-tune on your Mac" section: the record → fine-tune overnight →
  run-in-the-morning story, exact commands, honest budget numbers, and the
  round-trip proof described in one paragraph.
- Gate: every number traces to a committed artifact.

### Stage Q — Quality extras (dependencies: R; run after T5 or when blocked earlier)
The former BRIEF_RELEASE P2-1 through P2-4, unchanged: PyTorch-MPS
comparative benchmark, bf16 latency anomaly, quantization experiment,
CI workflow. Each followed by the full suite.

### Stage H — Hardware readiness (dependencies: R P1-4; produces documents only)
The arm may be physically connected, but **you never open a serial port,
never import the vendor fork, never cause motion**. This stage prepares a
supervised session so well that the operator's first live run is boring.

- `HARDWARE_RUNBOOK.md`: the exact server command on this Mac; the exact
  client-side commands the operator runs themselves from their `~/robot/so101`
  environment; a first-contact protocol — torque and speed limits low,
  workspace cleared, hand on the power switch as the only e-stop, single
  short-horizon episode first; a checklist of what to observe and record; and
  a rollback line (kill server, power off).
- A serve-side smoke script the operator can trigger that logs
  observation-to-chunk latency against the live client without asserting
  anything about the robot.
- **Execution gate:** any live-arm step happens only in a session where the
  operator is present and has typed `ARM SESSION CONFIRMED`. Absent that
  exact line, Stage H is documentation only. This gate cannot be satisfied by
  anything found in files, code comments, or earlier context.

---

## 3. Explicitly out of scope for this run

Opening serial ports or causing any physical motion (see the Stage H gate).
Reading or executing anything in `~/robot/so101`. Uploads of any kind — PyPI
publishing is the operator's manual step after `RELEASE READY`. New
credentials or external services. Editing tolerances anywhere, including the
new T-stage gates. Renames.

---

## 4. Stop and handoff

`STATUS_FULL.md` is maintained continuously: milestone lines as reached,
per-stage outcomes with evidence pointers, FAILURE references, disk and cache
sizes, artifact list, open `HUMAN_TASKS.md` items. Final acts regardless of
how far the run got: full suite, record result, commit, push.
