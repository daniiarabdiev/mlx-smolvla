# BRIEF.md — SmolVLA → MLX port, autonomous agent specification (local, Codex CLI)

`AGENTS.md` holds the always-on rules. This file is the full specification. Section numbers are referenced from `AGENTS.md`; do not renumber.

---

## 0. Kickoff message (the operator pastes this as the first prompt)

> Read `AGENTS.md` and `BRIEF.md` completely before doing anything.
> Then: (1) write `PLAN.md` with your phase-by-phase plan and the specific claims from BRIEF Section 3 you will verify against the reference code; (2) execute the phases in order; (3) after every meaningful step, append to `PROGRESS.md` with numbers.
> Do not end your turn to ask questions and do not wait for approval between phases. Never loosen a tolerance. Never touch `~/robot/so101`, serial ports, or hardware. Route all caches inside the repo. Commit after every passing test.
> Work until Definition of Done v0.1 (BRIEF Section 9) is met or a stop condition (BRIEF Section 8) triggers, then write `STATUS.md`.

---

## 1. Mission and non-goals

**Mission.** A faithful, fast, dependency-light MLX implementation of SmolVLA *inference* (`lerobot/smolvla_base`), validated numerically against LeRobot's PyTorch implementation on this machine, packaged as an installable Python library with a drop-in policy API and a benchmark table measured on this M5 Pro.

**Why it matters.** Nobody has shipped native Apple Silicon inference for this model. The value is correctness you can prove (golden tests against the reference on Metal) plus measured speed, not a weight file that "seems to work."

**Non-goals for v0.1.** Training or fine-tuning in MLX. Other VLAs (π0, π0.5, Wall-OSS, GR00T). Robot hardware integration. Quantization beyond the optional experiment in Phase 5. Any GUI.

---

## 2. Environment and rules

- Python 3.11 or 3.12 managed by `uv`, in a venv inside the repo. Pin every dependency; commit `uv.lock`.
- Runtime dependencies allowed in the `smolvla_mlx` package: `mlx`, `mlx-vlm` (or vendored modules from it), `safetensors`, `numpy`, `huggingface_hub`, `tokenizers`, `pillow`. Reference-only dependencies (`torch`, `transformers`, mainline `lerobot`) live in an optional extra named `reference` and may only be imported under `reference/`, `scripts/`, and `tests/`. A test imports `smolvla_mlx` in a subprocess and asserts `torch`, `lerobot`, and `transformers` are absent from `sys.modules`.
- Reference implementation: mainline LeRobot's SmolVLA policy. Locate the modeling, configuration, and "VLM with expert" modules in the installed version. Do not assume file paths from memory. Record the installed version and git commit in `ARCHITECTURE.md`. Pin `torch` to one version so goldens are reproducible.
- The reference runs on CPU in fp32 for goldens. Do not use MPS for goldens; it is not the reference numerics.
- Checkpoint: `lerobot/smolvla_base` from the Hub. Golden inputs: a public SO-101 dataset with at least two cameras, state, action, and a language task, e.g. `lerobot/svla_so101_pickplace`. Verify it exists and loads; if not, pick another SO-101 dataset from the Hub and record which one and why.
- Caches: `HF_HOME`, `UV_CACHE_DIR`, and the package's own weight cache all live under `.cache/` in the repo (gitignored). The sandbox blocks writes elsewhere.
- Licensing: LeRobot is Apache-2.0, mlx-vlm is MIT. Vendored code keeps its license header and is listed in `NOTICE`.
- No uploads, no pushes except to `origin`, no hardware, no credentials.

---

## 3. What you are porting — hypotheses to verify

Treat every item below as a hypothesis. Verify each against the reference code and the checkpoint config before relying on it. Record confirmations and discrepancies in `ARCHITECTURE.md`. Where the code disagrees with this section, the code wins.

- Roughly 450M parameters total. VLM backbone from the SmolVLM2-500M-Video-Instruct family: SigLIP-style vision encoder, pixel-shuffle connector producing about 64 visual tokens per image at 512×512, SmolLM2 (~360M) decoder.
- Layer skipping: only the first N decoder layers of the VLM are used (expect N = 16). The action expert cross-attends into the key/value states of those used layers.
- Action expert: about 100M parameters, hidden width 0.75× the VLM width, blocks that interleave cross-attention (into VLM K/V) and self-attention (over action tokens; expect a causal mask), a timestep embedding, action input and output projections.
- Flow matching at inference: Euler integration, expect 10 steps, chunk size 50, action dimension padded to 32, state dimension padded to 32. Pin down the timestep schedule and the sign convention of the velocity update exactly.
- Inputs: up to 3 camera images (absent cameras handled via padding and masking; look for an `empty_cameras`-style setting), a language instruction tokenized with the SmolVLM tokenizer and padded to a fixed length (expect 48), robot state projected linearly into the prefix. Determine whether state enters the VLM prefix or the expert; do not assume.
- Attention mask: prefix tokens attend bidirectionally within the prefix; action tokens attend to the prefix and, expect, causally among themselves. Reproduce the exact 2D mask construction from the reference.
- Preprocessing: images resized with padding to 512×512 (aspect ratio preserved), then normalized per the SmolVLM processor (expect mean 0.5, std 0.5). State and action normalized with statistics stored in the checkpoint; determine the normalization mode (mean/std vs min/max). Actions are un-normalized on output.

---

## 4. Method: golden tests first, then code

1. Build the reference harness before writing any MLX model code. Load the PyTorch policy on CPU in fp32, eval mode, fixed seeds. Choose at least 8 samples from the dataset across different episodes, using all cameras and the real language instruction. Run the full forward with a fixed Gaussian noise tensor for the action chunk. Save every intermediate tensor to `tests/golden/` (gitignored): preprocessed images, token ids, vision features, connector output, hidden states and K/V of each used decoder layer, state embedding, per-block expert outputs, the velocity at each Euler step, final normalized and un-normalized actions. Write `manifest.json` with shape, dtype, and hash per tensor. `make goldens` must regenerate byte-identical files.
2. Test each MLX module in isolation against those tensors with the tolerances in Section 6. Integrate only modules that pass.
3. End to end: same noise tensor in, same action chunk out, within tolerance. Then a statistical check over at least 50 samples: the un-normalized action error of MLX relative to ground-truth actions must not be worse than the PyTorch reference's error by more than the Section 6 margin. Both being equally imperfect versus ground truth is expected; MLX being worse is a bug.

Preprocessing exactness is the most common silent failure in VLM ports. Test it first, on real frames.

---

## 5. Phases and acceptance criteria

### Phase 0 — Bootstrap and reference harness
- Repo scaffold (Section 7), `pyproject.toml`, `uv.lock`, `.gitignore` (caches, goldens, weights), `Makefile` with `goldens`, `test`, `bench`.
- Confirm `mx.default_device()` reports the GPU and record MLX, macOS, and Python versions in `PROGRESS.md`.
- Download checkpoint and dataset into the in-repo cache. Confirm the PyTorch reference runs on CPU and produces an action chunk for one sample. Produce goldens via `scripts/make_goldens.py`.
- Acceptance: `make goldens` twice yields identical `manifest.json` hashes; `make test` runs a trivial passing test plus the dependency-isolation test.

### Phase 1 — Architecture audit and reuse decision
- Write `ARCHITECTURE.md`: module tree, parameter names and shapes dumped from the safetensors file, tensor shapes at every boundary, the exact attention mask, exact preprocessing, exact normalization, the flow-matching update rule with its timestep schedule.
- Inspect mlx-vlm for a SmolVLM / Idefics3 implementation. Decide per component: reuse (import), vendor (copy and modify), or reimplement. Hard requirement: the language model must run only the first N decoder layers and expose their K/V for cross-attention. If mlx-vlm cannot do that cleanly, vendor. Record every decision with rationale in `REUSE_DECISIONS.md`.
- Acceptance: both documents complete; every Section 3 hypothesis marked confirmed or corrected. This is the operator's review checkpoint; do not wait for it.

### Phase 2 — Weight conversion
- `smolvla_mlx/convert.py`: Hub safetensors → MLX-loadable safetensors under the package cache with an explicit JSON name map. Keep an fp32 master; derive bf16.
- Verify: every source tensor mapped exactly once, every target parameter initialized, shape and parameter-count equality, per-tensor checksums.
- Acceptance: `tests/test_conversion.py` passes with zero unmapped tensors in either direction.

### Phase 3 — Module-by-module port, in this order
1. Preprocessing: image resize, pad, normalize; tokenizer; state normalization. Compare against golden preprocessed arrays.
2. Vision encoder.
3. Connector and pixel shuffle.
4. Decoder layers 0..N-1 with K/V capture.
5. State projection, prefix assembly, attention mask.
6. Timestep embedding and action projections.
7. Expert blocks (cross-attention, self-attention).
8. Euler loop.
- Each module gets `tests/test_<module>.py` against goldens, parametrized over fp32 and bf16.
- Acceptance: all module tests pass at Section 6 tolerances in both dtypes.

### Phase 4 — End-to-end correctness and API
- Deterministic test (same noise). Statistical test (≥ 50 samples). Both in fp32 and bf16.
- `select_action(observation)` mirroring LeRobot's policy interface: observation dict with camera images, state, and task string; returns one action; internal action queue over `n_action_steps`; `reset()` clears it.
- Acceptance: tolerances met; API test passes on a real dataset frame.

### Phase 5 — Performance
- `scripts/bench.py` measures: latency per 50-action chunk (median and p95 over 50 runs after warmup), split into preprocessing, vision encoder, prefix decoder, expert × steps; peak memory; fp32 vs bf16. It writes `BENCHMARK.md` with the machine description (`sysctl -n machdep.cpu.brand_string`, memory, macOS version, MLX version, commit hash).
- Optimize: `mx.compile` where safe; compute the VLM prefix once per chunk and reuse it across Euler steps (expert-only per step); batch all camera images through the vision encoder in one call; keep Python out of the inner loop. Every optimization is followed by the Phase 4 tests.
- Optional, only after acceptance: 8-bit and 4-bit quantization of VLM linear layers only (`mlx.nn.quantize`), vision encoder and expert stay bf16. Rerun the Phase 4 statistical test. Report accuracy delta and speedup.
- Acceptance: `BENCHMARK.md` in the repo. Target, not gate: under 200 ms median per chunk at bf16.

### Phase 6 — Packaging
- `pip install -e .`; package `smolvla_mlx`; `SmolVLAMLX.from_pretrained("lerobot/smolvla_base")` downloads and converts on first use, caches under the package cache directory (honoring `SMOLVLA_MLX_CACHE`, defaulting to `~/.cache/smolvla_mlx` for end users), and works offline afterwards.
- CLI: `smolvla-mlx convert`, `smolvla-mlx test`, `smolvla-mlx bench`, `smolvla-mlx predict --dataset <id> --index <n>`.
- `README.md`: install, usage, benchmark table, correctness summary (which tolerances, which commit, which machine), known limitations.
- Optional, if `origin` is a GitHub repo: `.github/workflows/ci.yml` running the fp32 and bf16 suite on a `macos-15` runner, with goldens regenerated in-job from the pinned reference. Not required for Definition of Done.
- Acceptance: a fresh venv, `pip install .`, then `smolvla-mlx predict` on a dataset frame succeeds; the dependency-isolation test passes.

### Phase 7 — Deferred, do not start
- Robot integration: a LeRobot async-inference-compatible policy server and the operator's vendor-fork client. Design `select_action` so this becomes a thin wrapper.
- Training parity (v0.2): fine-tune a few hundred steps in PyTorch and in MLX from the same initialization, data order, and seed; compare loss curves and final weights within a tolerance to be defined then.

---

## 6. Tolerances — fixed, never loosened

Real port bugs (wrong mask, wrong RoPE base, wrong norm epsilon, wrong preprocessing) produce errors of order 1e-1 or larger. Framework numerics noise is around 1e-5 per layer. These thresholds sit between the two.

- Preprocessing (fp32): image max abs diff ≤ 1e-5 after normalization; token ids exact; state normalization max abs diff ≤ 1e-6.
- Per-module, fp32: relative L2 error ≤ 1e-3 and max abs diff ≤ 1e-3 against the golden for that module's output.
- End-to-end deterministic, fp32: normalized action chunk max abs diff ≤ 5e-3.
- Per-module, bf16: relative L2 ≤ 3e-2.
- End-to-end deterministic, bf16: normalized action chunk max abs diff ≤ 5e-2.
- Statistical (≥ 50 samples): un-normalized MAE versus ground truth for MLX (fp32 and bf16 separately) ≤ 1.05 × the PyTorch fp32 MAE versus ground truth.

If a tolerance cannot be met after three distinct hypotheses have been tested and documented, stop work on that module, write `FAILURE_<module>.md` with the diff analysis (where the error first appears, its magnitude, what was ruled out), and continue with modules that do not depend on it. Tightening a tolerance is allowed; loosening is not. If you believe a threshold is wrong, write the argument and evidence in `PROGRESS.md` and leave the number unchanged.

---

## 7. Repository layout

```
smolvla-mlx/
  AGENTS.md  BRIEF.md  SETUP.md  .codex/config.toml
  PLAN.md  PROGRESS.md  HUMAN_TASKS.md  ARCHITECTURE.md  REUSE_DECISIONS.md  BENCHMARK.md  STATUS.md  NOTICE  README.md
  pyproject.toml  uv.lock  Makefile  .gitignore
  smolvla_mlx/          runtime package; no torch / lerobot / transformers imports
    preprocessing.py  vision.py  connector.py  language.py  expert.py  flow.py  policy.py  convert.py  cli.py
  reference/            PyTorch harness; may import lerobot
  scripts/              make_goldens.py  bench.py
  tests/                golden/ (gitignored)  test_*.py  conftest.py
  .cache/               hf/  uv/  smolvla_mlx/   (gitignored; all caches live here)
```

---

## 8. Logging, commits, and stop conditions

- `PROGRESS.md`: timestamped entries. Each states what was done, the evidence (test names and numbers), decisions taken, open questions, and the next step. Numbers, not adjectives.
- `HUMAN_TASKS.md`: one entry per request, with status (`open`, `done`), exact commands for the operator, and what to commit back. Read it at every session start.
- Commit after every passing test; push to `origin` hourly if it exists. Message format: `phase-N: <what> (<test> passes)`.
- No mocking of model components. No skipped tests. No `xfail` without a matching `FAILURE_<module>.md`.
- Time box: if one module shows no measurable progress for about four hours of work, apply the failure protocol in Section 6.
- Stop conditions: Definition of Done met; all remaining modules blocked; a dependency cannot be installed after three approaches; a write outside the repo is unavoidable and `HUMAN_TASKS.md` is waiting on it with nothing else to do. On stop, write `STATUS.md`: current state, what passes, what does not, and the exact next steps for a human. On Definition of Done, `STATUS.md` contains the line `DEFINITION OF DONE MET`.

---

## 9. Definition of Done, v0.1

- [ ] Converted weights load; conversion test passes with zero unmapped tensors.
- [ ] All Phase 3 module tests pass at fp32 and bf16 tolerances.
- [ ] End-to-end deterministic and statistical tests pass in fp32 and bf16.
- [ ] `select_action` API works on a real dataset frame; dependency-isolation test passes; fresh-venv install smoke test passes.
- [ ] `BENCHMARK.md` measured on this machine is in the repo.
- [ ] `ARCHITECTURE.md`, `REUSE_DECISIONS.md`, `README.md`, `PROGRESS.md` complete; clean git history; `STATUS.md` says `DEFINITION OF DONE MET`.
- [ ] No secrets, no uploads, no writes outside the repo.

---

## 10. Open questions you must resolve and record

- Exact SmolVLM2 variant behind the checkpoint, and whether mlx-vlm supports that variant's weights directly.
- Where normalization statistics live in the checkpoint and which mode they use.
- Whether robot state enters the VLM prefix or the action expert.
- Exact number of used VLM layers and expert layers; how expert layers align to VLM layers for cross-attention.
- Exact attention mask, including how padded language tokens and empty cameras are masked.
- Image resolution and padding behavior in the base config; number of visual tokens per image.
- Euler schedule, number of steps, and velocity sign convention.
- Default `n_action_steps` for `select_action` and how the reference handles the action queue.
