# Full-Scope Status

FULL-SCOPE SOFTWARE COMPLETE — FINAL AUDIT PASSED

FINAL VERIFICATION COMPLETE — 652/652 TESTS PASS

HARDWARE VALIDATION NOT RUN — SUPERVISED OPERATOR SESSION REQUIRED

T3B-1 COMPLETE — SELF-CONSISTENCY FLOOR RECORDED

T3B-2 COMPLETE — PROSPECTIVE PARITY PROCEDURE FROZEN

T3B-3A PRE-LAUNCH VERIFIED — REAL UPDATE-1 GATE PASSED

TRAINING ALPHA (STATISTICAL)

T3B-3 COMPLETE — FIXED GATES PASSED; DERIVED DETERMINISTIC GATE DOCUMENTED

STAGE R P1-3 COMPLETE — RELEASE DOCUMENTATION AND GPU HANDOFF VERIFIED

STAGE R P1-1 COMPLETE — DEFAULT PRODUCTION METAL EVIDENCE RECORDED

STAGE R P1-4 COMPLETE — LEROBOT ASYNC SERVER VERIFIED IN LOOPBACK

RELEASE READY

T4 COMPLETE — NATIVE TRAINING UX AND EXACT RESUME VERIFIED

T5 COMPLETE — IDLE NATIVE TRAINING BENCHMARK RECORDED

STAGE Q P2-1 COMPLETE — PYTORCH-MPS COMPARISON RECORDED

STAGE Q P2-2 COMPLETE — BF16 SLOWDOWN LOCALIZED; DEFAULT UNCHANGED

STAGE Q P2-3 COMPLETE — VLM 8-BIT/4-BIT OPT-INS SHIPPED; DEFAULT UNCHANGED

STAGE Q P2-4 COMPLETE — HONESTLY DISABLED MACOS-15 WORKFLOW COMMITTED

STAGE Q COMPLETE

STAGE H COMPLETE — DOCUMENTS/SOFTWARE ONLY; HARDWARE VALIDATION NOT RUN

The protected SmolVLA MLX v0.1 inference baseline is intact. At full-scope
kickoff on 2026-08-31, `make test` passed **179/179** in **158.71 seconds** on
the M5 Pro. The repository had **553 GiB** free, above the mandatory 40 GiB
floor.

Stage T0 added 14 training/readiness regression cases. The exact post-T0 tree
passed **193/193 tests in 158.14 seconds**. `TRAINING_FEASIBILITY.md` records
the complete measured artifact and training-path decision.

Stage T1 now proves the real checkpoint-backed step-zero derivative. The fixed
PyTorch and MLX losses differ by **9.074298999059449e-7**; all **155/155**
gradient tensors pass, with worst relative L2 **8.673578115837066e-6** and
minimum cosine **0.9999999999623879**. See `GRADIENT_PARITY.md`.

Stage T2 now proves 25 cumulative optimizer updates. All **25/25** losses pass
with maximum relative difference **1.3529624562582406e-6**, and all **155/155**
final selected tensors pass with maximum relative-L2 drift
**2.8499913470883435e-8**. See `OPTIMIZER_LOCKSTEP.md`.

The exact post-T1 source and documentation tree passed **224/224 tests in
172.58 seconds**. The dependency lock resolves all **103 packages**, import
isolation is green, and **543 GiB** remains free.

The exact post-T2 source and documentation tree passed **236/236 tests in
189.07 seconds**. The dependency lock still resolves all **103 packages**,
import isolation remains green, and **542 GiB** remains free.

The checkpoint-hardened Stage T3 tree passes **278/278 tests in 215.81
seconds**. A full 458-trainable-tensor rebuild/restore proof reproduced update
2 exactly, including loss, gradient norm, model bytes, and optimizer bytes.
The dependency lock still resolves all **103 packages**, and **580 GiB**
remains free.

The fixed Stage T3 run completed all **3,000** Metal updates in
**4,956.693033957996 seconds** and exported all **500** fp32 tensors. Held-out
MLX MAE improved from **4.639846293521779** to **2.1164464077779224**, and the
Torch/MLX MAE ratio was **0.9927553940871358**; both gates passed. The unchanged
all-56-case stats-active parity gate failed at raw physical max absolute
**6.632053375244141** (normalized max **0.17762404680252075**) versus **0.005**. See
`FAILURE_LORA_FINETUNE.md`. `TRAINING ALPHA` was not reached.

The outcome-evidence-hardened tree passes **308/308 tests in 205.10 seconds**.
The current repository has **573 GiB** free, above the
mandatory 40 GiB floor.

The operator supplied the normative release specification and the T3B
amendment, closing the prior stop condition. The protected kickoff baseline
passed **308/308 tests in 226.41 seconds**. T3B-1 measured the failed T3
checkpoint's PyTorch self-consistency envelope with nine fresh workers on all
56 frozen cases:
`F = 0.00003549918286283038`, with `F64 = 0.00003549918286283038`. This result is
diagnostic only and does not alter the original T3 failure. See
`SELF_CONSISTENCY_T3.md`. The final v3 T3B-1 tree passes **347/347 tests in
244.83 seconds**; its floor-focused suite passes **39/39**.

T3B-2 froze the trained-checkpoint decision before a T3B checkpoint or MLX
comparison existed. It retains every fixed tolerance and derives normalized
action parity only as `max(0.005, 3 * F(C))`. The file evaluator reconstructs
all nine raw floor workers, enforces the real-clock floor/marker/comparison
order, rehashes every floor-bound input and complete export/evaluation tree,
validates the canonical fp32 conversion from private descriptor snapshots, and
installs one non-overwriting result. The focused suite passes **52/52**, the
combined floor/distribution/evaluator suite passes **97/97**, and the complete
repository passes **402/402 tests in 230.01 seconds**. Independent review found
no remaining substantive issue. See `PARITY_PROCEDURE_TRAINED.md`.

T3B-3a's pre-launch implementation is now verified. The explicit expert-only
scope contains 112 adapters / 224 fp32 tensors / 1,708,032 scalars; the legacy
T3 scope remains unchanged. The launcher enters the repository interpreter
with `-I -S`, provenance is frozen only after a real disposable video decode
and preprocessing pass, and every retained checkpoint identity is bound across
run-state publication. The first real disposable run exposed and safely
rejected a late PyAV import before update 1. After the test-first fix, a fresh
isolated run completed 29 updates, atomically bound the step-1 checkpoint, and
was intentionally interrupted. Independent review validated its PID, run,
metrics, pointer, metadata, model, and optimizer evidence and found no source
blocker. The focused exact-byte suite passes **142/142 in 235.70 seconds**.
With no training or floor process active at `2026-09-01T22:44:56Z`, the full
repository passed **536/536 tests in 490.35 seconds**. Those exact bytes were
committed and pushed as `75b5361`. The canonical launch file is now frozen at
SHA-256 `95f76513...bea81` / configuration `fe8a937e...748fb3`.

The canonical T3B run subsequently completed all **3,000** updates without
resume or recovery and exported **500** fp32 tensors. Expert-only LoRA uses
112 adapters / 224 tensors / 1,708,032 scalars. Held-out MLX MAE improved from
**4.639846293521779** to **2.2550044155546596**, and the Torch/MLX MAE ratio was
**0.9999447391574267**; every fixed gate passed. The prospective PyTorch-only
floor was written and hashed before comparison (`F =
0.00002467632293701172`), making the derived threshold **0.005**. The bound
all-56-case comparison measured normalized max absolute
**0.013038858771324158**, so only the derived deterministic gate failed. Per
`BRIEF_T3B.md`, the milestone is `TRAINING ALPHA (STATISTICAL)`. See
`LORA_SCOPE_COMPARISON.md` and `FAILURE_LORA_FINETUNE_B.md`.
The post-verdict focused suite passes **132/132 in 17.89 seconds**, and the
complete repository passes **544/544 in 482.16 seconds**.

Stage R P0-2 is complete. The base checkpoint retains effective identity
normalization; a pinned base-plus-real-dataset-stats variant and a pinned public
multitask fine-tune both pass all eight deterministic cases and the 50-frame
statistical gate in fp32 and bf16 at unchanged tolerances. The public target's
MLX/reference MAE ratios are `1.0000005749451057` / `0.996050278176297`; its
checkpoint-derived camera keys are `wrist_camera` and `top_camera`. The exact
P0-2 tree passes **567/567 tests in 519.76 seconds**.

Stage R P0-3 is complete. The frozen cleanup boundary removed only 23
top-level `debug-*` experiment trees and exact `benchmark-debug`, reducing
`.cache/smolvla_mlx` from **91,447,880 KiB** to **52,764,872 KiB** (36.89 GiB
of allocated space). Retained model, training, and golden fingerprints stayed
stable; the Hugging Face cache kept the same size and changed only six
zero-byte lock timestamps. The post-cleanup tree passes **576/576 tests in
522.53 seconds**, and the follow-up dry run reports no candidates.

Stage R P1-2 is complete. The base distribution now supports CPython
3.11–3.13, while the optional LeRobot reference lane is guarded to 3.12+.
Exact native CPU primitives and a true extension-free pure-MLX fallback are
both tested. The sdist and three `macosx_14_0_arm64` wheels each pass a fresh
install, dependency-isolation check, native-backend import, and offline real
saved-observation prediction. The pre-artifact source tree passes **584/584
tests in 523.20 seconds**. `DIST_MANIFEST.md` records all hashes and the honest
limitation that pinned MLX 0.32.2's own dylib declares `minos 26.2`.

Stage R P1-3 is complete. The release README's ten-line API example executes
offline, all native CLI forms parse, real conversion/prediction and the idle
50-run benchmark command succeed, and the pinned LeRobot 0.6.1 GPU training
flags are accepted with both upload paths disabled. Performance framing,
active-statistics correctness, strict CPU versus default Metal behavior,
audited inputs, cache safety, troubleshooting, and license attribution all
trace to recorded evidence. The exact P1-3 tree passes **584/584 tests in
529.75 seconds**.

Stage R P1-1 is complete. The public default is now explicit
`production`/Metal, while `execution_mode="strict"` and CLI
`--execution-mode strict` own the CPU compatibility path. On the same eight
goldens, production fp32 records normalized max **0.04730653762817383** versus
the unchanged **0.005** deterministic gate (fail), while bf16 records
**0.044106483459472656** versus **0.05** (pass). Production fp32/bf16
50-frame MAE ratios are **1.0000127999805857** /
**1.0000216963394593**, both below **1.05**. Idle production timing is
110.54/130.44 ms median and 2.94/2.44 GiB peak for fp32/bf16. The exact P1-1
tree passes **593/593 tests in 530.24 seconds**; no tolerance changed.

Stage R P1-4 is complete in software. `smolvla-mlx serve` speaks the pinned
LeRobot 0.6.1 four-RPC protobuf/pickle protocol behind a Python-3.12+ optional
extra, defaults to a trusted loopback bind, serializes MLX inference, and
preserves newest-observation and timed-action semantics. The reference client
transport drove a recorded real observation through localhost; its `[3, 6]`
chunk was exactly equal to direct `select_action` output (max difference
**0.0**, SHA-256 `46a4b280...7981`). Validation, error propagation,
cancellation, concurrency, base import isolation, and lifecycle/security
boundaries are tested. The pre-artifact-refresh tree passes **601/601 tests in
521.24 seconds**; hardware validation remains pending by design. Final artifacts
were then rebuilt from pushed source `a50cd3b`: four fresh base installs pass
offline prediction and a fifth serve-extra install passes an ephemeral
loopback `Ready` RPC. The exact Stage R closing tree passes **601/601 tests in
537.54 seconds**.

Stage T4 is complete. The public `smolvla-mlx train` surface accepts a dataset
repo ID or path and requires explicit LoRA or full mode. Full mode matches the
reference state-projection/action-expert freeze policy with 155 fp32 master
tensors / 99,880,992 scalars; expert-only LoRA exposes 224 fp32 tensors /
1,708,032 scalars. Both optimizers have exact two-moment coverage. Real
100-update direct and step-50-resumed trajectories both decrease their
first-ten to last-ten mean loss, retain only checkpoints 50/75/100, export all
500 tensors, reload, and emit finite actions. The evaluator records zero
parameter, loss, and all-metric drift with exact optimizer, draw-chain,
sampler, and canonical step-state identity for both modes. See
`TRAINING_UX.md`. The exact T4 tree passes **608/608 tests in 533.65 seconds**.

Stage T5 is complete. A clean protocol commit froze a four-cell Metal matrix:
expert-only LoRA and the full reference trainable set, each with bf16 or fp32
base storage, effective batch eight, three excluded warmups, and ten measured
updates. Median update rates are **0.873/0.914 steps/s** for LoRA bf16/fp32 and
**0.836/0.857 steps/s** for full bf16/fp32. Peak MLX memory is **2.27/3.24
GiB** and **3.55/4.32 GiB**, respectively. Every published value and projection
recomputes from `TRAINING_BENCHMARK.json`; its ignored full source artifact has
SHA-256 `7112806471e55e55d98ae101bc2af8172c2cc18f01b3e0c2c0646446adba9423`.
The exact T5 tree passes **613/613 tests in 520.69 seconds**.

Stage Q P2-1 is complete. With the same pinned checkpoint, saved observation,
saved noise, fp32 dtype, preprocessing-to-normalized-chunk boundary, 5 excluded
warmups, and 50 synchronized samples, native MLX measured **110.751 ms** median
and **9.029 chunks/s** versus PyTorch-MPS at **204.579 ms** and **4.888
chunks/s**. MLX is **1.847×** faster for this bounded case. MPS fallback was
enabled before Torch import; whether any individual operation used it was not
instrumented. Framework memory counters are reported with an explicit
non-equivalence caveat. `INFERENCE_COMPARISON.json` retains all raw timings and
has SHA-256 `115ad58c0c618b65a6275018614f3ee6cf17dd02a9d4ad9c94aaf7e5a9842e48`.
The exact P2-1 tree passes **620/620 tests in 522.71 seconds**.

Stage Q P2-2 is complete. The isolated six-boundary profile reproduces bf16 at
**130.217 ms** versus fp32 at **110.714 ms**, a **19.504 ms / 17.62%**
slowdown. The ten-step expert loop explains **74.44%** of that delta, prefix
prefill **14.92%**, the vision encoder **8.78%**, the connector **1.66%**, and
preprocessing **0.07%**. Dtype inspection confirms compact bf16 weights execute
with fp32 boundary activations and outputs; the result is consistent with
mixed-dtype conversion/kernel cost in MLX 0.32.2 but does not identify a
private Metal kernel. No arithmetic or default behavior changed. All 600 raw
durations are in `BF16_PROFILE.json`, SHA-256
`74da9f937cb8bfeba4066d5518187490ff96a1447e4a2ad2253e2493245be1cf`.
The exact P2-2 tree passes **627/627 tests in 533.26 seconds**.

Stage Q P2-3 is complete. The fixed 50-case gate passes for both VLM-only
presets: 8-bit has a **0.9999625007** PyTorch-fp32 MAE ratio and 4-bit has a
**1.0011790017** ratio, each within the unchanged `<=1.05` limit. Dense bf16,
8-bit, and 4-bit median chunk latency is **132.832/132.370/132.347 ms**;
reported peak MLX memory is **2.437/2.337/2.237 GiB**. The two presets quantize
exactly 114 connector/language linears and keep vision, state projection,
token embedding, and expert dense bf16. They ship only through explicit
`vlm-8bit`/`vlm-4bit` API and CLI flags; dense bf16 remains the default.
`QUANTIZATION_EXPERIMENT.json` retains every raw measurement and error record
at SHA-256
`40060b0eaa63efee471ce2966f8fd578ade6ba2e8d9923435e14ef2466be393b`.
The first closing run exposed a server cancellation race; a deterministic
regression now proves a canceled waiter restores an observation consumed
during cancellation. The corrected exact P2-3 tree passes **645/645 tests in
544.29 seconds**.

Stage Q P2-4 is complete. `origin` is a private GitHub repository, but the
current standard arm64 `macos-15` runner has only **7 GB RAM / 14 GB SSD** and
the paid arm64 XLarge has **14 GB RAM / 14 GB SSD**. The required ignored
evidence and environment exceed 20 GiB before native scratch space, while this
brief requires 40 GB to remain free. `.github/workflows/macos-15.yml` is
therefore valid, manual-dispatch-only, and unconditionally disabled. It
contains the full golden-regeneration/test sequence and an exact activation
contract for a self-hosted Apple Silicon runner with at least 48 GiB unified
memory and 80 GiB free SSD. `CI.md` records current official sources, cost/time
caveats, persistent T3/T3B/T4 evidence requirements, and that no secrets are
needed. Three workflow contracts and `actionlint` pass; the package-closing
suite passes **648/648 tests in 527.15 seconds**.

Stage H is complete at its documents-only gate. `HARDWARE_RUNBOOK.md` contains
the exact in-session `ARM SESSION CONFIRMED` gate, separate server/operator
commands, low-limit one-action first-contact protocol, physical power-switch
stop authority, verified-torque/speed prerequisite, rollback, and evidence
checklist. `scripts/serve_latency_smoke.py` starts only the policy server and
routes no-clobber JSONL telemetry through `ServeConfig.latency_log`. Each
successful chunk records wall-clock and monotonic latency boundaries,
timesteps, action count, and policy identity; no images, state, task, or action
values are stored. A fake-policy gRPC loopback proves the record and existing-
session refusal. The gate phrase was not supplied in-session: **hardware
validation remains NOT RUN**, and no robot environment, serial port, camera,
motor, or arm was accessed. The exact Stage H tree passes **652/652 tests in
537.24 seconds**.

Final verification is complete. The finished-tree distributions were rebuilt
from pushed code commit `691ce84fd9ba740239a9c39a458b3e2cc2a375be` and
contain the T4 training UX, Q3 quantization presets, and H server telemetry.
The sdist and CPython 3.11/3.12/3.13 wheels each passed a fresh base install,
isolated native-backend import, and offline finite-action prediction; the
CPython 3.12 serve-extra wheel also passed the pinned descriptor and ephemeral
loopback `Ready` RPC. Both installed quantization presets emitted finite
actions. Exact artifact hashes are in `DIST_MANIFEST.md`; the preceding release
artifacts remain recoverable under the ignored cache, and nothing was uploaded.

The final audit found all 33 required deliverables, no tracked private home
path, high-confidence secret pattern, model/build binary, file over 10 MiB,
code placeholder, skip, or xfail. All 18 relative Markdown links resolve, the
111-package lock is current, and `actionlint` passes with only its intentional
constant-false workflow diagnostic excluded. The T3B verdict recomputes its
embedded arithmetic and chronology; its frozen comparison implementation
digest resolves to Git commit `54a4e0b55dbabbbfd0ecbeb5c58caf80523f02d2`,
so the later T4 implementation is not misrepresented as historical evidence.
Four unreachable dangling blobs are the only `git fsck` findings; reachable
history is intact. The closing suite passes **652/652 tests in 538.08 seconds**
with no skips. The machine had **517 GiB** free and repository caches occupied
**122,897,608 KiB**, comfortably above the mandatory 40 GiB free-space floor.

## Stage state

| Stage | State | Evidence / next action |
| --- | --- | --- |
| R — Release | Complete — `RELEASE READY` | All P0/P1 criteria pass. The finished-tree sdist and CPython 3.11–3.13 wheels were refreshed from pushed code commit `691ce84`; four base smokes, one serve-extra loopback smoke, both installed quantization smokes, and the 652-test final suite pass. Metal fp32's fixed deterministic failure and pinned MLX's 26.2 dylib floor are documented limitations. |
| T0 — Training-readiness | Complete | 155/155 gradients finite and nonzero over 99,880,992 trainable scalars; 196.799 ms forward+backward and 2,509,594,126-byte peak MLX memory. See `TRAINING_FEASIBILITY.md`. |
| T1 — Gradient parity | Complete | Identical real batch/draws; loss and all 155 gradients pass immutable gates. See `GRADIENT_PARITY.md`. |
| T2 — Optimizer lockstep | Complete | 25/25 losses and 155/155 final tensors pass immutable gates. See `OPTIMIZER_LOCKSTEP.md`. |
| T3 — LoRA fine-tune | Failure-documented | The 3,000-update run, merge, held-out improvement, and Torch round trip completed. All 56 frozen cases were parity-checked; raw physical max was `6.632053375244141` versus the unchanged `0.005` gate; see `FAILURE_LORA_FINETUNE.md`. |
| T3B-1 — Reference floor | Complete | Nine PyTorch workers, including a fixed five-process MPS empirical envelope, evaluated 56 cases each; `F = F64 = 0.00003549918286283038`; report SHA-256 `cba4a856...f0585`; informational only, with no statistical-bound claim. |
| T3B-2 — Prospective evaluator | Complete | Fixed gates unchanged; derived `max(0.005, 3F)` gate, chronology, complete input provenance, semantic conversion validation, and no-clobber output are frozen and pass 52 focused tests. |
| T3B-3 — Expert-only LoRA | Statistical alpha | All 3,000 updates and the strict export completed. Fixed preprocessing, held-out-improvement, and round-trip gates pass. Prospective normalized parity is `0.013038858771324158` versus derived `0.005`; the new failure record preserves this result without changing tolerances. |
| T4 — Training UX/full fine-tune | Complete | Unified CLI, explicit reference-full topology, 100-update real smoke, finite complete export, three-checkpoint retention, and LoRA/full exact-resume gates all pass. See `TRAINING_UX.md`. |
| T5 — Training docs/benchmark | Complete | The frozen four-cell idle Metal matrix, exact commands, overnight projections, and Torch round-trip proof are published in `TRAINING_BENCHMARK.json`, `BENCHMARK.md`, and `README.md`. |
| Q — Quality extras | Complete | P2-1 comparison, P2-2 profile, P2-3 gated quantization, and P2-4 honestly disabled CI workflow are committed; every package closed with the full suite. |
| H — Hardware readiness | Complete — documents/software only | Exact supervised runbook and no-clobber observation-to-chunk latency logger are committed and loopback-tested. Hardware validation is explicitly NOT RUN because the in-session gate was absent. |

`TRAINING ALPHA (STATISTICAL)`, `RELEASE READY`, `T4 COMPLETE`, `T5 COMPLETE`,
and `FINAL VERIFICATION COMPLETE` have been reached.

## Current local evidence

| Artifact / cache | Size | Integrity |
| --- | ---: | --- |
| `.cache/training/t0-audit.json` | JSON | SHA-256 `88dacde30996c2d9cbad90681204e2583c92f6d51f0f3747c6e37e57b709fd51` |
| `.cache/training/gradient_goldens` | 769 MiB | manifest SHA-256 `b029a0ed66312e785cb8aa3f1db0affb16c9502ad7b5d0fe0feea3177bf8c145` |
| `.cache/training/t1-parity.json` | 52 KiB | SHA-256 `f4da0c16771a462e45bd615728bc02a059633db19eb77883342203426cb4d634` |
| `.cache/training/optimizer_goldens` | 381.329 MiB | manifest SHA-256 `88c3febc7da3e553bcb7c26f261721369ed1f56efd457887b7d43d50a077807c` |
| `.cache/training/t2-lockstep.json` | 60 KiB | SHA-256 `da8cabf5eecf4379065771b3a74407c47290b8aee9c2d0a9756893b6dd87a6a4` |
| `.cache/training/t3-benchmark.json` | JSON | SHA-256 `3598214cecd083cd3d5d143edd3edbe614dc899d30622152573e87dd104fe442` |
| `.cache/training/t3-evaluation` | 100 MiB | Tensor-manifest SHA-256 `9cabca6cd21e8658a94e42980af3e91ecd8ff5ed5daca5f75eb7a1ebd1d261a3`; full metadata SHA-256 `f49ee54aead7ce3ede7b94d5638864afd2e12ef57ae2622eb6574333820cd107` |
| `.cache/training/t3-base-evaluation.json` | JSON | SHA-256 `211d6778b0530208ca2e81abe6f4002cc683e24d496a09ddbe39c100ebd4f7ce`; base MAE `4.639846293521779` |
| `.cache/training/t3` | 1.8 GiB | Completed run SHA-256 `c7c3b86361c0872e26f2088cbd33ada865cf450b6711a9b737ece933c1868c82`; adapter/final-model SHA-256 `814e6f4b2a78a46b609aa7b48a28b4509f709d3e851e588dcd9a4bd2ca1408dc`; final optimizer SHA-256 `c9440be75315e04c1812ba18da0e0daccd2990fb0f6fdb1841d40ef7b01ffb5a`; retained checkpoints 2,800/2,900/3,000; export audit manifest SHA-256 `55ad6834cbb3acb9dd565a57296a274d78e7cdc863aa81c3e6ef25da8b66ba03` |
| `.cache/training/t3-outcome.json` | JSON | SHA-256 `8b74faf8f9cc96341090f91cfa795ed874c838026416944e4b77a550ad91bc44`; 15 source digests include the validated native conversion; improvement and round-trip pass, all-56-case parity fails |
| `.cache/training/t3/floor.json` | JSON | SHA-256 `cba4a856f9c907d986ffc8703789673611e54bad983c2afd0a987830466f0585`; all nine worker artifacts and every input file hash; combined input SHA-256 `d31a0835867116d7bfbe63f6cd23666eecfdc0a660aba620730cd09320295299`; `F = F64 = 0.00003549918286283038`; retrospective diagnostic only |
| `.cache/training/t3b` | Completed run | Run-state SHA-256 `2af527bea4691862e89eb6daa674e6d99309668ac45f57397786976aab3c301e`; adapter SHA-256 `cce4eed18a7311594950f6d4da33a44dd337f66fbc29162d686c5338ec044826`; retained checkpoints 2,800/2,900/3,000; export model SHA-256 `858704fa572501d9e5a048076f8da692693b90c463feda29201a72f3f0b18883` |
| `.cache/training/t3b/floor.json` | JSON | SHA-256 `28d83926a70e507671bfd694e032f81b71093d475075aad627b3c24c5b334efc`; floor input SHA-256 `3688cdad4f40724fa82765bb1c2ba89369aed056e29cecb8b1c074d6069939bb`; `F = 0.00002467632293701172`; prospective and older than the comparison marker |
| `.cache/training/t3b/comparison.json` | JSON | SHA-256 `6aa8e3771bbbd81ecd9599ec9605a4e1efb804fa9ec66c4f82d2d6aea3eb00c6`; fixed gates pass; normalized max `0.013038858771324158` |
| `.cache/training/t3b/parity-evaluation.json` | JSON | SHA-256 `1e337f0bb87aa66a4270c526dd918bd18807aa6aa5291a59b119780080ea9eca`; derived deterministic gate fails at threshold `0.005` |
| `.cache/training/t4-resume-lora` | Exact-resume smoke | Evidence SHA-256 `44325aa73c012d5b9dfb5499a549eeb689b90c64ebd07b137ee024cefa797b57`; 100 vs 50+resume has zero parameter/loss/metric drift and exact optimizer/draw/sampler/state identity; both exports finite |
| `.cache/training/t4-resume-full-v2` | Exact-resume smoke | Evidence SHA-256 `2c46c621a08b59584701b1bc2171690cfc03c7a116e41d4e4fff35f217699748`; 155 fp32 master tensors, zero parameter/loss/metric drift, exact continuation state, and finite 500-tensor exports |
| `TRAINING_BENCHMARK.json` | Tracked public record | SHA-256 `bca3ad9d0c2285fa70f4083885a6a6708e8c9d98b6c999d4cabd87b061cef07a`; four measured cells and mechanically verified derivations |
| `.cache/training/t5-benchmark.json` | Full timing evidence | SHA-256 `7112806471e55e55d98ae101bc2af8172c2cc18f01b3e0c2c0646446adba9423`; clean protocol commit, raw synchronized timings, environment, idle declaration, and source hashes |
| `INFERENCE_COMPARISON.json` | Tracked P2-1 timing evidence | SHA-256 `115ad58c0c618b65a6275018614f3ee6cf17dd02a9d4ad9c94aaf7e5a9842e48`; two isolated engines, 100 raw synchronized timings, fixed input/noise hashes, idle/environment/source evidence |
| `BF16_PROFILE.json` | Tracked P2-2 component evidence | SHA-256 `74da9f937cb8bfeba4066d5518187490ff96a1447e4a2ad2253e2493245be1cf`; 600 raw timings, exact component summaries/attribution, clean idle/environment/source evidence |
| `QUANTIZATION_EXPERIMENT.json` | Tracked P2-3 experiment evidence | SHA-256 `40060b0eaa63efee471ce2966f8fd578ade6ba2e8d9923435e14ef2466be393b`; three exact topology manifests, 150 raw timings, 150 per-case error records, fixed-gate decisions, and clean idle/source evidence |
| `HARDWARE_RUNBOOK.md` | Tracked documents-only H handoff | SHA-256 `c164bbe1a749905dbf96f790942ffe5d46668d2c8b8d64c70477b1bb9db58e84`; gate, exact commands, safety, rollback, and evidence checklist; hardware NOT RUN |
| `scripts/serve_latency_smoke.py` | Tracked serve-only H entrypoint | SHA-256 `73145031b4ec65800ba5b1fe9a55a16fb1bcccec840cc700d4dc9940bde7459d`; exclusive JSONL path wired to loopback-default server |
| `.cache/hf` | 2,614,416 KiB | repository-local pinned source/dataset cache |
| `.cache/smolvla_mlx` | 74,751,612 KiB | native conversion/evaluation cache after isolated Q experiments |
| `dist/` | 4 artifacts | Finished-tree sdist plus CPython 3.11/3.12/3.13 `macosx_14_0_arm64` wheels from code commit `691ce84`; exact hashes in `DIST_MANIFEST.md` |
| `.cache/production-deterministic.json` | 2,515 bytes | SHA-256 `3268f88be5ea854ff5162373146d1b2fd23cdbcc26bacbc060f0f8fa5b850398`; production fp32 fails fixed deterministic gate, bf16 passes |
| `.cache/statistical-strict-production-report.json` | 13,737 bytes | SHA-256 `b292736e3ec82b3eae8702c065c7c642326d226f06d35aa2214ac83fa1c23db5`; explicit strict CPU ratios both pass |
| `.cache/statistical-production.json` | 13,752 bytes | SHA-256 `c506ddcfdde50297e97b9905a299d55f117680a93b257e0af335ae6c9ad5fe07`; explicit production Metal ratios both pass |

Training evidence is intentionally ignored by Git and has not been uploaded.

## Safety and external state

- `origin`: `git@github.com:daniiarabdiev/smolvla_mlx.git`.
- No PyPI or Hugging Face uploads were made.
- No credentials, robot environment, vendor fork, serial ports, or hardware
  were accessed.
- The release brief and T3B amendment are committed. Full-scope software
  execution is complete; only the separately gated supervised hardware
  validation remains for an operator-present session.
## 2026-09-02 public-release continuation

- **Canonical identity migration:** software complete — distribution and CLI
  `mlx-smolvla`, import package `mlx_smolvla`, cache variable
  `MLX_SMOLVLA_CACHE`, default cache `~/.cache/mlx_smolvla`, and a tested
  one-release warning shim for `SMOLVLA_MLX_CACHE`. Public documentation and
  final release artifacts remain in Stage C/D.
- **Stage B macOS/MLX floor:** complete — official macOS 14 arm64 wheels for
  MLX 0.32.0–0.32.2 were directly inspected and every version passed strict
  conversion/goldens, fixed 50-frame statistical gates, installed offline
  prediction, doctor, and loopback `Ready`. The full suite passes 664/664.
- **GitHub rename:** blocked externally — the requested new SSH endpoint
  returns `Repository not found`, so `origin` remains on the working old
  endpoint pending the exact task in `HUMAN_TASKS.md`.
- **Hardware Stage A:** blocked by the unchanged live authorization gate; no
  device or vendor-tree access occurred.

## 2026-09-02 supervised hardware continuation

This section supersedes the earlier hardware and origin snapshots above.

- **Source checkpoints:** the fail-closed client was implemented and pushed in
  four reviewable commits ending at `4044671`. The canonical origin is
  `git@github.com:daniiarabdiev/mlx-smolvla.git`; local and fetched `main`
  matched at that checkpoint.
- **Authorization and boundary:** the operator supplied `ARM SESSION
  CONFIRMED`. Only the follower port was opened. The separate leader was
  detected but never opened. The vendor checkout remained at `a24998f` and its
  696 tracked files retained the identical composite SHA-256
  `d280efa881ab9e412cd071bbef38d8d9ec5050e484b49b5a2397df2a39bdb764`.
- **Physical preflight:** all six controller torque bits were zero before and
  after access, and calibration matched controller readback. Both cameras were
  live, but the wrist view was obstructed and the fixed view omitted the robot
  workspace. Lift and elbow were outside the required 10%-inset start range.
  Existing controller settings were recorded but were not assumed safe; no
  operator-attested low-limit profile exists.
- **No-motion result:** two bounded 60-second native MLX RPC loops completed at
  about 4.915 sampled camera FPS, with zero action timeouts and zero motor or
  torque writes. Raw-base observation-to-chunk latency was 149.313 ms median /
  151.139 ms p95; the stats-active run was 149.746 / 152.502 ms. The latter
  processed 294 chunks and held 12 outside the full public action domain.
- **Safety implementation:** motion modes now require exact follower identity,
  torque-off and calibration readback, finite camera data, effective six-axis
  state/action statistics, an exact nine-register operator profile, an inset
  start pose, valid `(1, 6)` finite actions, domain/envelope/rate checks, fixed
  watchdog/session caps, and verified torque-off cleanup. No-motion mode has no
  torque or actuator write path.
- **Environment isolation:** fresh ignored Python 3.12 `serve` and `hardware`
  environments were installed separately. The server environment has no PyAV;
  CLI import and loopback-only startup/shutdown emitted no duplicate
  AVFoundation-class warning.
- **Current gate:** the connected no-motion milestone is complete; single-
  action and bounded-continuous motion remain blocked on camera framing,
  manual neutral placement, a known-good low-limit profile, and the
  workspace/base/physical-power checklist. No tag, upload, release, visibility
  change, or hardware-motion claim is authorized.
- **Verification at this checkpoint:** focused hardware/server/import/
  distribution coverage passes 115/115. Artifact rebuild and closing fast/full
  suites remain to be recorded in a later section.

## 2026-09-02 hardware release-candidate closure

- The final package-surface correction explicitly includes the `hardware`
  extra in the Python-3.12+ requirements line and is pushed at source commit
  `9c549557f2e3a355bf5c0206e6c86fa54ad191bf`.
- A clean detached worktree at that commit produced one sdist and CPython
  3.11/3.12/3.13 `macosx_14_0_arm64` wheels. All four pass Twine and canonical
  archive inspection; the sdist has 148 entries, each wheel has 73, and every
  packaged or sdist-installed project extension reports `minos 14.0`.
- Four new base environments pass canonical import/isolation,
  `native-reference`, `doctor`, dependency integrity, and finite offline
  prediction. A fifth serve-extra environment passes protobuf descriptor
  identity, ephemeral loopback `Ready`, CLI surfaces, and both VLM quantization
  presets. A sixth hardware-extra environment passes dependency integrity,
  installed-module origin, stats-active checkpoint validation, camera mapping,
  and all graduated-mode help without device access.
- Ignored `dist/` now contains only the four verified `mlx_smolvla-0.1.0`
  files; its retired 0.0.1 contents were preserved under ignored `.cache/`.
  Exact sizes and SHA-256 values are in `docs/evidence/DIST_MANIFEST.md`.
- A clean idle `make test-fast` passes 479/479 selected tests with 291 slow
  tests deselected in 104.95 test seconds / 108.16 seconds wall. The closing
  `make test` passes all 770/770 tests in 703.76 test seconds / 707.22 seconds
  wall. Neither reports a skip or xfail.
- The closing static audit resolves 122 lockfile packages, passes dependency
  checking and all focused public/hardware/link/privacy tests, and finds no
  exact hardware serial, private home path, credential pattern, explicit
  skip/xfail, or tracked file over 10 MiB. `actionlint` reports only the
  intentional constant-false hosted-CI guard. The original LoRA failure hash
  remains `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
- Software verification is complete. `PUBLIC RELEASE READY` is not reached:
  the camera, neutral-pose, low-limit-profile, and physical checklist blockers
  still prohibit a single action, bounded continuous motion, tagging,
  publication, visibility changes, release creation, and hardware claims.
