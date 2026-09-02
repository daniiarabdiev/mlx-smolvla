# SmolVLA MLX Port Design

## Status and authority

This document records the approved implementation design for the SmolVLA MLX
port. `BRIEF.md` is the authoritative product and acceptance specification, and
`AGENTS.md` is the authoritative operating policy. If this document conflicts
with either one, those files win.

## Goal

Build a native MLX inference implementation of `lerobot/smolvla_base` that is
designed for Apple Silicon and does not require PyTorch, Transformers, or
LeRobot at runtime. It must reproduce the reference policy numerically within
the fixed tolerances in `BRIEF.md`, expose a practical policy API, and publish
measured latency and memory results from the operator's M5 Pro.

Version 0.1 covers inference, conversion, validation, packaging, and
benchmarking. Training, robot hardware integration, other VLA families, and a
GUI remain out of scope.

## Architecture strategy

Use a hybrid, evidence-led port:

1. Treat the installed mainline LeRobot implementation and checkpoint metadata
   as the behavioral source of truth.
2. Reuse compatible `mlx-vlm` implementations where their public behavior and
   parameter layout match the reference.
3. Vendor the smallest license-preserving subset when SmolVLA needs behavior
   that the reusable API cannot expose, especially partial decoder execution
   and per-layer key/value capture.
4. Implement SmolVLA-specific state conditioning, action expert, flow-matching
   loop, action queue, and conversion logic in focused local modules.

A full rewrite is avoided because it enlarges the numerical-parity surface. A
PyTorch or MPS wrapper is rejected because it would not produce a native MLX
runtime or meet dependency isolation.

## Runtime boundaries

The installable `smolvla_mlx` package owns preprocessing, MLX model modules,
weight conversion, policy state, and the CLI. It may depend only on the runtime
libraries allowed by `BRIEF.md`.

The `reference/`, `scripts/`, and `tests/` areas may use PyTorch, Transformers,
and LeRobot through a pinned optional `reference` dependency group. Runtime
imports are checked in a fresh subprocess so accidental transitive imports are
visible.

All development caches and generated goldens live below the repository's
`.cache/` or `tests/golden/` paths and remain untracked. End-user cache behavior
honors `SMOLVLA_MLX_CACHE` and otherwise uses the documented platform default.

## Inference data flow

1. Validate an observation containing camera images, robot state, and task
   text.
2. Reproduce the reference processor exactly: camera selection and masking,
   resize/pad/normalization, tokenization/padding, and state normalization.
3. Batch camera images through the vision encoder and connector.
4. Assemble the multimodal prefix and execute only the reference-selected
   language-model layers while capturing the key/value states required by the
   action expert.
5. Project state and noisy action tokens according to the verified reference
   architecture.
6. Run the expert under the exact verified attention masks and timestep
   embedding.
7. Integrate velocity with the verified Euler schedule and sign convention.
8. Un-normalize the action chunk, enqueue it, and return one action per
   `select_action` call. `reset()` clears the queue.

The VLM prefix is computed once per action chunk and reused across Euler steps.
No optimization may change this data flow until the corresponding fp32 and
bf16 parity tests pass.

## Weight conversion

Conversion reads the original Hub safetensors and checkpoint configuration,
produces an explicit source-to-target JSON name map, and writes MLX-loadable
safetensors. Every source tensor must map exactly once, every target parameter
must be initialized, shapes and total parameter counts must agree, and checksums
must be recorded. An fp32 master is retained and bf16 is derived from it.

## Correctness method

Correctness is established from the inside out:

1. Pin the reference environment and generate deterministic fp32 CPU goldens
   from real public SO-101 observations and fixed noise.
2. Save and hash preprocessing outputs and intermediate tensors at each model
   boundary.
3. Re-run golden generation and require byte-identical manifests.
4. Implement each MLX module test-first and compare it with its matching
   golden in fp32 and bf16.
5. Run deterministic end-to-end comparison with identical noise.
6. Run the specified statistical comparison over at least 50 observations.
7. Re-run correctness after every performance optimization.

The fixed tolerances and failure protocol in `BRIEF.md` are gates. They are not
tuning parameters.

## Failure handling

Input, configuration, checkpoint, and conversion errors fail early with the
offending field or tensor named. Missing network assets include the requested
Hub identifier and in-repo cache path in the error. Unsupported reference
architecture differences are documented before implementation continues.

After three evidence-backed hypotheses fail for a numerical mismatch, the
module receives the required `FAILURE_<module>.md`; downstream work that does
not depend on it may continue. External write or credential blockers go into
`HUMAN_TASKS.md` without weakening repository isolation.

## Delivery surfaces

The deliverable is an installable `smolvla_mlx` library with:

- `SmolVLAMLX.from_pretrained(...)` and the LeRobot-like `select_action` and
  `reset` policy interface;
- conversion, test, benchmark, and dataset prediction CLI commands;
- reproducible dependency locks and Make targets;
- architecture, reuse, benchmark, progress, status, licensing, and usage
  documentation;
- a clean runtime dependency-isolation test and fresh-environment smoke test.

## Completion criteria

Completion means every item in `BRIEF.md` Section 9 is supported by fresh test
or benchmark evidence and `STATUS.md` contains `DEFINITION OF DONE MET`. A fast
model without parity, or a correct model that still executes through PyTorch,
is not complete.
