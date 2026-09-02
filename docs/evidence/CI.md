# Continuous integration feasibility

Stage Q P2-4 leaves
[`.github/workflows/macos-15.yml`](../../.github/workflows/macos-15.yml) checked in,
syntactically valid, manually dispatchable, and unconditionally disabled at
the job level. It has no `push` or `pull_request` trigger. A manual dispatch
therefore creates a skipped job rather than spending runner time.

## Local test lanes

`make test-fast` runs every test not marked `slow` with all declared extras installed. The marker is reserved for
tests that load pinned model/dataset artifacts, execute complete model/evidence
pipelines, or run the isolated provenance/optimizer processes those artifacts
require. It does not skip or alter those tests: `make test` still runs the
complete suite with the same semantics and gates.

On 2026-09-02 the closing unfiltered fast lane selected 385 tests and passed them
in 94.88 seconds (97.83 seconds wall clock) on an otherwise idle Apple Silicon
validation host, with no skips or xfails.

## Why GitHub-hosted macOS is disabled

GitHub's runner reference, checked on 2026-09-02, lists the standard arm64
`macos-15` runner as an M1 with **7 GB** RAM and **14 GB** SSD. The paid arm64
macOS XLarge runner raises memory to 14 GB but retains the same **14 GB** SSD.
See GitHub's current
[hosted-runner specification](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and [larger-runner specification](https://docs.github.com/en/actions/reference/runners/larger-runners).

Disk is a deterministic blocker. The full suite consumes ignored, locally
generated evidence rather than accepting checked-in binary goldens. The three
inference-golden trees, reference artifacts, T1/T2 goldens, T3 and T3B runs,
T3 evaluation set, T4 exact-resume runs, pinned Hub cache, uv cache, and virtual
environment occupy more than 20 GiB on the validated machine before native
conversion scratch space. The project brief also requires at least 40 GB to
remain free. A 14 GB runner cannot hold that input set, regardless of cache
policy. GitHub's macOS larger runner does not offer a larger disk.

The standard runner's 7 GB RAM is also outside the validated execution class
for the combined Torch-reference/native-MLX and training coverage. The local
suite is verified on an Apple M5 Pro with 48 GiB unified memory; lowering the
runner to 7 GB without evidence would turn CI into an OOM/flakiness signal,
not a correctness signal. The current cached suite takes about nine minutes,
while regeneration and first-time model acquisition add material time. GitHub
caps a hosted job at six hours; see the current
[Actions limits](https://docs.github.com/en/actions/reference/limits).

The repository is private. If a suitable hosted option existed, standard
macOS time would currently be billed at $0.062/minute and macOS XLarge at
$0.102/minute; see GitHub's
[runner pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing).
The workflow is disabled, so it intentionally schedules no billable work.

## Exact activation requirements

Before removing the workflow's `if: ${{ false }}` guard:

1. Register a dedicated **self-hosted Apple Silicon** runner with the labels
   `self-hosted`, `macOS`, `ARM64`, and `mlx-smolvla-ci`.
2. Provide at least **48 GiB** unified memory and **80 GiB** free SSD at job
   start. The 80 GiB floor leaves room for regenerated inputs and MLX scratch
   while preserving the brief's mandatory 40 GiB free-space floor.
3. Seed the exact operator-verified **T3/T3B/T4** run evidence in the checkout's
   repository-local `.cache/training` directory, or first implement complete
   deterministic reconstruction targets for those long runs. No unverified
   substitute or reduced test selection is acceptable.
4. Change `runs-on` to
   `[self-hosted, macOS, ARM64, mlx-smolvla-ci]`. Keep checkout `clean: false`
   so the ignored operator evidence is not deleted.
5. Keep all caches repository-local. The workflow itself regenerates base,
   stats-active, public-fine-tune, gradient, and optimizer goldens before
   calling `make test`.
6. Run manually until a complete green execution is recorded. Automatic PR
   triggers may be considered only after the capacity and evidence lane is
   stable.

No secrets are required. The automatic read-only `GITHUB_TOKEN` used by the
official checkout action is sufficient; no model, dataset, package, or
artifact upload step exists.
