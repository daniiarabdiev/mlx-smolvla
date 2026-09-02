# Public-release status

Date: 2026-09-02

`mlx-smolvla` is a verified 0.1.0 software release candidate on the renamed
GitHub repository. Four of the five public-sharing blockers are clear. The
supervised SO-101 hardware gate is not clear, so the version tag, uploads,
visibility change, release creation, announcement, and hardware claim remain
withheld.

## Stage outcomes

| Stage | Outcome | Evidence |
| --- | --- | --- |
| A — real SO-101 | **BLOCKED** | The operator has not supplied the exact standalone live-session authorization. [`hardware/FIRST_CONTACT.md`](../../hardware/FIRST_CONTACT.md) therefore remains explicitly not performed; no device, vendor tree, or robot was accessed. |
| B — macOS / MLX floor | **Complete** | Official macOS 14 arm64 wheels for MLX 0.32.0, 0.32.1, and 0.32.2 were hash- and Mach-O-verified, then passed the unchanged correctness and installed-runtime gates. See [`MLX_COMPATIBILITY.md`](../evidence/MLX_COMPATIBILITY.md). |
| C — public-release preparation | **Complete** | Canonical distribution/import/CLI/cache/GitHub identities, prior-project acknowledgment, community metadata, hobbyist-first README, agent guide, root hygiene, and a sub-two-minute fast lane are committed. |
| D — software verification | **Complete; publication held** | The canonical artifacts passed archive, Twine, fresh-install, offline prediction, doctor, quantization, and loopback-serving checks. The fast suite passed 387/387 and the complete suite passed 678/678. See [`DIST_MANIFEST.md`](../evidence/DIST_MANIFEST.md) and [`DOCTOR.txt`](../evidence/DOCTOR.txt). |

## Closing verification

- `make test-fast`: 387 passed, 291 deliberately slow tests deselected, no
  skips or xfails, 93.71 test seconds / 96.50 seconds wall. An idle-process
  preflight found no trainer, floor worker, pytest, or other Make test process.
- `make test`: 678 passed, no skips or xfails, 572.61 test seconds / 575.49
  seconds wall. The same idle-process preflight passed first.
- Focused public/hygiene/distribution/rename/cache/compatibility/hardware slice:
  50/50.
- Sdist plus CPython 3.11/3.12/3.13 native wheels: canonical names only,
  project-extension `minos 14.0`, all four accepted by Twine and all four
  freshly installed outside the checkouts.
- Fresh installed behavior: four base `doctor` and finite offline prediction
  passes; one serve-extra descriptor/ephemeral-loopback `Ready` pass; finite
  VLM 8-bit and 4-bit offline predictions; cache-variable compatibility shim
  warning and precedence pass.
- `uv lock --check` resolved 111 packages. Actionlint passed with only the
  intentional constant-false diagnostic for the documented disabled hosted
  macOS workflow excluded.
- Public-document first-port-claim scan, stale-project-identity scan,
  tracked-root/link/personal-path tests, secret-pattern scan, tracked-file
  size audit, skip/xfail scan, and `git diff --check` passed.
- The protected first LoRA failure record remains byte-identical at SHA-256
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
  The protected bf16, inference-comparison, quantization, second-attempt, and
  training-benchmark evidence also matches the clean source checkpoint.
- A final read-only fetch confirmed the canonical origin. Both official PyPI
  endpoints returned HTTP 404 at 2026-09-02T14:45:08Z, so `mlx-smolvla`
  appeared unclaimed at that instant but is not reserved.
- Nothing was uploaded. No tolerance changed. No training run, floor
  computation, hardware access, camera access, serial access, or vendor-tree
  access occurred during this release pass.

## Public-sharing blocker table

| Blocker | Status | Reason |
| --- | --- | --- |
| Serve untested on hardware | **BLOCKED** | The separately authorized no-motion, single-action, and bounded-continuous protocol has not run. |
| macOS / MLX floor | **Clear** | MLX 0.32.0–0.32.2 and their official macOS 14 wheel family passed the fixed software gates. |
| Claims exceed evidence | **Clear** | Public first-port language was removed, the earlier independent Hub port is acknowledged factually, performance numbers resolve to evidence, training is a preview, and hardware is explicitly unvalidated. |
| Operator material in tree | **Clear** | The public root is allowlisted, local agent configuration is ignored/untracked, no private home path is present outside history, and scans found no credential pattern or oversized tracked artifact. |
| First-page friction | **Clear** | The README leads with requirements/install/run/serve/train paths, installed `doctor` works, and the unfiltered fast lane is under two minutes. |

## Exact next gate

The operator actions and commands are maintained in
[`HUMAN_TASKS.md`](HUMAN_TASKS.md). The next prerequisite is a genuinely live,
supervised hardware session beginning with the exact standalone authorization
specified there. Until its graduated protocol passes and its evidence is
committed, do not tag, publish, make the repository public, create a GitHub
Release, announce hardware support, or share the release as complete.
