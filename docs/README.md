# Documentation

- [Architecture](ARCHITECTURE.md): model boundary, conversion, execution modes,
  and serving protocol.
- [Benchmarks](BENCHMARK.md): pinned inference/training measurements and scope.
- [Reuse decisions](REUSE_DECISIONS.md): upstream components inspected and the
  implementation choice for each native module.
- [Hardware runbook](HARDWARE_RUNBOOK.md): separately authorized SO-101 safety
  and validation procedure.
- [Hardware preflight](../hardware/PREFLIGHT.md): redacted connected-device,
  camera, calibration, register, no-motion, and bounded-motion evidence.
- [Hardware client design](../hardware/CLIENT_DESIGN.md): fail-closed client
  architecture and motion-gate contract.
- [First-contact status](../hardware/FIRST_CONTACT.md): bounded single-action
  and two-chunk continuous evidence plus the sustained-return limitation.
- [Prepared model card](MODEL_CARD.md): intended use, sources, evidence, and
  limitations for a future Hub presentation.
- [Evidence index](evidence/README.md): correctness, compatibility, training,
  distribution, and negative-result artifacts.
- [Development](dev/RELEASE_CHECKLIST.md): research-backed public-release gate.
- [History](history/README.md): specifications, plans, status snapshots, and
  operator provenance retained without rewriting the record.

The package and model caches are deliberately not documentation artifacts and
remain untracked.
