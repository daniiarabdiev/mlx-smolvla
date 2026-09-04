#!/usr/bin/env python3
"""Validate the retained T3B reference-loader repair, not a new T3B milestone.

Run only after a corrected-loader informational nine-worker envelope. This
separate, fixed-limit check never replaces the checkpoint's original floor or
verdict. Outputs live in a newly reserved directory and inputs are rehashed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlx_smolvla.training.trained_parity import (  # noqa: E402
    _atomic_json_no_clobber,
    _load_floor_bundle,
    _pretty_json,
    _snapshot_file,
    _snapshot_floor_inputs,
    _snapshot_json,
    _revalidate_floor_input_locations,
    _validate_base_evaluation,
    _validate_mae_evaluation,
    _validate_parity_evidence,
    _validated_identities,
)


FIXED_LIMITS = {
    "fine_to_base_mae_ratio_maximum": 0.9,
    "torch_to_mlx_mae_ratio_minimum": 0.95,
    "torch_to_mlx_mae_ratio_maximum": 1.05,
    "image_preprocessing_max_abs": 1e-5,
    "state_preprocessing_max_abs": 1e-6,
    "stats_active_parity_max_abs": 0.005,
}
# This is a reproducible repair of one retained experiment, not a generic way
# to grant post-hoc prospective acceptance to an arbitrary checkpoint.
PROTECTED = {
    "docs/evidence/FAILURE_LORA_FINETUNE.md": "d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6",
    "docs/evidence/FAILURE_LORA_FINETUNE_B.md": "3bc6bb3cb7302797fa39c80309375c343a9348da18a18fbe2c08e504f4b37276",
    ".cache/training/t3b/export/model.safetensors": "858704fa572501d9e5a048076f8da692693b90c463feda29201a72f3f0b18883",
    ".cache/training/t3b/run.json": "2af527bea4691862e89eb6daa674e6d99309668ac45f57397786976aab3c301e",
    ".cache/training/t3b/metrics.csv": "33f00adc5316cbc295e6f3fa1e153963b64fadec59a7db5401074794245f6278",
    ".cache/training/t3b/adapter.safetensors": "cce4eed18a7311594950f6d4da33a44dd337f66fbc29162d686c5338ec044826",
    ".cache/training/t3b/outcome.json": "75dff4b750dd1e8c8bc4d8426fe9af297bedb33ea1bfa7523dcc49d51460b33f",
    ".cache/training/t3b/comparison.json": "6aa8e3771bbbd81ecd9599ec9605a4e1efb804fa9ec66c4f82d2d6aea3eb00c6",
    ".cache/training/t3b/parity-evaluation.json": "1e337f0bb87aa66a4270c526dd918bd18807aa6aa5291a59b119780080ea9eca",
    ".cache/training/t3b/floor.json": "28d83926a70e507671bfd694e032f81b71093d475075aad627b3c24c5b334efc",
    ".cache/training/t3-base-evaluation.json": "211d6778b0530208ca2e81abe6f4002cc683e24d496a09ddbe39c100ebd4f7ce",
    ".cache/mlx_smolvla/policy-float32/converted/b83f340c260da3a8/float32/model.float32.safetensors": "eedc59a06fa0b7977638631c176fec315cb983446710ab81571f1199e00b427e",
    ".cache/mlx_smolvla/policy-float32/converted/b83f340c260da3a8/float32/name_map.json": "149b85fa0eb95d3bfde0d3d2b3daa3bf3261f5ba7cf34c5f534336861b2ed505",
}


def check_chronology(*values: int) -> None:
    """Require envelope creation/write, marker creation/write, outcome, finish."""

    if (len(values) not in {3, 4, 6}
            or any(type(value) is not int or value <= 0 for value in values)
            or not values[0] <= values[1] < values[2]
            or (len(values) >= 4 and values[2] > values[3])
            or (len(values) == 6 and not values[3] < values[4] <= values[5])):
        raise ValueError("repair evidence chronology is invalid")


def reserve_output(directory: Path) -> None:
    """A rerun must use a new private directory, never overwrite evidence."""

    if directory.parent.resolve(strict=True) != directory.parent.absolute():
        raise ValueError("repair output parent must not be a symlink")
    directory.mkdir(mode=0o700, exist_ok=False)


def capture_file(path: Path) -> dict[str, object]:
    """Bind bytes and filesystem identity without retaining large model payloads."""

    snapshot = _snapshot_file(path, label="repair input")
    return binding_from_snapshot(snapshot)


def binding_from_snapshot(snapshot) -> dict[str, object]:
    """Keep the bytes already validated; do not silently adopt a replacement."""

    return {
        "path": str(snapshot.path.absolute()),
        "sha256": snapshot.sha256,
        "mtime_ns": snapshot.mtime_ns,
        "size": snapshot.size,
        "device": snapshot.device,
        "inode": snapshot.inode,
    }


def revalidate_files(bindings: list[dict[str, object]]) -> None:
    for binding in bindings:
        if capture_file(Path(str(binding["path"]))) != binding:
            raise RuntimeError(f"repair input changed: {binding['path']}")


def validate_fixed_outcome(outcome: dict[str, object]) -> dict[str, object]:
    """Recompute all 56-case summaries and apply the original fixed limits."""

    from training.evaluation import evaluate_outcome_gates
    from training.t3_contract import FROZEN_EVALUATION_MANIFEST_SHA256

    if outcome["thresholds"] != FIXED_LIMITS:
        raise ValueError("repair outcome differs from the original fixed limits")
    if (hashlib.sha256(_pretty_json(outcome["base_mlx_evaluation"])).hexdigest()
            != PROTECTED[".cache/training/t3-base-evaluation.json"]):
        raise ValueError("repair outcome differs from the frozen base report")
    identities = _validated_identities([
        {key: sample[key] for key in ("ordinal", "episode", "frame_index", "absolute_index")}
        for sample in outcome["base_mlx_evaluation"]["samples"]
    ])
    _, base = _validate_base_evaluation(
        outcome["base_mlx_evaluation"], identities,
        evaluation_manifest_sha256=FROZEN_EVALUATION_MANIFEST_SHA256,
    )
    _, fine = _validate_mae_evaluation(outcome["fine_mlx_evaluation"], identities, framework="mlx")
    _, torch = _validate_mae_evaluation(outcome["torch_evaluation"], identities, framework="torch")
    parity = {key: value for key, value in outcome["stats_active_parity"].items() if key != "passed"}
    _validate_parity_evidence(parity, identities)
    decision = asdict(evaluate_outcome_gates(
        base_mlx_mae=base, fine_mlx_mae=fine, torch_mae=torch,
        parity_max_abs=parity["gate_max_abs"],
        image_preprocessing_max_abs=parity["image_preprocessing_max_abs"],
        state_preprocessing_max_abs=parity["state_preprocessing_max_abs"],
    ))
    if (decision != outcome["gates"]
            or decision["parity_passed"] != outcome["stats_active_parity"]["passed"]):
        raise ValueError("repair outcome decision differs from its sample evidence")
    return decision


def install_outcome(staging: Path, output: Path, expected_sha: str):
    """Promote validated staging bytes without replacing any existing outcome."""

    outcome, snapshot = _snapshot_json(staging, label="staged repair outcome")
    if snapshot.sha256 != expected_sha:
        raise RuntimeError("staged repair outcome changed after evaluation")
    validate_fixed_outcome(outcome)
    digest = _atomic_json_no_clobber(output, outcome)
    binding = capture_file(output)
    if binding["sha256"] != digest:
        raise RuntimeError("installed repair outcome changed")
    return outcome, binding


def reject_output_overlap(output: Path, inputs: list[Path]) -> None:
    candidate = output.resolve()
    for source in inputs:
        source = source.resolve()
        if candidate.is_relative_to(source) or source.is_relative_to(candidate):
            raise ValueError("repair output overlaps a protected input")


def install_verdict(path: Path, verdict: dict[str, object], bindings: list[dict[str, object]]) -> str:
    """Recheck even the promoted outcome immediately before verdict publication."""

    revalidate_files(bindings)
    return _atomic_json_no_clobber(path, verdict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.absolute()
    if not output_dir.is_relative_to(ROOT / ".cache"):
        raise ValueError("repair outputs must remain in the repository .cache")
    reject_output_overlap(output_dir, [
        ROOT / ".cache/training/t3b", ROOT / ".cache/training/t3-evaluation",
        ROOT / ".cache/hf", ROOT / ".cache/mlx_smolvla/policy-float32",
        args.variants, args.envelope,
    ])

    from training.self_consistency import collect_floor_input_hashes, collect_floor_input_evidence
    from training.evaluation import run_finetune_outcome_evaluation

    locations = {
        "checkpoint_dir": ROOT / ".cache/training/t3b/export",
        "evaluation_dir": ROOT / ".cache/training/t3-evaluation",
        "cache_dir": ROOT / ".cache/hf",
    }
    floor, envelope_snapshot = _snapshot_json(args.envelope, label="informational envelope")
    bundle = _load_floor_bundle(floor, variant_root=args.variants)
    if (floor["purpose"] != "retrospective_diagnostic"
            or floor["checkpoint_path"] != ".cache/training/t3b/export"):
        raise ValueError("repair requires the retained T3B informational envelope")
    inputs, _ = collect_floor_input_hashes(**locations)
    if inputs != floor["input_sha256"]:
        raise ValueError("current implementation/inputs differ from the envelope")
    input_evidence = collect_floor_input_evidence(**locations)
    input_bundle = _snapshot_floor_inputs(
        evidence_root=ROOT, recorded_evidence=input_evidence, floor_inputs=inputs)
    bindings = [binding_from_snapshot(snapshot)
                for group in input_bundle.files.values() for snapshot in group.values()]
    for name, digest in PROTECTED.items():
        binding = capture_file(ROOT / name)
        if binding["sha256"] != digest:
            raise ValueError(f"protected history differs: {name}")
        bindings.append(binding)
    # Bind the executing native runtime, native binaries, reference, and runner
    # as well as the floor's complete checkpoint/data/dependency hash groups.
    source_paths = {Path(__file__), ROOT / "pyproject.toml", ROOT / "uv.lock"}
    for directory in ("mlx_smolvla", "training", "reference"):
        source_paths.update(path for path in (ROOT / directory).rglob("*")
                            if path.is_file() and path.suffix in {".py", ".so", ".cpp", ".metal"})
    bindings.extend(capture_file(path) for path in sorted(source_paths))
    bindings.extend(binding_from_snapshot(snapshot) for snapshot in bundle.snapshots)
    envelope_binding = binding_from_snapshot(envelope_snapshot)
    bindings.append(envelope_binding)
    original_floor, original_floor_snapshot = _snapshot_json(
        ROOT / ".cache/training/t3b/floor.json", label="original prospective floor")
    original_bundle = _load_floor_bundle(
        original_floor, variant_root=ROOT / ".cache/training/t3b/self-consistency/variants")
    if (original_bundle.bundle_sha256
            != "31ce3db6619294432742b38214132267cfecf735dc0ce1d98199bbd223e8a889"):
        raise ValueError("original floor raw bundle differs")
    bindings.append(binding_from_snapshot(original_floor_snapshot))
    bindings.extend(binding_from_snapshot(snapshot) for snapshot in original_bundle.snapshots)
    revalidate_files(bindings)
    start_ns = time.time_ns()
    check_chronology(floor["created_at_ns"], envelope_binding["mtime_ns"], start_ns)
    reserve_output(output_dir)
    outcome_path = output_dir / "outcome.json"
    staging_path = output_dir / ".staging-outcome.json"
    report_path = output_dir / "repair-verdict.json"
    marker_path = output_dir / "repair-start.json"
    marker = {
        "artifact_type": "mlx-smolvla-reference-loader-repair-start",
        "format_version": 1,
        "purpose": "fixed-limit-software-repair-validation-not-original-T3B-acceptance",
        "created_at_ns": start_ns,
        "fixed_limits": FIXED_LIMITS,
        "informational_envelope_sha256": envelope_binding["sha256"],
        "informational_raw_bundle_sha256": bundle.bundle_sha256,
        "floor_input_sha256": inputs,
        "floor_input_evidence": input_evidence,
        "file_bindings": bindings,
        "outcome_path": str(outcome_path),
        "private_staging_path": str(staging_path),
        "repair_report_path": str(report_path),
    }
    marker_sha = _atomic_json_no_clobber(marker_path, marker)
    marker_binding = capture_file(marker_path)
    if marker_binding["sha256"] != marker_sha:
        raise RuntimeError("repair start marker changed after installation")
    revalidate_files(bindings)
    # This is the final guard before any repaired MLX/Torch model work.
    fresh_inputs, _ = collect_floor_input_hashes(**locations)
    if fresh_inputs != inputs:
        raise RuntimeError("repair inputs changed before inference")
    check_chronology(floor["created_at_ns"], envelope_binding["mtime_ns"],
                     start_ns, marker_binding["mtime_ns"])
    if marker_binding["mtime_ns"] > time.time_ns():
        raise ValueError("repair marker chronology is in the future")

    def progress(framework: str, completed: int, total: int) -> None:
        if completed == 1 or completed % 8 == 0 or completed == total:
            print(f"repair {framework} evaluation {completed}/{total}", flush=True)

    outcome, outcome_sha = run_finetune_outcome_evaluation(
        cache_dir=locations["cache_dir"],
        native_cache=ROOT / ".cache/mlx_smolvla/policy-float32",
        run_dir=ROOT / ".cache/training/t3b",
        evaluation_dir=locations["evaluation_dir"],
        base_report_path=ROOT / ".cache/training/t3-base-evaluation.json",
        output_path=staging_path, progress=progress,
    )
    staging_binding = capture_file(staging_path)
    if staging_binding["sha256"] != outcome_sha:
        raise RuntimeError("repair outcome changed after evaluation")
    original = json.loads((ROOT / ".cache/training/t3b/outcome.json").read_bytes())
    if outcome["source_sha256"] != original["source_sha256"]:
        raise ValueError("repair changed the original training/export/evaluation sources")
    identities = [
        {key: sample[key] for key in ("ordinal", "episode", "frame_index", "absolute_index")}
        for sample in outcome["base_mlx_evaluation"]["samples"]
    ]
    if identities != floor["case_identities"]:
        raise ValueError("repair population differs from the informational envelope")
    decision = validate_fixed_outcome(outcome)
    final_inputs, _ = collect_floor_input_hashes(**locations)
    if final_inputs != inputs:
        raise RuntimeError("repair inputs changed during evaluation")
    _revalidate_floor_input_locations(input_bundle, evidence_root=ROOT)
    revalidate_files([*bindings, marker_binding, staging_binding])
    check_chronology(floor["created_at_ns"], envelope_binding["mtime_ns"],
                     start_ns, marker_binding["mtime_ns"], staging_binding["mtime_ns"], time.time_ns())
    installed_outcome, outcome_binding = install_outcome(staging_path, outcome_path, outcome_sha)
    if installed_outcome != outcome:
        raise RuntimeError("installed outcome differs from the evaluated outcome")
    finished_ns = time.time_ns()
    check_chronology(floor["created_at_ns"], envelope_binding["mtime_ns"],
                     start_ns, marker_binding["mtime_ns"], outcome_binding["mtime_ns"], finished_ns)
    verdict = {
        "artifact_type": "mlx-smolvla-reference-loader-repair-verdict",
        "format_version": 1,
        "purpose": marker["purpose"],
        "created_at_ns": finished_ns,
        "start_marker": marker_binding,
        "informational_envelope": envelope_binding,
        "informational_F": floor["F"], "informational_F64": floor["F64"],
        "outcome": outcome_binding,
        "fixed_limits": FIXED_LIMITS,
        "gates": decision,
        "inputs_unchanged": True,
        "historical_verdicts_unchanged": True,
        "original_T3B_milestone_reassigned": False,
    }
    _revalidate_floor_input_locations(input_bundle, evidence_root=ROOT)
    digest = install_verdict(
        report_path, verdict, [*bindings, marker_binding, staging_binding, outcome_binding])
    print(json.dumps({"gates": decision, "report": str(report_path), "sha256": digest}, indent=2))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
