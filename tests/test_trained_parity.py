"""Prospective trained-checkpoint parity procedure and evidence contracts."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.test_training_self_consistency import (
    _actions,
    _case_identities,
    _input_hashes,
    _runtime_metadata,
    _variant_metadata,
)


FLOOR_CREATED_NS = 1_788_264_000_000_000_000
FLOOR_MTIME_NS = FLOOR_CREATED_NS + 1_000_000
MARKER_CREATED_NS = FLOOR_CREATED_NS + 2_000_000
MARKER_MTIME_NS = FLOOR_CREATED_NS + 3_000_000
COMPARISON_CREATED_NS = FLOOR_CREATED_NS + 4_000_000
COMPARISON_MTIME_NS = FLOOR_CREATED_NS + 5_000_000
EVALUATED_NS = FLOOR_CREATED_NS + 6_000_000
BUNDLE_SHA256 = "f" * 64


def _module():
    return __import__(
        "mlx_smolvla.training.trained_parity",
        fromlist=["evaluate_trained_parity_documents"],
    )


def _utc(value_ns: int) -> str:
    seconds, nanoseconds = divmod(value_ns, 1_000_000_000)
    value = datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(
        microseconds=nanoseconds // 1_000
    )
    return value.isoformat(timespec="microseconds")


def _json_payload(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_json_payload(value)).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_group(files: dict[str, bytes]) -> dict[str, object]:
    digests = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(files.items())
    }
    return {"tree_sha256": _canonical_digest(digests), "files": digests}


def _synthetic_floor_input_evidence(
    floor: dict[str, object],
) -> dict[str, object]:
    inputs = floor["input_sha256"]
    return {
        "checkpoint_export": {
            "mode": "exact_tree",
            "root": floor["checkpoint_path"],
        },
        "evaluation_artifact": {
            "mode": "exact_tree",
            "root": ".cache/training/t3b/evaluation",
        },
        "pinned_dataset": {
            "mode": "named_files",
            "paths": {
                name: f".cache/training/t3b/dataset/{name}"
                for name in inputs["pinned_dataset"]["files"]
            },
        },
        "tokenizer_snapshot": {
            "mode": "contained_symlink_tree",
            "root": ".cache/training/t3b/tokenizer",
            "allowed_root": ".cache",
        },
        "implementation": {
            "mode": "named_files",
            "paths": {
                name: f".cache/training/t3b/implementation/{name}"
                for name in inputs["implementation"]["files"]
            },
        },
    }


def _prospective_floor(
    *,
    floor_value: float = 0.002,
    created_at_ns: int = FLOOR_CREATED_NS,
) -> dict[str, object]:
    floor_module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["assemble_floor_report"],
    )
    actions = {
        item.name: _actions()
        for item in floor_module.perturbation_plan(max_threads=12)
    }
    actions["cpu_float64"][9, 4, 2] = floor_value
    return floor_module.assemble_floor_report(
        actions=actions,
        variant_metadata=_variant_metadata(floor_module),
        case_identities=_case_identities(),
        input_sha256=_input_hashes(),
        checkpoint_path=".cache/training/t3b/export",
        purpose="prospective_gate",
        created_at_utc=_utc(created_at_ns),
        created_at_ns=created_at_ns,
    )


def _write_floor_bundle(
    root: Path,
    *,
    floor_value: float = 0.002,
    case_identities: list[dict[str, int]] | None = None,
    input_hashes: dict[str, object] | None = None,
    checkpoint_path: str = ".cache/training/t3b/export",
) -> tuple[dict[str, object], Path, Path, str]:
    floor_module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=[
            "assemble_floor_report",
            "perturbation_plan",
            "read_variant_artifact",
            "write_floor_report",
            "write_variant_artifact",
        ],
    )
    import numpy as np

    variant_root = root / "variants"
    input_hashes = deepcopy(input_hashes or _input_hashes())
    actions = {}
    metadata = {}
    for item in floor_module.perturbation_plan(max_threads=12):
        dtype = "float64" if item.dtype == "float64" else "float32"
        value = np.zeros((56, 50, 6), dtype=dtype)
        if item.name == floor_module.FLOAT64_VARIANT:
            value[9, 4, 2] = floor_value
        floor_module.write_variant_artifact(
            variant_root / item.name,
            variant=item,
            normalized_actions=value,
            input_combined_sha256=input_hashes["combined_sha256"],
            metadata=_runtime_metadata(item),
        )
        loaded, document = floor_module.read_variant_artifact(
            variant_root / item.name,
            expected_variant=item,
            expected_input_combined_sha256=input_hashes["combined_sha256"],
        )
        actions[item.name] = loaded
        metadata[item.name] = document
    floor = floor_module.assemble_floor_report(
        actions=actions,
        variant_metadata=metadata,
        case_identities=case_identities or _case_identities(),
        input_sha256=input_hashes,
        checkpoint_path=checkpoint_path,
        purpose="prospective_gate",
        created_at_utc=_utc(FLOOR_CREATED_NS),
        created_at_ns=FLOOR_CREATED_NS,
    )
    floor_path = root / "floor.json"
    floor_sha256 = floor_module.write_floor_report(floor_path, floor)
    os.utime(floor_path, ns=(FLOOR_MTIME_NS, FLOOR_MTIME_NS))
    return floor, floor_path, variant_root, floor_sha256


def _mae_samples(
    identities: list[dict[str, int]],
    *,
    total: float,
) -> list[dict[str, object]]:
    return [
        {
            **identity,
            "absolute_error_sum": total if index == 0 else 0.0,
            "element_count": 6,
        }
        for index, identity in enumerate(identities)
    ]


def _base_evaluation(
    module,
    identities: list[dict[str, int]],
    *,
    mae: float = 10.0,
) -> dict[str, object]:
    total = mae * 336
    samples = _mae_samples(identities, total=total)
    total = math.fsum(sample["absolute_error_sum"] for sample in samples)
    return {
        "format_version": 1,
        "artifact_type": "smolvla-lora-base-heldout-evaluation",
        "evaluation_manifest_sha256": __import__(
            "mlx_smolvla._lab.training.t3_contract", fromlist=["FROZEN_EVALUATION_MANIFEST_SHA256"]
        ).FROZEN_EVALUATION_MANIFEST_SHA256,
        "train_statistics_sha256": module.FROZEN_TRAIN_STATISTICS_SHA256,
        "sample_count": 56,
        "element_count": 336,
        "device": "Device(cpu, 0)",
        "dtype": "float32",
        "mlx_mae": total / 336,
        "absolute_error_sum": total,
        "samples": samples,
    }


def _mae_evaluation(
    identities: list[dict[str, int]],
    *,
    framework: str,
    mae: float,
) -> dict[str, object]:
    total = mae * 336
    samples = _mae_samples(identities, total=total)
    total = math.fsum(sample["absolute_error_sum"] for sample in samples)
    return {
        "framework": framework,
        "device": "cpu",
        "dtype": "float32",
        "sample_count": 56,
        "element_count": 336,
        "mae": total / 336,
        "absolute_error_sum": total,
        "samples": samples,
    }


def _parity_evidence(
    identities: list[dict[str, int]],
    *,
    image: float = 1e-5,
    state: float = 1e-6,
    normalized: float = 0.006,
) -> dict[str, object]:
    first = {
        "image_preprocessing_max_abs": image,
        "state_preprocessing_max_abs": state,
        "preprocessing_max_abs": max(image, state),
        "normalized_action_max_abs": normalized,
        "physical_action_max_abs": min(normalized, 0.004),
        "physical_action_standardized_max_abs": min(normalized, 0.003),
    }
    zero = {field: 0.0 for field in first}
    samples = [
        {**identity, **(first if index == 0 else zero)}
        for index, identity in enumerate(identities)
    ]
    summary = {
        field: max(float(sample[field]) for sample in samples)
        for field in first
    }
    return {
        "sample_count": 56,
        **summary,
        "gate_max_abs": max(summary.values()),
        "samples": samples,
    }


def _evidence(
    module,
    identities: list[dict[str, int]],
    *,
    base_mae: float = 10.0,
    fine_mae: float = 9.0,
    torch_mae: float = 8.55,
    image: float = 1e-5,
    state: float = 1e-6,
    normalized: float = 0.006,
    base: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "base_mlx_evaluation": base
        or _base_evaluation(module, identities, mae=base_mae),
        "fine_mlx_evaluation": _mae_evaluation(
            identities, framework="mlx", mae=fine_mae
        ),
        "torch_evaluation": _mae_evaluation(
            identities, framework="torch", mae=torch_mae
        ),
        "stats_active_parity": _parity_evidence(
            identities, image=image, state=state, normalized=normalized
        ),
    }


def _metrics(evidence: dict[str, object]) -> dict[str, float]:
    parity = evidence["stats_active_parity"]
    return {
        "base_mlx_mae": evidence["base_mlx_evaluation"]["mlx_mae"],
        "fine_mlx_mae": evidence["fine_mlx_evaluation"]["mae"],
        "torch_mae": evidence["torch_evaluation"]["mae"],
        "image_preprocessing_max_abs": parity["image_preprocessing_max_abs"],
        "state_preprocessing_max_abs": parity["state_preprocessing_max_abs"],
        "normalized_action_max_abs": parity["normalized_action_max_abs"],
    }


def _marker(
    floor: dict[str, object],
    *,
    floor_sha256: str,
    bundle_sha256: str = BUNDLE_SHA256,
    created_at_ns: int = MARKER_CREATED_NS,
    comparison_path: str = "/tmp/comparison.json",
) -> dict[str, object]:
    module = _module()
    return {
        "format_version": 1,
        "artifact_type": module.START_MARKER_ARTIFACT_TYPE,
        "procedure_id": module.PROCEDURE_ID,
        "created_at_utc": _utc(created_at_ns),
        "created_at_ns": created_at_ns,
        "comparison_path": comparison_path,
        "floor_sha256": floor_sha256,
        "floor_procedure_id": floor["procedure_id"],
        "floor_created_at_ns": floor["created_at_ns"],
        "floor_file_mtime_ns": FLOOR_MTIME_NS,
        "floor_bundle_sha256": bundle_sha256,
        "checkpoint_path": floor["checkpoint_path"],
        "input_combined_sha256": floor["input_sha256"]["combined_sha256"],
    }


def _comparison(
    floor: dict[str, object],
    marker: dict[str, object],
    evidence: dict[str, object],
    *,
    floor_sha256: str,
    marker_sha256: str,
    evidence_files: dict[str, dict[str, str]],
    bundle_sha256: str = BUNDLE_SHA256,
    marker_mtime_ns: int = MARKER_MTIME_NS,
    created_at_ns: int = COMPARISON_CREATED_NS,
    conversion_tensor_count: int = 1,
    conversion_parameter_count: int = 1,
    floor_input_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    module = _module()
    return {
        "format_version": 1,
        "artifact_type": module.COMPARISON_ARTIFACT_TYPE,
        "procedure_id": module.PROCEDURE_ID,
        "created_at_utc": _utc(created_at_ns),
        "created_at_ns": created_at_ns,
        "checkpoint_path": floor["checkpoint_path"],
        "sample_count": floor["sample_count"],
        "normalized_action_chunk_shape": floor["normalized_action_chunk_shape"],
        "floor_binding": {
            "floor_sha256": floor_sha256,
            "floor_procedure_id": floor["procedure_id"],
            "floor_created_at_ns": floor["created_at_ns"],
            "floor_file_mtime_ns": FLOOR_MTIME_NS,
            "input_combined_sha256": floor["input_sha256"]["combined_sha256"],
            "floor_bundle_sha256": bundle_sha256,
        },
        "start_marker_binding": {
            "marker_sha256": marker_sha256,
            "marker_created_at_ns": marker["created_at_ns"],
            "marker_file_mtime_ns": marker_mtime_ns,
            "floor_bundle_sha256": bundle_sha256,
        },
        "source_identity": deepcopy(floor["source_identity"]),
        "input_sha256": deepcopy(floor["input_sha256"]),
        "floor_input_evidence": deepcopy(
            floor_input_evidence or _synthetic_floor_input_evidence(floor)
        ),
        "case_identities": deepcopy(floor["case_identities"]),
        "evidence_files": evidence_files,
        "conversion_validation": {
            "source_model_sha256": floor["input_sha256"]["checkpoint_export"][
                "files"
            ]["model.safetensors"],
            "converted_model_sha256": evidence_files["native_conversion_model"][
                "sha256"
            ],
            "name_map_sha256": evidence_files["native_conversion_name_map"][
                "sha256"
            ],
            "dtype": "float32",
            "tensor_count": conversion_tensor_count,
            "parameter_count": conversion_parameter_count,
        },
        **deepcopy(evidence),
        "metrics": _metrics(evidence),
    }


def _documents(
    monkeypatch: pytest.MonkeyPatch,
    *,
    floor_value: float = 0.002,
    **evidence_overrides,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    module = _module()
    floor = _prospective_floor(floor_value=floor_value)
    floor_sha256 = _digest(floor)
    marker = _marker(floor, floor_sha256=floor_sha256)
    marker_sha256 = _digest(marker)
    evidence = _evidence(module, floor["case_identities"], **evidence_overrides)
    base_sha256 = _digest(evidence["base_mlx_evaluation"])
    monkeypatch.setattr(module, "FROZEN_BASE_REPORT_SHA256", base_sha256)
    evidence_files = {
        "base_report": {"path": "base.json", "sha256": base_sha256},
        "native_conversion_model": {"path": "model.safetensors", "sha256": "a" * 64},
        "native_conversion_name_map": {"path": "name-map.json", "sha256": "b" * 64},
        "comparison_implementation": {"path": "compare.py", "sha256": "c" * 64},
    }
    comparison = _comparison(
        floor,
        marker,
        evidence,
        floor_sha256=floor_sha256,
        marker_sha256=marker_sha256,
        evidence_files=evidence_files,
    )
    arguments = {
        "floor": floor,
        "floor_sha256": floor_sha256,
        "floor_file_mtime_ns": FLOOR_MTIME_NS,
        "floor_bundle_sha256": BUNDLE_SHA256,
        "start_marker": marker,
        "start_marker_sha256": marker_sha256,
        "start_marker_file_mtime_ns": MARKER_MTIME_NS,
        "comparison": comparison,
        "comparison_sha256": _digest(comparison),
        "comparison_file_mtime_ns": COMPARISON_MTIME_NS,
        "evaluated_at_ns": EVALUATED_NS,
    }
    return floor, marker, comparison, arguments


def test_fixed_constants_and_all_inclusive_boundaries_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _, _, _, arguments = _documents(monkeypatch)

    result = module.evaluate_trained_parity_documents(**arguments)

    assert module.IMAGE_PREPROCESSING_MAX_ABS == 1e-5
    assert module.STATE_PREPROCESSING_MAX_ABS == 1e-6
    assert module.FINE_TO_BASE_MAE_RATIO_MAXIMUM == 0.9
    assert module.TORCH_TO_MLX_MAE_RATIO_MINIMUM == 0.95
    assert module.TORCH_TO_MLX_MAE_RATIO_MAXIMUM == 1.05
    assert module.DETERMINISTIC_FALLBACK_MAX_ABS == 0.005
    assert module.REFERENCE_FLOOR_MULTIPLIER == 3.0
    assert result["thresholds"]["normalized_action_max_abs"] == 0.006
    assert result["metrics"]["fine_to_base_mae_ratio"] == 0.9
    assert result["metrics"]["torch_to_mlx_mae_ratio"] >= 0.95
    assert all(result["gates"].values())

    _, _, _, upper = _documents(monkeypatch, torch_mae=9.0 * 1.05)
    assert _module().evaluate_trained_parity_documents(**upper)["gates"][
        "torch_mlx_roundtrip_passed"
    ]


def test_fallback_branch_passes_at_exact_fixed_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, arguments = _documents(
        monkeypatch, floor_value=0.001, normalized=0.005
    )
    result = _module().evaluate_trained_parity_documents(**arguments)
    assert result["thresholds"]["normalized_action_max_abs"] == 0.005
    assert result["gates"]["deterministic_parity_passed"]


def test_derived_deterministic_threshold_cannot_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, arguments = _documents(
        monkeypatch,
        floor_value=sys.float_info.max,
    )
    with pytest.raises(ValueError, match="derived deterministic threshold must be finite"):
        _module().evaluate_trained_parity_documents(**arguments)


@pytest.mark.parametrize(
    "evidence_overrides",
    [
        {"base_mae": 5e-324, "fine_mae": 1.0, "torch_mae": 1.0},
        {"base_mae": 5e-324, "fine_mae": 5e-324, "torch_mae": 1.0},
    ],
)
def test_derived_mae_ratios_cannot_overflow(
    monkeypatch: pytest.MonkeyPatch,
    evidence_overrides: dict[str, float],
) -> None:
    _, _, _, arguments = _documents(monkeypatch, **evidence_overrides)
    with pytest.raises(ValueError, match="derived MAE ratios must be finite"):
        _module().evaluate_trained_parity_documents(**arguments)


@pytest.mark.parametrize(
    ("overrides", "gate"),
    [
        ({"image": math.nextafter(1e-5, math.inf)}, "image_preprocessing_passed"),
        ({"state": math.nextafter(1e-6, math.inf)}, "state_preprocessing_passed"),
        ({"fine_mae": math.nextafter(9.0, math.inf)}, "heldout_improvement_passed"),
        ({"torch_mae": 9.0 * math.nextafter(0.95, 0.0)}, "torch_mlx_roundtrip_passed"),
        ({"torch_mae": 9.0 * math.nextafter(1.05, math.inf)}, "torch_mlx_roundtrip_passed"),
        ({"normalized": math.nextafter(0.006, math.inf)}, "deterministic_parity_passed"),
    ],
)
def test_each_gate_fails_immediately_beyond_its_boundary(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, float],
    gate: str,
) -> None:
    _, _, _, arguments = _documents(monkeypatch, **overrides)
    result = _module().evaluate_trained_parity_documents(**arguments)
    assert not result["gates"][gate]
    assert not result["gates"]["passed"]


def test_chronology_rejection_precedes_bad_metric_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, marker, comparison, arguments = _documents(monkeypatch)
    marker["created_at_ns"] = FLOOR_MTIME_NS
    marker["created_at_utc"] = _utc(FLOOR_MTIME_NS)
    comparison["metrics"]["normalized_action_max_abs"] = math.nan
    arguments["start_marker"] = marker
    arguments["start_marker_sha256"] = _digest(marker)

    with pytest.raises(ValueError, match="strictly follow"):
        _module().evaluate_trained_parity_documents(**arguments)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.__setitem__(
                "checkpoint_path", ".cache/training/other/export"
            ),
            "checkpoint",
        ),
        (
            lambda value: value["source_identity"].__setitem__(
                "checkpoint_role", "other"
            ),
            "source identity",
        ),
        (
            lambda value: value["input_sha256"]["checkpoint_export"]["files"].__setitem__(
                "model.safetensors", "d" * 64
            ),
            "input hashes",
        ),
        (
            lambda value: value["case_identities"][0].__setitem__(
                "frame_index", 999
            ),
            "case identities",
        ),
        (
            lambda value: value["floor_binding"].__setitem__(
                "floor_procedure_id", "other"
            ),
            "floor binding",
        ),
        (
            lambda value: value["start_marker_binding"].__setitem__(
                "marker_sha256", "e" * 64
            ),
            "marker binding",
        ),
    ],
)
def test_comparison_binds_exact_floor_marker_population_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    _, _, comparison, arguments = _documents(monkeypatch)
    mutation(comparison)
    arguments["comparison"] = comparison
    arguments["comparison_sha256"] = _digest(comparison)
    with pytest.raises(ValueError, match=message):
        _module().evaluate_trained_parity_documents(**arguments)


def test_scalar_summary_cannot_replace_or_hide_per_case_mae_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, comparison, arguments = _documents(monkeypatch)
    comparison["fine_mlx_evaluation"]["samples"][0]["absolute_error_sum"] += 1.0
    arguments["comparison"] = comparison
    arguments["comparison_sha256"] = _digest(comparison)
    with pytest.raises(ValueError, match="aggregate differs from sample evidence"):
        _module().evaluate_trained_parity_documents(**arguments)


def test_parity_summary_is_recomputed_from_all_56_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, comparison, arguments = _documents(monkeypatch)
    comparison["stats_active_parity"]["samples"][17][
        "normalized_action_max_abs"
    ] = 0.5
    arguments["comparison"] = comparison
    arguments["comparison_sha256"] = _digest(comparison)
    with pytest.raises(ValueError, match="summary differs from sample evidence"):
        _module().evaluate_trained_parity_documents(**arguments)


def test_frozen_base_report_hash_is_not_an_opaque_replaceable_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, comparison, arguments = _documents(monkeypatch)
    comparison["evidence_files"]["base_report"]["sha256"] = "0" * 64
    arguments["comparison"] = comparison
    arguments["comparison_sha256"] = _digest(comparison)
    with pytest.raises(ValueError, match="pre-training frozen base report"):
        _module().evaluate_trained_parity_documents(**arguments)


def test_frozen_base_report_body_must_match_its_bound_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, comparison, arguments = _documents(monkeypatch)
    samples = comparison["base_mlx_evaluation"]["samples"]
    samples[0]["absolute_error_sum"] -= 1.0
    samples[1]["absolute_error_sum"] += 1.0
    arguments["comparison"] = comparison
    arguments["comparison_sha256"] = _digest(comparison)
    with pytest.raises(ValueError, match="base report body digest"):
        _module().evaluate_trained_parity_documents(**arguments)


def test_native_conversion_source_must_be_the_floor_bound_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    floor, _, comparison, arguments = _documents(monkeypatch)
    comparison["conversion_validation"]["source_model_sha256"] = "0" * 64
    assert floor["input_sha256"]["checkpoint_export"]["files"][
        "model.safetensors"
    ] != "0" * 64
    arguments["comparison"] = comparison
    arguments["comparison_sha256"] = _digest(comparison)
    with pytest.raises(ValueError, match="conversion source model differs"):
        _module().evaluate_trained_parity_documents(**arguments)


def test_floor_input_evidence_named_inventory_is_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, comparison, arguments = _documents(monkeypatch)
    comparison["floor_input_evidence"]["implementation"]["paths"] = {
        "different.py": ".cache/training/t3b/implementation/different.py"
    }
    arguments["comparison"] = comparison
    arguments["comparison_sha256"] = _digest(comparison)
    with pytest.raises(ValueError, match="paths differ from the hashed inventory"):
        _module().evaluate_trained_parity_documents(**arguments)


def test_persisted_result_recomputes_evidence_thresholds_and_verdicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _, _, _, arguments = _documents(monkeypatch)
    result = module.evaluate_trained_parity_documents(**arguments)
    module.validate_trained_parity_report(result)

    changed = deepcopy(result)
    changed["evidence"]["torch_evaluation"]["samples"][0]["absolute_error_sum"] += 1
    with pytest.raises(ValueError, match="aggregate differs from sample evidence"):
        module.validate_trained_parity_report(changed)

    changed = deepcopy(result)
    base_samples = changed["evidence"]["base_mlx_evaluation"]["samples"]
    base_samples[0]["absolute_error_sum"] -= 1.0
    base_samples[1]["absolute_error_sum"] += 1.0
    with pytest.raises(ValueError, match="base report body digest"):
        module.validate_trained_parity_report(changed)

    changed = deepcopy(result)
    changed["thresholds"]["reference_floor_multiplier"] = 4.0
    with pytest.raises(ValueError, match="fixed procedure"):
        module.validate_trained_parity_report(changed)

    changed = deepcopy(result)
    changed["gates"]["passed"] = False
    with pytest.raises(ValueError, match="decisions are inconsistent"):
        module.validate_trained_parity_report(changed)


def test_raw_variant_bundle_defeats_a_coherent_floor_json_tamper(
    tmp_path: Path,
) -> None:
    module = _module()
    floor, _, variant_root, _ = _write_floor_bundle(tmp_path)
    changed = deepcopy(floor)
    variant = changed["variants"]["cpu_float64"]
    variant["case_max_abs"] = [0.0] * 56
    variant["max_abs_vs_baseline"] = 0.0
    variant["worst_case"] = {**changed["case_identities"][0], "max_abs_vs_baseline": 0.0}
    changed["F"] = 0.0
    changed["F64"] = 0.0
    with pytest.raises(ValueError, match="raw variant"):
        module.validate_floor_bundle(changed, variant_root=variant_root)


def test_raw_variant_bundle_rejects_a_symlinked_worker_directory(
    tmp_path: Path,
) -> None:
    module = _module()
    floor, _, variant_root, _ = _write_floor_bundle(tmp_path)
    worker = variant_root / "cpu_fp32_threads_1"
    real_worker = variant_root / "cpu_fp32_threads_1-real"
    worker.rename(real_worker)
    worker.symlink_to(real_worker.name, target_is_directory=True)

    with pytest.raises(FileNotFoundError, match="raw variant.*unsafe"):
        module.validate_floor_bundle(floor, variant_root=variant_root)


def test_start_marker_is_real_clocked_floor_bound_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _, floor_path, variant_root, floor_sha256 = _write_floor_bundle(tmp_path)
    marker_path = tmp_path / "comparison-start.json"
    comparison_path = tmp_path / "comparison.json"
    monkeypatch.setattr(module.time, "time_ns", lambda: MARKER_CREATED_NS)

    marker, marker_sha256 = module.create_comparison_start_marker(
        floor_path=floor_path,
        variant_root=variant_root,
        output_path=marker_path,
        comparison_path=comparison_path,
    )

    assert marker["created_at_ns"] == MARKER_CREATED_NS
    assert marker["floor_sha256"] == floor_sha256
    assert marker["comparison_path"] == str(comparison_path.resolve())
    assert hashlib.sha256(marker_path.read_bytes()).hexdigest() == marker_sha256
    with pytest.raises(FileExistsError, match="already exists"):
        module.create_comparison_start_marker(
            floor_path=floor_path,
            variant_root=variant_root,
            output_path=marker_path,
            comparison_path=comparison_path,
        )


def test_start_marker_refuses_a_preexisting_comparison_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _, floor_path, variant_root, _ = _write_floor_bundle(tmp_path)
    marker_path = tmp_path / "comparison-start.json"
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_bytes(b"post-hoc comparison\n")
    monkeypatch.setattr(module.time, "time_ns", lambda: MARKER_CREATED_NS)

    with pytest.raises(FileExistsError, match="comparison target already exists"):
        module.create_comparison_start_marker(
            floor_path=floor_path,
            variant_root=variant_root,
            output_path=marker_path,
            comparison_path=comparison_path,
        )
    assert not marker_path.exists()


def test_start_marker_refuses_a_comparison_created_during_floor_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _, floor_path, variant_root, _ = _write_floor_bundle(tmp_path)
    marker_path = tmp_path / "comparison-start.json"
    comparison_path = tmp_path / "comparison.json"
    original = module._revalidate_snapshots

    def race(snapshots):
        original(snapshots)
        comparison_path.write_bytes(b"concurrent comparison\n")

    monkeypatch.setattr(module, "_revalidate_snapshots", race)
    monkeypatch.setattr(module.time, "time_ns", lambda: MARKER_CREATED_NS)
    with pytest.raises(FileExistsError, match="comparison target already exists"):
        module.create_comparison_start_marker(
            floor_path=floor_path,
            variant_root=variant_root,
            output_path=marker_path,
            comparison_path=comparison_path,
        )
    assert not marker_path.exists()


def test_start_marker_outputs_cannot_overlap_raw_floor_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _, floor_path, variant_root, _ = _write_floor_bundle(tmp_path)
    monkeypatch.setattr(module.time, "time_ns", lambda: MARKER_CREATED_NS)
    with pytest.raises(ValueError, match="overlaps protected raw floor variant tree"):
        module.create_comparison_start_marker(
            floor_path=floor_path,
            variant_root=variant_root,
            output_path=variant_root / "comparison-start.json",
            comparison_path=tmp_path / "comparison.json",
        )


def test_start_marker_target_cannot_overlap_checkpoint_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _, floor_path, variant_root, _ = _write_floor_bundle(tmp_path)
    checkpoint_root = tmp_path / ".cache" / "training" / "t3b" / "export"
    checkpoint_root.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.time, "time_ns", lambda: MARKER_CREATED_NS)
    with pytest.raises(ValueError, match="overlaps protected checkpoint export tree"):
        module.create_comparison_start_marker(
            floor_path=floor_path,
            variant_root=variant_root,
            output_path=tmp_path / "comparison-start.json",
            comparison_path=checkpoint_root / "comparison.json",
        )


def test_atomic_report_install_cannot_clobber_a_concurrent_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    output = tmp_path / "winner.json"
    original_link = module.os.link

    def racing_link(source, destination):
        output.write_bytes(b"concurrent winner\n")
        return original_link(source, destination)

    monkeypatch.setattr(module.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        module._atomic_json_no_clobber(output, {"value": 1})
    assert output.read_bytes() == b"concurrent winner\n"


def _file_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest_tensor_count: int = 2,
    manifest_parameter_count: int = 60,
) -> dict[str, object]:
    module = _module()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    export_dir = evidence_root / ".cache" / "training" / "t3b" / "export"
    export_dir.mkdir(parents=True)
    import mlx.core as mx
    from mlx_smolvla._lab.reference.discovery import CHECKPOINT_ID, CHECKPOINT_REVISION
    from mlx_smolvla.convert import convert_checkpoint

    mx.save_safetensors(
        str(export_dir / "model.safetensors"),
        {
            "model.state_proj.weight": mx.arange(12, dtype=mx.float32).reshape(3, 4),
            (
                "model.vlm_with_expert.vlm.model.vision_model.embeddings."
                "patch_embedding.weight"
            ): mx.arange(48, dtype=mx.float32).reshape(2, 3, 2, 4),
        },
    )
    model_sha256 = hashlib.sha256(
        (export_dir / "model.safetensors").read_bytes()
    ).hexdigest()
    training_manifest = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-merged-training-checkpoint",
        "dtype": "float32",
        "tensor_count": manifest_tensor_count,
        "parameter_count": manifest_parameter_count,
        "source_checkpoint": {
            "repo_id": CHECKPOINT_ID,
            "revision": CHECKPOINT_REVISION,
        },
        "metadata": {"run": "model-free-test"},
        "file_sha256": {"model.safetensors": model_sha256},
    }
    (export_dir / "training_manifest.json").write_bytes(
        _json_payload(training_manifest)
    )
    conversion = convert_checkpoint(
        export_dir, evidence_root / "converted", dtype="float32"
    )
    input_root = evidence_root / "floor-inputs"
    tree_payloads = {
        "evaluation_artifact": {
            "manifest.json": b'{"artifact":"heldout-cases-and-noise"}\n',
            "noise.bin": b"frozen-noise\n",
        },
        "tokenizer_snapshot": {
            "tokenizer.json": b'{"tokenizer":"pinned"}\n',
        },
    }
    named_payloads = {
        "pinned_dataset": {"meta/info.json": b'{"dataset":"pinned"}\n'},
        "implementation": {
            "training/self_consistency.py": b"# frozen floor implementation\n"
        },
    }
    for group, files in {**tree_payloads, **named_payloads}.items():
        for relative, payload in files.items():
            path = input_root / group / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    checkpoint_payloads = {
        path.relative_to(export_dir).as_posix(): path.read_bytes()
        for path in sorted(export_dir.rglob("*"))
        if path.is_file()
    }
    input_hashes = {
        "checkpoint_export": _hash_group(checkpoint_payloads),
        **{group: _hash_group(files) for group, files in tree_payloads.items()},
        **{group: _hash_group(files) for group, files in named_payloads.items()},
    }
    input_hashes["combined_sha256"] = _canonical_digest(
        {
            name: input_hashes[name]
            for name in sorted(input_hashes)
            if name != "combined_sha256"
        }
    )
    actual_base_path = Path(".cache/training/t3-base-evaluation.json")
    base_payload = actual_base_path.read_bytes()
    base = json.loads(base_payload)
    actual_identities = [
        {field: sample[field] for field in ("ordinal", "episode", "frame_index", "absolute_index")}
        for sample in base["samples"]
    ]
    floor, floor_path, variant_root, floor_sha256 = _write_floor_bundle(
        tmp_path,
        case_identities=actual_identities,
        input_hashes=input_hashes,
        checkpoint_path=".cache/training/t3b/export",
    )
    base_path = evidence_root / "base.json"
    base_path.write_bytes(base_payload)
    assert hashlib.sha256(base_payload).hexdigest() == module.FROZEN_BASE_REPORT_SHA256
    source_payloads = {
        "native_conversion_model": (
            conversion.output_path.relative_to(evidence_root).as_posix(),
            conversion.output_path.read_bytes(),
        ),
        "native_conversion_name_map": (
            conversion.name_map_path.relative_to(evidence_root).as_posix(),
            conversion.name_map_path.read_bytes(),
        ),
        "comparison_implementation": ("compare.py", b"# frozen comparison producer\n"),
    }
    evidence_files = {
        "base_report": {
            "path": "base.json",
            "sha256": hashlib.sha256(base_payload).hexdigest(),
        }
    }
    for name, (filename, payload) in source_payloads.items():
        path = evidence_root / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        evidence_files[name] = {
            "path": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    floor_input_evidence = {
        "checkpoint_export": {
            "mode": "exact_tree",
            "root": export_dir.relative_to(evidence_root).as_posix(),
        },
        "evaluation_artifact": {
            "mode": "exact_tree",
            "root": (input_root / "evaluation_artifact")
            .relative_to(evidence_root)
            .as_posix(),
        },
        "pinned_dataset": {
            "mode": "named_files",
            "paths": {
                name: (input_root / "pinned_dataset" / name)
                .relative_to(evidence_root)
                .as_posix()
                for name in named_payloads["pinned_dataset"]
            },
        },
        "tokenizer_snapshot": {
            "mode": "contained_symlink_tree",
            "root": (input_root / "tokenizer_snapshot")
            .relative_to(evidence_root)
            .as_posix(),
            "allowed_root": input_root.relative_to(evidence_root).as_posix(),
        },
        "implementation": {
            "mode": "named_files",
            "paths": {
                name: (input_root / "implementation" / name)
                .relative_to(evidence_root)
                .as_posix()
                for name in named_payloads["implementation"]
            },
        },
    }

    comparison_path = tmp_path / "comparison.json"
    marker_path = tmp_path / "comparison-start.json"
    monkeypatch.setattr(module.time, "time_ns", lambda: MARKER_CREATED_NS)
    marker, marker_sha256 = module.create_comparison_start_marker(
        floor_path=floor_path,
        variant_root=variant_root,
        output_path=marker_path,
        comparison_path=comparison_path,
    )
    os.utime(marker_path, ns=(MARKER_MTIME_NS, MARKER_MTIME_NS))
    evidence = _evidence(
        module,
        floor["case_identities"],
        base=base,
        fine_mae=float(base["mlx_mae"]) * 0.8,
        torch_mae=float(base["mlx_mae"]) * 0.8,
    )
    bundle_sha256 = marker["floor_bundle_sha256"]
    comparison = _comparison(
        floor,
        marker,
        evidence,
        floor_sha256=floor_sha256,
        marker_sha256=marker_sha256,
        marker_mtime_ns=MARKER_MTIME_NS,
        evidence_files=evidence_files,
        bundle_sha256=bundle_sha256,
        conversion_tensor_count=2,
        conversion_parameter_count=60,
        floor_input_evidence=floor_input_evidence,
    )
    comparison_path.write_bytes(_json_payload(comparison))
    os.utime(comparison_path, ns=(COMPARISON_MTIME_NS, COMPARISON_MTIME_NS))
    return {
        "module": module,
        "floor_path": floor_path,
        "variant_root": variant_root,
        "marker_path": marker_path,
        "comparison_path": comparison_path,
        "comparison": comparison,
        "export_dir": export_dir,
        "input_root": input_root,
        "evidence_root": evidence_root,
        "output_path": tmp_path / "evaluation.json",
    }


def _file_arguments(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "floor_path": fixture["floor_path"],
        "variant_root": fixture["variant_root"],
        "start_marker_path": fixture["marker_path"],
        "comparison_path": fixture["comparison_path"],
        "output_path": fixture["output_path"],
        "evidence_root": fixture["evidence_root"],
        "evaluated_at_ns": EVALUATED_NS,
    }


def _rewrite_comparison(fixture: dict[str, object]) -> None:
    fixture["comparison_path"].write_bytes(_json_payload(fixture["comparison"]))
    os.utime(
        fixture["comparison_path"],
        ns=(COMPARISON_MTIME_NS, COMPARISON_MTIME_NS),
    )


def test_file_evaluator_verifies_all_files_and_writes_auditable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    module = fixture["module"]
    report, digest = module.evaluate_trained_parity_files(**_file_arguments(fixture))
    output = fixture["output_path"]
    assert report["gates"]["passed"]
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert json.loads(output.read_bytes()) == report
    module.validate_trained_parity_report(report)
    with pytest.raises(FileExistsError, match="already exists"):
        module.evaluate_trained_parity_files(**_file_arguments(fixture))


@pytest.mark.parametrize(
    ("protected_group", "relative_output"),
    [
        ("checkpoint_export", "parity-evaluation.json"),
        ("evaluation_artifact", "parity-evaluation.json"),
        ("tokenizer_snapshot", "parity-evaluation.json"),
        ("raw_variants", "parity-evaluation.json"),
    ],
)
def test_parity_output_cannot_overlap_floor_bound_input_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_group: str,
    relative_output: str,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    if protected_group == "checkpoint_export":
        root = fixture["export_dir"]
    elif protected_group == "raw_variants":
        root = fixture["variant_root"]
    else:
        root = fixture["input_root"] / protected_group
    arguments = _file_arguments(fixture)
    arguments["output_path"] = root / relative_output

    with pytest.raises(ValueError, match="parity output overlaps protected"):
        fixture["module"].evaluate_trained_parity_files(**arguments)
    assert not arguments["output_path"].exists()


def test_floor_bound_checkpoint_export_rejects_extra_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    (fixture["export_dir"] / "unexpected-index.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checkpoint export inventory differs"):
        fixture["module"].evaluate_trained_parity_files(**_file_arguments(fixture))
    assert not fixture["output_path"].exists()


def test_floor_bound_checkpoint_source_file_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    (fixture["export_dir"] / "model.safetensors").write_bytes(b"replacement\n")
    with pytest.raises(ValueError, match="checkpoint_export input differs"):
        fixture["module"].evaluate_trained_parity_files(**_file_arguments(fixture))
    assert not fixture["output_path"].exists()


def test_floor_bound_evaluation_artifact_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    (fixture["input_root"] / "evaluation_artifact" / "noise.bin").write_bytes(
        b"different-noise\n"
    )
    with pytest.raises(ValueError, match="evaluation_artifact input differs"):
        fixture["module"].evaluate_trained_parity_files(**_file_arguments(fixture))
    assert not fixture["output_path"].exists()


def test_floor_bound_named_implementation_input_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    path = (
        fixture["input_root"]
        / "implementation"
        / "training"
        / "self_consistency.py"
    )
    path.write_bytes(b"# replacement implementation\n")
    with pytest.raises(ValueError, match="implementation input differs"):
        fixture["module"].evaluate_trained_parity_files(**_file_arguments(fixture))
    assert not fixture["output_path"].exists()


def test_contained_hugging_face_style_tokenizer_symlink_is_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    tokenizer = fixture["input_root"] / "tokenizer_snapshot" / "tokenizer.json"
    payload = tokenizer.read_bytes()
    blob = fixture["input_root"] / "blobs" / hashlib.sha256(payload).hexdigest()
    blob.parent.mkdir()
    blob.write_bytes(payload)
    tokenizer.unlink()
    tokenizer.symlink_to(Path("..") / "blobs" / blob.name)

    report, _ = fixture["module"].evaluate_trained_parity_files(
        **_file_arguments(fixture)
    )
    assert report["gates"]["passed"]


def test_tokenizer_symlink_must_remain_within_its_allowed_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    tokenizer = fixture["input_root"] / "tokenizer_snapshot" / "tokenizer.json"
    outside = fixture["evidence_root"] / "outside-tokenizer.json"
    outside.write_bytes(tokenizer.read_bytes())
    tokenizer.unlink()
    tokenizer.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink escapes its allowed root"):
        fixture["module"].evaluate_trained_parity_files(**_file_arguments(fixture))
    assert not fixture["output_path"].exists()


def test_training_manifest_inventory_counts_bind_native_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(
        tmp_path,
        monkeypatch,
        manifest_tensor_count=3,
        manifest_parameter_count=61,
    )
    with pytest.raises(ValueError, match="training manifest inventory differs"):
        fixture["module"].evaluate_trained_parity_files(**_file_arguments(fixture))
    assert not fixture["output_path"].exists()


def test_converted_tensor_tamper_cannot_be_blessed_by_updated_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    import mlx.core as mx

    comparison = fixture["comparison"]
    recorded = comparison["evidence_files"]["native_conversion_model"]
    converted_path = fixture["evidence_root"] / recorded["path"]
    tensors = mx.load(str(converted_path))
    mx.eval(*tensors.values())
    name = sorted(tensors)[0]
    tensors[name] = tensors[name] + mx.ones_like(tensors[name])
    mx.eval(*tensors.values())
    replacement = converted_path.with_name("replacement.safetensors")
    mx.save_safetensors(str(replacement), tensors)
    replacement.replace(converted_path)
    changed_sha256 = hashlib.sha256(converted_path.read_bytes()).hexdigest()
    recorded["sha256"] = changed_sha256
    comparison["conversion_validation"]["converted_model_sha256"] = changed_sha256
    _rewrite_comparison(fixture)

    with pytest.raises(ValueError, match="converted tensor differs from source export"):
        fixture["module"].evaluate_trained_parity_files(**_file_arguments(fixture))
    assert not fixture["output_path"].exists()


def test_semantic_conversion_uses_private_descriptor_snapshot_copies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    import mlx_smolvla.convert as conversion_module

    comparison = fixture["comparison"]
    original_paths = {
        fixture["export_dir"] / "model.safetensors",
        fixture["evidence_root"]
        / comparison["evidence_files"]["native_conversion_model"]["path"],
        fixture["evidence_root"]
        / comparison["evidence_files"]["native_conversion_name_map"]["path"],
    }
    original_validator = conversion_module.validate_converted_checkpoint

    def race_original_paths(source_dir, converted_path, name_map_path, *, dtype):
        validator_paths = {
            Path(source_dir) / "model.safetensors",
            Path(converted_path),
            Path(name_map_path),
        }
        assert validator_paths.isdisjoint(original_paths)
        captured = {
            path: (path.read_bytes(), path.stat().st_atime_ns, path.stat().st_mtime_ns)
            for path in original_paths
        }
        for path in original_paths:
            path.write_bytes(b"racing alternate bundle\n")
        try:
            return original_validator(
                source_dir,
                converted_path,
                name_map_path,
                dtype=dtype,
            )
        finally:
            for path, (payload, atime_ns, mtime_ns) in captured.items():
                path.write_bytes(payload)
                os.utime(path, ns=(atime_ns, mtime_ns))

    monkeypatch.setattr(
        conversion_module,
        "validate_converted_checkpoint",
        race_original_paths,
    )
    report, _ = fixture["module"].evaluate_trained_parity_files(
        **_file_arguments(fixture)
    )
    assert report["gates"]["passed"]


def test_actual_source_file_digest_is_verified_not_copied_opaquely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    (fixture["evidence_root"] / "compare.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="actual file"):
        fixture["module"].evaluate_trained_parity_files(**_file_arguments(fixture))
    assert not fixture["output_path"].exists()


def test_recorded_evidence_path_and_ancestors_must_not_be_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    evidence_root = fixture["evidence_root"]
    implementation = evidence_root / "compare.py"
    real_implementation = evidence_root / "compare-real.py"
    implementation.rename(real_implementation)
    implementation.symlink_to(real_implementation.name)

    with pytest.raises(ValueError, match="evidence path is symlinked"):
        fixture["module"].evaluate_trained_parity_files(**_file_arguments(fixture))
    assert not fixture["output_path"].exists()


def test_input_replacement_between_validation_and_install_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    module = fixture["module"]
    original = module._revalidate_snapshots

    def replace_then_validate(snapshots):
        recorded = fixture["comparison"]["evidence_files"][
            "native_conversion_name_map"
        ]["path"]
        (fixture["evidence_root"] / recorded).write_bytes(b"replacement\n")
        return original(snapshots)

    monkeypatch.setattr(module, "_revalidate_snapshots", replace_then_validate)
    with pytest.raises(RuntimeError, match="changed before report install"):
        module.evaluate_trained_parity_files(**_file_arguments(fixture))
    assert not fixture["output_path"].exists()


def test_touched_or_reused_start_marker_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    module = fixture["module"]
    os.utime(
        fixture["marker_path"],
        ns=(MARKER_MTIME_NS + 1, MARKER_MTIME_NS + 1),
    )
    with pytest.raises(ValueError, match="marker binding"):
        module.evaluate_trained_parity_files(**_file_arguments(fixture))

    second = tmp_path / "second-comparison.json"
    second.write_bytes(fixture["comparison_path"].read_bytes())
    os.utime(second, ns=(COMPARISON_MTIME_NS, COMPARISON_MTIME_NS))
    arguments = _file_arguments(fixture)
    arguments["comparison_path"] = second
    with pytest.raises(ValueError, match="different comparison path"):
        module.evaluate_trained_parity_files(**arguments)


def test_cli_evaluates_complete_file_bound_evidence_without_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _file_fixture(tmp_path, monkeypatch)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_trained_parity.py",
            "--floor",
            str(fixture["floor_path"]),
            "--variants",
            str(fixture["variant_root"]),
            "--start-marker",
            str(fixture["marker_path"]),
            "--comparison",
            str(fixture["comparison_path"]),
            "--evidence-root",
            str(fixture["evidence_root"]),
            "--output",
            str(fixture["output_path"]),
            "--evaluated-at-ns",
            str(EVALUATED_NS),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["passed"] is True
    assert hashlib.sha256(fixture["output_path"].read_bytes()).hexdigest() == summary[
        "report_sha256"
    ]
