"""Floor-first producer tests for the trained-checkpoint comparison artifact."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from tests.test_trained_parity import (
    BUNDLE_SHA256,
    COMPARISON_CREATED_NS,
    FLOOR_MTIME_NS,
    MARKER_MTIME_NS,
    _digest,
    _evidence,
    _marker,
    _module as parity_module,
    _prospective_floor,
    _synthetic_floor_input_evidence,
    _write_floor_bundle,
)


def test_producer_assembles_the_exact_frozen_comparison_schema() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.evaluation",
        fromlist=["assemble_trained_comparison_report"],
    )
    parity = parity_module()
    floor = _prospective_floor()
    floor_sha256 = _digest(floor)
    marker = _marker(floor, floor_sha256=floor_sha256)
    marker_sha256 = _digest(marker)
    evidence = _evidence(parity, floor["case_identities"])
    evidence_files = {
        "base_report": {
            "path": ".cache/training/t3-base-evaluation.json",
            "sha256": parity.FROZEN_BASE_REPORT_SHA256,
        },
        "native_conversion_model": {
            "path": ".cache/mlx_smolvla/converted/model.float32.safetensors",
            "sha256": "a" * 64,
        },
        "native_conversion_name_map": {
            "path": ".cache/mlx_smolvla/converted/name_map.json",
            "sha256": "b" * 64,
        },
        "comparison_implementation": {
            "path": "training/evaluation.py",
            "sha256": "c" * 64,
        },
    }
    conversion = {
        "source_model_sha256": floor["input_sha256"]["checkpoint_export"][
            "files"
        ]["model.safetensors"],
        "converted_model_sha256": "a" * 64,
        "name_map_sha256": "b" * 64,
        "dtype": "float32",
        "tensor_count": 500,
        "parameter_count": 450_046_176,
    }

    comparison = module.assemble_trained_comparison_report(
        floor=floor,
        floor_sha256=floor_sha256,
        floor_file_mtime_ns=FLOOR_MTIME_NS,
        floor_bundle_sha256=BUNDLE_SHA256,
        start_marker=marker,
        start_marker_sha256=marker_sha256,
        start_marker_file_mtime_ns=MARKER_MTIME_NS,
        floor_input_evidence=_synthetic_floor_input_evidence(floor),
        evidence_files=evidence_files,
        conversion_validation=conversion,
        base_mlx_evaluation=evidence["base_mlx_evaluation"],
        fine_mlx_evaluation=evidence["fine_mlx_evaluation"],
        torch_evaluation=evidence["torch_evaluation"],
        stats_active_parity=evidence["stats_active_parity"],
        created_at_ns=COMPARISON_CREATED_NS,
    )

    validated, created_at_ns, _ = parity._validate_comparison(comparison)
    assert validated == comparison
    assert created_at_ns == COMPARISON_CREATED_NS
    assert comparison["metrics"] == {
        "base_mlx_mae": 10.0,
        "fine_mlx_mae": 9.0,
        "torch_mae": 8.55,
        "image_preprocessing_max_abs": 1e-5,
        "state_preprocessing_max_abs": 1e-6,
        "normalized_action_max_abs": 0.006,
    }


def test_producer_validates_the_real_floor_bundle_and_one_shot_marker(
    tmp_path: Path,
) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.evaluation",
        fromlist=["validate_trained_comparison_start_files"],
    )
    parity = parity_module()
    floor, floor_path, variant_root, floor_sha256 = _write_floor_bundle(tmp_path)
    comparison_path = tmp_path / "comparison.json"
    marker_path = tmp_path / "comparison-start.json"
    marker, marker_sha256 = parity.create_comparison_start_marker(
        floor_path=floor_path,
        variant_root=variant_root,
        output_path=marker_path,
        comparison_path=comparison_path,
    )

    start = module.validate_trained_comparison_start_files(
        floor_path=floor_path,
        variant_root=variant_root,
        start_marker_path=marker_path,
        comparison_path=comparison_path,
    )

    assert start.floor == floor
    assert start.floor_sha256 == floor_sha256
    assert start.start_marker == marker
    assert start.start_marker_sha256 == marker_sha256
    assert start.floor_file_mtime_ns < marker["created_at_ns"]


def test_invalid_floor_or_marker_stops_before_any_model_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.evaluation",
        fromlist=["run_trained_comparison_evaluation"],
    )
    evaluated = False

    def reject_start(**_kwargs):
        raise ValueError("comparison marker floor SHA-256 differs from the floor file")

    def forbidden_evaluation(**_kwargs):
        nonlocal evaluated
        evaluated = True
        raise AssertionError("model evaluation ran before the floor/marker check")

    monkeypatch.setattr(module, "validate_trained_comparison_start_files", reject_start)
    monkeypatch.setattr(module, "run_finetune_outcome_evaluation", forbidden_evaluation)

    with pytest.raises(ValueError, match="marker floor SHA-256"):
        module.run_trained_comparison_evaluation(
            floor_path=Path(".cache/training/t3b/floor.json"),
            variant_root=Path(".cache/training/t3b/self-consistency/variants"),
            start_marker_path=Path(".cache/training/t3b/comparison-start.json"),
            comparison_path=Path(".cache/training/t3b/comparison.json"),
            outcome_path=Path(".cache/training/t3b/outcome.json"),
            cache_dir=Path(".cache/hf"),
            native_cache=Path(".cache/mlx_smolvla/policy-float32"),
            run_dir=Path(".cache/training/t3b"),
            evaluation_dir=Path(".cache/training/t3-evaluation"),
            base_report_path=Path(".cache/training/t3-base-evaluation.json"),
        )
    assert not evaluated


def test_comparison_producer_cli_is_available_without_loading_a_checkpoint() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/produce_trained_comparison.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--start-marker" in completed.stdout
    assert "--outcome" in completed.stdout
