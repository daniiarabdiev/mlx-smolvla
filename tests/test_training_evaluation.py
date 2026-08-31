"""Frozen held-out evidence and immutable Stage T3 outcome-gate contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from training.data import TrainingArtifact


_EVALUATION_DIR = Path(".cache/training/t3-evaluation")
_BASE_REPORT = Path(".cache/training/t3-base-evaluation.json")


def test_t3_outcome_thresholds_are_immutable() -> None:
    module = __import__("training.evaluation", fromlist=["MAE_IMPROVEMENT_RATIO_MAXIMUM"])

    assert module.MAE_IMPROVEMENT_RATIO_MAXIMUM == 0.9
    assert module.TORCH_MLX_MAE_RATIO_MINIMUM == 0.95
    assert module.TORCH_MLX_MAE_RATIO_MAXIMUM == 1.05
    assert module.INFERENCE_MAX_ABSOLUTE_TOLERANCE == 5e-3
    assert module.EVALUATION_SAMPLE_COUNT == 56


def test_frozen_evaluation_artifact_is_complete_integral_and_unseen() -> None:
    module = __import__("training.evaluation", fromlist=["load_evaluation_cases"])
    artifact = TrainingArtifact(_EVALUATION_DIR)
    names = artifact.verify_all()
    metadata = artifact.metadata
    cases = module.load_evaluation_cases(_EVALUATION_DIR)

    assert metadata["artifact_type"] == "smolvla-lora-heldout-evaluation"
    assert metadata["sample_count"] == 56
    assert metadata["noise_seed"] == 20260902
    assert metadata["heldout_episodes"] == [2, 7, 21, 28, 31, 34, 35, 41]
    assert metadata["train_statistics_sha256"] == (
        "5aa5ab85e0c71c0adee97782be37907b0918050a8539bb3aab88fe392953948e"
    )
    assert len(names) == 280
    assert len(cases) == 56
    assert len({case.absolute_index for case in cases}) == 56
    assert {case.episode for case in cases} == {2, 7, 21, 28, 31, 34, 35, 41}
    assert all(case.camera1.dtype == np.uint8 for case in cases)
    assert all(case.camera2.dtype == np.uint8 for case in cases)
    assert all(case.state.shape == (6,) for case in cases)
    assert all(case.target_action.shape == (6,) for case in cases)
    assert all(case.noise.shape == (1, 50, 32) for case in cases)
    assert all(np.isfinite(case.noise).all() for case in cases)


def test_absolute_error_aggregates_physical_action_mae_without_rounding() -> None:
    module = __import__("training.evaluation", fromlist=["absolute_error"])
    prediction = np.array([1.0, 3.0, -2.0], dtype=np.float32)
    target = np.array([0.0, 5.0, 2.0], dtype=np.float32)

    error_sum, count = module.absolute_error(prediction, target)

    assert error_sum == 7.0
    assert count == 3


def test_outcome_gate_requires_all_three_independent_results() -> None:
    module = __import__("training.evaluation", fromlist=["evaluate_outcome_gates"])

    passing = module.evaluate_outcome_gates(
        base_mlx_mae=10.0,
        fine_mlx_mae=9.0,
        torch_mae=9.45,
        parity_max_abs=0.005,
    )
    failed_improvement = module.evaluate_outcome_gates(
        base_mlx_mae=10.0,
        fine_mlx_mae=9.00001,
        torch_mae=9.0,
        parity_max_abs=0.0,
    )
    failed_roundtrip = module.evaluate_outcome_gates(
        base_mlx_mae=10.0,
        fine_mlx_mae=8.0,
        torch_mae=8.401,
        parity_max_abs=0.0,
    )
    failed_parity = module.evaluate_outcome_gates(
        base_mlx_mae=10.0,
        fine_mlx_mae=8.0,
        torch_mae=8.0,
        parity_max_abs=0.005001,
    )

    assert passing.passed
    assert passing.improvement_passed
    assert passing.roundtrip_passed
    assert passing.parity_passed
    assert passing.fine_to_base_ratio == 0.9
    assert math.isclose(passing.torch_to_mlx_ratio, 1.05, rel_tol=0, abs_tol=1e-15)
    assert not failed_improvement.passed
    assert not failed_improvement.improvement_passed
    assert not failed_roundtrip.passed
    assert not failed_roundtrip.roundtrip_passed
    assert not failed_parity.passed
    assert not failed_parity.parity_passed


def test_base_report_is_bound_to_frozen_manifest_before_training() -> None:
    artifact = TrainingArtifact(_EVALUATION_DIR)
    report = json.loads(_BASE_REPORT.read_text(encoding="utf-8"))

    assert report["artifact_type"] == "smolvla-lora-base-heldout-evaluation"
    assert report["evaluation_manifest_sha256"] == artifact.metadata["manifest_sha256"]
    assert report["sample_count"] == 56
    assert report["element_count"] == 336
    assert report["mlx_mae"] > 0
    assert len(report["samples"]) == 56
    assert report["device"] == "Device(cpu, 0)"
    assert report["dtype"] == "float32"
