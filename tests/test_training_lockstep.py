"""Immutable native MLX 25-step optimizer-lockstep gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.data import TrainingArtifact


_T1_DIR = Path(".cache/training/gradient_goldens")
_T2_DIR = Path(".cache/training/optimizer_goldens")
_NATIVE_CACHE = Path(".cache/smolvla_mlx/policy-float32")


def test_optimizer_lockstep_thresholds_are_immutable() -> None:
    module = __import__(
        "training.lockstep",
        fromlist=["LOSS_RELATIVE_TOLERANCE", "PARAMETER_RELATIVE_L2_TOLERANCE"],
    )

    assert module.LOSS_RELATIVE_TOLERANCE == 1e-3
    assert module.PARAMETER_RELATIVE_L2_TOLERANCE == 5e-3


def test_optimizer_artifact_link_and_step_draw_consumption_are_exact() -> None:
    module = __import__(
        "training.lockstep",
        fromlist=["validate_optimizer_artifact_link", "load_lockstep_training_batch"],
    )
    t1_artifact = TrainingArtifact(_T1_DIR)
    optimizer_artifact = TrainingArtifact(_T2_DIR)

    link = module.validate_optimizer_artifact_link(t1_artifact, optimizer_artifact)
    batch = module.load_lockstep_training_batch(t1_artifact, optimizer_artifact, step=17)

    assert link == (
        t1_artifact.metadata["manifest_sha256"],
        optimizer_artifact.metadata["manifest_sha256"],
    )
    np.testing.assert_array_equal(
        np.asarray(batch.noise),
        optimizer_artifact.load("draws/017/noise"),
    )
    np.testing.assert_array_equal(
        np.asarray(batch.timesteps),
        optimizer_artifact.load("draws/017/timesteps"),
    )
    np.testing.assert_array_equal(
        np.asarray(batch.actions),
        t1_artifact.load("batch/actions"),
    )
    np.testing.assert_array_equal(
        np.asarray(batch.action_is_pad),
        t1_artifact.load("batch/action_is_pad"),
    )


def test_real_25_step_optimizer_lockstep_passes_every_gate() -> None:
    module = __import__("training.lockstep", fromlist=["run_optimizer_lockstep"])

    result = module.run_optimizer_lockstep(
        t1_dir=_T1_DIR,
        optimizer_golden_dir=_T2_DIR,
        native_cache=_NATIVE_CACHE,
    )

    assert result.passed
    assert result.parameter_match_count == 155
    assert len(result.loss_comparisons) == 25
    assert len(result.parameter_comparisons) == 155
    assert result.maximum_loss_relative_difference <= 1e-3
    assert result.maximum_parameter_relative_l2 <= 5e-3
    assert all(item.relative_difference <= 1e-3 for item in result.loss_comparisons)
    assert all(item.relative_l2 <= 5e-3 for item in result.parameter_comparisons)
    assert len(result.worst_loss_steps) == 5
    assert len(result.worst_parameters) == 5
    assert list(result.worst_loss_steps) == sorted(
        result.loss_comparisons,
        key=lambda item: (-item.relative_difference, item.step),
    )[:5]
    assert list(result.worst_parameters) == sorted(
        result.parameter_comparisons,
        key=lambda item: (-item.relative_l2, item.name),
    )[:5]
    report = result.as_dict()
    assert report["thresholds"] == {
        "per_step_loss_relative_difference_maximum": 1e-3,
        "final_parameter_relative_l2_maximum": 5e-3,
    }
    assert len(report["loss_comparisons"]) == 25
    assert len(report["parameter_comparisons"]) == 155
    assert len(report["worst_loss_steps"]) == 5
    assert len(report["worst_parameters"]) == 5
