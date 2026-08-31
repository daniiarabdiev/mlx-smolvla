"""Immutable checkpoint-backed MLX/Torch step-zero gradient-parity gates."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from training.data import TrainingArtifact


_GOLDEN_DIR = Path(".cache/training/gradient_goldens")
_NATIVE_CACHE = Path(".cache/smolvla_mlx/policy-float32")


def test_gradient_parity_thresholds_are_immutable() -> None:
    module = __import__(
        "training.parity",
        fromlist=[
            "LOSS_RELATIVE_TOLERANCE",
            "GRADIENT_RELATIVE_L2_TOLERANCE",
            "GRADIENT_COSINE_MINIMUM",
        ],
    )

    assert module.LOSS_RELATIVE_TOLERANCE == 1e-4
    assert module.GRADIENT_RELATIVE_L2_TOLERANCE == 1e-2
    assert module.GRADIENT_COSINE_MINIMUM == 0.999


def test_serialized_training_batch_consumes_exact_artifact_draws() -> None:
    module = __import__("training.parity", fromlist=["load_serialized_training_batch"])
    artifact = TrainingArtifact(_GOLDEN_DIR)

    batch = module.load_serialized_training_batch(artifact)

    np.testing.assert_array_equal(
        np.asarray(batch.processed.pixel_values),
        artifact.load("batch/pixel_values"),
    )
    np.testing.assert_array_equal(
        np.asarray(batch.processed.pixel_attention_mask),
        artifact.load("batch/pixel_attention_mask"),
    )
    np.testing.assert_array_equal(
        np.asarray(batch.processed.input_ids),
        artifact.load("batch/input_ids"),
    )
    np.testing.assert_array_equal(
        np.asarray(batch.processed.text_attention_mask),
        artifact.load("batch/text_attention_mask"),
    )
    np.testing.assert_array_equal(
        np.asarray(batch.processed.state),
        artifact.load("batch/state"),
    )
    np.testing.assert_array_equal(np.asarray(batch.actions), artifact.load("batch/actions"))
    np.testing.assert_array_equal(
        np.asarray(batch.action_is_pad),
        artifact.load("batch/action_is_pad"),
    )
    np.testing.assert_array_equal(np.asarray(batch.noise), artifact.load("draws/noise"))
    np.testing.assert_array_equal(
        np.asarray(batch.timesteps),
        artifact.load("draws/timesteps"),
    )
    assert batch.action_dim == 6


def test_checkpoint_training_parameters_equal_reference_artifact() -> None:
    model_module = __import__("training.model", fromlist=["SmolVLATrainingModel"])
    parity_module = __import__(
        "training.parity",
        fromlist=["validate_checkpoint_parameter_identity"],
    )
    artifact = TrainingArtifact(_GOLDEN_DIR)

    with mx.stream(mx.cpu):
        model = model_module.SmolVLATrainingModel.from_pretrained(
            cache_dir=_NATIVE_CACHE,
            dtype=mx.float32,
        )
        matched_names = parity_module.validate_checkpoint_parameter_identity(model, artifact)

    expected_names = tuple(
        sorted(item["canonical"] for item in artifact.metadata["parameter_map"])
    )
    assert matched_names == expected_names
    assert len(matched_names) == 155


def test_real_step_zero_gradient_parity_passes_every_immutable_gate() -> None:
    module = __import__("training.parity", fromlist=["run_gradient_parity"])

    result = module.run_gradient_parity(
        golden_dir=_GOLDEN_DIR,
        native_cache=_NATIVE_CACHE,
    )

    assert result.passed
    assert result.parameter_match_count == 155
    assert result.gradient_count == 155
    assert len(result.comparisons) == 155
    assert result.loss_relative_difference <= 1e-4
    assert result.maximum_gradient_relative_l2 <= 1e-2
    assert result.minimum_gradient_cosine >= 0.999
    assert all(comparison.relative_l2 <= 1e-2 for comparison in result.comparisons)
    assert all(comparison.cosine_similarity >= 0.999 for comparison in result.comparisons)
    assert len(result.worst_relative_l2) == 5
    assert len(result.worst_cosine) == 5
    assert list(result.worst_relative_l2) == sorted(
        result.comparisons,
        key=lambda comparison: (-comparison.relative_l2, comparison.name),
    )[:5]
    assert list(result.worst_cosine) == sorted(
        result.comparisons,
        key=lambda comparison: (comparison.cosine_similarity, comparison.name),
    )[:5]
    report = result.as_dict()
    assert report["thresholds"] == {
        "loss_relative_difference_maximum": 1e-4,
        "gradient_relative_l2_maximum": 1e-2,
        "gradient_cosine_minimum": 0.999,
    }
    assert len(report["comparisons"]) == 155
    assert len(report["worst_relative_l2"]) == 5
    assert len(report["worst_cosine"]) == 5
