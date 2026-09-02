"""Actual LeRobot CPU/fp32 training-case and gradient-golden contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from training.data import TrainingArtifact


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def reference_training_case():
    module = __import__("training.reference", fromlist=["prepare_reference_training_case"])
    return module.prepare_reference_training_case(
        Path(".cache/hf"),
        episode=0,
        frame_index=100,
    )


def test_reference_training_case_matches_the_real_train_loop(reference_training_case) -> None:
    case = reference_training_case

    assert (case.episode, case.frame_index, case.absolute_index) == (0, 100, 100)
    assert case.pixel_values.shape == (2, 3, 512, 512)
    assert case.pixel_attention_mask.shape == (2, 1)
    assert case.input_ids.shape == (1, 48)
    assert case.text_attention_mask.shape == (1, 48)
    assert case.state.shape == (1, 32)
    assert case.actions.shape == (1, 50, 32)
    assert case.action_is_pad.shape == (1, 50)
    assert not bool(torch.any(case.action_is_pad))
    assert case.physical_action_dim == 6
    assert case.task == "pink lego brick into the transparent box\n"
    assert case.pixel_values.dtype == torch.float32
    assert float(case.pixel_values.min()) >= -1.0
    assert float(case.pixel_values.max()) <= 1.0


def test_reference_training_case_uses_dataset_statistics(reference_training_case) -> None:
    case = reference_training_case
    state_stats = case.dataset_stats["observation.state"]
    action_stats = case.dataset_stats["action"]
    expected_state = (
        case.raw_state[:, -1, :] - torch.as_tensor(state_stats["mean"], dtype=torch.float32)
    ) / (torch.as_tensor(state_stats["std"], dtype=torch.float32) + 1e-8)
    expected_actions = (
        case.raw_actions - torch.as_tensor(action_stats["mean"], dtype=torch.float32)
    ) / (torch.as_tensor(action_stats["std"], dtype=torch.float32) + 1e-8)

    torch.testing.assert_close(case.state[:, :6], expected_state, rtol=0, atol=0)
    torch.testing.assert_close(case.actions[:, :, :6], expected_actions, rtol=0, atol=0)
    assert not torch.equal(case.state[:, :6], case.raw_state[:, -1, :])
    assert not torch.equal(case.actions[:, :, :6], case.raw_actions)


def test_reference_trainable_names_are_the_strict_canonical_bijection(
    reference_training_case,
) -> None:
    specs = reference_training_case.parameter_specs
    source_names = tuple(spec.source_name for spec in specs)
    canonical_names = tuple(spec.canonical_name for spec in specs)

    assert len(specs) == 155
    assert sum(spec.scalar_count for spec in specs) == 99_880_992
    assert len(set(source_names)) == 155
    assert len(set(canonical_names)) == 155
    assert all(
        name.startswith(("expert.", "state_proj.", "action_"))
        for name in canonical_names
    )


def test_reference_gradient_artifact_is_complete_and_integral() -> None:
    artifact = TrainingArtifact(Path(".cache/training/gradient_goldens"))
    names = artifact.verify_all()
    metadata = artifact.metadata

    assert metadata["artifact_type"] == "smolvla-gradient-golden"
    assert metadata["device"] == "cpu"
    assert metadata["dtype"] == "float32"
    assert metadata["seed"] == 20_260_831
    assert metadata["episode"] == 0
    assert metadata["frame_index"] == 100
    assert metadata["absolute_index"] == 100
    assert metadata["trainable_tensor_count"] == 155
    assert metadata["trainable_scalar_count"] == 99_880_992
    assert metadata["tensor_count"] == 324
    assert len(names) == 324
    assert len([name for name in names if name.startswith("parameters/")]) == 155
    assert len([name for name in names if name.startswith("gradients/")]) == 155
    saved_loss = artifact.load("flow/loss")
    assert saved_loss.shape == ()
    assert saved_loss.item() == 2.101923942565918
    np.testing.assert_array_equal(
        artifact.load("draws/timesteps"),
        np.array([0.8003060817718506], dtype=np.float32),
    )
    for name in names:
        if name.startswith("gradients/"):
            gradient = artifact.load(name)
            assert np.all(np.isfinite(gradient)), name
            assert float(np.linalg.norm(gradient.reshape(-1))) > 0.0, name
