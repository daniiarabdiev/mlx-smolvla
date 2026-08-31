"""Reference PyTorch 25-step optimizer artifact contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.data import TrainingArtifact


_T1_DIR = Path(".cache/training/gradient_goldens")
_T2_DIR = Path(".cache/training/optimizer_goldens")


def test_reference_lockstep_constants_fix_the_observation_window() -> None:
    module = __import__(
        "training.reference_lockstep",
        fromlist=[
            "OPTIMIZER_LOCKSTEP_STEPS",
            "OPTIMIZER_TRAINING_HORIZON",
            "OPTIMIZER_LOCKSTEP_SEED",
        ],
    )

    assert module.OPTIMIZER_LOCKSTEP_STEPS == 25
    assert module.OPTIMIZER_TRAINING_HORIZON == 100_000
    assert module.OPTIMIZER_LOCKSTEP_SEED == 20_260_831


def test_reference_optimizer_artifact_is_complete_and_linked_to_t1() -> None:
    artifact = TrainingArtifact(_T2_DIR)
    t1_artifact = TrainingArtifact(_T1_DIR)
    names = artifact.verify_all()
    metadata = artifact.metadata

    assert metadata["artifact_type"] == "smolvla-optimizer-golden"
    assert metadata["device"] == "cpu"
    assert metadata["dtype"] == "float32"
    assert metadata["seed"] == 20_260_831
    assert metadata["step_count"] == 25
    assert metadata["training_horizon"] == 100_000
    assert metadata["batch_schedule"] == "repeat fixed T1 batch for every step"
    assert metadata["t1_manifest_sha256"] == t1_artifact.metadata["manifest_sha256"]
    assert metadata["t1_batch_verified"] is True
    assert metadata["initial_parameters_verified"] == 155
    assert metadata["trainable_tensor_count"] == 155
    assert metadata["trainable_scalar_count"] == 99_880_992
    assert metadata["tensor_count"] == 330
    assert len(names) == 330
    assert len([name for name in names if name.startswith("final_parameters/")]) == 155


def test_reference_optimizer_step_metrics_and_draws_are_exact() -> None:
    optimizer_module = __import__(
        "training.optimizer",
        fromlist=["SmolVLAOptimizerConfig", "cosine_decay_with_warmup_lr"],
    )
    artifact = TrainingArtifact(_T2_DIR)
    t1_artifact = TrainingArtifact(_T1_DIR)
    config = optimizer_module.SmolVLAOptimizerConfig()

    losses = []
    clipped_steps = 0
    for step in range(25):
        prefix = f"steps/{step:03d}"
        loss = artifact.load(f"{prefix}/loss")
        lr_used = artifact.load(f"{prefix}/lr_used")
        lr_next = artifact.load(f"{prefix}/lr_next")
        gradient_norm = artifact.load(f"{prefix}/gradient_norm")
        clip_coefficient = artifact.load(f"{prefix}/clip_coefficient")
        assert loss.shape == () and np.isfinite(loss.item()) and loss.item() > 0
        assert lr_used.item() == optimizer_module.cosine_decay_with_warmup_lr(step, config)
        assert lr_next.item() == optimizer_module.cosine_decay_with_warmup_lr(step + 1, config)
        assert np.isfinite(gradient_norm.item()) and gradient_norm.item() > 0.0
        expected_coefficient = min(1.0, 10.0 / (gradient_norm.item() + 1e-6))
        assert clip_coefficient.item() == expected_coefficient
        assert 0.0 < clip_coefficient.item() <= 1.0
        clipped_steps += int(clip_coefficient.item() < 1.0)
        assert artifact.load(f"draws/{step:03d}/noise").shape == (1, 50, 32)
        assert artifact.load(f"draws/{step:03d}/timesteps").shape == (1,)
        losses.append(loss.item())

    assert np.all(np.isfinite(losses))
    assert clipped_steps == 24
    np.testing.assert_array_equal(
        artifact.load("draws/000/noise"),
        t1_artifact.load("draws/noise"),
    )
    np.testing.assert_array_equal(
        artifact.load("draws/000/timesteps"),
        t1_artifact.load("draws/timesteps"),
    )


def test_reference_optimizer_final_parameters_are_all_finite_fp32() -> None:
    artifact = TrainingArtifact(_T2_DIR)
    parameter_map = artifact.metadata["parameter_map"]

    assert len(parameter_map) == 155
    for item in parameter_map:
        parameter = artifact.load(f"final_parameters/{item['canonical']}")
        assert parameter.dtype == np.float32
        assert list(parameter.shape) == item["shape"]
        assert np.all(np.isfinite(parameter)), item["canonical"]
