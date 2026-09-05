from pathlib import Path

import pytest
import torch


@pytest.mark.slow
def test_reference_policy_loads_pinned_checkpoint_on_cpu() -> None:
    from mlx_smolvla._lab.reference.discovery import BASE_VLM_REVISION
    from mlx_smolvla._lab.reference.policy import ReferencePolicy

    reference = ReferencePolicy.load(cache_dir=Path(".cache/hf"))

    assert reference.device == torch.device("cpu")
    assert reference.dtype == torch.float32
    assert reference.config.chunk_size == 50
    assert reference.config.max_state_dim == 32
    assert reference.config.max_action_dim == 32
    assert reference.config.num_steps == 10
    assert reference.parameter_count == 450_046_176
    assert reference.vlm_snapshot.name == BASE_VLM_REVISION
    assert Path(reference.config.vlm_model_name) == reference.vlm_snapshot


@pytest.mark.slow
def test_real_dataset_observation_maps_two_cameras_and_task() -> None:
    from mlx_smolvla._lab.reference.policy import load_dataset_observation

    sample = load_dataset_observation(cache_dir=Path(".cache/hf"), index=0)

    assert set(sample.observation) == {
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.state",
        "task",
    }
    assert sample.observation["observation.images.camera1"].shape == (3, 480, 640)
    assert sample.observation["observation.images.camera2"].shape == (3, 480, 640)
    assert sample.observation["observation.state"].shape == (6,)
    assert sample.action.shape == (6,)
    assert isinstance(sample.observation["task"], str)
    assert sample.observation["task"]


@pytest.mark.slow
def test_checkpoint_preprocessor_batches_tokenizes_and_normalizes() -> None:
    from mlx_smolvla._lab.reference.policy import ReferencePolicy, load_dataset_observation

    reference = ReferencePolicy.load(cache_dir=Path(".cache/hf"))
    sample = load_dataset_observation(cache_dir=Path(".cache/hf"), index=0)
    batch = reference.prepare(sample.observation)

    assert batch["observation.images.camera1"].shape == (1, 3, 480, 640)
    assert batch["observation.images.camera2"].shape == (1, 3, 480, 640)
    assert batch["observation.state"].shape == (1, 6)
    assert batch["observation.language.tokens"].shape == (1, 48)
    assert batch["observation.language.attention_mask"].shape == (1, 48)
    assert batch["observation.state"].device.type == "cpu"
    assert torch.isfinite(batch["observation.state"]).all()
    assert torch.equal(
        batch["observation.state"], sample.observation["observation.state"].unsqueeze(0)
    )


@pytest.mark.slow
def test_reference_predicts_finite_action_chunk_from_fixed_noise() -> None:
    from mlx_smolvla._lab.reference.policy import ReferencePolicy, load_dataset_observation

    reference = ReferencePolicy.load(cache_dir=Path(".cache/hf"))
    sample = load_dataset_observation(cache_dir=Path(".cache/hf"), index=0)
    generator = torch.Generator(device="cpu").manual_seed(20260831)
    noise = torch.randn((1, 50, 32), generator=generator, dtype=torch.float32)

    prediction = reference.predict(sample.observation, noise=noise)

    assert prediction.normalized_actions.shape == (1, 50, 6)
    assert prediction.actions.shape == (1, 50, 6)
    assert prediction.normalized_actions.device.type == "cpu"
    assert prediction.normalized_actions.dtype == torch.float32
    assert prediction.actions.dtype == torch.float32
    assert torch.isfinite(prediction.normalized_actions).all()
    assert torch.isfinite(prediction.actions).all()
    assert torch.equal(prediction.actions, prediction.normalized_actions)
