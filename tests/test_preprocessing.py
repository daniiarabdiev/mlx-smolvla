from pathlib import Path
import shutil

import mlx.core as mx
import numpy as np
import pytest


def _stats_active_checkpoint(tmp_path: Path, checkpoint_dir: Path) -> Path:
    active = tmp_path / "stats-active"
    active.mkdir()
    shutil.copy2(checkpoint_dir / "config.json", active / "config.json")
    tensors = {
        "observation.state.mean": mx.array([1, 2, 3, 4, 5, 6], dtype=mx.float32),
        "observation.state.std": mx.array([2, 4, 5, 8, 10, 20], dtype=mx.float32),
        "action.mean": mx.array([10, 20, 30, 40, 50, 60], dtype=mx.float32),
        "action.std": mx.array([1, 2, 4, 5, 10, 20], dtype=mx.float32),
    }
    mx.save_safetensors(
        str(active / "policy_preprocessor_step_5_normalizer_processor.safetensors"),
        tensors,
    )
    mx.save_safetensors(
        str(active / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"),
        tensors,
    )
    return active


@pytest.mark.parametrize("golden", range(8), indirect=True)
def test_preprocessing_matches_reference(
    golden,
    checkpoint_dir: Path,
    base_vlm_dir: Path,
) -> None:
    from smolvla_mlx.preprocessing import SmolVLAPreprocessor

    preprocessor = SmolVLAPreprocessor.from_pretrained_files(checkpoint_dir, base_vlm_dir)
    actual = preprocessor(golden.observation())

    np.testing.assert_allclose(
        np.array(actual.pixel_values),
        golden.array("preprocessed/pixel_values"),
        atol=1e-5,
        rtol=0,
    )
    np.testing.assert_array_equal(np.array(actual.pixel_attention_mask), golden.array("preprocessed/pixel_mask"))
    np.testing.assert_array_equal(np.array(actual.input_ids), golden.array("preprocessed/input_ids"))
    np.testing.assert_array_equal(
        np.array(actual.text_attention_mask),
        golden.array("preprocessed/text_attention_mask"),
    )
    np.testing.assert_allclose(
        np.array(actual.state),
        golden.array("preprocessed/state_normalized"),
        atol=1e-6,
        rtol=0,
    )


def test_effective_action_normalization_is_identity(checkpoint_dir: Path, base_vlm_dir: Path) -> None:
    from smolvla_mlx.preprocessing import SmolVLAPreprocessor

    preprocessor = SmolVLAPreprocessor.from_pretrained_files(checkpoint_dir, base_vlm_dir)
    action = mx.array([[[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]]], dtype=mx.float32)

    np.testing.assert_array_equal(np.array(preprocessor.normalize_actions(action)), np.array(action))
    np.testing.assert_array_equal(np.array(preprocessor.unnormalize_actions(action)), np.array(action))


def test_active_mean_std_state_and_action_match_reference_math(
    tmp_path: Path,
    checkpoint_dir: Path,
    base_vlm_dir: Path,
) -> None:
    from smolvla_mlx.preprocessing import SmolVLAPreprocessor

    active = _stats_active_checkpoint(tmp_path, checkpoint_dir)
    preprocessor = SmolVLAPreprocessor.from_pretrained_files(active, base_vlm_dir)
    observation = {
        "observation.images.camera1": np.zeros((3, 256, 256), dtype=np.float32),
        "observation.state": np.array([3, 6, 8, 12, 15, 26], dtype=np.float32),
        "task": "test task",
    }

    processed = preprocessor(observation)
    np.testing.assert_allclose(
        np.asarray(processed.state),
        np.array([[1, 1, 1, 1, 1, 1]], dtype=np.float32),
        atol=1e-7,
        rtol=0,
    )
    physical = mx.array([[[11, 24, 38, 50, 70, 100]]], dtype=mx.float32)
    normalized = preprocessor.normalize_actions(physical)
    np.testing.assert_allclose(
        np.asarray(normalized),
        np.array([[[1, 2, 2, 2, 2, 2]]], dtype=np.float32),
        atol=1e-6,
        rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(preprocessor.unnormalize_actions(normalized)),
        np.asarray(physical),
        atol=1e-6,
        rtol=0,
    )


def test_active_stats_reject_wrong_checkpoint_shape(
    tmp_path: Path,
    checkpoint_dir: Path,
    base_vlm_dir: Path,
) -> None:
    from smolvla_mlx.preprocessing import SmolVLAPreprocessor

    active = _stats_active_checkpoint(tmp_path, checkpoint_dir)
    mx.save_safetensors(
        str(active / "policy_preprocessor_step_5_normalizer_processor.safetensors"),
        {
            "observation.state.mean": mx.zeros((5,), dtype=mx.float32),
            "observation.state.std": mx.ones((5,), dtype=mx.float32),
            "action.mean": mx.zeros((6,), dtype=mx.float32),
            "action.std": mx.ones((6,), dtype=mx.float32),
        },
    )

    with pytest.raises(ValueError, match=r"observation\.state\.mean.*shape \(6,\)"):
        SmolVLAPreprocessor.from_pretrained_files(active, base_vlm_dir)


def test_missing_observation_keys_name_checkpoint_contract(
    checkpoint_dir: Path,
    base_vlm_dir: Path,
) -> None:
    from smolvla_mlx.preprocessing import SmolVLAPreprocessor

    preprocessor = SmolVLAPreprocessor.from_pretrained_files(checkpoint_dir, base_vlm_dir)
    with pytest.raises(ValueError) as missing_camera:
        preprocessor(
            {
                "observation.state": np.zeros((6,), dtype=np.float32),
                "task": "test task",
            }
        )
    message = str(missing_camera.value)
    assert "observation.images.camera1" in message
    assert "observation.images.camera2" in message
    assert "observation.images.camera3" in message
    assert "(3, 256, 256)" in message
    assert "observation.state" in message
    assert "(6,)" in message

    with pytest.raises(ValueError, match=r"observation\.state.*\(6,\)"):
        preprocessor(
            {
                "observation.images.camera1": np.zeros((3, 256, 256), dtype=np.float32),
                "task": "test task",
            }
        )


def test_reference_three_camera_slot_uses_each_present_stream_and_no_implicit_padding(
    checkpoint_dir: Path,
    base_vlm_dir: Path,
) -> None:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as TorchConfig
    from smolvla_mlx.preprocessing import SmolVLAPreprocessor
    import torch

    torch_config = TorchConfig.from_pretrained(checkpoint_dir)
    torch_config.device = "cpu"
    torch_config.vlm_model_name = str(base_vlm_dir)
    torch_preprocessor, _ = make_pre_post_processors(
        torch_config,
        pretrained_path=checkpoint_dir,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "tokenizer_processor": {"tokenizer_name": str(base_vlm_dir)},
        },
    )
    native = SmolVLAPreprocessor.from_pretrained_files(checkpoint_dir, base_vlm_dir)
    observation = {
        "observation.images.camera1": torch.zeros((3, 256, 256), dtype=torch.float32),
        "observation.images.camera2": torch.full((3, 256, 256), 0.5, dtype=torch.float32),
        "observation.images.camera3": torch.ones((3, 256, 256), dtype=torch.float32),
        "observation.state": torch.zeros((6,), dtype=torch.float32),
        "task": "test task",
    }
    torch_batch = torch_preprocessor(dict(observation))
    torch_images, torch_masks = [], []
    from lerobot.policies.common.vla_utils import resize_with_pad

    for key in torch_config.image_features:
        image = torch_batch[key]
        width, height = torch_config.resize_imgs_with_padding
        image = resize_with_pad(image, height, width, pad_value=0)
        torch_images.append(image * 2.0 - 1.0)
        torch_masks.append(torch.ones(image.shape[0], dtype=torch.bool))
    actual = native(observation)
    np.testing.assert_allclose(
        np.asarray(actual.pixel_values),
        torch.cat(torch_images, dim=0).numpy(),
        atol=1e-5,
        rtol=0,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.pixel_attention_mask),
        torch.cat([mask.reshape(-1, 1) for mask in torch_masks], dim=0).numpy(),
    )
    assert actual.pixel_values.shape[0] == 3

    two_camera = dict(observation)
    del two_camera["observation.images.camera3"]
    assert native(two_camera).pixel_values.shape[0] == 2
