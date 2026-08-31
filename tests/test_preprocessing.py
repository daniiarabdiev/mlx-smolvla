from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


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
