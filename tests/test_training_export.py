"""Stats-active preprocessing and merged standard checkpoint export gates."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest
from safetensors import safe_open

from smolvla_mlx.types import ProcessedObservation


class IdentityBasePreprocessor:
    def __init__(self) -> None:
        self.config = object()

    def __call__(self, observation):
        return ProcessedObservation(
            pixel_values=mx.zeros((2, 3, 4, 4), dtype=mx.float32),
            pixel_attention_mask=mx.ones((2, 1), dtype=mx.bool_),
            input_ids=mx.zeros((1, 3), dtype=mx.int32),
            text_attention_mask=mx.ones((1, 3), dtype=mx.bool_),
            state=mx.array(observation["observation.state"], dtype=mx.float32)[None],
        )


def test_stats_aware_preprocessor_matches_lerobot_mean_std_math() -> None:
    module = __import__("training.preprocessing", fromlist=["StatsAwareSmolVLAPreprocessor"])
    processor = module.StatsAwareSmolVLAPreprocessor(
        base=IdentityBasePreprocessor(),
        state_mean=mx.array([10.0, -2.0], dtype=mx.float32),
        state_std=mx.array([2.0, 4.0], dtype=mx.float32),
        action_mean=mx.array([1.0, 5.0], dtype=mx.float32),
        action_std=mx.array([0.5, 2.0], dtype=mx.float32),
    )

    processed = processor({"observation.state": np.array([14.0, 6.0], dtype=np.float32)})
    actions = mx.array([[[2.0, 1.0], [0.0, 9.0]]], dtype=mx.float32)
    normalized = processor.normalize_actions(actions)
    restored = processor.unnormalize_actions(normalized)
    mx.eval(processed.state, normalized, restored)

    np.testing.assert_allclose(np.asarray(processed.state), [[2.0, 2.0]], rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(normalized),
        [[[2.0, -2.0], [-2.0, 2.0]]],
        rtol=0,
        atol=2e-7,
    )
    np.testing.assert_allclose(np.asarray(restored), np.asarray(actions), rtol=0, atol=2e-7)


def test_stats_aware_preprocessor_reads_standard_processor_safetensors(tmp_path: Path) -> None:
    module = __import__("training.preprocessing", fromlist=["StatsAwareSmolVLAPreprocessor"])
    mx.save_safetensors(
        str(tmp_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"),
        {
            "observation.state.mean": mx.array([1.0, 2.0], dtype=mx.float32),
            "observation.state.std": mx.array([3.0, 4.0], dtype=mx.float32),
            "action.mean": mx.array([5.0, 6.0], dtype=mx.float32),
            "action.std": mx.array([7.0, 8.0], dtype=mx.float32),
        },
    )
    mx.save_safetensors(
        str(tmp_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"),
        {
            "action.mean": mx.array([5.0, 6.0], dtype=mx.float32),
            "action.std": mx.array([7.0, 8.0], dtype=mx.float32),
        },
    )

    processor = module.StatsAwareSmolVLAPreprocessor.from_pretrained_files(
        IdentityBasePreprocessor(),
        tmp_path,
    )

    np.testing.assert_array_equal(np.asarray(processor.state_mean), [1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(processor.state_std), [3.0, 4.0])
    np.testing.assert_array_equal(np.asarray(processor.action_mean), [5.0, 6.0])
    np.testing.assert_array_equal(np.asarray(processor.action_std), [7.0, 8.0])


def test_inverse_checkpoint_mapping_is_strict_and_reverses_patch_layout() -> None:
    module = __import__("training.export", fromlist=["source_name_map"])
    source_names = (
        "model.state_proj.weight",
        "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight",
    )
    mapping = module.source_name_map(source_names)

    assert mapping == {
        "state_proj.weight": "model.state_proj.weight",
        "vision.embeddings.patch_embedding.weight": (
            "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"
        ),
    }
    native_patch = mx.arange(2 * 3 * 4 * 5, dtype=mx.float32).reshape(2, 3, 4, 5)
    source_patch = module.source_layout_tensor(source_names[1], native_patch)
    assert source_patch.shape == (2, 5, 3, 4)
    np.testing.assert_array_equal(np.asarray(source_patch), np.asarray(native_patch).transpose(0, 3, 1, 2))


@pytest.fixture(scope="module")
def standard_export(tmp_path_factory: pytest.TempPathFactory):
    dataset = __import__("training.dataset", fromlist=["compute_train_statistics"])
    export = __import__("training.export", fromlist=["export_merged_checkpoint"])
    model_module = __import__("training.model", fromlist=["SmolVLATrainingModel"])
    root = tmp_path_factory.mktemp("standard-export")
    source = export.resolve_base_checkpoint(Path(".cache/hf"))
    split = dataset.make_episode_split(num_episodes=50, seed=20260901)
    stats = dataset.compute_train_statistics(
        Path(".cache/hf/datasets/svla_so101_pickplace"),
        split.train_episodes,
    )
    model = model_module.SmolVLATrainingModel.from_pretrained(
        cache_dir=Path(".cache/smolvla_mlx/policy-float32"),
        dtype=mx.bfloat16,
    )
    output = root / "checkpoint"
    report = export.export_merged_checkpoint(
        model=model,
        source_checkpoint_dir=source,
        output_dir=output,
        processor_stats=stats.processor_stats,
        metadata={
            "split_seed": split.seed,
            "train_statistics_sha256": stats.sha256,
            "test_fixture": True,
        },
    )
    del model
    gc.collect()
    mx.clear_cache()
    return output, report, source, root


def test_export_is_atomic_complete_fp32_and_standard(standard_export) -> None:
    output, report, source, _ = standard_export
    expected_files = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "training_manifest.json",
    }

    assert expected_files <= {path.name for path in output.iterdir()}
    assert report.tensor_count == 500
    assert report.parameter_count == 450_046_176
    assert report.dtype == "float32"
    assert report.output_dir == output.resolve()
    assert len(report.file_sha256) >= len(expected_files) - 1
    manifest = json.loads((output / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tensor_count"] == 500
    assert manifest["parameter_count"] == 450_046_176
    assert manifest["metadata"]["split_seed"] == 20260901
    assert manifest["metadata"]["train_statistics_sha256"] == (
        "5aa5ab85e0c71c0adee97782be37907b0918050a8539bb3aab88fe392953948e"
    )
    with safe_open(output / "model.safetensors", framework="np") as tensors:
        keys = tuple(tensors.keys())
        assert len(keys) == 500
        assert all("lora_" not in key and ".base." not in key for key in keys)
        assert all(tensors.get_slice(key).get_dtype() == "F32" for key in keys)
        patch = tensors.get_tensor(
            "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"
        )
    assert patch.shape == (768, 3, 16, 16)
    assert (output / "config.json").read_bytes() == (source / "config.json").read_bytes()


def test_complete_export_can_be_validated_and_reused_after_finalization_interrupt(
    standard_export,
) -> None:
    output, original, _, _ = standard_export
    module = __import__(
        "training.export", fromlist=["validate_merged_checkpoint_export"]
    )
    expected_metadata = json.loads(
        (output / "training_manifest.json").read_text(encoding="utf-8")
    )["metadata"]

    recovered = module.validate_merged_checkpoint_export(
        output,
        expected_metadata=expected_metadata,
    )

    assert recovered.output_dir == original.output_dir
    assert recovered.tensor_count == original.tensor_count
    assert recovered.parameter_count == original.parameter_count
    assert recovered.file_sha256 == original.file_sha256
    try:
        module.validate_merged_checkpoint_export(
            output,
            expected_metadata={**expected_metadata, "split_seed": 0},
        )
    except ValueError as error:
        assert "metadata" in str(error)
    else:
        raise AssertionError("export with different run metadata was reused")


def test_exported_processor_contains_exact_train_only_stats(standard_export) -> None:
    output, _, _, _ = standard_export
    pre = mx.load(str(output / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post = mx.load(str(output / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))

    np.testing.assert_allclose(
        np.asarray(pre["observation.state.mean"]),
        [
            7.7335728942111,
            -55.131248744440754,
            66.75018237690006,
            69.09834068655765,
            -53.38197588213268,
            8.229249810138885,
        ],
        rtol=0,
        atol=4e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(pre["action.mean"]),
        np.asarray(post["action.mean"]),
    )
    np.testing.assert_array_equal(
        np.asarray(pre["action.std"]),
        np.asarray(post["action.std"]),
    )


def test_export_loads_strictly_in_mlx_and_torch(standard_export) -> None:
    output, _, _, root = standard_export
    preprocessing = __import__("training.preprocessing", fromlist=["load_stats_aware_policy"])
    reference = __import__("training.reference_export", fromlist=["TorchExportPolicy"])

    mlx_policy = preprocessing.load_stats_aware_policy(
        output,
        cache_dir=root / "mlx-cache",
        dtype=mx.bfloat16,
    )
    assert len(mlx_policy.loaded_parameter_names) == 500
    assert len(tuple(tree_flatten(mlx_policy.expert.parameters()))) > 0
    assert mlx_policy.preprocessor.action_mean.shape == (6,)
    del mlx_policy
    gc.collect()
    mx.clear_cache()

    torch_export = reference.TorchExportPolicy.load(
        output,
        cache_dir=Path(".cache/hf"),
    )
    assert torch_export.parameter_count == 450_046_176
    assert str(torch_export.device) == "cpu"
    assert str(torch_export.dtype) == "torch.float32"
    assert torch_export.preprocessor is not None
    assert torch_export.postprocessor is not None
