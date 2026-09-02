"""Stats-active preprocessing and merged standard checkpoint export gates."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest
from safetensors import safe_open

from mlx_smolvla.types import ProcessedObservation


pytestmark = pytest.mark.slow


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
        cache_dir=Path(".cache/mlx_smolvla/policy-float32"),
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


def test_tokenizer_snapshot_resolution_is_strictly_local(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.reference_export",
        fromlist=["resolve_tokenizer_snapshot"],
    )
    resolved = tmp_path / "tokenizer"
    resolved.mkdir()
    calls = []

    def snapshot_download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(resolved)

    monkeypatch.setattr(module, "snapshot_download", snapshot_download)
    assert module.resolve_tokenizer_snapshot(tmp_path / "cache") == resolved
    assert len(calls) == 1
    assert calls[0][1]["local_files_only"] is True


def test_bound_export_validator_reads_one_retained_export_tree(
    standard_export,
) -> None:
    output, original, source, root = standard_export
    module = __import__(
        "training.export",
        fromlist=["validate_bound_merged_checkpoint_export"],
    )
    dataset = __import__("training.dataset", fromlist=["compute_train_statistics"])
    model_module = __import__(
        "training.model", fromlist=["SmolVLATrainingModel"]
    )
    split = dataset.make_episode_split(num_episodes=50, seed=20260901)
    stats = dataset.compute_train_statistics(
        Path(".cache/hf/datasets/svla_so101_pickplace"),
        split.train_episodes,
    )
    model = model_module.SmolVLATrainingModel.from_pretrained(
        cache_dir=Path(".cache/mlx_smolvla/policy-float32"),
        dtype=mx.bfloat16,
    )
    manifest = json.loads(
        (output / "training_manifest.json").read_text(encoding="utf-8")
    )
    private_source = root / "bound-source"
    private_source.mkdir()
    os.link(
        (source / "model.safetensors").resolve(strict=True),
        private_source / "model.safetensors",
    )
    output_descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    source_descriptor = os.open(private_source, os.O_RDONLY | os.O_DIRECTORY)
    try:
        validated = module.validate_bound_merged_checkpoint_export(
            output_descriptor=output_descriptor,
            source_checkpoint_descriptor=source_descriptor,
            expected_metadata=manifest["metadata"],
            expected_support_sha256={
                name: original.file_sha256[name]
                for name in module._SUPPORT_FILES
            },
            model=model,
        )
    finally:
        os.close(source_descriptor)
        os.close(output_descriptor)
        del model
        gc.collect()
        mx.clear_cache()

    assert validated.file_sha256 == original.file_sha256
    assert validated.tensor_count == original.tensor_count
    assert validated.parameter_count == original.parameter_count


def test_reused_export_requires_exact_manifest_and_directory_inventory(
    standard_export,
    tmp_path: Path,
) -> None:
    output, _, _, _ = standard_export
    module = __import__(
        "training.export",
        fromlist=["validate_merged_checkpoint_export"],
    )
    for case, declare in (("declared", True), ("unmanifested", False)):
        altered = tmp_path / case
        shutil.copytree(output, altered)
        extra = altered / "model.safetensors.index.json"
        extra.write_text('{"weight_map":{}}\n', encoding="utf-8")
        manifest_path = altered / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if declare:
            manifest["file_sha256"][extra.name] = hashlib.sha256(
                extra.read_bytes()
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        try:
            module.validate_merged_checkpoint_export(
                altered,
                expected_metadata=manifest["metadata"],
            )
        except ValueError as error:
            assert "inventory" in str(error) or "manifest" in str(error)
        else:
            raise AssertionError(f"{case} extra export file was accepted")


def test_reused_export_support_files_are_bound_to_source_and_statistics(
    standard_export,
    tmp_path: Path,
) -> None:
    output, _, source, _ = standard_export
    dataset = __import__("training.dataset", fromlist=["compute_train_statistics"])
    module = __import__(
        "training.export",
        fromlist=[
            "expected_merged_checkpoint_support_file_sha256",
            "validate_merged_checkpoint_support_files",
        ],
    )
    split = dataset.make_episode_split(num_episodes=50, seed=20260901)
    stats = dataset.compute_train_statistics(
        Path(".cache/hf/datasets/svla_so101_pickplace"),
        split.train_episodes,
    )
    expected = module.expected_merged_checkpoint_support_file_sha256(
        source_checkpoint_dir=source,
        processor_stats=stats.processor_stats,
    )
    assert module.validate_merged_checkpoint_support_files(
        output,
        expected_sha256=expected,
    ) == expected

    altered = tmp_path / "altered-export"
    shutil.copytree(output, altered)
    processor = altered / "policy_preprocessor.json"
    processor.write_bytes(processor.read_bytes() + b" ")
    manifest_path = altered / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_sha256"][processor.name] = hashlib.sha256(
        processor.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    module.validate_merged_checkpoint_export(
        altered,
        expected_metadata=manifest["metadata"],
    )
    try:
        module.validate_merged_checkpoint_support_files(
            altered,
            expected_sha256=expected,
        )
    except ValueError as error:
        assert "support file" in str(error)
    else:
        raise AssertionError("self-declared altered support file was reused")


def test_support_file_generation_cleanup_preserves_a_replacement_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.export",
        fromlist=["expected_merged_checkpoint_support_file_sha256"],
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    replacement: Path | None = None
    original: Path | None = None

    def replace_temporary(directory, **_kwargs):
        nonlocal replacement, original
        replacement = Path(directory)
        original = replacement.with_name(f"{replacement.name}.original")
        replacement.rename(original)
        replacement.mkdir()
        (replacement / "owner.txt").write_text("competitor\n", encoding="utf-8")
        raise RuntimeError("injected processor failure")

    monkeypatch.setattr(module, "_save_processors", replace_temporary)
    try:
        module.expected_merged_checkpoint_support_file_sha256(
            source_checkpoint_dir=source,
            processor_stats={},
        )
    except RuntimeError as error:
        assert "injected processor failure" in str(error)
    else:
        raise AssertionError("injected support generation failure was accepted")

    assert replacement is not None and original is not None
    try:
        assert (replacement / "owner.txt").read_text(encoding="utf-8") == (
            "competitor\n"
        )
        assert (original / "config.json").is_file()
    finally:
        shutil.rmtree(replacement)
        shutil.rmtree(original)


def test_reused_export_model_is_bound_to_current_merged_values(
    standard_export,
    tmp_path: Path,
) -> None:
    output, _, source, _ = standard_export
    module = __import__(
        "training.export",
        fromlist=["validate_merged_checkpoint_model_values"],
    )
    model_module = __import__(
        "training.model",
        fromlist=["SmolVLATrainingModel"],
    )
    altered = tmp_path / "altered-model-export"
    shutil.copytree(output, altered)
    model_path = altered / "model.safetensors"
    weights = mx.load(str(model_path))
    changed_name = sorted(weights)[0]
    weights[changed_name] = weights[changed_name] + mx.ones_like(weights[changed_name])
    mx.eval(*weights.values())
    replacement = model_path.with_name("replacement.safetensors")
    mx.save_safetensors(str(replacement), weights)
    replacement.replace(model_path)
    del weights
    gc.collect()
    mx.clear_cache()
    manifest_path = altered / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_sha256"]["model.safetensors"] = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    module.validate_merged_checkpoint_export(
        altered,
        expected_metadata=manifest["metadata"],
    )
    model = model_module.SmolVLATrainingModel.from_pretrained(
        cache_dir=Path(".cache/mlx_smolvla/policy-float32"),
        dtype=mx.bfloat16,
    )
    try:
        module.validate_merged_checkpoint_model_values(
            altered,
            model=model,
            source_checkpoint_dir=source,
            expected_model_sha256=manifest["file_sha256"]["model.safetensors"],
        )
    except ValueError as error:
        assert "current merged model" in str(error)
    else:
        raise AssertionError("self-rehashed different merged model was reused")
    finally:
        del model
        gc.collect()
        mx.clear_cache()


def test_export_publication_never_clobbers_an_inserted_symlink(tmp_path: Path) -> None:
    module = __import__(
        "training.export",
        fromlist=["_publish_directory_no_clobber"],
    )
    staged = tmp_path / ".staged"
    staged.mkdir()
    (staged / "payload.bin").write_bytes(b"complete")
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "export"
    destination.symlink_to(outside, target_is_directory=True)

    try:
        module._publish_directory_no_clobber(staged, destination)
    except FileExistsError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("export publication clobbered an inserted symlink")
    assert destination.is_symlink()
    assert list(outside.iterdir()) == []
    assert (staged / "payload.bin").read_bytes() == b"complete"


def test_export_publication_rejects_staged_directory_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.export",
        fromlist=["_publish_directory_no_clobber"],
    )
    staged = tmp_path / ".staged"
    staged.mkdir()
    (staged / "payload.bin").write_bytes(b"complete")
    original = tmp_path / "original-staged"
    destination = tmp_path / "export"
    real_open = module.os.open
    swapped = False

    def swap_before_parent_bind(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(path) == tmp_path:
            swapped = True
            staged.rename(original)
            staged.mkdir()
            (staged / "payload.bin").write_bytes(b"replacement")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", swap_before_parent_bind)
    try:
        module._publish_directory_no_clobber(staged, destination)
    except RuntimeError as error:
        assert "staged export changed" in str(error)
    else:
        raise AssertionError("replaced staged export directory was published")
    assert not destination.exists()
    assert (original / "payload.bin").read_bytes() == b"complete"


def test_export_publication_quarantines_a_stage_replaced_during_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.export",
        fromlist=["_publish_directory_no_clobber"],
    )
    staged = tmp_path / ".staged"
    staged.mkdir()
    (staged / "payload.bin").write_bytes(b"complete")
    expected = staged.stat()
    original = tmp_path / ".staged.original"
    destination = tmp_path / "export"
    real_rename = module._renameatx_np
    swapped = False

    def replace_stage_during_rename(**kwargs):
        nonlocal swapped
        if not swapped and kwargs["source_name"] == staged.name:
            swapped = True
            staged.rename(original)
            staged.mkdir()
            (staged / "payload.bin").write_bytes(b"replacement")
        return real_rename(**kwargs)

    monkeypatch.setattr(module, "_renameatx_np", replace_stage_during_rename)
    try:
        module._publish_directory_no_clobber(
            staged,
            destination,
            expected_staged_identity=(expected.st_dev, expected.st_ino),
        )
    except RuntimeError as error:
        assert "changed during publication" in str(error)
    else:
        raise AssertionError("replacement export stage became public")

    assert not destination.exists()
    failures = tuple(tmp_path.glob(".export.publication-failed-*"))
    assert len(failures) == 1
    assert (failures[0] / "payload.bin").read_bytes() == b"replacement"
    assert (original / "payload.bin").read_bytes() == b"complete"

    original.rename(staged)
    module._publish_directory_no_clobber(
        staged,
        destination,
        expected_staged_identity=(expected.st_dev, expected.st_ino),
    )
    assert (destination / "payload.bin").read_bytes() == b"complete"


def _install_tiny_bound_export_fakes(module, monkeypatch) -> None:
    monkeypatch.setattr(module, "_CHECKPOINT_FILES", ("config.json", "model.safetensors"))
    monkeypatch.setattr(module, "_MINIMUM_FREE_BYTES", 0)
    monkeypatch.setattr(module, "_save_processors", lambda *_args, **_kwargs: None)

    def validate(output_dir, *, expected_metadata):
        manifest = json.loads(
            (Path(output_dir) / "training_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["metadata"] == dict(expected_metadata)
        return type(
            "ValidatedExport",
            (),
            {
                "output_dir": Path(output_dir).resolve(),
                "tensor_count": 1,
                "parameter_count": 1,
                "dtype": "float32",
                "file_sha256": manifest["file_sha256"],
            },
        )()

    monkeypatch.setattr(module, "validate_merged_checkpoint_export", validate)


def test_bound_export_never_writes_into_a_replacement_output_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.export",
        fromlist=["_export_merged_checkpoint_under_bound_parent"],
    )
    _install_tiny_bound_export_fakes(module, monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    parent = tmp_path / "run"
    parent.mkdir()
    snapshot = module._snapshot_export_directory(parent)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    detached = tmp_path / "detached-run"
    real_create = module._create_staged_export_directory_at
    swapped = False

    def replace_parent_before_staging(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            parent.rename(detached)
            parent.mkdir()
        return real_create(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "_create_staged_export_directory_at",
        replace_parent_before_staging,
    )
    try:
        module._export_merged_checkpoint_under_bound_parent(
            source_checkpoint_dir=source,
            output_name="export",
            parent_descriptor=descriptor,
            parent_snapshot=snapshot,
            source_values={"weight": mx.array([1.0], dtype=mx.float32)},
            parameter_count=1,
            processor_stats={},
            metadata={"test": True},
            disk_free_before=1,
        )
    except RuntimeError as error:
        assert "export directory changed" in str(error)
    else:
        raise AssertionError("bound export accepted a replaced output parent")
    finally:
        os.close(descriptor)

    assert tuple(parent.iterdir()) == ()
    assert (detached / "export" / "training_manifest.json").is_file()


def test_bound_export_cleanup_preserves_a_replacement_staging_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.export",
        fromlist=["_export_merged_checkpoint_under_bound_parent"],
    )
    _install_tiny_bound_export_fakes(module, monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    parent = tmp_path / "run"
    parent.mkdir()
    snapshot = module._snapshot_export_directory(parent)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    real_publish = module._publish_directory_no_clobber
    replacement: Path | None = None
    original: Path | None = None

    def replace_staging_before_publish(staged, destination, **kwargs):
        nonlocal replacement, original
        replacement = Path(staged)
        original = replacement.with_name(f"{replacement.name}.original")
        replacement.rename(original)
        replacement.mkdir()
        (replacement / "owner.txt").write_text("competitor\n", encoding="utf-8")
        return real_publish(staged, destination, **kwargs)

    monkeypatch.setattr(
        module,
        "_publish_directory_no_clobber",
        replace_staging_before_publish,
    )
    try:
        module._export_merged_checkpoint_under_bound_parent(
            source_checkpoint_dir=source,
            output_name="export",
            parent_descriptor=descriptor,
            parent_snapshot=snapshot,
            source_values={"weight": mx.array([1.0], dtype=mx.float32)},
            parameter_count=1,
            processor_stats={},
            metadata={"test": True},
            disk_free_before=1,
        )
    except RuntimeError as error:
        assert "staged export changed" in str(error)
    else:
        raise AssertionError("replaced export staging directory was accepted")
    finally:
        os.close(descriptor)

    assert replacement is not None and original is not None
    assert (replacement / "owner.txt").read_text(encoding="utf-8") == "competitor\n"
    assert (original / "model.safetensors").is_file()
    assert not (parent / "export").exists()


def test_bound_export_stage_uses_its_inode_for_the_first_model_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.export",
        fromlist=["_export_merged_checkpoint_under_bound_parent"],
    )
    _install_tiny_bound_export_fakes(module, monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    parent = tmp_path / "run"
    parent.mkdir()
    snapshot = module._snapshot_export_directory(parent)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    real_save = module.mx.save_safetensors
    replacement: Path | None = None
    original: Path | None = None
    swapped = False

    def replace_stage_before_first_write(path, tensors):
        nonlocal replacement, original, swapped
        if not swapped:
            swapped = True
            replacement = next(parent.glob(".export.*"))
            original = replacement.with_name(f"{replacement.name}.original")
            replacement.rename(original)
            replacement.mkdir()
            (replacement / "owner.txt").write_text("competitor\n", encoding="utf-8")
        return real_save(path, tensors)

    monkeypatch.setattr(module.mx, "save_safetensors", replace_stage_before_first_write)
    try:
        module._export_merged_checkpoint_under_bound_parent(
            source_checkpoint_dir=source,
            output_name="export",
            parent_descriptor=descriptor,
            parent_snapshot=snapshot,
            source_values={"weight": mx.array([1.0], dtype=mx.float32)},
            parameter_count=1,
            processor_stats={},
            metadata={"test": True},
            disk_free_before=1,
        )
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("renamed export stage was accepted")
    finally:
        os.close(descriptor)

    assert replacement is not None and original is not None
    assert tuple(path.name for path in replacement.iterdir()) == ("owner.txt",)
    assert (original / "model.safetensors").is_file()
    assert not (parent / "export").exists()


@pytest.mark.parametrize(
    "attacked_name",
    ("model.safetensors", "policy_preprocessor.json"),
)
def test_bound_export_child_writes_never_follow_inserted_symlinks(
    tmp_path: Path,
    monkeypatch,
    attacked_name: str,
) -> None:
    module = __import__(
        "training.export",
        fromlist=["_export_merged_checkpoint_under_bound_parent"],
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    parent = tmp_path / "run"
    parent.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"guard\n")
    snapshot = module._snapshot_export_directory(parent)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    real_create = module._create_exclusive_export_file_at
    inserted = False

    if attacked_name == "model.safetensors":
        _install_tiny_bound_export_fakes(module, monkeypatch)
    else:
        monkeypatch.setattr(module, "_MINIMUM_FREE_BYTES", 0)

        def save_processor_support(
            _directory,
            *,
            directory_descriptor,
            **_kwargs,
        ):
            module._write_export_child_bytes_at(
                directory_descriptor,
                attacked_name,
                b"{}",
            )

        monkeypatch.setattr(module, "_save_processors", save_processor_support)

    def insert_symlink(parent_descriptor, name, **kwargs):
        nonlocal inserted
        if not inserted and name == attacked_name:
            inserted = True
            os.symlink(outside, name, dir_fd=parent_descriptor)
        return real_create(parent_descriptor, name, **kwargs)

    monkeypatch.setattr(module, "_create_exclusive_export_file_at", insert_symlink)
    try:
        module._export_merged_checkpoint_under_bound_parent(
            source_checkpoint_dir=source,
            output_name="export",
            parent_descriptor=descriptor,
            parent_snapshot=snapshot,
            source_values={"weight": mx.array([1.0], dtype=mx.float32)},
            parameter_count=1,
            processor_stats={},
            metadata={"test": True},
            disk_free_before=1,
        )
    except (FileExistsError, RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("export serializer followed an inserted child symlink")
    finally:
        os.close(descriptor)

    assert outside.read_bytes() == b"guard\n"
    assert inserted
    assert not (parent / "export").exists()


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
