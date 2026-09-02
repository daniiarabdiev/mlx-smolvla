from pathlib import Path
import hashlib
import json

import mlx.core as mx
import pytest


def _tensor_data_sha256(path: Path, tensor_name: str) -> str:
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), byteorder="little")
        header = json.loads(handle.read(header_size))
        start, end = header[tensor_name]["data_offsets"]
        handle.seek(8 + header_size + start)
        return hashlib.sha256(handle.read(end - start)).hexdigest()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_conversion_maps_every_checkpoint_tensor_once(
    checkpoint_dir: Path,
    dtype: str,
) -> None:
    from mlx_smolvla.convert import convert_checkpoint

    output_dir = Path(".cache/mlx_smolvla") / f"conversion-{dtype}"
    report = convert_checkpoint(checkpoint_dir, output_dir, dtype=dtype)

    assert report.unmapped_source == ()
    assert report.uninitialized_target == ()
    assert report.source_parameter_count == 450_046_176
    assert report.target_parameter_count == 450_046_176
    assert len(report.source_names) == 500
    assert len(report.target_names) == 500
    assert len(set(report.source_names)) == 500
    assert len(set(report.target_names)) == 500
    assert len(report.checksums) == 500
    assert report.output_path.is_file()
    converted = mx.load(str(report.output_path))
    assert len(converted) == 500
    assert converted["vision.embeddings.patch_embedding.weight"].shape == (768, 16, 16, 3)
    assert converted["language.layers.0.self_attn.q_proj.weight"].shape == (960, 960)
    assert converted["expert.layers.1.self_attn.k_proj.weight"].shape == (320, 320)
    expected_dtype = mx.float32 if dtype == "float32" else mx.bfloat16
    assert converted["vision.embeddings.patch_embedding.weight"].dtype == expected_dtype
    name_map = json.loads(report.name_map_path.read_text(encoding="utf-8"))
    patch_record = next(
        record
        for record in name_map["tensors"]
        if record["target"] == "vision.embeddings.patch_embedding.weight"
    )
    assert patch_record["transform"] == "OIHW_to_OHWI"
    assert len(patch_record["source_sha256"]) == 64
    assert len(patch_record["target_sha256"]) == 64
    assert report.checksums[patch_record["target"]] == patch_record["target_sha256"]


def test_name_map_is_a_strict_bijection(checkpoint_dir: Path) -> None:
    from mlx_smolvla.convert import build_name_map, source_tensor_names, target_name_for_source

    source_names = source_tensor_names(checkpoint_dir / "model.safetensors")
    target_names = tuple(target_name_for_source(name) for name in source_names)
    mapping = build_name_map(source_names, target_names)

    assert len(mapping) == len(source_names) == len(target_names)
    assert mapping["model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"] == (
        "vision.embeddings.patch_embedding.weight"
    )
    assert mapping["model.vlm_with_expert.lm_expert.layers.0.mlp.down_proj.weight"] == (
        "expert.layers.0.mlp.down_proj.weight"
    )


def test_conversion_validation_rejects_a_stale_source_checkpoint(
    tmp_path: Path,
) -> None:
    from mlx_smolvla.convert import (
        convert_checkpoint,
        validate_converted_checkpoint,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "model.safetensors"
    source_tensors = {
        "model.state_proj.weight": mx.arange(12, dtype=mx.float32).reshape(3, 4),
        (
            "model.vlm_with_expert.vlm.model.vision_model.embeddings."
            "patch_embedding.weight"
        ): mx.arange(48, dtype=mx.float32).reshape(2, 3, 2, 4),
    }
    mx.save_safetensors(str(source_path), source_tensors)
    report = convert_checkpoint(source_dir, tmp_path / "converted", dtype="float32")

    validation = validate_converted_checkpoint(
        source_dir,
        report.output_path,
        report.name_map_path,
        dtype="float32",
        expected_tensor_count=2,
    )
    assert validation.tensor_count == 2
    assert validation.parameter_count == 60

    changed = dict(source_tensors)
    changed["model.state_proj.weight"] = changed["model.state_proj.weight"] + 1
    mx.save_safetensors(str(source_path), changed)

    with pytest.raises(ValueError, match="converted tensor differs from source export"):
        validate_converted_checkpoint(
            source_dir,
            report.output_path,
            report.name_map_path,
            dtype="float32",
            expected_tensor_count=2,
        )


def test_conversion_validation_derives_bfloat16_values_from_source(
    tmp_path: Path,
) -> None:
    from mlx_smolvla.convert import (
        convert_checkpoint,
        validate_converted_checkpoint,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensors = {
        "model.state_proj.weight": mx.arange(12, dtype=mx.float32).reshape(3, 4),
        (
            "model.vlm_with_expert.vlm.model.vision_model.embeddings."
            "patch_embedding.weight"
        ): mx.arange(48, dtype=mx.float32).reshape(2, 3, 2, 4),
    }
    mx.save_safetensors(str(source_dir / "model.safetensors"), source_tensors)
    report = convert_checkpoint(source_dir, tmp_path / "converted", dtype="bfloat16")

    validation = validate_converted_checkpoint(
        source_dir,
        report.output_path,
        report.name_map_path,
        dtype="bfloat16",
        expected_tensor_count=2,
    )
    assert validation.dtype == "bfloat16"

    tampered = mx.load(str(report.output_path))
    target_name = "state_proj.weight"
    tampered[target_name] = tampered[target_name] + mx.array(1, dtype=mx.bfloat16)
    tampered_path = report.output_path.with_name("tampered.safetensors")
    mx.save_safetensors(str(tampered_path), tampered)
    tampered_path.replace(report.output_path)
    name_map = json.loads(report.name_map_path.read_text(encoding="utf-8"))
    record = next(
        item for item in name_map["tensors"] if item["target"] == target_name
    )
    record["target_sha256"] = _tensor_data_sha256(report.output_path, target_name)
    report.name_map_path.write_text(
        json.dumps(name_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="converted tensor differs from source export"):
        validate_converted_checkpoint(
            source_dir,
            report.output_path,
            report.name_map_path,
            dtype="bfloat16",
            expected_tensor_count=2,
        )


def test_conversion_validation_accepts_the_pinned_mixed_source_dtypes(
    tmp_path: Path,
) -> None:
    from mlx_smolvla.convert import (
        convert_checkpoint,
        validate_converted_checkpoint,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensors = {
        "model.state_proj.weight": mx.arange(12, dtype=mx.float32).reshape(3, 4),
        "model.vlm_with_expert.lm_expert.layers.0.input_layernorm.weight": (
            mx.arange(4, dtype=mx.float32).astype(mx.bfloat16)
        ),
    }
    mx.save_safetensors(str(source_dir / "model.safetensors"), source_tensors)
    report = convert_checkpoint(source_dir, tmp_path / "converted", dtype="bfloat16")

    validation = validate_converted_checkpoint(
        source_dir,
        report.output_path,
        report.name_map_path,
        dtype="bfloat16",
        expected_tensor_count=2,
    )

    assert validation.tensor_count == 2
    assert validation.parameter_count == 16
