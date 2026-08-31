from pathlib import Path
import json

import mlx.core as mx
import pytest


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_conversion_maps_every_checkpoint_tensor_once(
    checkpoint_dir: Path,
    dtype: str,
) -> None:
    from smolvla_mlx.convert import convert_checkpoint

    output_dir = Path(".cache/smolvla_mlx") / f"conversion-{dtype}"
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
    from smolvla_mlx.convert import build_name_map, source_tensor_names, target_name_for_source

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
