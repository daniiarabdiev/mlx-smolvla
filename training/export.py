"""Merged native SmolVLA export in the standard LeRobot safetensors layout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from huggingface_hub import snapshot_download
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from safetensors import safe_open

from reference.discovery import (
    BASE_VLM_ID,
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
)
from smolvla_mlx.convert import source_tensor_names, target_name_for_source
from training.gradients import canonical_parameter_name


_PATCH_CONV_SOURCE = "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"
_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
_EXPECTED_TENSORS = 500
_EXPECTED_PARAMETERS = 450_046_176
_MINIMUM_FREE_BYTES = 40 * 1024**3
_CAMERA_RENAME_MAP = {
    "observation.images.side": "observation.images.camera1",
    "observation.images.up": "observation.images.camera2",
}


@dataclass(frozen=True)
class ExportReport:
    """Auditable identity of one complete merged checkpoint export."""

    output_dir: Path
    tensor_count: int
    parameter_count: int
    dtype: str
    file_sha256: Mapping[str, str]
    disk_free_before_bytes: int
    disk_free_after_bytes: int


def resolve_base_checkpoint(cache_dir: str | Path) -> Path:
    """Resolve the complete immutable source checkpoint into a local cache."""

    return Path(
        snapshot_download(
            CHECKPOINT_ID,
            revision=CHECKPOINT_REVISION,
            cache_dir=str(cache_dir),
            allow_patterns=list(_CHECKPOINT_FILES),
        )
    )


def source_name_map(source_names: tuple[str, ...]) -> dict[str, str]:
    """Return the strict inverse canonical-name map for a source tensor set."""

    mapping: dict[str, str] = {}
    for source_name in source_names:
        target_name = target_name_for_source(source_name)
        if target_name in mapping:
            raise ValueError(f"source names collide at canonical target {target_name}")
        mapping[target_name] = source_name
    if len(mapping) != len(source_names):
        raise ValueError("source/canonical tensor mapping is not one-to-one")
    return mapping


def source_layout_tensor(source_name: str, value: mx.array) -> mx.array:
    """Reverse the sole native layout transform and return fp32 source layout."""

    value = value.astype(mx.float32)
    if source_name == _PATCH_CONV_SOURCE:
        if value.ndim != 4:
            raise ValueError(f"native patch convolution must be OHWI, got {value.shape}")
        return value.transpose(0, 3, 1, 2)
    return value


def canonical_model_tensors(model: nn.Module) -> dict[str, mx.array]:
    """Collect the complete plain training composition under checkpoint names."""

    tensors: dict[str, mx.array] = {}
    for name, value in tree_flatten(model.parameters()):
        if "lora_" in name or ".base." in name:
            raise ValueError("merged export cannot contain adapter wrappers")
        canonical = canonical_parameter_name(name)
        if canonical in tensors:
            raise ValueError(f"duplicate canonical model tensor: {canonical}")
        tensors[canonical] = value
    if len(tensors) != _EXPECTED_TENSORS:
        raise ValueError(f"merged model has {len(tensors)} tensors, expected {_EXPECTED_TENSORS}")
    parameter_count = sum(value.size for value in tensors.values())
    if parameter_count != _EXPECTED_PARAMETERS:
        raise ValueError(
            f"merged model has {parameter_count} scalars, expected {_EXPECTED_PARAMETERS}"
        )
    return tensors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_merged_checkpoint_export(
    output_dir: str | Path,
    *,
    expected_metadata: Mapping[str, object],
) -> ExportReport:
    """Validate and describe a completely published merged export."""

    output_candidate = Path(output_dir)
    if output_candidate.is_symlink() or not output_candidate.is_dir():
        raise ValueError(f"merged export directory is missing or unsafe: {output_dir}")
    output_dir = output_candidate.resolve()
    manifest_path = output_dir / "training_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("merged export manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_manifest_fields = {
        "format_version",
        "artifact_type",
        "dtype",
        "tensor_count",
        "parameter_count",
        "source_checkpoint",
        "metadata",
        "file_sha256",
    }
    if set(manifest) != required_manifest_fields:
        raise ValueError("merged export manifest fields differ from the frozen schema")
    if (
        manifest["format_version"] != 1
        or manifest["artifact_type"] != "smolvla-mlx-merged-training-checkpoint"
        or manifest["dtype"] != "float32"
        or manifest["tensor_count"] != _EXPECTED_TENSORS
        or manifest["parameter_count"] != _EXPECTED_PARAMETERS
        or manifest["source_checkpoint"]
        != {"repo_id": CHECKPOINT_ID, "revision": CHECKPOINT_REVISION}
    ):
        raise ValueError("merged export manifest identity is invalid")
    if manifest["metadata"] != dict(expected_metadata):
        raise ValueError("merged export metadata differs from the requested run")
    hashes = manifest["file_sha256"]
    if not isinstance(hashes, Mapping) or not set(_CHECKPOINT_FILES) <= set(hashes):
        raise ValueError("merged export file manifest is incomplete")
    for name, expected_digest in hashes.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("merged export manifest contains an unsafe filename")
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"merged export file is missing or unsafe: {name}")
        if _sha256_file(path) != expected_digest:
            raise ValueError(f"merged export file digest is invalid: {name}")

    tensor_count = 0
    parameter_count = 0
    with safe_open(output_dir / "model.safetensors", framework="np") as tensors:
        for name in tensors.keys():
            tensor_count += 1
            tensor = tensors.get_slice(name)
            if tensor.get_dtype() != "F32":
                raise ValueError(f"merged export model tensor is not fp32: {name}")
            parameter_count += math.prod(tensor.get_shape())
    if tensor_count != _EXPECTED_TENSORS or parameter_count != _EXPECTED_PARAMETERS:
        raise ValueError("merged export model tensor inventory is invalid")
    disk_free = shutil.disk_usage(output_dir.parent).free
    return ExportReport(
        output_dir=output_dir,
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        dtype="float32",
        file_sha256=dict(hashes),
        disk_free_before_bytes=disk_free,
        disk_free_after_bytes=disk_free,
    )


def _save_processors(
    directory: Path,
    *,
    source_checkpoint_dir: Path,
    processor_stats: Mapping[str, Mapping[str, object]],
) -> None:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig.from_pretrained(source_checkpoint_dir)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=source_checkpoint_dir,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {
                "features": {
                    **config.input_features,
                    **config.output_features,
                },
                "norm_map": config.normalization_mapping,
                "stats": processor_stats,
            },
            "rename_observations_processor": {"rename_map": _CAMERA_RENAME_MAP},
            "tokenizer_processor": {"tokenizer_name": BASE_VLM_ID},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
                "stats": processor_stats,
            },
        },
    )
    preprocessor.save_pretrained(directory)
    postprocessor.save_pretrained(directory)


def export_merged_checkpoint(
    *,
    model: nn.Module,
    source_checkpoint_dir: str | Path,
    output_dir: str | Path,
    processor_stats: Mapping[str, Mapping[str, object]],
    metadata: Mapping[str, object],
) -> ExportReport:
    """Atomically export all 500 merged tensors and stats-active processors."""

    source_checkpoint_dir = Path(source_checkpoint_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing export {output_dir}")
    if not (source_checkpoint_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"source checkpoint is incomplete: {source_checkpoint_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    disk_free_before = shutil.disk_usage(output_dir.parent).free
    if disk_free_before < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"merged export requires at least {_MINIMUM_FREE_BYTES} free bytes, got {disk_free_before}"
        )

    canonical = canonical_model_tensors(model)
    source_names = source_tensor_names(source_checkpoint_dir / "model.safetensors")
    inverse = source_name_map(source_names)
    if set(inverse) != set(canonical):
        raise ValueError(
            "merged model/source name sets differ; "
            f"missing={sorted(set(inverse) - set(canonical))}, "
            f"unexpected={sorted(set(canonical) - set(inverse))}"
        )
    source_values = {
        source_name: source_layout_tensor(source_name, canonical[target_name])
        for target_name, source_name in inverse.items()
    }
    parameter_count = sum(value.size for value in source_values.values())
    if parameter_count != _EXPECTED_PARAMETERS:
        raise RuntimeError("source-layout export changed the checkpoint scalar count")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        mx.save_safetensors(str(temporary / "model.safetensors"), source_values)
        shutil.copyfile(source_checkpoint_dir / "config.json", temporary / "config.json")
        _save_processors(
            temporary,
            source_checkpoint_dir=source_checkpoint_dir,
            processor_stats=processor_stats,
        )
        required = set(_CHECKPOINT_FILES)
        present = {path.name for path in temporary.iterdir() if path.is_file()}
        if not required <= present:
            raise RuntimeError(f"standard export is missing files: {sorted(required - present)}")
        file_hashes = {
            path.name: _sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        manifest = {
            "format_version": 1,
            "artifact_type": "smolvla-mlx-merged-training-checkpoint",
            "dtype": "float32",
            "tensor_count": len(source_values),
            "parameter_count": parameter_count,
            "source_checkpoint": {
                "repo_id": CHECKPOINT_ID,
                "revision": CHECKPOINT_REVISION,
            },
            "metadata": dict(metadata),
            "file_sha256": file_hashes,
        }
        _write_json(temporary / "training_manifest.json", manifest)
        for path in temporary.iterdir():
            if path.is_file():
                _sync_file(path)
        _sync_directory(temporary)
        temporary.replace(output_dir)
        _sync_directory(output_dir.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    disk_free_after = shutil.disk_usage(output_dir.parent).free
    if disk_free_after < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"merged export left less than {_MINIMUM_FREE_BYTES} free bytes: {disk_free_after}"
        )
    return ExportReport(
        output_dir=output_dir,
        tensor_count=len(source_values),
        parameter_count=parameter_count,
        dtype="float32",
        file_sha256=file_hashes,
        disk_free_before_bytes=disk_free_before,
        disk_free_after_bytes=disk_free_after,
    )
