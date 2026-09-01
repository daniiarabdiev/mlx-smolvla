"""Explicit, dependency-isolated SmolVLA safetensors conversion for MLX."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mlx.core as mx
import numpy as np


_NAME_PREFIX_RULES = (
    ("model.vlm_with_expert.vlm.model.vision_model.", "vision."),
    ("model.vlm_with_expert.vlm.model.connector.", "connector."),
    ("model.vlm_with_expert.vlm.model.text_model.", "language."),
    ("model.vlm_with_expert.vlm.lm_head.", "language.lm_head."),
    ("model.vlm_with_expert.lm_expert.", "expert."),
    ("model.", ""),
)
_PATCH_CONV_SOURCE = "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"


@dataclass(frozen=True)
class ConversionReport:
    """Auditable result of one source-checkpoint conversion."""

    source_names: tuple[str, ...]
    target_names: tuple[str, ...]
    unmapped_source: tuple[str, ...]
    uninitialized_target: tuple[str, ...]
    source_parameter_count: int
    target_parameter_count: int
    checksums: Mapping[str, str]
    output_path: Path
    name_map_path: Path
    dtype: str


@dataclass(frozen=True)
class ConversionValidationReport:
    """Digests and inventory for a conversion proven to match its source."""

    tensor_count: int
    parameter_count: int
    source_model_sha256: str
    converted_model_sha256: str
    name_map_sha256: str
    dtype: str


def _read_header(path: Path) -> tuple[dict[str, dict[str, object]], int]:
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), byteorder="little")
        header = json.loads(handle.read(header_size))
    return {name: spec for name, spec in header.items() if name != "__metadata__"}, 8 + header_size


def source_tensor_names(source_path: Path) -> tuple[str, ...]:
    """Read source names from the safetensors header without loading model values."""

    header, _ = _read_header(source_path)
    return tuple(sorted(header))


def target_name_for_source(source_name: str) -> str:
    """Apply one of the committed canonical source-to-native tree rules."""

    for source_prefix, target_prefix in _NAME_PREFIX_RULES:
        if source_name.startswith(source_prefix):
            return f"{target_prefix}{source_name.removeprefix(source_prefix)}"
    raise ValueError(f"No canonical MLX target name exists for {source_name!r}")


def build_name_map(source_names: tuple[str, ...], target_names: tuple[str, ...]) -> dict[str, str]:
    """Validate that the supplied target tree is exactly the canonical bijection."""

    if len(source_names) != len(set(source_names)):
        raise ValueError("Source tensor names are not unique")
    if len(target_names) != len(set(target_names)):
        raise ValueError("Target tensor names are not unique")
    mapping = {source_name: target_name_for_source(source_name) for source_name in source_names}
    if len(mapping) != len(source_names) or len(set(mapping.values())) != len(mapping):
        raise ValueError("Canonical source-to-target mapping is not one-to-one")
    expected = set(mapping.values())
    actual = set(target_names)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"Target tree does not match canonical mapping; missing={missing}, unexpected={unexpected}")
    return mapping


def _sha256_source_slice(source_path: Path, data_start: int, offsets: object) -> str:
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"Invalid safetensors data_offsets: {offsets!r}")
    start, end = (int(offset) for offset in offsets)
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        handle.seek(data_start + start)
        remaining = end - start
        while remaining:
            chunk = handle.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"Unexpected end of source tensor data in {source_path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _parameter_count(shape: object) -> int:
    if not isinstance(shape, list):
        raise ValueError(f"Invalid tensor shape: {shape!r}")
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_dtype(dtype: str) -> tuple[mx.Dtype, str]:
    if dtype == "float32":
        return mx.float32, "float32"
    if dtype == "bfloat16":
        return mx.bfloat16, "bfloat16"
    raise ValueError("dtype must be 'float32' or 'bfloat16'")


def _atomic_save_safetensors(path: Path, tensors: dict[str, mx.array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".safetensors", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        mx.save_safetensors(str(temporary), tensors)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def convert_checkpoint(source_dir: Path, output_dir: Path, dtype: str) -> ConversionReport:
    """Convert all 500 source tensors into the canonical MLX parameter tree.

    Only the patch convolution changes layout: PyTorch OIHW becomes MLX OHWI.
    Linear, embedding, normalization, and projection weights retain their source
    shape and name suffixes.
    """

    source_path = source_dir / "model.safetensors"
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint is missing {source_path}")
    target_dtype, dtype_name = _target_dtype(dtype)
    header, data_start = _read_header(source_path)
    source_names = tuple(sorted(header))
    target_names = tuple(target_name_for_source(name) for name in source_names)
    mapping = build_name_map(source_names, target_names)
    source_values = mx.load(str(source_path))
    if set(source_values) != set(source_names):
        raise ValueError("MLX loaded a different tensor set than the safetensors header")

    converted: dict[str, mx.array] = {}
    records = []
    for source_name in source_names:
        source_value = source_values[source_name]
        target_name = mapping[source_name]
        transform = "identity"
        if source_name == _PATCH_CONV_SOURCE:
            target_value = source_value.transpose(0, 2, 3, 1)
            transform = "OIHW_to_OHWI"
        else:
            target_value = source_value
        target_value = target_value.astype(target_dtype)
        converted[target_name] = target_value
        source_spec = header[source_name]
        checksum = _sha256_source_slice(source_path, data_start, source_spec["data_offsets"])
        records.append(
            {
                "source": source_name,
                "target": target_name,
                "source_shape": source_spec["shape"],
                "target_shape": list(target_value.shape),
                "transform": transform,
                "source_sha256": checksum,
            }
        )

    output_dir = output_dir.resolve()
    output_path = output_dir / f"model.{dtype_name}.safetensors"
    name_map_path = output_dir / "name_map.json"
    _atomic_save_safetensors(output_path, converted)
    target_header, target_data_start = _read_header(output_path)
    target_checksums: dict[str, str] = {}
    for record in records:
        target_name = record["target"]
        if target_name not in target_header:
            raise ValueError(f"Converted safetensors output omitted {target_name}")
        checksum = _sha256_source_slice(output_path, target_data_start, target_header[target_name]["data_offsets"])
        record["target_sha256"] = checksum
        target_checksums[target_name] = checksum
    _atomic_write_json(
        name_map_path,
        {
            "format_version": 1,
            "dtype": dtype_name,
            "rules": [
                {"source_prefix": source_prefix, "target_prefix": target_prefix}
                for source_prefix, target_prefix in _NAME_PREFIX_RULES
            ],
            "tensors": records,
        },
    )

    source_parameter_count = sum(_parameter_count(header[name]["shape"]) for name in source_names)
    target_parameter_count = sum(value.size for value in converted.values())
    uninitialized_target = tuple(sorted(set(target_names) - set(converted)))
    return ConversionReport(
        source_names=source_names,
        target_names=tuple(sorted(converted)),
        unmapped_source=tuple(sorted(set(source_names) - set(mapping))),
        uninitialized_target=uninitialized_target,
        source_parameter_count=source_parameter_count,
        target_parameter_count=target_parameter_count,
        checksums=target_checksums,
        output_path=output_path,
        name_map_path=name_map_path,
        dtype=dtype_name,
    )


def validate_converted_checkpoint(
    source_dir: str | Path,
    output_path: str | Path,
    name_map_path: str | Path,
    *,
    dtype: str,
    expected_tensor_count: int | None = None,
) -> ConversionValidationReport:
    """Prove cached converted tensor bytes derive from the current source export."""

    source_path = Path(source_dir) / "model.safetensors"
    output_path = Path(output_path)
    name_map_path = Path(name_map_path)
    for label, path in (
        ("source checkpoint", source_path),
        ("converted checkpoint", output_path),
        ("conversion name map", name_map_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    target_mx_dtype, dtype_name = _target_dtype(dtype)
    target_dtype = {"float32": "F32", "bfloat16": "BF16"}[dtype_name]
    source_header, source_data_start = _read_header(source_path)
    target_header, target_data_start = _read_header(output_path)
    source_names = tuple(sorted(source_header))
    target_names = tuple(sorted(target_header))
    mapping = build_name_map(source_names, target_names)
    if expected_tensor_count is not None and len(source_names) != expected_tensor_count:
        raise ValueError(
            f"converted tensor count changed: {len(source_names)} != {expected_tensor_count}"
        )

    name_map_payload = name_map_path.read_bytes()
    name_map = json.loads(name_map_payload)
    expected_rules = [
        {"source_prefix": source_prefix, "target_prefix": target_prefix}
        for source_prefix, target_prefix in _NAME_PREFIX_RULES
    ]
    if (
        not isinstance(name_map, dict)
        or set(name_map) != {"format_version", "dtype", "rules", "tensors"}
        or name_map.get("format_version") != 1
        or name_map.get("dtype") != dtype_name
        or name_map.get("rules") != expected_rules
        or not isinstance(name_map.get("tensors"), list)
        or len(name_map["tensors"]) != len(source_names)
    ):
        raise ValueError("conversion name map differs from the canonical schema")
    records = {}
    for record in name_map["tensors"]:
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "source",
                "target",
                "source_shape",
                "target_shape",
                "transform",
                "source_sha256",
                "target_sha256",
            }
            or not isinstance(record.get("source"), str)
            or record["source"] in records
        ):
            raise ValueError("conversion name map tensor records are invalid")
        records[record["source"]] = record
    if set(records) != set(source_names):
        raise ValueError("conversion name map tensor inventory changed")

    source_values = mx.load(str(source_path))
    target_values = mx.load(str(output_path))
    if set(source_values) != set(source_names) or set(target_values) != set(
        target_names
    ):
        raise ValueError("MLX loaded a different tensor set than the safetensors headers")
    mx.eval(source_values, target_values)

    parameter_count = 0
    for source_name in source_names:
        target_name = mapping[source_name]
        source_spec = source_header[source_name]
        target_spec = target_header[target_name]
        source_shape = source_spec.get("shape")
        expected_target_shape = source_shape
        transform = "identity"
        if source_name == _PATCH_CONV_SOURCE:
            if not isinstance(source_shape, list) or len(source_shape) != 4:
                raise ValueError("source patch convolution shape is invalid")
            expected_target_shape = [
                source_shape[0],
                source_shape[2],
                source_shape[3],
                source_shape[1],
            ]
            transform = "OIHW_to_OHWI"
        source_checksum = _sha256_source_slice(
            source_path,
            source_data_start,
            source_spec.get("data_offsets"),
        )
        target_checksum = _sha256_source_slice(
            output_path,
            target_data_start,
            target_spec.get("data_offsets"),
        )
        expected_record_identity = {
            "source": source_name,
            "target": target_name,
            "source_shape": source_shape,
            "target_shape": expected_target_shape,
            "transform": transform,
        }
        record = records[source_name]
        if (
            source_spec.get("dtype") not in {"F32", "BF16"}
            or target_spec.get("dtype") != target_dtype
            or target_spec.get("shape") != expected_target_shape
            or any(
                record.get(name) != value
                for name, value in expected_record_identity.items()
            )
        ):
            raise ValueError(f"converted tensor metadata differs from source export: {source_name}")
        if (
            record.get("source_sha256") != source_checksum
            or record.get("target_sha256") != target_checksum
        ):
            raise ValueError(f"converted tensor differs from source export: {source_name}")
        expected_value = source_values[source_name]
        if transform == "OIHW_to_OHWI":
            expected_value = expected_value.transpose(0, 2, 3, 1)
        expected_value = expected_value.astype(target_mx_dtype)
        target_value = target_values[target_name]
        expected_bytes = np.asarray(expected_value.astype(mx.float32)).tobytes()
        target_bytes = np.asarray(target_value.astype(mx.float32)).tobytes()
        if target_bytes != expected_bytes:
            raise ValueError(
                f"converted tensor differs from source export: {source_name}"
            )
        parameter_count += _parameter_count(source_shape)

    return ConversionValidationReport(
        tensor_count=len(source_names),
        parameter_count=parameter_count,
        source_model_sha256=_file_sha256(source_path),
        converted_model_sha256=_file_sha256(output_path),
        name_map_sha256=hashlib.sha256(name_map_payload).hexdigest(),
        dtype=dtype_name,
    )
