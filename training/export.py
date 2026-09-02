"""Merged native SmolVLA export in the standard LeRobot safetensors layout."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Iterator, Mapping

from huggingface_hub import snapshot_download
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
from safetensors import safe_open

from reference.discovery import (
    BASE_VLM_ID,
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
)
from mlx_smolvla.convert import source_tensor_names, target_name_for_source
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
_SUPPORT_FILES = tuple(
    name for name in _CHECKPOINT_FILES if name != "model.safetensors"
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


@dataclass(frozen=True)
class _ExportDirectorySnapshot:
    path: Path
    components: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class _ExportFileSnapshot:
    name: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str
    payload: bytes | None = None


def _snapshot_export_file_at(
    parent_descriptor: int,
    name: str,
    *,
    capture_payload: bool = False,
) -> _ExportFileSnapshot:
    """Hash one exact regular child of a retained export directory."""

    if Path(name).name != name:
        raise ValueError("export child name must be direct")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"export child is not a regular file: {name}")
        digest = hashlib.sha256()
        payload = bytearray() if capture_payload else None
        byte_count = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
            if payload is not None:
                payload.extend(chunk)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            or (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or byte_count != after.st_size
        ):
            raise RuntimeError(f"export child changed while read: {name}")
        return _ExportFileSnapshot(
            name=name,
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
            sha256=digest.hexdigest(),
            payload=None if payload is None else bytes(payload),
        )
    finally:
        os.close(descriptor)


def _same_export_file_snapshot(
    left: _ExportFileSnapshot,
    right: _ExportFileSnapshot,
) -> bool:
    return (
        left.name,
        left.device,
        left.inode,
        left.size,
        left.mtime_ns,
        left.sha256,
    ) == (
        right.name,
        right.device,
        right.inode,
        right.size,
        right.mtime_ns,
        right.sha256,
    )


@contextmanager
def _bound_export_file_at(
    parent_descriptor: int,
    name: str,
    *,
    expected: _ExportFileSnapshot,
) -> Iterator[int]:
    """Keep an exact export child open across a path-only binary reader."""

    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (expected.device, expected.inode):
            raise RuntimeError(f"export child changed before open: {name}")
        yield descriptor
        current = _snapshot_export_file_at(parent_descriptor, name)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (expected.device, expected.inode)
            or not _same_export_file_snapshot(current, expected)
        ):
            raise RuntimeError(f"export child changed while in use: {name}")
    finally:
        os.close(descriptor)


def _snapshot_export_directory(path: str | Path) -> _ExportDirectorySnapshot:
    """Bind a lexical directory path while rejecting symlinked ancestry."""

    absolute = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(absolute.anchor)
    components: list[tuple[str, int, int]] = []
    candidates = [current]
    for part in absolute.parts[1:]:
        current /= part
        candidates.append(current)
    for candidate in candidates:
        value = os.lstat(candidate)
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise ValueError(f"export directory has unsafe ancestry: {candidate}")
        components.append((str(candidate), value.st_dev, value.st_ino))
    return _ExportDirectorySnapshot(path=absolute, components=tuple(components))


def _revalidate_export_directory(snapshot: _ExportDirectorySnapshot) -> None:
    if _snapshot_export_directory(snapshot.path) != snapshot:
        raise RuntimeError(f"export directory changed while in use: {snapshot.path}")


def _export_descriptor_path(descriptor: int) -> Path:
    """Resolve the current Darwin path naming an open export directory."""

    if sys.platform != "darwin" or not hasattr(fcntl, "F_GETPATH"):
        raise RuntimeError("export descriptor path resolution requires macOS F_GETPATH")
    payload = fcntl.fcntl(descriptor, fcntl.F_GETPATH, b"\0" * 1024)
    path = Path(os.fsdecode(payload.rstrip(b"\0")))
    named = os.stat(path, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
        raise RuntimeError("export descriptor path does not name the opened inode")
    return path


def _create_staged_export_directory_at(
    parent_descriptor: int,
    *,
    prefix: str,
) -> tuple[int, str, Path]:
    """Create and retain one unpredictable staging directory below a bound fd."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _ in range(256):
        name = f"{prefix}{os.urandom(12).hex()}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            os.close(descriptor)
            raise RuntimeError("staged export changed during creation")
        return descriptor, name, _export_descriptor_path(descriptor)
    raise RuntimeError("export staging namespace is exhausted")


def _create_exclusive_export_file_at(
    parent_descriptor: int,
    name: str,
    *,
    mode: int = 0o600,
) -> tuple[int, os.stat_result]:
    """Create and bind one exact export child without following a path."""

    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("bound export filename must be direct")
    descriptor = os.open(
        name,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
        dir_fd=parent_descriptor,
    )
    opened = os.fstat(descriptor)
    try:
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except BaseException:
        os.close(descriptor)
        raise
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        os.close(descriptor)
        raise RuntimeError(f"bound export child changed during creation: {name}")
    return descriptor, opened


def _write_export_child_bytes_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
) -> _ExportFileSnapshot:
    """Write exact bytes through a newly bound export-child descriptor."""

    descriptor, identity = _create_exclusive_export_file_at(
        parent_descriptor,
        name,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while serializing export child: {name}")
            view = view[written:]
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino) != (identity.st_dev, identity.st_ino):
            raise RuntimeError(f"bound export child changed while writing: {name}")
    finally:
        os.close(descriptor)
    snapshot = _snapshot_export_file_at(parent_descriptor, name)
    if (snapshot.device, snapshot.inode) != (identity.st_dev, identity.st_ino):
        raise RuntimeError(f"bound export child name changed: {name}")
    os.fsync(parent_descriptor)
    return snapshot


def _save_export_safetensors_at(
    parent_descriptor: int,
    name: str,
    tensors: Mapping[str, mx.array],
) -> _ExportFileSnapshot:
    """Serialize MLX safetensors through a bound export-child descriptor."""

    descriptor, identity = _create_exclusive_export_file_at(
        parent_descriptor,
        name,
    )
    with os.fdopen(descriptor, "w+b") as handle:
        mx.save_safetensors(handle, tensors)
        handle.flush()
        os.fsync(handle.fileno())
        after = os.fstat(handle.fileno())
        if (after.st_dev, after.st_ino) != (identity.st_dev, identity.st_ino):
            raise RuntimeError(f"bound export safetensors child changed: {name}")
    snapshot = _snapshot_export_file_at(parent_descriptor, name)
    if (snapshot.device, snapshot.inode) != (identity.st_dev, identity.st_ino):
        raise RuntimeError(f"bound export safetensors child name changed: {name}")
    os.fsync(parent_descriptor)
    return snapshot


@contextmanager
def _working_export_directory_at(descriptor: int) -> Iterator[Path]:
    """Run path-only serializers relative to one retained export-stage inode."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    previous = os.open(".", flags)
    expected = os.fstat(descriptor)
    try:
        os.fchdir(descriptor)
        current = os.stat(".", follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("bound export working directory changed during entry")
        yield Path(".")
        current = os.stat(".", follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("bound export working directory changed while in use")
    finally:
        os.fchdir(previous)
        os.close(previous)


def _renameatx_np(
    *,
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
    flags: int,
) -> None:
    """Invoke Darwin's descriptor-relative atomic rename primitive."""

    if sys.platform != "darwin":
        raise RuntimeError("atomic no-clobber directory publication requires macOS")
    libc = ctypes.CDLL(None, use_errno=True)
    renameatx = libc.renameatx_np
    renameatx.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameatx.restype = ctypes.c_int
    result = renameatx(
        source_descriptor,
        os.fsencode(source_name),
        destination_descriptor,
        os.fsencode(destination_name),
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {
        errno.EEXIST,
        errno.ENOTEMPTY,
        errno.EISDIR,
        errno.ELOOP,
    }:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    raise OSError(error_number, os.strerror(error_number), source_name)


def _publish_directory_no_clobber(
    staged: str | Path,
    destination: str | Path,
    *,
    expected_parent: _ExportDirectorySnapshot | None = None,
    parent_descriptor: int | None = None,
    expected_staged_identity: tuple[int, int] | None = None,
) -> None:
    """Atomically install one complete directory without replacing any entry."""

    staged = Path(os.path.abspath(Path(staged).expanduser()))
    destination = Path(os.path.abspath(Path(destination).expanduser()))
    if staged.parent != destination.parent:
        raise ValueError("staged export and destination must share one parent")
    staged_before_parent_open = _snapshot_export_directory(staged)
    if expected_staged_identity is None:
        expected_staged_identity = (
            staged_before_parent_open.components[-1][1],
            staged_before_parent_open.components[-1][2],
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    parent = (
        _snapshot_export_directory(destination.parent)
        if expected_parent is None
        else expected_parent
    )
    bound_parent_descriptor = (
        os.open(parent.path, flags)
        if parent_descriptor is None
        else os.dup(parent_descriptor)
    )
    try:
        parent_value = os.fstat(bound_parent_descriptor)
        _, expected_device, expected_inode = parent.components[-1]
        if (parent_value.st_dev, parent_value.st_ino) != (
            expected_device,
            expected_inode,
        ):
            raise RuntimeError("export parent changed before publication")
        bound_parent = _export_descriptor_path(bound_parent_descriptor)
        bound_parent_snapshot = _snapshot_export_directory(bound_parent)
        staged = bound_parent / staged.name
        destination = bound_parent / destination.name
        staged_snapshot = _snapshot_export_directory(staged)
        staged_value = os.lstat(staged)
        _, staged_device, staged_inode = staged_snapshot.components[-1]
        if (
            stat.S_ISLNK(staged_value.st_mode)
            or not stat.S_ISDIR(staged_value.st_mode)
            or (staged_value.st_dev, staged_value.st_ino)
            != (staged_device, staged_inode)
            or (staged_device, staged_inode) != expected_staged_identity
        ):
            raise RuntimeError("staged export changed before publication")
        staged_before_publish = os.stat(
            staged.name,
            dir_fd=bound_parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(staged_before_publish.st_mode)
            or (staged_before_publish.st_dev, staged_before_publish.st_ino)
            != (staged_device, staged_inode)
        ):
            raise RuntimeError("staged export changed before publication")
        rename_excl = 0x00000004
        rename_nofollow_any = 0x00000010
        try:
            _renameatx_np(
                source_descriptor=bound_parent_descriptor,
                source_name=staged.name,
                destination_descriptor=bound_parent_descriptor,
                destination_name=destination.name,
                flags=rename_excl | rename_nofollow_any,
            )
        except FileExistsError as error:
            raise FileExistsError(
                error.errno,
                f"refusing to overwrite existing export {destination}",
            ) from error
        published = os.stat(
            destination.name,
            dir_fd=bound_parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(published.st_mode)
            or (published.st_dev, published.st_ino)
            != (staged_device, staged_inode)
        ):
            for _ in range(1_000_000):
                failed_name = (
                    f".{destination.name}.publication-failed-"
                    f"{os.urandom(12).hex()}"
                )
                try:
                    _renameatx_np(
                        source_descriptor=bound_parent_descriptor,
                        source_name=destination.name,
                        destination_descriptor=bound_parent_descriptor,
                        destination_name=failed_name,
                        flags=rename_excl | rename_nofollow_any,
                    )
                except FileExistsError:
                    continue
                quarantined = os.stat(
                    failed_name,
                    dir_fd=bound_parent_descriptor,
                    follow_symlinks=False,
                )
                if (quarantined.st_dev, quarantined.st_ino) != (
                    published.st_dev,
                    published.st_ino,
                ):
                    raise RuntimeError(
                        "published export changed during quarantine"
                    )
                os.fsync(bound_parent_descriptor)
                break
            else:
                raise RuntimeError("export publication failure namespace is exhausted")
            raise RuntimeError("staged export changed during publication")
        os.fsync(bound_parent_descriptor)
        _revalidate_export_directory(bound_parent_snapshot)
    finally:
        os.close(bound_parent_descriptor)


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
    output_snapshot = _snapshot_export_directory(output_candidate)
    output_dir = output_snapshot.path
    expected_inventory = set(_CHECKPOINT_FILES) | {"training_manifest.json"}

    def validate_inventory() -> None:
        entries = tuple(output_dir.iterdir())
        names = {entry.name for entry in entries}
        unsafe = sorted(
            entry.name
            for entry in entries
            if entry.is_symlink() or not entry.is_file()
        )
        if names != expected_inventory or unsafe:
            raise ValueError(
                "merged export directory inventory differs; "
                f"missing={sorted(expected_inventory - names)}, "
                f"extra={sorted(names - expected_inventory)}, unsafe={unsafe}"
            )

    validate_inventory()
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
    if not isinstance(manifest, Mapping) or set(manifest) != required_manifest_fields:
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
    if not isinstance(hashes, Mapping) or set(hashes) != set(_CHECKPOINT_FILES):
        raise ValueError("merged export file manifest inventory differs")
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
    validate_inventory()
    _revalidate_export_directory(output_snapshot)
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
    tokenizer_dir: Path | None = None,
    directory_descriptor: int | None = None,
) -> None:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig.from_pretrained(
        source_checkpoint_dir,
        local_files_only=True,
    )
    tokenizer_override: dict[str, object] = {"tokenizer_name": BASE_VLM_ID}
    local_tokenizer = None
    if tokenizer_dir is not None:
        from transformers import AutoTokenizer

        tokenizer_dir = Path(tokenizer_dir)
        if tokenizer_dir.is_symlink() or not tokenizer_dir.is_dir():
            raise ValueError("export tokenizer directory is missing or unsafe")
        local_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
        )
        tokenizer_override["tokenizer"] = local_tokenizer
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
            "tokenizer_processor": tokenizer_override,
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
                "stats": processor_stats,
            },
        },
    )
    if local_tokenizer is not None:
        tokenizer_steps = [
            step
            for step in preprocessor.steps
            if getattr(step.__class__, "_registry_name", None)
            == "tokenizer_processor"
        ]
        if len(tokenizer_steps) != 1:
            raise RuntimeError("export preprocessor tokenizer topology changed")
        tokenizer_step = tokenizer_steps[0]
        if tokenizer_step.input_tokenizer is not local_tokenizer:
            raise RuntimeError("export preprocessor did not retain the pinned tokenizer")
        tokenizer_step.tokenizer_name = BASE_VLM_ID
        tokenizer_step.tokenizer = None
    descriptor = (
        os.open(
            directory,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if directory_descriptor is None
        else os.dup(directory_descriptor)
    )
    try:
        from safetensors.torch import save as serialize_safetensors

        for pipeline in (preprocessor, postprocessor):
            sanitized_name = pipeline._get_sanitized_name()
            config_name = f"{sanitized_name}.json"
            configuration = json.dumps(pipeline.get_config(), indent=2).encode("utf-8")
            state = pipeline.state_dict()
            for state_key, state_tensors in state.items():
                state_name = f"{state_key}.safetensors"
                _write_export_child_bytes_at(
                    descriptor,
                    state_name,
                    serialize_safetensors(state_tensors),
                )
            _write_export_child_bytes_at(
                descriptor,
                config_name,
                configuration,
            )
    finally:
        os.close(descriptor)


def expected_merged_checkpoint_support_file_sha256(
    *,
    source_checkpoint_dir: str | Path,
    processor_stats: Mapping[str, Mapping[str, object]],
    tokenizer_dir: str | Path | None = None,
) -> dict[str, str]:
    """Regenerate the deterministic non-model export files and hash them."""

    source_candidate = Path(source_checkpoint_dir)
    if source_candidate.is_symlink() or not source_candidate.is_dir():
        raise ValueError("merged export source checkpoint is missing or unsafe")
    source_checkpoint_dir = source_candidate.resolve()
    resolved_tokenizer_dir = (
        None if tokenizer_dir is None else Path(tokenizer_dir).resolve(strict=True)
    )
    temporary_parent = Path(os.path.realpath(tempfile.gettempdir()))
    parent_snapshot = _snapshot_export_directory(temporary_parent)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(temporary_parent, directory_flags)
    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    temporary_device: int | None = None
    temporary_inode: int | None = None
    try:
        opened_parent = os.fstat(parent_descriptor)
        _, parent_device, parent_inode = parent_snapshot.components[-1]
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_device,
            parent_inode,
        ):
            raise RuntimeError("export support temporary parent changed")
        temporary_descriptor, temporary_name, temporary = (
            _create_staged_export_directory_at(
                parent_descriptor,
                prefix="smolvla-export-support-",
            )
        )
        temporary_identity = os.fstat(temporary_descriptor)
        temporary_device = temporary_identity.st_dev
        temporary_inode = temporary_identity.st_ino
        _write_export_child_bytes_at(
            temporary_descriptor,
            "config.json",
            (source_checkpoint_dir / "config.json").read_bytes(),
        )
        _save_processors(
            _export_descriptor_path(temporary_descriptor),
            source_checkpoint_dir=source_checkpoint_dir,
            processor_stats=processor_stats,
            tokenizer_dir=resolved_tokenizer_dir,
            directory_descriptor=temporary_descriptor,
        )
        temporary = _export_descriptor_path(temporary_descriptor)
        missing = [name for name in _SUPPORT_FILES if not (temporary / name).is_file()]
        if missing:
            raise RuntimeError(f"generated export support files are incomplete: {missing}")
        return {
            name: _sha256_file(temporary / name)
            for name in _SUPPORT_FILES
        }
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if (
            temporary_name is not None
            and temporary_device is not None
            and temporary_inode is not None
        ):
            try:
                remaining = os.stat(
                    temporary_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                remaining = None
            if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
                temporary_device,
                temporary_inode,
            ):
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise RuntimeError(
                        "safe export-support cleanup requires fd-based rmtree"
                    )
                shutil.rmtree(temporary_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
        os.close(parent_descriptor)


def validate_merged_checkpoint_support_files(
    output_dir: str | Path,
    *,
    expected_sha256: Mapping[str, str],
) -> dict[str, str]:
    """Bind an existing export's support files to independently derived bytes."""

    if set(expected_sha256) != set(_SUPPORT_FILES) or any(
        not isinstance(value, str) or len(value) != 64
        for value in expected_sha256.values()
    ):
        raise ValueError("expected merged export support file evidence is invalid")
    output_candidate = Path(output_dir)
    if output_candidate.is_symlink() or not output_candidate.is_dir():
        raise ValueError("merged export support directory is missing or unsafe")
    output_dir = output_candidate.resolve()
    actual: dict[str, str] = {}
    for name in _SUPPORT_FILES:
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"merged export support file is missing or unsafe: {name}")
        actual[name] = _sha256_file(path)
    expected = dict(expected_sha256)
    if actual != expected:
        raise ValueError("merged export support file bytes differ from frozen inputs")
    return actual


def validate_merged_checkpoint_model_values(
    output_dir: str | Path,
    *,
    model: nn.Module,
    source_checkpoint_dir: str | Path,
    expected_model_sha256: str,
) -> str:
    """Prove an existing export exactly equals the current merged model values."""

    if (
        not isinstance(expected_model_sha256, str)
        or len(expected_model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_model_sha256)
    ):
        raise ValueError("expected merged model digest is invalid")
    output_candidate = Path(output_dir)
    source_candidate = Path(source_checkpoint_dir)
    if output_candidate.is_symlink() or not output_candidate.is_dir():
        raise ValueError("merged export model directory is missing or unsafe")
    if source_candidate.is_symlink() or not source_candidate.is_dir():
        raise ValueError("merged export source checkpoint is missing or unsafe")
    model_path = output_candidate.resolve() / "model.safetensors"
    if model_path.is_symlink() or not model_path.is_file():
        raise ValueError("merged export model file is missing or unsafe")
    before_sha256 = _sha256_file(model_path)
    if before_sha256 != expected_model_sha256:
        raise ValueError("merged export model digest changed before value validation")

    canonical = canonical_model_tensors(model)
    source_names = source_tensor_names(
        source_candidate.resolve() / "model.safetensors"
    )
    inverse = source_name_map(source_names)
    if set(inverse) != set(canonical):
        raise ValueError("current merged model/source tensor names differ")
    with safe_open(model_path, framework="np") as exported:
        if set(exported.keys()) != set(source_names):
            raise ValueError("exported model tensor names differ from the frozen source")
        for target_name, source_name in sorted(inverse.items()):
            expected = np.asarray(
                source_layout_tensor(source_name, canonical[target_name])
            )
            actual = exported.get_tensor(source_name)
            if (
                actual.dtype != np.float32
                or actual.shape != expected.shape
                or not np.array_equal(actual, expected)
            ):
                raise ValueError(
                    f"exported tensor differs from the current merged model: {source_name}"
                )
    after_sha256 = _sha256_file(model_path)
    if after_sha256 != before_sha256:
        raise RuntimeError("merged export model changed during value validation")
    return after_sha256


def validate_bound_merged_checkpoint_export(
    *,
    output_descriptor: int,
    source_checkpoint_descriptor: int,
    expected_metadata: Mapping[str, object],
    expected_support_sha256: Mapping[str, str],
    model: nn.Module,
) -> ExportReport:
    """Validate every release-facing export claim against one retained inode."""

    output_identity = os.fstat(output_descriptor)
    source_identity = os.fstat(source_checkpoint_descriptor)
    if not stat.S_ISDIR(output_identity.st_mode) or not stat.S_ISDIR(
        source_identity.st_mode
    ):
        raise ValueError("bound export validation requires directory descriptors")
    expected_inventory = set(_CHECKPOINT_FILES) | {"training_manifest.json"}
    if set(os.listdir(output_descriptor)) != expected_inventory:
        raise ValueError("bound merged export inventory differs from the frozen schema")
    snapshots = {
        name: _snapshot_export_file_at(
            output_descriptor,
            name,
            capture_payload=name == "training_manifest.json",
        )
        for name in sorted(expected_inventory)
    }
    manifest_payload = snapshots["training_manifest.json"].payload
    assert manifest_payload is not None
    try:
        manifest = json.loads(manifest_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("bound merged export manifest is not valid JSON") from error
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
    if not isinstance(manifest, Mapping) or set(manifest) != required_manifest_fields:
        raise ValueError("merged export manifest fields differ from the frozen schema")
    if (
        manifest["format_version"] != 1
        or manifest["artifact_type"]
        != "smolvla-mlx-merged-training-checkpoint"
        or manifest["dtype"] != "float32"
        or manifest["tensor_count"] != _EXPECTED_TENSORS
        or manifest["parameter_count"] != _EXPECTED_PARAMETERS
        or manifest["source_checkpoint"]
        != {"repo_id": CHECKPOINT_ID, "revision": CHECKPOINT_REVISION}
    ):
        raise ValueError("merged export manifest identity is invalid")
    if manifest["metadata"] != dict(expected_metadata):
        raise ValueError("merged export metadata differs from the requested run")
    manifest_hashes = manifest["file_sha256"]
    actual_hashes = {
        name: snapshots[name].sha256 for name in _CHECKPOINT_FILES
    }
    if not isinstance(manifest_hashes, Mapping) or dict(manifest_hashes) != actual_hashes:
        raise ValueError("merged export file digests differ from its manifest")
    if (
        set(expected_support_sha256) != set(_SUPPORT_FILES)
        or {
            name: actual_hashes[name] for name in _SUPPORT_FILES
        }
        != dict(expected_support_sha256)
    ):
        raise ValueError("merged export support file bytes differ from frozen inputs")

    canonical = canonical_model_tensors(model)
    source_model = _snapshot_export_file_at(
        source_checkpoint_descriptor,
        "model.safetensors",
    )
    with _bound_export_file_at(
        source_checkpoint_descriptor,
        "model.safetensors",
        expected=source_model,
    ) as source_descriptor:
        with safe_open(f"/dev/fd/{source_descriptor}", framework="np") as source:
            source_names = tuple(source.keys())
    inverse = source_name_map(source_names)
    if set(inverse) != set(canonical):
        raise ValueError("current merged model/source tensor names differ")

    tensor_count = 0
    parameter_count = 0
    model_snapshot = snapshots["model.safetensors"]
    with _bound_export_file_at(
        output_descriptor,
        "model.safetensors",
        expected=model_snapshot,
    ) as model_descriptor:
        with safe_open(f"/dev/fd/{model_descriptor}", framework="np") as exported:
            if set(exported.keys()) != set(source_names):
                raise ValueError(
                    "exported model tensor names differ from the frozen source"
                )
            for target_name, source_name in sorted(inverse.items()):
                expected = np.asarray(
                    source_layout_tensor(source_name, canonical[target_name])
                )
                actual = exported.get_tensor(source_name)
                tensor_count += 1
                parameter_count += actual.size
                if (
                    actual.dtype != np.float32
                    or actual.shape != expected.shape
                    or not np.array_equal(actual, expected)
                ):
                    raise ValueError(
                        "exported tensor differs from the current merged model: "
                        f"{source_name}"
                    )
    if tensor_count != _EXPECTED_TENSORS or parameter_count != _EXPECTED_PARAMETERS:
        raise ValueError("merged export model tensor inventory is invalid")

    if set(os.listdir(output_descriptor)) != expected_inventory:
        raise RuntimeError("bound merged export inventory changed while in use")
    after = {
        name: _snapshot_export_file_at(output_descriptor, name)
        for name in sorted(expected_inventory)
    }
    if any(
        not _same_export_file_snapshot(snapshots[name], after[name])
        for name in expected_inventory
    ):
        raise RuntimeError("bound merged export bytes changed while in use")
    output_after = os.fstat(output_descriptor)
    source_after = os.fstat(source_checkpoint_descriptor)
    if (output_after.st_dev, output_after.st_ino) != (
        output_identity.st_dev,
        output_identity.st_ino,
    ) or (source_after.st_dev, source_after.st_ino) != (
        source_identity.st_dev,
        source_identity.st_ino,
    ):
        raise RuntimeError("bound export validation directory changed")
    output_dir = _export_descriptor_path(output_descriptor)
    disk_free = shutil.disk_usage(output_dir.parent).free
    return ExportReport(
        output_dir=output_dir,
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        dtype="float32",
        file_sha256=actual_hashes,
        disk_free_before_bytes=disk_free,
        disk_free_after_bytes=disk_free,
    )


def _export_merged_checkpoint_under_bound_parent(
    *,
    source_checkpoint_dir: Path,
    output_name: str,
    parent_descriptor: int,
    parent_snapshot: _ExportDirectorySnapshot,
    source_values: Mapping[str, mx.array],
    parameter_count: int,
    processor_stats: Mapping[str, Mapping[str, object]],
    metadata: Mapping[str, object],
    disk_free_before: int,
    tokenizer_dir: Path | None = None,
) -> ExportReport:
    """Build, publish, validate, and clean below one retained parent dirfd."""

    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    temporary_device: int | None = None
    temporary_inode: int | None = None
    file_hashes: dict[str, str]
    try:
        temporary_descriptor, temporary_name, temporary = (
            _create_staged_export_directory_at(
                parent_descriptor,
                prefix=f".{output_name}.",
            )
        )
        temporary_identity = os.fstat(temporary_descriptor)
        temporary_device = temporary_identity.st_dev
        temporary_inode = temporary_identity.st_ino
        source_checkpoint_dir = source_checkpoint_dir.resolve(strict=True)
        _save_export_safetensors_at(
            temporary_descriptor,
            "model.safetensors",
            source_values,
        )
        _write_export_child_bytes_at(
            temporary_descriptor,
            "config.json",
            (source_checkpoint_dir / "config.json").read_bytes(),
        )
        _save_processors(
            _export_descriptor_path(temporary_descriptor),
            source_checkpoint_dir=source_checkpoint_dir,
            processor_stats=processor_stats,
            tokenizer_dir=tokenizer_dir,
            directory_descriptor=temporary_descriptor,
        )
        temporary = _export_descriptor_path(temporary_descriptor)
        required = set(_CHECKPOINT_FILES)
        present = {path.name for path in temporary.iterdir() if path.is_file()}
        if not required <= present:
            raise RuntimeError(
                f"standard export is missing files: {sorted(required - present)}"
            )
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
        _write_export_child_bytes_at(
            temporary_descriptor,
            "training_manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        for path in temporary.iterdir():
            if path.is_file():
                _sync_file(path)
        os.fsync(temporary_descriptor)
        named_staging = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named_staging.st_mode)
            or (named_staging.st_dev, named_staging.st_ino)
            != (temporary_device, temporary_inode)
        ):
            raise RuntimeError("staged export changed before publication")
        _publish_directory_no_clobber(
            _export_descriptor_path(parent_descriptor) / temporary_name,
            _export_descriptor_path(parent_descriptor) / output_name,
            expected_parent=parent_snapshot,
            parent_descriptor=parent_descriptor,
            expected_staged_identity=(temporary_device, temporary_inode),
        )
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if (
            temporary_name is not None
            and temporary_device is not None
            and temporary_inode is not None
        ):
            try:
                remaining = os.stat(
                    temporary_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                remaining = None
            if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
                temporary_device,
                temporary_inode,
            ):
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise RuntimeError("safe export cleanup requires fd-based rmtree")
                shutil.rmtree(temporary_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)

    bound_parent = _export_descriptor_path(parent_descriptor)
    output_dir = bound_parent / output_name
    disk_free_after = shutil.disk_usage(bound_parent).free
    if disk_free_after < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"merged export left less than {_MINIMUM_FREE_BYTES} free bytes: {disk_free_after}"
        )
    validated = validate_merged_checkpoint_export(
        output_dir,
        expected_metadata=metadata,
    )
    if (
        validated.tensor_count != len(source_values)
        or validated.parameter_count != parameter_count
        or validated.dtype != "float32"
        or dict(validated.file_sha256) != file_hashes
    ):
        raise RuntimeError("published export differs from the staged export report")
    _revalidate_export_directory(parent_snapshot)
    return ExportReport(
        output_dir=validated.output_dir,
        tensor_count=validated.tensor_count,
        parameter_count=validated.parameter_count,
        dtype=validated.dtype,
        file_sha256=validated.file_sha256,
        disk_free_before_bytes=disk_free_before,
        disk_free_after_bytes=disk_free_after,
    )


def export_merged_checkpoint(
    *,
    model: nn.Module,
    source_checkpoint_dir: str | Path,
    output_dir: str | Path,
    processor_stats: Mapping[str, Mapping[str, object]],
    metadata: Mapping[str, object],
    tokenizer_dir: str | Path | None = None,
    output_parent_descriptor: int | None = None,
    expected_output_parent: tuple[str | Path, int, int] | None = None,
) -> ExportReport:
    """Atomically export all 500 merged tensors and stats-active processors."""

    source_checkpoint_dir = Path(source_checkpoint_dir).resolve()
    resolved_tokenizer_dir = (
        None if tokenizer_dir is None else Path(tokenizer_dir).resolve(strict=True)
    )
    output_dir = Path(os.path.abspath(Path(output_dir).expanduser()))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if output_parent_descriptor is None:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        parent_snapshot = _snapshot_export_directory(output_dir.parent)
        parent_descriptor = os.open(output_dir.parent, directory_flags)
    else:
        if expected_output_parent is None:
            raise ValueError("bound merged export requires parent identity")
        expected_path = Path(
            os.path.abspath(Path(expected_output_parent[0]).expanduser())
        )
        if output_dir.parent != expected_path:
            raise ValueError("merged export output differs from its bound parent")
        parent_descriptor = os.dup(output_parent_descriptor)
        opened_parent = os.fstat(parent_descriptor)
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            expected_output_parent[1],
            expected_output_parent[2],
        ):
            os.close(parent_descriptor)
            raise RuntimeError("merged export parent descriptor changed")
        bound_parent = _export_descriptor_path(parent_descriptor)
        parent_snapshot = _snapshot_export_directory(bound_parent)
        output_dir = bound_parent / output_dir.name
    try:
        opened_parent = os.fstat(parent_descriptor)
        _, parent_device, parent_inode = parent_snapshot.components[-1]
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_device,
            parent_inode,
        ):
            raise RuntimeError("merged export parent changed before use")
        try:
            os.stat(
                output_dir.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite existing export {output_dir}")
        if not (source_checkpoint_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"source checkpoint is incomplete: {source_checkpoint_dir}"
            )
        disk_free_before = shutil.disk_usage(_export_descriptor_path(parent_descriptor)).free
        if disk_free_before < _MINIMUM_FREE_BYTES:
            raise RuntimeError(
                f"merged export requires at least {_MINIMUM_FREE_BYTES} free bytes, "
                f"got {disk_free_before}"
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

        return _export_merged_checkpoint_under_bound_parent(
            source_checkpoint_dir=source_checkpoint_dir,
            output_name=output_dir.name,
            parent_descriptor=parent_descriptor,
            parent_snapshot=parent_snapshot,
            source_values=source_values,
            parameter_count=parameter_count,
            processor_stats=processor_stats,
            tokenizer_dir=resolved_tokenizer_dir,
            metadata=metadata,
            disk_free_before=disk_free_before,
        )
    finally:
        os.close(parent_descriptor)
