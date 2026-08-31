"""Framework-neutral, manifest-backed arrays for deterministic training evidence."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np


_MANIFEST_NAME = "manifest.json"
_METADATA_NAME = "metadata.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_payload(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _validated_relative_path(name: str, *, suffix: str = "") -> Path:
    candidate = Path(name)
    if (
        not name
        or candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        raise ValueError(f"training artifact name must be a safe relative path: {name!r}")
    return Path(f"{name}{suffix}")


class TrainingArtifactWriter:
    """Atomically write contiguous NumPy arrays and their integrity manifest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._entries: dict[str, dict[str, object]] = {}
        self._finalized = False

    def add(self, name: str, value: object) -> None:
        if self._finalized:
            raise RuntimeError("training artifact has already been finalized")
        relative_path = _validated_relative_path(name, suffix=".npy")
        if name in self._entries:
            raise ValueError(f"duplicate training artifact tensor name: {name!r}")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"training artifact arrays cannot have object dtype: {name!r}")
        array = np.ascontiguousarray(array)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        payload = buffer.getvalue()
        _atomic_write(self.root / relative_path, payload)
        self._entries[name] = {
            "path": relative_path.as_posix(),
            "shape": list(array.shape),
            "dtype": array.dtype.name,
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def finalize(self, metadata: Mapping[str, object]) -> dict[str, object]:
        if self._finalized:
            raise RuntimeError("training artifact has already been finalized")
        format_version = metadata.get("format_version")
        if not isinstance(format_version, int) or format_version < 1:
            raise ValueError("training artifact metadata requires a positive integer format_version")
        artifact_type = metadata.get("artifact_type")
        if not isinstance(artifact_type, str) or not artifact_type:
            raise ValueError("training artifact metadata requires a non-empty artifact_type")
        if not self._entries:
            raise ValueError("training artifact must contain at least one tensor")

        manifest = {name: self._entries[name] for name in sorted(self._entries)}
        manifest_payload = _json_payload(manifest)
        _atomic_write(self.root / _MANIFEST_NAME, manifest_payload)
        finalized = dict(metadata)
        finalized["manifest_sha256"] = hashlib.sha256(manifest_payload).hexdigest()
        finalized["tensor_count"] = len(manifest)
        _atomic_write(self.root / _METADATA_NAME, _json_payload(finalized))
        self._finalized = True
        return finalized


class TrainingArtifact:
    """Load and verify a completed training artifact without framework imports."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / _MANIFEST_NAME
        metadata_path = self.root / _METADATA_NAME
        if not manifest_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"training artifact requires {manifest_path} and {metadata_path}"
            )
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        metadata = json.loads(metadata_path.read_bytes())
        if not isinstance(manifest, dict) or not isinstance(metadata, dict):
            raise ValueError("training artifact manifest and metadata must be JSON objects")
        expected_manifest_hash = metadata.get("manifest_sha256")
        actual_manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
        if expected_manifest_hash != actual_manifest_hash:
            raise ValueError(
                "training artifact manifest hash mismatch: "
                f"{actual_manifest_hash} != {expected_manifest_hash}"
            )
        if metadata.get("tensor_count") != len(manifest):
            raise ValueError("training artifact tensor_count does not match its manifest")
        if not isinstance(metadata.get("format_version"), int) or not isinstance(
            metadata.get("artifact_type"), str
        ):
            raise ValueError("training artifact metadata is incomplete")

        for name, record in manifest.items():
            if not isinstance(name, str) or not isinstance(record, dict):
                raise ValueError("training artifact manifest entries must be named JSON objects")
            _validated_relative_path(name)
            required = {"path", "shape", "dtype", "byte_count", "sha256"}
            if set(record) != required:
                raise ValueError(f"training artifact manifest entry is incomplete: {name!r}")
            path = record["path"]
            if not isinstance(path, str):
                raise ValueError(f"training artifact path is not a string: {name!r}")
            _validated_relative_path(path)

        self.manifest: dict[str, dict[str, object]] = manifest
        self.metadata: dict[str, object] = metadata

    def load(self, name: str) -> np.ndarray:
        try:
            record = self.manifest[name]
        except KeyError as error:
            raise KeyError(f"training artifact tensor is absent: {name!r}") from error
        path = self.root / str(record["path"])
        payload = path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != record["sha256"]:
            raise ValueError(
                f"training artifact tensor hash mismatch for {name}: "
                f"{actual_hash} != {record['sha256']}"
            )
        if len(payload) != record["byte_count"]:
            raise ValueError(f"training artifact byte count mismatch for {name}")
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        if list(array.shape) != record["shape"] or array.dtype.name != record["dtype"]:
            raise ValueError(f"training artifact array metadata mismatch for {name}")
        return np.ascontiguousarray(array)

    def verify_all(self) -> tuple[str, ...]:
        names = tuple(sorted(self.manifest))
        for name in names:
            self.load(name)
        return names
