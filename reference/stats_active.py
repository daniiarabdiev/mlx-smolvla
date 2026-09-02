"""Construct a local base checkpoint with pinned dataset normalization statistics."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import mlx.core as mx
import numpy as np


_COPIED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
_PREPROCESSOR_STATE = "policy_preprocessor_step_5_normalizer_processor.safetensors"
_POSTPROCESSOR_STATE = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_dataset_stats(stats: Mapping[str, object]) -> dict[str, mx.array]:
    """Flatten LeRobot's JSON stats into its saved processor state schema."""

    flattened: dict[str, mx.array] = {}
    for feature_name in sorted(stats):
        feature_stats = stats[feature_name]
        if not isinstance(feature_stats, Mapping):
            raise ValueError(f"Dataset stats for {feature_name!r} must be a mapping")
        for stat_name in sorted(feature_stats):
            array = np.asarray(feature_stats[stat_name], dtype=np.float32)
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError(f"Dataset stat {feature_name}.{stat_name} must be finite and nonempty")
            flattened[f"{feature_name}.{stat_name}"] = mx.array(array).astype(mx.float32)
    required = {
        "observation.state.mean",
        "observation.state.std",
        "action.mean",
        "action.std",
    }
    if not required <= set(flattened):
        raise ValueError(f"Dataset stats are missing {sorted(required - set(flattened))}")
    mx.eval(flattened)
    return flattened


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_stats_active_artifact(
    *,
    source_checkpoint: Path,
    dataset_stats_path: Path,
    output_dir: Path,
    checkpoint_id: str,
    checkpoint_revision: str,
    dataset_id: str,
    dataset_revision: str,
) -> dict[str, object]:
    """Create one no-clobber local checkpoint using explicit processor state surgery."""

    source_checkpoint = source_checkpoint.resolve()
    dataset_stats_path = dataset_stats_path.resolve()
    output_dir = output_dir.expanduser().absolute()
    missing = [name for name in _COPIED_FILES if not (source_checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Source checkpoint is missing {missing}")
    if output_dir.exists() or output_dir.is_symlink():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise FileExistsError(f"Stats-active reference artifact path is unsafe: {output_dir}")
        artifact_path = output_dir / "artifact.json"
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise FileExistsError(f"Existing stats-active artifact has no safe manifest: {output_dir}")
        report = json.loads(artifact_path.read_text(encoding="utf-8"))
        expected_identity = {
            "checkpoint": {"id": checkpoint_id, "revision": checkpoint_revision},
            "dataset": {"id": dataset_id, "revision": dataset_revision},
            "dataset_stats_sha256": _sha256(dataset_stats_path),
        }
        if any(report.get(key) != value for key, value in expected_identity.items()):
            raise FileExistsError("Existing stats-active artifact identity differs from the requested input")
        files = report.get("files")
        if not isinstance(files, dict):
            raise FileExistsError("Existing stats-active artifact has an invalid file manifest")
        expected_names = {*_COPIED_FILES, _PREPROCESSOR_STATE, _POSTPROCESSOR_STATE}
        if set(files) != expected_names:
            raise FileExistsError("Existing stats-active artifact file inventory changed")
        for name in expected_names:
            path = output_dir / name
            if path.is_symlink() or not path.is_file() or _sha256(path) != files[name]:
                raise FileExistsError(f"Existing stats-active artifact differs at {name}")
        for name in _COPIED_FILES:
            if _sha256(source_checkpoint / name) != files[name]:
                raise FileExistsError(f"Existing stats-active artifact source changed at {name}")
        return report
    raw_stats = json.loads(dataset_stats_path.read_text(encoding="utf-8"))
    if not isinstance(raw_stats, Mapping):
        raise ValueError("Dataset stats JSON must contain an object")
    tensors = flatten_dataset_stats(raw_stats)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name in _COPIED_FILES:
            _link_or_copy(source_checkpoint / name, temporary / name)
        mx.save_safetensors(str(temporary / _PREPROCESSOR_STATE), tensors)
        mx.save_safetensors(str(temporary / _POSTPROCESSOR_STATE), tensors)
        files = {
            name: _sha256(temporary / name)
            for name in (*_COPIED_FILES, _PREPROCESSOR_STATE, _POSTPROCESSOR_STATE)
        }
        report: dict[str, object] = {
            "format_version": 1,
            "artifact_type": "smolvla-stats-active-reference-checkpoint",
            "checkpoint": {"id": checkpoint_id, "revision": checkpoint_revision},
            "dataset": {"id": dataset_id, "revision": dataset_revision},
            "dataset_stats_sha256": _sha256(dataset_stats_path),
            "normalization": {
                "state": "mean_std",
                "action": "mean_std",
                "epsilon": 1e-8,
                "mechanism": "explicit LeRobot processor state-dict surgery",
            },
            "files": files,
        }
        (temporary / "artifact.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
