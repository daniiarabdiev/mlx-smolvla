"""Integrity contracts for manifest-backed training artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def test_training_artifact_round_trips_sorted_hashed_arrays(tmp_path: Path) -> None:
    module = __import__("training.data", fromlist=["TrainingArtifactWriter", "TrainingArtifact"])
    writer = module.TrainingArtifactWriter(tmp_path)
    noncontiguous = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]

    writer.add("z/value", noncontiguous)
    writer.add("a/mask", np.array([[True, False]], dtype=np.bool_))
    finalized = writer.finalize({"format_version": 1, "artifact_type": "unit-test"})

    manifest_payload = (tmp_path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_payload)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert list(manifest) == ["a/mask", "z/value"]
    assert metadata == finalized
    assert metadata["manifest_sha256"] == hashlib.sha256(manifest_payload).hexdigest()
    assert metadata["tensor_count"] == 2
    assert manifest["z/value"]["shape"] == [4, 3]
    assert manifest["z/value"]["dtype"] == "float32"
    assert manifest["z/value"]["byte_count"] == (tmp_path / "z/value.npy").stat().st_size

    artifact = module.TrainingArtifact(tmp_path)
    assert artifact.verify_all() == ("a/mask", "z/value")
    np.testing.assert_array_equal(artifact.load("z/value"), noncontiguous)
    assert artifact.load("z/value").flags.c_contiguous
    np.testing.assert_array_equal(
        artifact.load("a/mask"),
        np.array([[True, False]], dtype=np.bool_),
    )


@pytest.mark.parametrize("name", ("", "/absolute", "../parent", "nested/../../escape"))
def test_training_artifact_rejects_unsafe_logical_names(tmp_path: Path, name: str) -> None:
    module = __import__("training.data", fromlist=["TrainingArtifactWriter"])
    writer = module.TrainingArtifactWriter(tmp_path)

    with pytest.raises(ValueError, match="relative"):
        writer.add(name, np.zeros((1,), dtype=np.float32))


def test_training_artifact_rejects_duplicates_objects_and_incomplete_metadata(tmp_path: Path) -> None:
    module = __import__("training.data", fromlist=["TrainingArtifactWriter"])
    writer = module.TrainingArtifactWriter(tmp_path)
    writer.add("value", np.ones((2,), dtype=np.float32))

    with pytest.raises(ValueError, match="duplicate"):
        writer.add("value", np.zeros((2,), dtype=np.float32))
    with pytest.raises(TypeError, match="object"):
        writer.add("objects", np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="format_version"):
        writer.finalize({"artifact_type": "unit-test"})


def test_training_artifact_detects_payload_tampering(tmp_path: Path) -> None:
    module = __import__("training.data", fromlist=["TrainingArtifactWriter", "TrainingArtifact"])
    writer = module.TrainingArtifactWriter(tmp_path)
    writer.add("value", np.arange(8, dtype=np.float32))
    writer.finalize({"format_version": 1, "artifact_type": "unit-test"})
    artifact = module.TrainingArtifact(tmp_path)
    payload_path = tmp_path / "value.npy"
    payload = payload_path.read_bytes()
    payload_path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))

    with pytest.raises(ValueError, match="hash mismatch"):
        artifact.load("value")


def test_training_artifact_detects_manifest_tampering(tmp_path: Path) -> None:
    module = __import__("training.data", fromlist=["TrainingArtifactWriter", "TrainingArtifact"])
    writer = module.TrainingArtifactWriter(tmp_path)
    writer.add("value", np.arange(3, dtype=np.int32))
    writer.finalize({"format_version": 1, "artifact_type": "unit-test"})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        module.TrainingArtifact(tmp_path)


def test_training_artifact_preserves_zero_dimensional_scalars(tmp_path: Path) -> None:
    module = __import__("training.data", fromlist=["TrainingArtifactWriter", "TrainingArtifact"])
    writer = module.TrainingArtifactWriter(tmp_path)
    writer.add("scalar", np.asarray(2.5, dtype=np.float32))
    writer.finalize({"format_version": 1, "artifact_type": "unit-test"})

    scalar = module.TrainingArtifact(tmp_path).load("scalar")

    assert scalar.shape == ()
    assert scalar.item() == 2.5
