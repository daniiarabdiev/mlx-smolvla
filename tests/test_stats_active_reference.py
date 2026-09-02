"""Local stats-active reference artifact construction contracts."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def test_flatten_dataset_stats_preserves_every_numeric_vector() -> None:
    module = __import__("reference.stats_active", fromlist=["flatten_dataset_stats"])
    stats = {
        "observation.state": {"mean": [1.0, 2.0], "std": [3.0, 4.0], "count": [5]},
        "action": {"mean": [6.0, 7.0], "std": [8.0, 9.0]},
    }

    tensors = module.flatten_dataset_stats(stats)

    assert set(tensors) == {
        "observation.state.mean",
        "observation.state.std",
        "observation.state.count",
        "action.mean",
        "action.std",
    }
    for value in tensors.values():
        assert value.dtype == mx.float32
    np.testing.assert_array_equal(np.asarray(tensors["action.mean"]), [6.0, 7.0])


def test_build_stats_active_artifact_binds_source_and_dataset(
    tmp_path: Path,
) -> None:
    module = __import__("reference.stats_active", fromlist=["build_stats_active_artifact"])
    source = tmp_path / "source"
    source.mkdir()
    for name, payload in {
        "config.json": b"{}",
        "model.safetensors": b"model",
        "policy_preprocessor.json": b"{}",
        "policy_postprocessor.json": b"{}",
    }.items():
        (source / name).write_bytes(payload)
    dataset_stats_path = tmp_path / "stats.json"
    dataset_stats_path.write_text(
        json.dumps(
            {
                "observation.state": {"mean": [1.0, 2.0], "std": [3.0, 4.0]},
                "action": {"mean": [5.0, 6.0], "std": [7.0, 8.0]},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "artifact"

    report = module.build_stats_active_artifact(
        source_checkpoint=source,
        dataset_stats_path=dataset_stats_path,
        output_dir=output,
        checkpoint_id="test/checkpoint",
        checkpoint_revision="abc123",
        dataset_id="test/dataset",
        dataset_revision="def456",
    )

    assert report["artifact_type"] == "smolvla-stats-active-reference-checkpoint"
    assert report["checkpoint"] == {"id": "test/checkpoint", "revision": "abc123"}
    assert report["dataset"] == {"id": "test/dataset", "revision": "def456"}
    assert set(report["files"]) == {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    assert json.loads((output / "artifact.json").read_text(encoding="utf-8")) == report
    pre = mx.load(str(output / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post = mx.load(str(output / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    assert set(pre) == set(post)
    np.testing.assert_array_equal(np.asarray(pre["observation.state.mean"]), [1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(post["action.std"]), [7.0, 8.0])
    assert module.build_stats_active_artifact(
        source_checkpoint=source,
        dataset_stats_path=dataset_stats_path,
        output_dir=output,
        checkpoint_id="test/checkpoint",
        checkpoint_revision="abc123",
        dataset_id="test/dataset",
        dataset_revision="def456",
    ) == report
