"""Command-line surface tests for the packaged native runtime."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest


def test_cli_exposes_required_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "smolvla_mlx.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for command in ("convert", "test", "bench", "predict", "serve", "train"):
        assert command in completed.stdout


def test_predict_requires_exactly_one_observation_source() -> None:
    from smolvla_mlx.cli import _parser

    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["predict"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["predict", "--dataset", "owner/data", "--observation", "sample"]
        )

    saved = parser.parse_args(["predict", "--observation", "sample"])
    assert saved.observation.name == "sample"
    assert saved.dataset is None
    assert saved.execution_mode == "production"

    strict = parser.parse_args(
        ["predict", "--observation", "sample", "--execution-mode", "strict"]
    )
    assert strict.execution_mode == "strict"


def test_serve_parser_defaults_to_safe_local_production() -> None:
    from smolvla_mlx.cli import _parser

    args = _parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.dtype == "bfloat16"
    assert args.execution_mode == "production"
    assert args.allow_remote is False


def test_saved_observation_loads_arrays_and_matching_task(tmp_path) -> None:
    from smolvla_mlx.cli import _saved_observation

    sample = tmp_path / "sample_007"
    raw = sample / "raw"
    raw.mkdir(parents=True)
    np.save(raw / "camera1.npy", np.zeros((3, 4, 5), dtype=np.uint8))
    np.save(raw / "camera2.npy", np.ones((3, 4, 5), dtype=np.uint8))
    np.save(raw / "state.npy", np.arange(6, dtype=np.float32))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        '{"samples":[{"name":"sample_000","task":"wrong"},'
        '{"name":"sample_007","task":"place the block"}]}',
        encoding="utf-8",
    )

    observation = _saved_observation(sample, metadata)

    assert observation["task"] == "place the block"
    np.testing.assert_array_equal(observation["observation.state"], np.arange(6, dtype=np.float32))
