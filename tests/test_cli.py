"""Command-line surface tests for the packaged native runtime."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_cli_exposes_required_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "mlx_smolvla.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for command in ("convert", "test", "bench", "predict", "serve", "train", "doctor"):
        assert command in completed.stdout


def test_doctor_command_prints_a_complete_json_report(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_smolvla.cli",
            "doctor",
            "--cache-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["cache_path"] == str(tmp_path.resolve())
    assert report["package_version"]
    assert report["mlx_version"]
    assert report["compatibility"]["message"]


def test_predict_requires_exactly_one_observation_source() -> None:
    from mlx_smolvla.cli import _parser

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
    assert saved.quantization is None

    strict = parser.parse_args(
        ["predict", "--observation", "sample", "--execution-mode", "strict"]
    )
    assert strict.execution_mode == "strict"

    quantized = parser.parse_args(
        ["predict", "--observation", "sample", "--quantization", "vlm-4bit"]
    )
    assert quantized.quantization == "vlm-4bit"

    benchmark = parser.parse_args(["bench", "--quantization", "vlm-8bit"])
    assert benchmark.quantization == "vlm-8bit"

    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "--quantization", "all-4bit"])


def test_serve_parser_defaults_to_safe_local_production() -> None:
    from mlx_smolvla.cli import _parser

    args = _parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.dtype == "bfloat16"
    assert args.execution_mode == "production"
    assert args.quantization is None
    assert args.latency_log is None
    assert args.allow_remote is False

    quantized = _parser().parse_args(
        [
            "serve",
            "--quantization",
            "vlm-8bit",
            "--latency-log",
            ".cache/hardware/session.jsonl",
        ]
    )
    assert quantized.quantization == "vlm-8bit"
    assert quantized.latency_log.as_posix() == ".cache/hardware/session.jsonl"


def test_saved_observation_loads_arrays_and_matching_task(tmp_path) -> None:
    from mlx_smolvla.cli import _saved_observation

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
