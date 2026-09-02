"""Stage T5 fixed training benchmark protocol tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def test_training_benchmark_protocol_is_fixed() -> None:
    from training.benchmark import TrainingBenchmarkConfig

    config = TrainingBenchmarkConfig()
    assert config.warmup_updates == 3
    assert config.measured_updates == 10
    assert config.batch_size == 8
    assert config.training_horizon == 3000
    assert config.modes == ("lora", "full")
    assert config.dtypes == ("bfloat16", "float32")


def test_training_timing_summary_uses_median_update_rate_and_per_1k_projection() -> None:
    from training.benchmark import summarize_update_seconds

    summary = summarize_update_seconds((1.0, 2.0, 3.0, 4.0))
    assert summary == {
        "update_seconds": [1.0, 2.0, 3.0, 4.0],
        "median_update_seconds": 2.5,
        "mean_update_seconds": 2.5,
        "p95_update_seconds": 3.8499999999999996,
        "steps_per_second": 0.4,
        "wall_seconds_per_1000_steps": 2500.0,
    }


def test_training_benchmark_matrix_validator_requires_all_four_cells() -> None:
    from training.benchmark import validate_training_benchmark

    cells = []
    for mode in ("lora", "full"):
        for dtype in ("bfloat16", "float32"):
            cells.append(
                {
                    "mode": mode,
                    "dtype": dtype,
                    "warmup_updates": 3,
                    "measured_updates": 10,
                    "batch_size": 8,
                    "training_horizon": 3000,
                    "update_seconds": [1.0] * 10,
                    "median_update_seconds": 1.0,
                    "mean_update_seconds": 1.0,
                    "p95_update_seconds": 1.0,
                    "steps_per_second": 1.0,
                    "wall_seconds_per_1000_steps": 1000.0,
                    "peak_memory_bytes": 1,
                    "trainable_tensor_count": 1,
                    "trainable_scalar_count": 1,
                }
            )
    document = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-training-benchmark",
        "idle": {"verified": True, "matching_processes": []},
        "protocol": {
            "warmup_updates": 3,
            "measured_updates": 10,
            "batch_size": 8,
            "training_horizon": 3000,
            "modes": ["lora", "full"],
            "dtypes": ["bfloat16", "float32"],
        },
        "cells": cells,
    }
    assert validate_training_benchmark(document)["cells"] == cells

    document["cells"] = cells[:-1]
    try:
        validate_training_benchmark(document)
    except ValueError as error:
        assert "matrix" in str(error)
    else:
        raise AssertionError("incomplete training benchmark matrix was accepted")


def test_committed_training_benchmark_record_recomputes_every_derived_number() -> None:
    record = json.loads(Path("TRAINING_BENCHMARK.json").read_text(encoding="utf-8"))
    assert record["artifact_type"] == "smolvla-mlx-training-benchmark-public-record"
    assert record["source_artifact"]["git_commit"] == (
        "0d897449b06d114d536756f2ed6850b52fd5bda4"
    )
    assert len(record["cells"]) == 4
    for cell in record["cells"]:
        median = cell["median_update_seconds"]
        assert cell["steps_per_second"] == 1 / median
        assert cell["wall_seconds_per_1000_steps"] == median * 1000
        assert cell["wall_minutes_per_1000_steps"] == median * 1000 / 60
        assert cell["projected_3000_minutes"] == median * 3000 / 60
        assert cell["projected_30000_hours"] == median * 30000 / 3600
        assert cell["peak_memory_gib"] == cell["peak_memory_bytes"] / 1024**3
        assert all(math.isfinite(value) and value > 0 for value in (
            median,
            cell["p95_update_seconds"],
            cell["peak_memory_gib"],
        ))

    local = Path(".cache/training/t5-benchmark.json")
    if local.is_file():
        assert hashlib.sha256(local.read_bytes()).hexdigest() == (
            record["source_artifact"]["file_sha256"]
        )
        measured = json.loads(local.read_text(encoding="utf-8"))
        for public, source in zip(record["cells"], measured["cells"], strict=True):
            for field in (
                "mode",
                "dtype",
                "median_update_seconds",
                "mean_update_seconds",
                "p95_update_seconds",
                "steps_per_second",
                "wall_seconds_per_1000_steps",
                "peak_memory_bytes",
            ):
                assert public[field] == source[field]


def test_training_benchmark_and_readme_numbers_trace_to_committed_record() -> None:
    record = json.loads(Path("TRAINING_BENCHMARK.json").read_text(encoding="utf-8"))
    benchmark = Path("BENCHMARK.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "## Native training performance (Metal)" in benchmark
    assert "## Fine-tune on your Mac" in readme
    assert record["source_artifact"]["file_sha256"] in benchmark
    for cell in record["cells"]:
        assert f"{cell['median_update_seconds']:.3f}" in benchmark
        assert f"{cell['steps_per_second']:.3f}" in benchmark
        assert f"{cell['wall_minutes_per_1000_steps']:.2f}" in benchmark
        assert f"{cell['peak_memory_gib']:.2f}" in benchmark
    for required in (
        "--lora",
        "--dtype bfloat16",
        "--steps 30000",
        "--batch-size 8",
        "--checkpoint-every 100",
        "--resume",
        "--model .cache/training/overnight/export",
    ):
        assert required in readme
