"""Stage T5 fixed training benchmark protocol tests."""

from __future__ import annotations


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
