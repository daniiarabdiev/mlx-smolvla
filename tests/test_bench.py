"""Schema and warmup-boundary tests for native inference benchmarks."""

from __future__ import annotations

from smolvla_mlx.benchmark import BenchmarkResult


def test_benchmark_result_has_required_metrics_and_excludes_warmups() -> None:
    measured = {
        "preprocessing": [1.0 + index * 0.01 for index in range(50)],
        "vision": [2.0 + index * 0.01 for index in range(50)],
        "prefix": [3.0 + index * 0.01 for index in range(50)],
        "expert": [4.0 + index * 0.01 for index in range(50)],
        "total": [10.0 + index * 0.04 for index in range(50)],
    }
    result = BenchmarkResult.from_stage_samples(
        measured,
        warmup_runs=5,
        peak_memory_bytes=123_456,
        device="Device(gpu, 0)",
        dtype="bfloat16",
    )

    assert result.measured_runs == 50
    assert result.warmup_runs == 5
    assert result.total_ms.median > 0
    assert result.total_ms.p95 >= result.total_ms.median
    assert set(result.stages) == {"preprocessing", "vision", "prefix", "expert"}
    assert result.peak_memory_bytes == 123_456
