"""Schema and warmup-boundary tests for native inference benchmarks."""

from __future__ import annotations

from smolvla_mlx.benchmark import BenchmarkResult, TimingSummary
from smolvla_mlx.production_evidence import (
    DeterministicDtypeEvidence,
    ProductionDeterministicEvidence,
)
from smolvla_mlx.statistical import StatisticalResult


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
        execution_mode="production",
    )

    assert result.measured_runs == 50
    assert result.warmup_runs == 5
    assert result.total_ms.median > 0
    assert result.total_ms.p95 >= result.total_ms.median
    assert set(result.stages) == {"preprocessing", "vision", "prefix", "expert"}
    assert result.peak_memory_bytes == 123_456
    assert result.execution_mode == "production"
    assert result.as_dict()["execution_mode"] == "production"


def test_benchmark_report_separates_strict_and_default_production_correctness() -> None:
    from scripts.bench import _render

    timing = TimingSummary(median=111.0, p95=112.0)
    benchmark = BenchmarkResult(
        measured_runs=50,
        warmup_runs=5,
        stages={name: timing for name in ("preprocessing", "vision", "prefix", "expert")},
        total_ms=timing,
        peak_memory_bytes=3_000_000_000,
        device="Device(gpu, 0)",
        dtype="float32",
        execution_mode="production",
    )
    samples = ({"name": "sample_000", "max_abs": 0.047},)
    production = ProductionDeterministicEvidence(
        execution_mode="production",
        device="Device(gpu, 0)",
        case_count=1,
        checkpoint={},
        golden={},
        results={
            "float32": DeterministicDtypeEvidence(0.005, 0.047, False, "sample_000", samples),
            "bfloat16": DeterministicDtypeEvidence(0.05, 0.044, True, "sample_000", samples),
        },
    )
    strict_stats = StatisticalResult(50, 1.0, 1.0, 1.0, 1.0, 1.0, "strict", "Device(cpu, 0)")
    production_stats = StatisticalResult(
        50,
        1.0,
        1.00001,
        1.00002,
        1.00001,
        1.00002,
        "production",
        "Device(gpu, 0)",
    )

    report = _render([benchmark], production, strict_stats, production_stats)

    assert "## Strict-parity correctness (CPU)" in report
    assert "## Default-production correctness (Metal)" in report
    assert "0.0470000000" in report
    assert "fail" in report
    assert "## Default-production performance (Metal)" in report
