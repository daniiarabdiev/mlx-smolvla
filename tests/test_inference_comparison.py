"""Frozen Stage Q P2-1 MLX-versus-PyTorch-MPS benchmark contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest


def _document() -> dict[str, object]:
    from reference.benchmark import ComparisonProtocol, EngineTiming
    from reference.discovery import CHECKPOINT_REVISION

    protocol = ComparisonProtocol()
    return {
        "artifact_type": "smolvla-mlx-inference-comparison",
        "format_version": 1,
        "protocol": protocol.as_dict(),
        "idle": {
            "verified": True,
            "checked_at_utc": "2026-09-02T00:00:00+00:00",
            "matching_processes": [],
        },
        "environment": {
            "cpu": "Apple M5 Pro",
            "unified_memory_bytes": 48 * 1024**3,
            "macos": "26.6.2",
            "python": "3.12.13",
            "mlx": "0.32.2",
            "torch": "2.11.0",
            "lerobot": "0.6.1",
        },
        "inputs": {
            "sample": "tests/golden/sample_000",
            "sample_sha256": "a" * 64,
            "noise_sha256": "b" * 64,
            "checkpoint_id": "lerobot/smolvla_base",
            "checkpoint_revision": CHECKPOINT_REVISION,
        },
        "source": {
            "git_commit": "d" * 40,
            "tracked_worktree_clean": True,
            "sha256": {
                "reference/benchmark.py": "e" * 64,
                "reference/policy.py": "e" * 64,
                "scripts/benchmark_inference_comparison.py": "f" * 64,
                "mlx_smolvla/benchmark.py": "e" * 64,
                "mlx_smolvla/policy.py": "e" * 64,
            },
        },
        "engines": [
            EngineTiming.from_samples(
                engine="mlx",
                samples_ms=[100.0 + index for index in range(50)],
                warmup_runs=5,
                peak_memory_bytes=2_000_000_000,
                device="Device(gpu, 0)",
                dtype="float32",
                fallback_enabled=False,
            ).as_dict(),
            EngineTiming.from_samples(
                engine="pytorch-mps",
                samples_ms=[200.0 + index for index in range(50)],
                warmup_runs=5,
                peak_memory_bytes=3_000_000_000,
                device="mps:0",
                dtype="float32",
                fallback_enabled=True,
            ).as_dict(),
        ],
    }


def test_comparative_protocol_is_fixed_to_same_case_dtype_and_counts() -> None:
    from reference.benchmark import ComparisonProtocol

    protocol = ComparisonProtocol()
    assert protocol.engines == ("mlx", "pytorch-mps")
    assert protocol.sample == "sample_000"
    assert protocol.dtype == "float32"
    assert protocol.warmup_runs == 5
    assert protocol.measured_runs == 50
    assert protocol.boundary == "preprocessing-through-normalized-action-chunk"
    with pytest.raises(ValueError, match="fixed"):
        ComparisonProtocol(measured_runs=49)


def test_engine_timing_recomputes_summary_from_all_raw_measurements() -> None:
    from reference.benchmark import EngineTiming

    samples = [float(index) for index in range(1, 51)]
    result = EngineTiming.from_samples(
        engine="mlx",
        samples_ms=samples,
        warmup_runs=5,
        peak_memory_bytes=123,
        device="gpu",
        dtype="float32",
        fallback_enabled=False,
    )
    assert result.median_ms == 25.5
    assert result.p95_ms == pytest.approx(47.55)
    assert result.measured_runs == 50
    assert result.samples_ms == tuple(samples)
    assert math.isfinite(result.chunks_per_second)
    assert result.chunks_per_second == 1000 / result.median_ms


def test_comparison_validator_rejects_incomplete_or_recomputed_evidence() -> None:
    from reference.benchmark import validate_comparison_document

    document = _document()
    validated = validate_comparison_document(document)
    assert [item["engine"] for item in validated["engines"]] == [
        "mlx",
        "pytorch-mps",
    ]

    missing = copy.deepcopy(document)
    missing["engines"] = missing["engines"][:1]
    with pytest.raises(ValueError, match="matrix"):
        validate_comparison_document(missing)

    changed = copy.deepcopy(document)
    changed["engines"][0]["median_ms"] += 0.001
    with pytest.raises(ValueError, match="summary"):
        validate_comparison_document(changed)

    not_idle = copy.deepcopy(document)
    not_idle["idle"]["matching_processes"] = ["pytest"]
    with pytest.raises(ValueError, match="idle"):
        validate_comparison_document(not_idle)


def test_torch_worker_enables_mps_fallback_before_framework_imports() -> None:
    source = Path("scripts/benchmark_inference_comparison.py").read_text(encoding="utf-8")
    assignment = source.index('os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"')
    clearing = source.rindex("os.environ.pop(name, None)", 0, assignment)
    torch_import = source.index("import torch", assignment)
    reference_import = source.index("from reference.policy import ReferencePolicy", assignment)
    assert clearing < assignment
    assert assignment < torch_import
    assert assignment < reference_import


def test_reference_loader_rejects_non_cpu_or_mps_devices_before_resolution() -> None:
    from reference.policy import ReferencePolicy

    with pytest.raises(ValueError, match="cpu.*mps"):
        ReferencePolicy.load(Path("unused"), device="cuda")


def test_committed_comparison_artifact_revalidates_from_raw_timings() -> None:
    from reference.benchmark import validate_comparison_document

    path = Path("INFERENCE_COMPARISON.json")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    validated = validate_comparison_document(artifact)
    assert validated == artifact
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "115ad58c0c618b65a6275018614f3ee6cf17dd02a9d4ad9c94aaf7e5a9842e48"
    )


def test_benchmark_document_traces_comparison_values_to_artifact() -> None:
    artifact = json.loads(Path("INFERENCE_COMPARISON.json").read_text(encoding="utf-8"))
    benchmark = Path("BENCHMARK.md").read_text(encoding="utf-8")
    assert "## MLX versus PyTorch-MPS" in benchmark
    assert "115ad58c0c618b65a6275018614f3ee6cf17dd02a9d4ad9c94aaf7e5a9842e48" in benchmark
    for engine in artifact["engines"]:
        assert f"{engine['median_ms']:.2f}" in benchmark
        assert f"{engine['p95_ms']:.2f}" in benchmark
        assert f"{engine['chunks_per_second']:.3f}" in benchmark
        assert f"{engine['peak_memory_gib']:.2f}" in benchmark
    speedup = artifact["engines"][1]["median_ms"] / artifact["engines"][0]["median_ms"]
    assert f"{speedup:.3f}×" in benchmark
