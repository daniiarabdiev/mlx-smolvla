"""Validation gates for default production-path correctness evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smolvla_mlx.production_evidence import ProductionDeterministicEvidence
from smolvla_mlx.statistical import StatisticalResult


def _valid_deterministic_payload() -> dict[str, object]:
    return {
        "format_version": 1,
        "execution_mode": "production",
        "device": "Device(gpu, 0)",
        "case_count": 2,
        "checkpoint": {
            "id": "lerobot/smolvla_base",
            "revision": "a" * 40,
            "model_sha256": "b" * 64,
        },
        "golden": {
            "manifest_sha256": "c" * 64,
            "metadata_sha256": "d" * 64,
        },
        "results": {
            "float32": {
                "fixed_max_abs_gate": 0.005,
                "max_abs": 0.01,
                "passed": False,
                "worst_case": "sample_001",
                "samples": [
                    {"name": "sample_000", "max_abs": 0.002},
                    {"name": "sample_001", "max_abs": 0.01},
                ],
            },
            "bfloat16": {
                "fixed_max_abs_gate": 0.05,
                "max_abs": 0.04,
                "passed": True,
                "worst_case": "sample_001",
                "samples": [
                    {"name": "sample_000", "max_abs": 0.03},
                    {"name": "sample_001", "max_abs": 0.04},
                ],
            },
        },
    }


def test_deterministic_evidence_recomputes_outcomes_and_freezes_gates(tmp_path: Path) -> None:
    path = tmp_path / "production.json"
    path.write_text(json.dumps(_valid_deterministic_payload()), encoding="utf-8")

    evidence = ProductionDeterministicEvidence.from_json(path)

    assert evidence.execution_mode == "production"
    assert evidence.device == "Device(gpu, 0)"
    assert evidence.results["float32"].passed is False
    assert evidence.results["bfloat16"].passed is True

    changed = _valid_deterministic_payload()
    changed["results"]["float32"]["fixed_max_abs_gate"] = 0.05  # type: ignore[index]
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        ProductionDeterministicEvidence.from_json(path)


def test_recorded_default_production_deterministic_evidence_is_self_consistent() -> None:
    evidence = ProductionDeterministicEvidence.from_json(
        Path(".cache/production-deterministic.json")
    )

    assert evidence.execution_mode == "production"
    assert evidence.device == "Device(gpu, 0)"
    assert evidence.case_count == 8
    assert set(evidence.results) == {"float32", "bfloat16"}


def test_recorded_default_production_statistical_evidence_uses_fifty_frames() -> None:
    result = StatisticalResult.from_json(Path(".cache/statistical-production.json"))

    assert result.execution_mode == "production"
    assert result.device == "Device(gpu, 0)"
    assert result.sample_count == 50
    assert result.mlx_fp32_ratio <= 1.05
    assert result.mlx_bf16_ratio <= 1.05
