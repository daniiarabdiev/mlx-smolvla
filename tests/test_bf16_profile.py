"""Stage Q P2-2 component-profile contracts for the bf16 anomaly."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


def _document() -> dict[str, object]:
    from mlx_smolvla.profile import (
        DtypeProfile,
        ProfileProtocol,
        derive_dtype_analysis,
    )

    protocol = ProfileProtocol()
    profiles = []
    for dtype, offset in (("float32", 0.0), ("bfloat16", 5.0)):
        samples = {
            stage: [
                float(index + stage_index + 1) + offset
                for index in range(protocol.measured_runs)
            ]
            for stage_index, stage in enumerate(protocol.stages)
        }
        profiles.append(
            DtypeProfile.from_samples(
                dtype=dtype,
                samples_ms=samples,
                warmup_runs=protocol.warmup_runs,
                peak_memory_bytes=2_000_000_000,
                device="Device(gpu, 0)",
            ).as_dict()
        )
    return {
        "artifact_type": "smolvla-mlx-bf16-component-profile",
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
        },
        "inputs": {
            "sample": "tests/golden/sample_000",
            "sample_sha256": "a" * 64,
            "noise_sha256": "b" * 64,
            "checkpoint_id": "lerobot/smolvla_base",
            "checkpoint_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        },
        "source": {
            "git_commit": "c" * 40,
            "tracked_worktree_clean": True,
            "sha256": {
                "scripts/benchmark_inference_comparison.py": "d" * 64,
                "scripts/profile_inference_dtypes.py": "e" * 64,
                "mlx_smolvla/benchmark.py": "f" * 64,
                "mlx_smolvla/policy.py": "1" * 64,
                "mlx_smolvla/profile.py": "2" * 64,
            },
        },
        "profiles": profiles,
        "analysis": derive_dtype_analysis(profiles),
    }


def test_profile_protocol_fixes_dtypes_boundaries_and_counts() -> None:
    from mlx_smolvla.profile import ProfileProtocol

    protocol = ProfileProtocol()
    assert protocol.dtypes == ("float32", "bfloat16")
    assert protocol.stages == (
        "preprocessing",
        "vision_encoder",
        "connector",
        "prefix",
        "expert_loop",
        "total",
    )
    assert protocol.warmup_runs == 5
    assert protocol.measured_runs == 50
    with pytest.raises(ValueError, match="fixed"):
        ProfileProtocol(warmup_runs=4)


def test_dtype_profile_recomputes_every_stage_summary() -> None:
    from mlx_smolvla.profile import DtypeProfile, ProfileProtocol

    protocol = ProfileProtocol()
    samples = {
        stage: [float(index + 1) for index in range(50)]
        for stage in protocol.stages
    }
    result = DtypeProfile.from_samples(
        dtype="float32",
        samples_ms=samples,
        warmup_runs=5,
        peak_memory_bytes=123,
        device="gpu",
    ).as_dict()
    assert result["summaries_ms"]["total"] == {
        "median": 25.5,
        "p95": pytest.approx(47.55),
    }
    assert result["samples_ms"] == samples


def test_profile_validator_recomputes_matrix_and_delta_attribution() -> None:
    from mlx_smolvla.profile import validate_profile_document

    document = _document()
    assert validate_profile_document(document) == document
    assert document["analysis"]["classification"] == "bf16-slower"
    assert document["analysis"]["total_median_delta_ms"] == 5.0

    missing = copy.deepcopy(document)
    del missing["profiles"][0]["samples_ms"]["connector"]
    with pytest.raises(ValueError, match="stages"):
        validate_profile_document(missing)

    changed = copy.deepcopy(document)
    changed["analysis"]["total_median_delta_ms"] += 1.0
    with pytest.raises(ValueError, match="analysis"):
        validate_profile_document(changed)


def test_profile_validator_accepts_canonical_sorted_json_key_order() -> None:
    from mlx_smolvla.profile import validate_profile_document

    canonical_round_trip = json.loads(json.dumps(_document(), sort_keys=True))
    assert validate_profile_document(canonical_round_trip) == canonical_round_trip


def test_profile_script_uses_clean_idle_coordinator_and_isolated_workers() -> None:
    source = Path("scripts/profile_inference_dtypes.py").read_text(encoding="utf-8")
    assert "_tracked_worktree_is_clean()" in source
    assert "_competing_processes()" in source
    assert 'choices=("float32", "bfloat16")' in source
    assert "subprocess.run(" in source


def test_committed_profile_revalidates_from_all_raw_component_timings() -> None:
    from mlx_smolvla.profile import validate_profile_document

    path = Path("BF16_PROFILE.json")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert validate_profile_document(artifact) == artifact
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "74da9f937cb8bfeba4066d5518187490ff96a1447e4a2ad2253e2493245be1cf"
    )


def test_benchmark_document_traces_profile_and_preserves_default_behavior() -> None:
    artifact = json.loads(Path("BF16_PROFILE.json").read_text(encoding="utf-8"))
    benchmark = Path("BENCHMARK.md").read_text(encoding="utf-8")
    assert "## bf16 latency diagnosis" in benchmark
    assert "74da9f937cb8bfeba4066d5518187490ff96a1447e4a2ad2253e2493245be1cf" in benchmark
    for profile in artifact["profiles"]:
        for stage in artifact["protocol"]["stages"]:
            assert f"{profile['summaries_ms'][stage]['median']:.2f}" in benchmark
    analysis = artifact["analysis"]
    assert f"{analysis['total_slowdown_percent']:.2f}%" in benchmark
    assert "No inference behavior changed" in benchmark
    assert "MLX 0.32.2" in benchmark
