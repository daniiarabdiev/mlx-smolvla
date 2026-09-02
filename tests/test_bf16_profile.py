"""Stage Q P2-2 component-profile contracts for the bf16 anomaly."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest


def _document() -> dict[str, object]:
    from smolvla_mlx.profile import (
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
                "smolvla_mlx/benchmark.py": "f" * 64,
                "smolvla_mlx/policy.py": "1" * 64,
                "smolvla_mlx/profile.py": "2" * 64,
            },
        },
        "profiles": profiles,
        "analysis": derive_dtype_analysis(profiles),
    }


def test_profile_protocol_fixes_dtypes_boundaries_and_counts() -> None:
    from smolvla_mlx.profile import ProfileProtocol

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
    from smolvla_mlx.profile import DtypeProfile, ProfileProtocol

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
    from smolvla_mlx.profile import validate_profile_document

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
    from smolvla_mlx.profile import validate_profile_document

    canonical_round_trip = json.loads(json.dumps(_document(), sort_keys=True))
    assert validate_profile_document(canonical_round_trip) == canonical_round_trip


def test_profile_script_uses_clean_idle_coordinator_and_isolated_workers() -> None:
    source = Path("scripts/profile_inference_dtypes.py").read_text(encoding="utf-8")
    assert "_tracked_worktree_is_clean()" in source
    assert "_competing_processes()" in source
    assert 'choices=("float32", "bfloat16")' in source
    assert "subprocess.run(" in source
