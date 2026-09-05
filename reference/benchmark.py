"""Pure validation helpers for the frozen MLX/PyTorch-MPS comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
import math
import re

import numpy as np

from mlx_smolvla._lab.reference.discovery import CHECKPOINT_ID, CHECKPOINT_REVISION


_ENGINES = ("mlx", "pytorch-mps")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ComparisonProtocol:
    """Immutable Stage Q P2-1 timing boundary and sample population."""

    engines: tuple[str, ...] = _ENGINES
    sample: str = "sample_000"
    dtype: str = "float32"
    warmup_runs: int = 5
    measured_runs: int = 50
    boundary: str = "preprocessing-through-normalized-action-chunk"
    fixed_noise: bool = True

    def __post_init__(self) -> None:
        if (
            self.engines != _ENGINES
            or self.sample != "sample_000"
            or self.dtype != "float32"
            or self.warmup_runs != 5
            or self.measured_runs != 50
            or self.boundary != "preprocessing-through-normalized-action-chunk"
            or self.fixed_noise is not True
        ):
            raise ValueError("comparison protocol differs from the fixed Stage Q P2-1 values")

    def as_dict(self) -> dict[str, object]:
        return {
            "engines": list(self.engines),
            "sample": self.sample,
            "dtype": self.dtype,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "boundary": self.boundary,
            "fixed_noise": self.fixed_noise,
        }


@dataclass(frozen=True)
class EngineTiming:
    """Raw synchronized durations and exactly recomputable summary values."""

    engine: str
    samples_ms: tuple[float, ...]
    measured_runs: int
    warmup_runs: int
    median_ms: float
    p95_ms: float
    chunks_per_second: float
    peak_memory_bytes: int
    device: str
    dtype: str
    fallback_enabled: bool

    @classmethod
    def from_samples(
        cls,
        *,
        engine: str,
        samples_ms: Sequence[float],
        warmup_runs: int,
        peak_memory_bytes: int,
        device: str,
        dtype: str,
        fallback_enabled: bool,
    ) -> "EngineTiming":
        if engine not in _ENGINES:
            raise ValueError(f"unknown comparison engine {engine!r}")
        if warmup_runs != 5:
            raise ValueError("engine warmup count differs from the fixed protocol")
        samples = tuple(float(value) for value in samples_ms)
        if len(samples) != 50:
            raise ValueError("engine measurement count differs from the fixed protocol")
        values = np.asarray(samples, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("engine timings must be finite positive milliseconds")
        if isinstance(peak_memory_bytes, bool) or not isinstance(peak_memory_bytes, int):
            raise ValueError("peak memory must be an integer byte count")
        if peak_memory_bytes <= 0:
            raise ValueError("peak memory must be positive")
        if not isinstance(device, str) or not device:
            raise ValueError("engine device must be a non-empty string")
        if dtype != "float32":
            raise ValueError("comparison engine dtype must be float32")
        expected_fallback = engine == "pytorch-mps"
        if fallback_enabled is not expected_fallback:
            raise ValueError("engine fallback flag differs from the fixed protocol")
        median = float(np.median(values))
        return cls(
            engine=engine,
            samples_ms=samples,
            measured_runs=len(samples),
            warmup_runs=warmup_runs,
            median_ms=median,
            p95_ms=float(np.percentile(values, 95)),
            chunks_per_second=1_000.0 / median,
            peak_memory_bytes=peak_memory_bytes,
            device=device,
            dtype=dtype,
            fallback_enabled=fallback_enabled,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "samples_ms": list(self.samples_ms),
            "measured_runs": self.measured_runs,
            "warmup_runs": self.warmup_runs,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "chunks_per_second": self.chunks_per_second,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_gib": self.peak_memory_bytes / 1024**3,
            "device": self.device,
            "dtype": self.dtype,
            "fallback_enabled": self.fallback_enabled,
        }


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha(value: object, label: str, *, git: bool = False) -> str:
    pattern = _GIT_SHA if git else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase {'git' if git else 'SHA-256'} digest")
    return value


def _validate_timestamp(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("idle timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("idle timestamp must be valid ISO-8601") from error
    if parsed.utcoffset() is None:
        raise ValueError("idle timestamp must include a UTC offset")


def validate_comparison_document(value: object) -> dict[str, object]:
    """Reject any P2-1 artifact whose protocol or summaries are not complete."""

    document = _mapping(value, "comparison document")
    if document.get("artifact_type") != "smolvla-mlx-inference-comparison":
        raise ValueError("comparison artifact identity is invalid")
    if document.get("format_version") != 1:
        raise ValueError("comparison artifact format version is invalid")
    if document.get("protocol") != ComparisonProtocol().as_dict():
        raise ValueError("comparison protocol differs from the fixed values")

    idle = _mapping(document.get("idle"), "idle declaration")
    _validate_timestamp(idle.get("checked_at_utc"))
    if idle.get("verified") is not True or idle.get("matching_processes") != []:
        raise ValueError("comparison requires a clean idle declaration")

    environment = _mapping(document.get("environment"), "environment")
    for key in ("cpu", "macos", "python", "mlx", "torch", "lerobot"):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(f"environment {key} is absent")
    memory = environment.get("unified_memory_bytes")
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise ValueError("environment unified memory is invalid")

    inputs = _mapping(document.get("inputs"), "inputs")
    if inputs.get("sample") != "tests/golden/sample_000":
        raise ValueError("comparison input sample differs from the fixed case")
    _sha(inputs.get("sample_sha256"), "sample digest")
    _sha(inputs.get("noise_sha256"), "noise digest")
    if inputs.get("checkpoint_id") != CHECKPOINT_ID:
        raise ValueError("comparison checkpoint id differs from the pin")
    if inputs.get("checkpoint_revision") != CHECKPOINT_REVISION:
        raise ValueError("comparison checkpoint revision differs from the pin")

    source = _mapping(document.get("source"), "source")
    _sha(source.get("git_commit"), "source commit", git=True)
    if source.get("tracked_worktree_clean") is not True:
        raise ValueError("comparison source worktree was not clean")
    source_hashes = _mapping(source.get("sha256"), "source hashes")
    required_sources = {
        "reference/benchmark.py",
        "reference/policy.py",
        "scripts/benchmark_inference_comparison.py",
        "mlx_smolvla/benchmark.py",
        "mlx_smolvla/policy.py",
    }
    normalized_sources = {
        name.replace("smolvla_mlx/", "mlx_smolvla/", 1)
        for name in source_hashes
    }
    if (
        len(normalized_sources) != len(source_hashes)
        or normalized_sources != required_sources
    ):
        raise ValueError("comparison source hash inventory is incomplete")
    for name, digest in source_hashes.items():
        _sha(digest, f"source digest for {name}")

    engines = document.get("engines")
    if not isinstance(engines, list) or len(engines) != 2:
        raise ValueError("comparison engine matrix is incomplete")
    rebuilt: list[dict[str, object]] = []
    for expected_engine, raw in zip(_ENGINES, engines, strict=True):
        record = _mapping(raw, "engine record")
        if record.get("engine") != expected_engine:
            raise ValueError("comparison engine matrix is incomplete or reordered")
        samples = record.get("samples_ms")
        if not isinstance(samples, list):
            raise ValueError("engine raw timing samples are absent")
        rebuilt_record = EngineTiming.from_samples(
            engine=expected_engine,
            samples_ms=samples,
            warmup_runs=record.get("warmup_runs"),
            peak_memory_bytes=record.get("peak_memory_bytes"),
            device=record.get("device"),
            dtype=record.get("dtype"),
            fallback_enabled=record.get("fallback_enabled"),
        ).as_dict()
        if dict(record) != rebuilt_record:
            raise ValueError(f"{expected_engine} summary does not recompute from raw timings")
        rebuilt.append(rebuilt_record)

    result = json.loads(json.dumps(document))
    result["engines"] = rebuilt
    return result


__all__ = [
    "ComparisonProtocol",
    "EngineTiming",
    "validate_comparison_document",
]
