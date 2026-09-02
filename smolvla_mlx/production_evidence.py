"""Fail-closed parser for default production-path deterministic evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping


_IMMUTABLE_MAX_ABS_GATES = {"float32": 0.005, "bfloat16": 0.05}


def _nonnegative_finite(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error
    return value


@dataclass(frozen=True)
class DeterministicDtypeEvidence:
    """One storage dtype's fixed-gate outcome across every golden case."""

    fixed_max_abs_gate: float
    max_abs: float
    passed: bool
    worst_case: str
    samples: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class ProductionDeterministicEvidence:
    """Validated production-Metal comparison against immutable CPU goldens."""

    execution_mode: str
    device: str
    case_count: int
    checkpoint: Mapping[str, object]
    golden: Mapping[str, object]
    results: Mapping[str, DeterministicDtypeEvidence]

    @classmethod
    def from_json(cls, path: Path) -> "ProductionDeterministicEvidence":
        if not path.is_file():
            raise FileNotFoundError(f"Production deterministic evidence is absent at {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("format_version") != 1:
            raise ValueError("Production deterministic evidence format is invalid")
        if raw.get("execution_mode") != "production":
            raise ValueError("Production deterministic evidence must label production mode")
        device = raw.get("device")
        if not isinstance(device, str) or "gpu" not in device.lower():
            raise ValueError("Production deterministic evidence must name its GPU device")
        case_count = raw.get("case_count")
        if not isinstance(case_count, int) or case_count <= 0:
            raise ValueError("Production deterministic case_count must be positive")

        checkpoint = raw.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ValueError("Production deterministic checkpoint provenance is invalid")
        if not isinstance(checkpoint.get("id"), str) or not isinstance(checkpoint.get("revision"), str):
            raise ValueError("Production deterministic checkpoint id/revision is invalid")
        _sha256(checkpoint.get("model_sha256"), name="checkpoint.model_sha256")

        golden = raw.get("golden")
        if not isinstance(golden, dict):
            raise ValueError("Production deterministic golden provenance is invalid")
        _sha256(golden.get("manifest_sha256"), name="golden.manifest_sha256")
        _sha256(golden.get("metadata_sha256"), name="golden.metadata_sha256")

        raw_results = raw.get("results")
        if not isinstance(raw_results, dict) or set(raw_results) != set(_IMMUTABLE_MAX_ABS_GATES):
            raise ValueError("Production deterministic results must contain float32 and bfloat16")
        results: dict[str, DeterministicDtypeEvidence] = {}
        for dtype, immutable_gate in _IMMUTABLE_MAX_ABS_GATES.items():
            result = raw_results[dtype]
            if not isinstance(result, dict):
                raise ValueError(f"{dtype} deterministic result must be an object")
            gate = _nonnegative_finite(result.get("fixed_max_abs_gate"), name=f"{dtype} gate")
            if gate != immutable_gate:
                raise ValueError(f"{dtype} immutable deterministic gate changed")
            raw_samples = result.get("samples")
            if not isinstance(raw_samples, list) or len(raw_samples) != case_count:
                raise ValueError(f"{dtype} deterministic samples do not match case_count")
            samples: list[Mapping[str, object]] = []
            names: set[str] = set()
            for sample in raw_samples:
                if not isinstance(sample, dict) or not isinstance(sample.get("name"), str):
                    raise ValueError(f"{dtype} deterministic sample is invalid")
                name = sample["name"]
                if name in names:
                    raise ValueError(f"{dtype} deterministic sample names must be unique")
                names.add(name)
                sample_max = _nonnegative_finite(sample.get("max_abs"), name=f"{dtype}/{name} max_abs")
                samples.append({"name": name, "max_abs": sample_max})
            worst = max(samples, key=lambda sample: float(sample["max_abs"]))
            computed_max = float(worst["max_abs"])
            reported_max = _nonnegative_finite(result.get("max_abs"), name=f"{dtype} max_abs")
            if reported_max != computed_max or result.get("worst_case") != worst["name"]:
                raise ValueError(f"{dtype} deterministic aggregate does not match its samples")
            computed_passed = computed_max <= immutable_gate
            if result.get("passed") is not computed_passed:
                raise ValueError(f"{dtype} deterministic pass/fail outcome is inconsistent")
            results[dtype] = DeterministicDtypeEvidence(
                fixed_max_abs_gate=immutable_gate,
                max_abs=computed_max,
                passed=computed_passed,
                worst_case=str(worst["name"]),
                samples=tuple(samples),
            )
        return cls(
            execution_mode="production",
            device=device,
            case_count=case_count,
            checkpoint=checkpoint,
            golden=golden,
            results=results,
        )
