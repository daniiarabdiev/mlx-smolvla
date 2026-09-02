"""VLM-only weight quantization and auditable Stage Q P2-3 evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
import math
import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten


_VARIANTS = ("dense-bf16", "vlm-8bit", "vlm-4bit")
_QUANTIZED_ROOTS = ("connector", "language")
_DENSE_ROOTS = ("vision", "state_proj", "expert")
_GROUP_SIZE = 64
_MODE = "affine"
_STATISTICAL_SAMPLES = 50
_STATISTICAL_RATIO_GATE = 1.05
_SEED_BASE = 20_260_831
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_ID = "lerobot/smolvla_base"
_CHECKPOINT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
_DATASET_ID = "lerobot/svla_so101_pickplace"
_DATASET_REVISION = "f641879e22172be7e8161d5e6c1503c2d2feb657"
_DENSE_STATISTICAL_SHA256 = "c506ddcfdde50297e97b9905a299d55f117680a93b257e0af335ae6c9ad5fe07"

EXPECTED_QUANTIZED_SCALARS = 216_391_680


def expected_vlm_linear_paths() -> tuple[str, ...]:
    """Return the exact audited connector/text-linear quantization population."""

    paths = ["connector.modality_projection.proj", "language.lm_head"]
    for layer in range(16):
        prefix = f"language.layers.{layer}"
        paths.extend(
            (
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.down_proj",
                f"{prefix}.mlp.up_proj",
            )
        )
    return tuple(sorted(paths))


@dataclass(frozen=True)
class QuantizationProtocol:
    """Immutable accuracy, topology, and timing experiment matrix."""

    variants: tuple[str, ...] = _VARIANTS
    quantized_roots: tuple[str, ...] = _QUANTIZED_ROOTS
    dense_roots: tuple[str, ...] = _DENSE_ROOTS
    base_dtype: str = "bfloat16"
    group_size: int = _GROUP_SIZE
    mode: str = _MODE
    statistical_samples: int = _STATISTICAL_SAMPLES
    statistical_ratio_gate: float = _STATISTICAL_RATIO_GATE
    noise_seed_base: int = _SEED_BASE
    latency_sample: str = "sample_000"
    latency_warmup_runs: int = 5
    latency_measured_runs: int = 50

    def __post_init__(self) -> None:
        if (
            self.variants != _VARIANTS
            or self.quantized_roots != _QUANTIZED_ROOTS
            or self.dense_roots != _DENSE_ROOTS
            or self.base_dtype != "bfloat16"
            or self.group_size != _GROUP_SIZE
            or self.mode != _MODE
            or self.statistical_samples != _STATISTICAL_SAMPLES
            or self.statistical_ratio_gate != _STATISTICAL_RATIO_GATE
            or self.noise_seed_base != _SEED_BASE
            or self.latency_sample != "sample_000"
            or self.latency_warmup_runs != 5
            or self.latency_measured_runs != 50
        ):
            raise ValueError("quantization protocol differs from the fixed Stage Q P2-3 values")

    def as_dict(self) -> dict[str, object]:
        return {
            "variants": list(self.variants),
            "quantized_roots": list(self.quantized_roots),
            "dense_roots": list(self.dense_roots),
            "base_dtype": self.base_dtype,
            "group_size": self.group_size,
            "mode": self.mode,
            "statistical_samples": self.statistical_samples,
            "statistical_ratio_gate": self.statistical_ratio_gate,
            "noise_seed_base": self.noise_seed_base,
            "latency_sample": self.latency_sample,
            "latency_warmup_runs": self.latency_warmup_runs,
            "latency_measured_runs": self.latency_measured_runs,
        }


@dataclass(frozen=True)
class QuantizationManifest:
    """Observed module scope after optional in-memory quantization."""

    variant: str
    bits: int | None
    group_size: int | None
    mode: str | None
    quantized_roots: tuple[str, ...]
    dense_roots: tuple[str, ...]
    base_dtype: str
    eligible_linear_count: int
    eligible_weight_scalar_count: int
    quantized_linear_count: int
    quantized_weight_scalar_count: int
    quantized_paths: tuple[str, ...]
    embedding_quantized: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "quantized_roots": list(self.quantized_roots),
            "dense_roots": list(self.dense_roots),
            "base_dtype": self.base_dtype,
            "eligible_linear_count": self.eligible_linear_count,
            "eligible_weight_scalar_count": self.eligible_weight_scalar_count,
            "quantized_linear_count": self.quantized_linear_count,
            "quantized_weight_scalar_count": self.quantized_weight_scalar_count,
            "quantized_paths": list(self.quantized_paths),
            "embedding_quantized": self.embedding_quantized,
        }


def _named_modules(root: nn.Module, kind: type[nn.Module]) -> dict[str, nn.Module]:
    return {name: module for name, module in root.named_modules() if isinstance(module, kind)}


def _root(policy: object, name: str) -> nn.Module:
    value = getattr(policy, name, None)
    if not isinstance(value, nn.Module):
        raise TypeError(f"policy {name} must be an MLX module")
    return value


def _floating_parameters_are_bf16(root: nn.Module) -> bool:
    values = [value for _, value in tree_flatten(root.parameters()) if hasattr(value, "dtype")]
    floating = [value for value in values if value.dtype in {mx.float16, mx.bfloat16, mx.float32}]
    return bool(floating) and all(value.dtype == mx.bfloat16 for value in floating)


def _preflight_dense_policy(policy: object) -> tuple[dict[str, nn.Linear], int]:
    all_roots = (*_QUANTIZED_ROOTS, *_DENSE_ROOTS)
    for name in all_roots:
        root = _root(policy, name)
        if _named_modules(root, nn.QuantizedLinear) or _named_modules(root, nn.QuantizedEmbedding):
            raise ValueError("policy is already quantized")
        if not _floating_parameters_are_bf16(root):
            raise ValueError(f"policy root {name} must contain dense bfloat16 parameters")
    eligible: dict[str, nn.Linear] = {}
    for root_name in _QUANTIZED_ROOTS:
        for path, module in _named_modules(_root(policy, root_name), nn.Linear).items():
            full_path = f"{root_name}.{path}" if path else root_name
            if module.weight.shape[-1] % _GROUP_SIZE != 0:
                raise ValueError(
                    f"VLM linear {full_path} input width is not divisible by group size {_GROUP_SIZE}"
                )
            eligible[full_path] = module
    if not eligible:
        raise ValueError("policy has no eligible VLM linear modules")
    return eligible, sum(int(module.weight.size) for module in eligible.values())


def describe_dense_vlm(policy: object) -> QuantizationManifest:
    """Validate and describe an unmodified dense-bf16 baseline policy."""

    eligible, scalars = _preflight_dense_policy(policy)
    return QuantizationManifest(
        variant="dense-bf16",
        bits=None,
        group_size=None,
        mode=None,
        quantized_roots=_QUANTIZED_ROOTS,
        dense_roots=_DENSE_ROOTS,
        base_dtype="bfloat16",
        eligible_linear_count=len(eligible),
        eligible_weight_scalar_count=scalars,
        quantized_linear_count=0,
        quantized_weight_scalar_count=0,
        quantized_paths=(),
        embedding_quantized=False,
    )


def quantize_vlm_linears(policy: object, *, bits: int) -> QuantizationManifest:
    """Quantize only connector/text Linear weights; leave every other root dense."""

    if bits not in {4, 8}:
        raise ValueError("VLM quantization bits must be 4 or 8")
    eligible, scalars = _preflight_dense_policy(policy)
    expected_paths = set(eligible)
    for root_name in _QUANTIZED_ROOTS:
        root = _root(policy, root_name)
        nn.quantize(
            root,
            group_size=_GROUP_SIZE,
            bits=bits,
            mode=_MODE,
            class_predicate=lambda _path, module: isinstance(module, nn.Linear),
        )
        mx.eval(root.parameters())

    observed: dict[str, nn.QuantizedLinear] = {}
    for root_name in _QUANTIZED_ROOTS:
        root = _root(policy, root_name)
        for path, module in _named_modules(root, nn.QuantizedLinear).items():
            observed[f"{root_name}.{path}" if path else root_name] = module
        if _named_modules(root, nn.QuantizedEmbedding):
            raise RuntimeError("VLM embedding was quantized despite the Linear-only predicate")
    if set(observed) != expected_paths:
        raise RuntimeError("quantized VLM module set differs from the dense eligible set")
    if any(module.bits != bits or module.group_size != _GROUP_SIZE for module in observed.values()):
        raise RuntimeError("quantized VLM module parameters differ from the requested preset")
    for root_name in _DENSE_ROOTS:
        root = _root(policy, root_name)
        if _named_modules(root, nn.QuantizedLinear) or not _floating_parameters_are_bf16(root):
            raise RuntimeError(f"excluded root {root_name} did not remain dense bfloat16")

    return QuantizationManifest(
        variant=f"vlm-{bits}bit",
        bits=bits,
        group_size=_GROUP_SIZE,
        mode=_MODE,
        quantized_roots=_QUANTIZED_ROOTS,
        dense_roots=_DENSE_ROOTS,
        base_dtype="bfloat16",
        eligible_linear_count=len(eligible),
        eligible_weight_scalar_count=scalars,
        quantized_linear_count=len(observed),
        quantized_weight_scalar_count=scalars,
        quantized_paths=tuple(sorted(observed)),
        embedding_quantized=False,
    )


def expected_topology_manifest(variant: str) -> dict[str, object]:
    """Return the exact audited real-checkpoint topology expected per variant."""

    if variant not in _VARIANTS:
        raise ValueError(f"unknown quantization variant {variant!r}")
    bits = {"dense-bf16": None, "vlm-8bit": 8, "vlm-4bit": 4}[variant]
    paths = () if bits is None else expected_vlm_linear_paths()
    return QuantizationManifest(
        variant=variant,
        bits=bits,
        group_size=None if bits is None else _GROUP_SIZE,
        mode=None if bits is None else _MODE,
        quantized_roots=_QUANTIZED_ROOTS,
        dense_roots=_DENSE_ROOTS,
        base_dtype="bfloat16",
        eligible_linear_count=114,
        eligible_weight_scalar_count=EXPECTED_QUANTIZED_SCALARS,
        quantized_linear_count=len(paths),
        quantized_weight_scalar_count=0 if bits is None else EXPECTED_QUANTIZED_SCALARS,
        quantized_paths=paths,
        embedding_quantized=False,
    ).as_dict()


@dataclass(frozen=True)
class QuantizationAccuracy:
    """Fifty-case current-action MAE and unchanged statistical verdict."""

    variant: str
    samples: tuple[Mapping[str, object], ...]
    sample_count: int
    element_count: int
    torch_fp32_mae: float
    dense_bf16_mae: float
    candidate_mae: float
    statistical_ratio: float
    statistical_ratio_gate: float
    statistical_passed: bool
    mae_delta_vs_dense_bf16: float
    relative_mae_delta_percent: float

    @classmethod
    def from_samples(
        cls,
        *,
        variant: str,
        samples: Sequence[Mapping[str, object]],
    ) -> "QuantizationAccuracy":
        if variant not in _VARIANTS:
            raise ValueError(f"unknown accuracy variant {variant!r}")
        if len(samples) != _STATISTICAL_SAMPLES:
            raise ValueError("quantization accuracy requires exactly 50 samples")
        normalized: list[dict[str, object]] = []
        totals = {"torch": 0.0, "dense": 0.0, "candidate": 0.0}
        element_count = 0
        for index, raw in enumerate(samples):
            if not isinstance(raw, Mapping):
                raise ValueError("quantization accuracy sample must be an object")
            expected_identity = (index, 0, _SEED_BASE + index, 6)
            identity = (
                raw.get("episode"),
                raw.get("frame_index"),
                raw.get("seed"),
                raw.get("element_count"),
            )
            if identity != expected_identity:
                raise ValueError("quantization accuracy sample identity differs from the fixed population")
            values: dict[str, float] = {}
            for key in (
                "torch_fp32_abs_error_sum",
                "dense_bf16_abs_error_sum",
                "candidate_abs_error_sum",
            ):
                value = raw.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("quantization accuracy error sum must be numeric")
                number = float(value)
                if not math.isfinite(number) or number < 0.0:
                    raise ValueError("quantization accuracy error sum must be finite and nonnegative")
                values[key] = number
            if variant == "dense-bf16" and values["candidate_abs_error_sum"] != values["dense_bf16_abs_error_sum"]:
                raise ValueError("dense baseline candidate errors must equal the bound dense evidence")
            totals["torch"] += values["torch_fp32_abs_error_sum"]
            totals["dense"] += values["dense_bf16_abs_error_sum"]
            totals["candidate"] += values["candidate_abs_error_sum"]
            element_count += 6
            normalized.append(
                {
                    "episode": index,
                    "frame_index": 0,
                    "seed": _SEED_BASE + index,
                    "element_count": 6,
                    **values,
                }
            )
        torch_mae = totals["torch"] / element_count
        dense_mae = totals["dense"] / element_count
        candidate_mae = totals["candidate"] / element_count
        if torch_mae <= 0.0 or dense_mae <= 0.0:
            raise ValueError("quantization accuracy reference MAEs must be positive")
        ratio = candidate_mae / torch_mae
        return cls(
            variant=variant,
            samples=tuple(normalized),
            sample_count=len(normalized),
            element_count=element_count,
            torch_fp32_mae=torch_mae,
            dense_bf16_mae=dense_mae,
            candidate_mae=candidate_mae,
            statistical_ratio=ratio,
            statistical_ratio_gate=_STATISTICAL_RATIO_GATE,
            statistical_passed=ratio <= _STATISTICAL_RATIO_GATE,
            mae_delta_vs_dense_bf16=candidate_mae - dense_mae,
            relative_mae_delta_percent=(candidate_mae / dense_mae - 1.0) * 100.0,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "samples": [dict(sample) for sample in self.samples],
            "sample_count": self.sample_count,
            "element_count": self.element_count,
            "torch_fp32_mae": self.torch_fp32_mae,
            "dense_bf16_mae": self.dense_bf16_mae,
            "candidate_mae": self.candidate_mae,
            "statistical_ratio": self.statistical_ratio,
            "statistical_ratio_gate": self.statistical_ratio_gate,
            "statistical_passed": self.statistical_passed,
            "mae_delta_vs_dense_bf16": self.mae_delta_vs_dense_bf16,
            "relative_mae_delta_percent": self.relative_mae_delta_percent,
        }


@dataclass(frozen=True)
class QuantizationLatency:
    """Raw fixed-sample total latency and allocator evidence for one variant."""

    variant: str
    samples_ms: tuple[float, ...]
    measured_runs: int
    warmup_runs: int
    median_ms: float
    p95_ms: float
    chunks_per_second: float
    peak_memory_bytes: int
    peak_memory_gib: float
    device: str

    @classmethod
    def from_samples(
        cls,
        *,
        variant: str,
        samples_ms: Sequence[float],
        warmup_runs: int,
        peak_memory_bytes: int,
        device: str,
    ) -> "QuantizationLatency":
        if variant not in _VARIANTS:
            raise ValueError(f"unknown latency variant {variant!r}")
        samples = tuple(float(value) for value in samples_ms)
        values = np.asarray(samples, dtype=np.float64)
        if warmup_runs != 5 or len(samples) != 50:
            raise ValueError("quantization latency differs from the fixed 5+50 protocol")
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("quantization latency samples must be finite and positive")
        if isinstance(peak_memory_bytes, bool) or not isinstance(peak_memory_bytes, int) or peak_memory_bytes <= 0:
            raise ValueError("quantization latency peak memory is invalid")
        if not isinstance(device, str) or not device:
            raise ValueError("quantization latency device is invalid")
        median = float(np.median(values))
        return cls(
            variant=variant,
            samples_ms=samples,
            measured_runs=50,
            warmup_runs=warmup_runs,
            median_ms=median,
            p95_ms=float(np.percentile(values, 95)),
            chunks_per_second=1_000.0 / median,
            peak_memory_bytes=peak_memory_bytes,
            peak_memory_gib=peak_memory_bytes / 1024**3,
            device=device,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "samples_ms": list(self.samples_ms),
            "measured_runs": self.measured_runs,
            "warmup_runs": self.warmup_runs,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "chunks_per_second": self.chunks_per_second,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_gib": self.peak_memory_gib,
            "device": self.device,
        }


def derive_quantization_decision(variants: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Ship only non-default variants whose unchanged statistical gate passes."""

    if len(variants) != 3 or tuple(item.get("variant") for item in variants) != _VARIANTS:
        raise ValueError("quantization decision requires the complete ordered variant matrix")
    eligible: list[str] = []
    rejected: list[str] = []
    for record in variants[1:]:
        accuracy = record.get("accuracy")
        if not isinstance(accuracy, Mapping) or not isinstance(accuracy.get("statistical_passed"), bool):
            raise ValueError("quantization decision lacks a statistical verdict")
        (eligible if accuracy["statistical_passed"] else rejected).append(str(record["variant"]))
    return {
        "default_changed": False,
        "eligible_opt_in_variants": eligible,
        "rejected_variants": rejected,
    }


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _digest(value: object, label: str, *, git: bool = False) -> None:
    pattern = _GIT_SHA if git else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} is not a valid digest")


def validate_quantization_document(value: object) -> dict[str, object]:
    """Recompute topology, accuracy, timing, and ship/no-ship evidence."""

    document = _mapping(value, "quantization document")
    if document.get("artifact_type") != "smolvla-mlx-vlm-quantization-experiment":
        raise ValueError("quantization artifact identity is invalid")
    if document.get("format_version") != 1:
        raise ValueError("quantization artifact format is invalid")
    if document.get("protocol") != QuantizationProtocol().as_dict():
        raise ValueError("quantization protocol differs from the fixed values")

    idle = _mapping(document.get("idle"), "quantization idle declaration")
    timestamp = idle.get("checked_at_utc")
    if not isinstance(timestamp, str):
        raise ValueError("quantization idle timestamp is absent")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("quantization idle timestamp is invalid") from error
    if parsed.utcoffset() is None or idle.get("verified") is not True or idle.get("matching_processes") != []:
        raise ValueError("quantization experiment requires a clean idle declaration")

    environment = _mapping(document.get("environment"), "quantization environment")
    for key in ("cpu", "macos", "python", "mlx", "lerobot"):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(f"quantization environment {key} is absent")
    memory = environment.get("unified_memory_bytes")
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise ValueError("quantization environment memory is invalid")

    inputs = _mapping(document.get("inputs"), "quantization inputs")
    expected_inputs = {
        "checkpoint_id": _CHECKPOINT_ID,
        "checkpoint_revision": _CHECKPOINT_REVISION,
        "dataset_id": _DATASET_ID,
        "dataset_revision": _DATASET_REVISION,
        "latency_sample": "tests/golden/sample_000",
        "dense_statistical_sha256": _DENSE_STATISTICAL_SHA256,
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise ValueError(f"quantization input {key} differs from the pin")
    _digest(inputs.get("latency_sample_sha256"), "quantization latency sample digest")
    _digest(inputs.get("latency_noise_sha256"), "quantization latency noise digest")

    source = _mapping(document.get("source"), "quantization source")
    _digest(source.get("git_commit"), "quantization source commit", git=True)
    if source.get("tracked_worktree_clean") is not True:
        raise ValueError("quantization source worktree was not clean")
    source_hashes = _mapping(source.get("sha256"), "quantization source hashes")
    expected_sources = {
        "scripts/benchmark_inference_comparison.py",
        "scripts/experiment_quantization.py",
        "mlx_smolvla/benchmark.py",
        "mlx_smolvla/policy.py",
        "mlx_smolvla/quantization.py",
    }
    normalized_sources = {
        name.replace("smolvla_mlx/", "mlx_smolvla/", 1)
        for name in source_hashes
    }
    if (
        len(normalized_sources) != len(source_hashes)
        or normalized_sources != expected_sources
    ):
        raise ValueError("quantization source hash inventory is incomplete")
    for name, digest in source_hashes.items():
        _digest(digest, f"quantization source digest for {name}")

    raw_variants = document.get("variants")
    if not isinstance(raw_variants, list) or len(raw_variants) != 3:
        raise ValueError("quantization variant matrix is incomplete")
    rebuilt: list[dict[str, object]] = []
    for variant, raw in zip(_VARIANTS, raw_variants, strict=True):
        record = _mapping(raw, "quantization variant")
        if record.get("variant") != variant:
            raise ValueError("quantization variant matrix is incomplete or reordered")
        if record.get("topology") != expected_topology_manifest(variant):
            raise ValueError(f"{variant} topology differs from the VLM-only contract")
        accuracy = _mapping(record.get("accuracy"), "quantization accuracy")
        samples = accuracy.get("samples")
        if not isinstance(samples, list):
            raise ValueError("quantization accuracy raw samples are absent")
        rebuilt_accuracy = QuantizationAccuracy.from_samples(
            variant=variant,
            samples=samples,
        ).as_dict()
        if dict(accuracy) != rebuilt_accuracy:
            raise ValueError(f"{variant} accuracy does not recompute from samples")
        latency = _mapping(record.get("latency"), "quantization latency")
        latency_samples = latency.get("samples_ms")
        if not isinstance(latency_samples, list):
            raise ValueError("quantization latency raw samples are absent")
        rebuilt_latency = QuantizationLatency.from_samples(
            variant=variant,
            samples_ms=latency_samples,
            warmup_runs=latency.get("warmup_runs"),
            peak_memory_bytes=latency.get("peak_memory_bytes"),
            device=latency.get("device"),
        ).as_dict()
        if dict(latency) != rebuilt_latency:
            raise ValueError(f"{variant} latency does not recompute from samples")
        rebuilt.append(
            {
                "variant": variant,
                "topology": expected_topology_manifest(variant),
                "accuracy": rebuilt_accuracy,
                "latency": rebuilt_latency,
            }
        )
    expected_decision = derive_quantization_decision(rebuilt)
    if document.get("decision") != expected_decision:
        raise ValueError("quantization decision does not follow the fixed gate")
    result = json.loads(json.dumps(document))
    result["variants"] = rebuilt
    result["decision"] = expected_decision
    return result


__all__ = [
    "EXPECTED_QUANTIZED_SCALARS",
    "QuantizationAccuracy",
    "QuantizationLatency",
    "QuantizationManifest",
    "QuantizationProtocol",
    "derive_quantization_decision",
    "describe_dense_vlm",
    "expected_topology_manifest",
    "expected_vlm_linear_paths",
    "quantize_vlm_linears",
    "validate_quantization_document",
]
