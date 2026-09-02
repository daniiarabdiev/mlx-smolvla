"""Frozen component profiling and validation for the bf16 latency anomaly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
import math
import re
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from smolvla_mlx.policy import SmolVLAMLX


_DTYPES = ("float32", "bfloat16")
_STAGES = (
    "preprocessing",
    "vision_encoder",
    "connector",
    "prefix",
    "expert_loop",
    "total",
)
_COMPONENT_STAGES = _STAGES[:-1]
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_ID = "lerobot/smolvla_base"
_CHECKPOINT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"


@dataclass(frozen=True)
class ProfileProtocol:
    """Immutable P2-2 dtype and stage matrix."""

    dtypes: tuple[str, ...] = _DTYPES
    stages: tuple[str, ...] = _STAGES
    sample: str = "sample_000"
    warmup_runs: int = 5
    measured_runs: int = 50
    fixed_noise: bool = True
    synchronization: str = "every-stage-boundary"

    def __post_init__(self) -> None:
        if (
            self.dtypes != _DTYPES
            or self.stages != _STAGES
            or self.sample != "sample_000"
            or self.warmup_runs != 5
            or self.measured_runs != 50
            or self.fixed_noise is not True
            or self.synchronization != "every-stage-boundary"
        ):
            raise ValueError("profile protocol differs from the fixed Stage Q P2-2 values")

    def as_dict(self) -> dict[str, object]:
        return {
            "dtypes": list(self.dtypes),
            "stages": list(self.stages),
            "sample": self.sample,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "fixed_noise": self.fixed_noise,
            "synchronization": self.synchronization,
        }


@dataclass(frozen=True)
class DtypeProfile:
    """One dtype's complete raw per-stage timing population."""

    dtype: str
    samples_ms: Mapping[str, tuple[float, ...]]
    summaries_ms: Mapping[str, Mapping[str, float]]
    measured_runs: int
    warmup_runs: int
    peak_memory_bytes: int
    peak_memory_gib: float
    device: str

    @classmethod
    def from_samples(
        cls,
        *,
        dtype: str,
        samples_ms: Mapping[str, Sequence[float]],
        warmup_runs: int,
        peak_memory_bytes: int,
        device: str,
    ) -> "DtypeProfile":
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported profile dtype {dtype!r}")
        if warmup_runs != 5:
            raise ValueError("profile warmup count differs from the fixed protocol")
        if len(samples_ms) != len(_STAGES) or set(samples_ms) != set(_STAGES):
            raise ValueError("profile samples must contain the exact stages")
        normalized: dict[str, tuple[float, ...]] = {}
        summaries: dict[str, dict[str, float]] = {}
        for stage in _STAGES:
            values = tuple(float(value) for value in samples_ms[stage])
            array = np.asarray(values, dtype=np.float64)
            if len(values) != 50:
                raise ValueError("profile stage measurement count differs from the fixed protocol")
            if not np.isfinite(array).all() or np.any(array <= 0.0):
                raise ValueError("profile timings must be finite positive milliseconds")
            normalized[stage] = values
            summaries[stage] = {
                "median": float(np.median(array)),
                "p95": float(np.percentile(array, 95)),
            }
        if isinstance(peak_memory_bytes, bool) or not isinstance(peak_memory_bytes, int):
            raise ValueError("profile peak memory must be an integer")
        if peak_memory_bytes <= 0:
            raise ValueError("profile peak memory must be positive")
        if not isinstance(device, str) or not device:
            raise ValueError("profile device must be a non-empty string")
        return cls(
            dtype=dtype,
            samples_ms=normalized,
            summaries_ms=summaries,
            measured_runs=50,
            warmup_runs=warmup_runs,
            peak_memory_bytes=peak_memory_bytes,
            peak_memory_gib=peak_memory_bytes / 1024**3,
            device=device,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "samples_ms": {
                stage: list(self.samples_ms[stage]) for stage in _STAGES
            },
            "summaries_ms": {
                stage: dict(self.summaries_ms[stage]) for stage in _STAGES
            },
            "measured_runs": self.measured_runs,
            "warmup_runs": self.warmup_runs,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_gib": self.peak_memory_gib,
            "device": self.device,
        }


def derive_dtype_analysis(profiles: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Derive all anomaly attribution from ordered fp32/bf16 summaries."""

    if len(profiles) != 2 or tuple(item.get("dtype") for item in profiles) != _DTYPES:
        raise ValueError("profile analysis requires ordered fp32 and bf16 records")
    summaries = [
        item.get("summaries_ms") if isinstance(item.get("summaries_ms"), Mapping) else None
        for item in profiles
    ]
    if any(item is None for item in summaries):
        raise ValueError("profile analysis summaries are absent")
    fp32, bf16 = summaries

    def median(summary: Mapping[str, object], stage: str) -> float:
        record = summary.get(stage)
        if not isinstance(record, Mapping):
            raise ValueError(f"profile analysis stage {stage!r} is absent")
        value = record.get("median")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"profile analysis stage {stage!r} median is invalid")
        return float(value)

    total_fp32 = median(fp32, "total")
    total_bf16 = median(bf16, "total")
    total_delta = total_bf16 - total_fp32
    stage_deltas = {
        stage: median(bf16, stage) - median(fp32, stage)
        for stage in _COMPONENT_STAGES
    }
    shares = {
        stage: (delta / total_delta if total_delta != 0.0 else None)
        for stage, delta in stage_deltas.items()
    }
    dominant = max(stage_deltas, key=stage_deltas.__getitem__)
    return {
        "classification": (
            "bf16-slower" if total_delta > 0.0 else "bf16-faster" if total_delta < 0.0 else "tie"
        ),
        "fp32_total_median_ms": total_fp32,
        "bf16_total_median_ms": total_bf16,
        "total_median_delta_ms": total_delta,
        "total_slowdown_percent": total_delta / total_fp32 * 100.0,
        "component_median_delta_ms": stage_deltas,
        "component_share_of_total_delta": shares,
        "largest_positive_delta_component": dominant,
    }


def _evaluate_processed(processed: object) -> None:
    import mlx.core as mx

    mx.eval(
        processed.pixel_values,
        processed.pixel_attention_mask,
        processed.input_ids,
        processed.text_attention_mask,
        processed.state,
    )


def run_profile_iteration(
    policy: "SmolVLAMLX",
    observation: Mapping[str, object],
    noise: object,
) -> dict[str, float]:
    """Run one chunk with a synchronization at every fixed component boundary."""

    import mlx.core as mx

    from smolvla_mlx.flow import euler_sample
    from smolvla_mlx.language import pad_state_to_width

    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    started = time.perf_counter()
    processed = policy.preprocessor(observation)
    _evaluate_processed(processed)
    timings["preprocessing"] = (time.perf_counter() - started) * 1_000.0

    started = time.perf_counter()
    vision_features = policy.vision(processed.pixel_values, processed.pixel_attention_mask)
    mx.eval(vision_features)
    timings["vision_encoder"] = (time.perf_counter() - started) * 1_000.0

    started = time.perf_counter()
    image_tokens = policy.connector(vision_features)
    mx.eval(image_tokens)
    timings["connector"] = (time.perf_counter() - started) * 1_000.0

    started = time.perf_counter()
    padded_state = pad_state_to_width(processed.state, width=policy.config.max_state_dim)
    state_embedding = policy.state_proj(padded_state)[:, None, :]
    prefix = policy.language.build_prefix(processed, image_tokens, state_embedding)
    cache = policy.language.encode_prefix(prefix)
    mx.eval(cache.hidden, *cache.keys, *cache.values)
    timings["prefix"] = (time.perf_counter() - started) * 1_000.0

    started = time.perf_counter()
    padded_actions = euler_sample(
        lambda x_t, timestep: policy.expert.denoise(cache, x_t, timestep).velocity,
        noise,
        num_steps=policy.config.num_steps,
    )
    actions = padded_actions[:, :, : policy.config.action_dim]
    mx.eval(actions)
    timings["expert_loop"] = (time.perf_counter() - started) * 1_000.0
    timings["total"] = (time.perf_counter() - total_started) * 1_000.0
    return timings


def collect_dtype_profile(
    policy: "SmolVLAMLX",
    observation: Mapping[str, object],
    noise: object,
) -> DtypeProfile:
    """Collect the exact fixed profile for one already-loaded policy."""

    import mlx.core as mx

    protocol = ProfileProtocol()
    noise = mx.array(noise).astype(mx.float32)
    expected_shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    if noise.shape != expected_shape:
        raise ValueError(f"profile noise must have shape {expected_shape}, got {noise.shape}")
    samples = {stage: [] for stage in protocol.stages}
    with mx.stream(policy.execution_device):
        for _ in range(protocol.warmup_runs):
            run_profile_iteration(policy, observation, noise)
        mx.clear_cache()
        mx.reset_peak_memory()
        for _ in range(protocol.measured_runs):
            result = run_profile_iteration(policy, observation, noise)
            for stage in protocol.stages:
                samples[stage].append(result[stage])
        peak = int(max(mx.get_peak_memory(), mx.get_active_memory()))
    storage_dtype = policy.expert.action_in_proj.weight.dtype
    dtype = "bfloat16" if storage_dtype == mx.bfloat16 else "float32"
    return DtypeProfile.from_samples(
        dtype=dtype,
        samples_ms=samples,
        warmup_runs=protocol.warmup_runs,
        peak_memory_bytes=peak,
        device=str(policy.execution_device),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _digest(value: object, label: str, *, git: bool = False) -> None:
    pattern = _GIT_SHA if git else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} is not a valid digest")


def validate_profile_document(value: object) -> dict[str, object]:
    """Recompute every profile summary and derived anomaly attribution."""

    document = _mapping(value, "profile document")
    if document.get("artifact_type") != "smolvla-mlx-bf16-component-profile":
        raise ValueError("profile artifact identity is invalid")
    if document.get("format_version") != 1:
        raise ValueError("profile format version is invalid")
    if document.get("protocol") != ProfileProtocol().as_dict():
        raise ValueError("profile protocol differs from the fixed values")

    idle = _mapping(document.get("idle"), "idle declaration")
    timestamp = idle.get("checked_at_utc")
    if not isinstance(timestamp, str):
        raise ValueError("profile idle timestamp is absent")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("profile idle timestamp is invalid") from error
    if parsed.utcoffset() is None or idle.get("verified") is not True or idle.get("matching_processes") != []:
        raise ValueError("profile requires a clean idle declaration")

    environment = _mapping(document.get("environment"), "environment")
    for key in ("cpu", "macos", "python", "mlx"):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(f"profile environment {key} is absent")
    memory = environment.get("unified_memory_bytes")
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise ValueError("profile environment memory is invalid")

    inputs = _mapping(document.get("inputs"), "inputs")
    if inputs.get("sample") != "tests/golden/sample_000":
        raise ValueError("profile sample differs from the fixed case")
    _digest(inputs.get("sample_sha256"), "profile sample digest")
    _digest(inputs.get("noise_sha256"), "profile noise digest")
    if inputs.get("checkpoint_id") != _CHECKPOINT_ID or inputs.get("checkpoint_revision") != _CHECKPOINT_REVISION:
        raise ValueError("profile checkpoint differs from the pin")

    source = _mapping(document.get("source"), "source")
    _digest(source.get("git_commit"), "profile source commit", git=True)
    if source.get("tracked_worktree_clean") is not True:
        raise ValueError("profile source worktree was not clean")
    source_hashes = _mapping(source.get("sha256"), "profile source hashes")
    expected_sources = {
        "scripts/benchmark_inference_comparison.py",
        "scripts/profile_inference_dtypes.py",
        "smolvla_mlx/benchmark.py",
        "smolvla_mlx/policy.py",
        "smolvla_mlx/profile.py",
    }
    if set(source_hashes) != expected_sources:
        raise ValueError("profile source hash inventory is incomplete")
    for name, digest in source_hashes.items():
        _digest(digest, f"profile source digest for {name}")

    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, list) or len(raw_profiles) != 2:
        raise ValueError("profile dtype matrix is incomplete")
    rebuilt: list[dict[str, object]] = []
    for dtype, raw in zip(_DTYPES, raw_profiles, strict=True):
        record = _mapping(raw, "dtype profile")
        if record.get("dtype") != dtype:
            raise ValueError("profile dtype matrix is incomplete or reordered")
        samples = _mapping(record.get("samples_ms"), "profile samples")
        rebuilt_record = DtypeProfile.from_samples(
            dtype=dtype,
            samples_ms=samples,
            warmup_runs=record.get("warmup_runs"),
            peak_memory_bytes=record.get("peak_memory_bytes"),
            device=record.get("device"),
        ).as_dict()
        if dict(record) != rebuilt_record:
            raise ValueError(f"{dtype} profile summary does not recompute")
        rebuilt.append(rebuilt_record)
    expected_analysis = derive_dtype_analysis(rebuilt)
    if document.get("analysis") != expected_analysis:
        raise ValueError("profile analysis does not recompute from dtype summaries")
    result = json.loads(json.dumps(document))
    result["profiles"] = rebuilt
    result["analysis"] = expected_analysis
    return result


__all__ = [
    "DtypeProfile",
    "ProfileProtocol",
    "collect_dtype_profile",
    "derive_dtype_analysis",
    "run_profile_iteration",
    "validate_profile_document",
]
