"""Synchronized, stage-level native SmolVLA benchmarking helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from smolvla_mlx.flow import euler_sample
from smolvla_mlx.language import pad_state_to_width

if TYPE_CHECKING:
    from smolvla_mlx.policy import SmolVLAMLX


_STAGE_NAMES = ("preprocessing", "vision", "prefix", "expert")


@dataclass(frozen=True)
class TimingSummary:
    """Median and p95 latency in milliseconds for one measured section."""

    median: float
    p95: float


@dataclass(frozen=True)
class BenchmarkResult:
    """A reproducible result for one device and compact-weight storage mode."""

    measured_runs: int
    warmup_runs: int
    stages: Mapping[str, TimingSummary]
    total_ms: TimingSummary
    peak_memory_bytes: int
    device: str
    dtype: str
    execution_mode: str

    @classmethod
    def from_stage_samples(
        cls,
        samples: Mapping[str, Sequence[float]],
        *,
        warmup_runs: int,
        peak_memory_bytes: int,
        device: str,
        dtype: str,
        execution_mode: str,
    ) -> "BenchmarkResult":
        """Summarize measured samples only; warmups are never added to a percentile."""

        expected = set(_STAGE_NAMES) | {"total"}
        if set(samples) != expected:
            raise ValueError(f"Benchmark samples must have exactly {sorted(expected)}, got {sorted(samples)}")
        lengths = {len(values) for values in samples.values()}
        if len(lengths) != 1:
            raise ValueError(f"All benchmark stages need equal measured-run counts, got {lengths}")
        measured_runs = lengths.pop()
        if measured_runs <= 0:
            raise ValueError("Benchmark needs at least one measured run")
        if warmup_runs < 0:
            raise ValueError(f"warmup_runs must be non-negative, got {warmup_runs}")
        if peak_memory_bytes < 0:
            raise ValueError(f"peak_memory_bytes must be non-negative, got {peak_memory_bytes}")
        if execution_mode not in {"production", "strict"}:
            raise ValueError(f"Unknown benchmark execution mode {execution_mode!r}")

        def summary(values: Sequence[float]) -> TimingSummary:
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all() or np.any(array <= 0.0):
                raise ValueError("Benchmark durations must be finite positive milliseconds")
            return TimingSummary(median=float(np.median(array)), p95=float(np.percentile(array, 95)))

        return cls(
            measured_runs=measured_runs,
            warmup_runs=warmup_runs,
            stages={name: summary(samples[name]) for name in _STAGE_NAMES},
            total_ms=summary(samples["total"]),
            peak_memory_bytes=peak_memory_bytes,
            device=device,
            dtype=dtype,
            execution_mode=execution_mode,
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation for scripts and CLI output."""

        return {
            "measured_runs": self.measured_runs,
            "warmup_runs": self.warmup_runs,
            "device": self.device,
            "dtype": self.dtype,
            "execution_mode": self.execution_mode,
            "peak_memory_bytes": self.peak_memory_bytes,
            "total_ms": {"median": self.total_ms.median, "p95": self.total_ms.p95},
            "stages": {
                name: {"median": timing.median, "p95": timing.p95}
                for name, timing in self.stages.items()
            },
        }


def _evaluate_processed(processed) -> None:
    mx.eval(
        processed.pixel_values,
        processed.pixel_attention_mask,
        processed.input_ids,
        processed.text_attention_mask,
        processed.state,
    )


def _run_staged(policy: "SmolVLAMLX", observation: Mapping[str, object], noise: mx.array) -> dict[str, float]:
    """Run one real chunk and synchronize every timing boundary."""

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    processed = policy.preprocessor(observation)
    _evaluate_processed(processed)
    timings["preprocessing"] = (time.perf_counter() - start) * 1_000.0

    start = time.perf_counter()
    vision_features = policy.vision(processed.pixel_values, processed.pixel_attention_mask)
    image_tokens = policy.connector(vision_features)
    mx.eval(image_tokens)
    timings["vision"] = (time.perf_counter() - start) * 1_000.0

    start = time.perf_counter()
    padded_state = pad_state_to_width(processed.state, width=policy.config.max_state_dim)
    state_embedding = policy.state_proj(padded_state)[:, None, :]
    prefix = policy.language.build_prefix(processed, image_tokens, state_embedding)
    cache = policy.language.encode_prefix(prefix)
    mx.eval(cache.hidden, *cache.keys, *cache.values)
    timings["prefix"] = (time.perf_counter() - start) * 1_000.0

    start = time.perf_counter()
    padded_actions = euler_sample(
        lambda x_t, timestep: policy.expert.denoise(cache, x_t, timestep).velocity,
        noise,
        num_steps=policy.config.num_steps,
    )
    actions = padded_actions[:, :, : policy.config.action_dim]
    mx.eval(actions)
    timings["expert"] = (time.perf_counter() - start) * 1_000.0
    timings["total"] = (time.perf_counter() - total_start) * 1_000.0
    return timings


def run_benchmark(
    policy: "SmolVLAMLX",
    observation: Mapping[str, object],
    *,
    measured_runs: int = 50,
    warmup_runs: int = 5,
    noise: mx.array | None = None,
) -> BenchmarkResult:
    """Measure actual native execution after unrecorded warmups."""

    if measured_runs <= 0:
        raise ValueError(f"measured_runs must be positive, got {measured_runs}")
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be non-negative, got {warmup_runs}")
    with mx.stream(policy.execution_device):
        if noise is None:
            noise = mx.random.normal(
                (1, policy.config.chunk_size, policy.config.max_action_dim)
            ).astype(mx.float32)
        else:
            noise = mx.array(noise).astype(mx.float32)
        expected_shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
        if noise.shape != expected_shape:
            raise ValueError(f"noise must have shape {expected_shape}, got {noise.shape}")

        for _ in range(warmup_runs):
            _run_staged(policy, observation, noise)
        mx.reset_peak_memory()
        stage_samples = {name: [] for name in (*_STAGE_NAMES, "total")}
        for _ in range(measured_runs):
            timings = _run_staged(policy, observation, noise)
            for name, value in timings.items():
                stage_samples[name].append(value)
        peak_memory_bytes = int(max(mx.get_peak_memory(), mx.get_active_memory()))
        storage_dtype = policy.expert.action_in_proj.weight.dtype
        dtype = "bfloat16" if storage_dtype == mx.bfloat16 else "float32"
        return BenchmarkResult.from_stage_samples(
            stage_samples,
            warmup_runs=warmup_runs,
            peak_memory_bytes=peak_memory_bytes,
            device=str(mx.default_device()),
            dtype=dtype,
            execution_mode=policy.execution_mode,
        )
