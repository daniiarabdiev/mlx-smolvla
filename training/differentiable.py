"""Pure-MLX differentiable primitives for the optional training path."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn


_CPU_PRIMITIVE_LOCK = threading.RLock()
_CPU_PRIMITIVES_ACTIVE = False


def differentiable_rms_norm(
    inputs: mx.array,
    weight: mx.array,
    eps: float,
) -> mx.array:
    """Apply fp32 RMSNorm using only operations with MLX autodiff support."""

    values = inputs.astype(mx.float32)
    variance = mx.mean(values * values, axis=-1, keepdims=True)
    normalized = values * mx.rsqrt(variance + mx.array(eps, dtype=mx.float32))
    return normalized * weight.astype(mx.float32)


def differentiable_rope(
    states: mx.array,
    position_ids: mx.array,
    max_wavelength: float = 10_000.0,
) -> mx.array:
    """Apply split-half fp32 RoPE using operations with MLX VJPs."""

    if states.ndim != 4:
        raise ValueError(
            "RoPE states must have [batch, sequence, heads, dimension] shape; "
            f"got {states.shape}"
        )
    if position_ids.shape != states.shape[:2]:
        raise ValueError(
            f"RoPE positions {position_ids.shape} do not match states {states.shape[:2]}"
        )
    dimension = states.shape[-1]
    if dimension % 2:
        raise ValueError(f"RoPE dimension must be even, got {dimension}")
    half = dimension // 2
    values = states.astype(mx.float32)
    exponents = (2.0 / dimension) * mx.arange(half, dtype=mx.float32)
    timescale = mx.power(mx.array(max_wavelength, dtype=mx.float32), exponents)
    radians = position_ids.astype(mx.float32)[..., None] / timescale[None, None, :]
    sine = mx.sin(radians)[..., None, :]
    cosine = mx.cos(radians)[..., None, :]
    first, second = values[..., :half], values[..., half:]
    return mx.concatenate(
        (first * cosine - second * sine, second * cosine + first * sine),
        axis=-1,
    )


def differentiable_softmax(inputs: mx.array) -> mx.array:
    """Apply last-axis fp32 softmax using MLX's differentiable implementation."""

    return mx.softmax(inputs.astype(mx.float32), axis=-1)


def differentiable_silu(inputs: mx.array) -> mx.array:
    """Apply fp32 SiLU using MLX's differentiable implementation."""

    return nn.silu(inputs.astype(mx.float32))


def _differentiable_rms_norm_call(normalizer: object, inputs: mx.array) -> mx.array:
    return differentiable_rms_norm(
        inputs,
        normalizer.weight,
        normalizer.eps,
    )


@contextmanager
def differentiable_cpu_primitives() -> Iterator[None]:
    """Temporarily route CPU language/expert math through pure MLX VJPs.

    Runtime inference modules retain their exact native CPU dispatch before and
    after this single-process scope. The lock prevents concurrent parity scopes,
    while explicit nesting is rejected so restoration remains unambiguous.
    """

    if mx.default_device() != mx.cpu:
        raise RuntimeError("differentiable_cpu_primitives requires an MLX CPU stream")

    global _CPU_PRIMITIVES_ACTIVE
    with _CPU_PRIMITIVE_LOCK:
        if _CPU_PRIMITIVES_ACTIVE:
            raise RuntimeError("differentiable CPU primitives are already active")
        _CPU_PRIMITIVES_ACTIVE = True

        from mlx_smolvla import expert as expert_module
        from mlx_smolvla import language as language_module
        from mlx_smolvla import rmsnorm as rms_module

        replacements = (
            (rms_module.ReferenceRMSNorm, "__call__", _differentiable_rms_norm_call),
            (rms_module, "reference_rope", differentiable_rope),
            (rms_module, "reference_softmax", differentiable_softmax),
            (rms_module, "reference_silu", differentiable_silu),
            (language_module, "reference_rope", differentiable_rope),
            (language_module, "reference_softmax", differentiable_softmax),
            (language_module, "reference_silu", differentiable_silu),
            (expert_module, "reference_softmax", differentiable_softmax),
            (expert_module, "reference_silu", differentiable_silu),
        )
        originals: list[tuple[object, str, object]] = []
        try:
            for owner, name, replacement in replacements:
                originals.append((owner, name, getattr(owner, name)))
                setattr(owner, name, replacement)
            yield
        finally:
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)
            _CPU_PRIMITIVES_ACTIVE = False
