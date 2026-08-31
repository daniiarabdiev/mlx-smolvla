"""Pure-MLX differentiable primitives for the optional training path."""

from __future__ import annotations

import mlx.core as mx


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
