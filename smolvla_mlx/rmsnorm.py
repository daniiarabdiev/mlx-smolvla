from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from . import _rmsnorm_native


def reference_rope(states: mx.array, position_ids: mx.array) -> mx.array:
    """Run the CPU RoPE arithmetic order used by the pinned PyTorch reference."""

    return _rmsnorm_native.reference_rope(states, position_ids)


def reference_softmax(input: mx.array) -> mx.array:
    """Run the CPU softmax arithmetic used by the pinned PyTorch reference."""

    return _rmsnorm_native.reference_softmax(input)


def reference_silu(input: mx.array) -> mx.array:
    """Run the CPU SiLU arithmetic used by the pinned PyTorch reference."""

    return _rmsnorm_native.reference_silu(input)


class ReferenceRMSNorm(nn.Module):
    """RMSNorm with PyTorch's CPU reduction order and MLX's GPU kernel."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        if width != 960:
            raise ValueError(f"ReferenceRMSNorm only supports width 960, got {width}")
        self.width = width
        self.eps = eps
        self.weight = mx.ones((width,), dtype=mx.float32)

    def __call__(self, input: mx.array) -> mx.array:
        if mx.default_device() == mx.cpu:
            return _rmsnorm_native.rms_norm(input, self.weight.astype(mx.float32), self.eps)
        return mx.fast.rms_norm(input, self.weight, self.eps)
