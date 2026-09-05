from __future__ import annotations

from functools import lru_cache
from importlib import metadata

import mlx.core as mx
import mlx.nn as nn

_native_import_failure: str | None = None
try:
    from . import _rmsnorm_native
except (ImportError, OSError, RuntimeError) as exc:
    _rmsnorm_native = None
    _native_import_failure = f"{type(exc).__name__}: {exc}"


_NATIVE_EXTENSION_MLX_ABI = "0.32.2"


@lru_cache(maxsize=1)
def _runtime_mlx_version() -> str:
    try:
        return metadata.version("mlx")
    except metadata.PackageNotFoundError:
        return "not installed"


def native_extension_available() -> bool:
    """Return whether exact CPU-reference primitives are installed."""

    return (
        _rmsnorm_native is not None
        and _runtime_mlx_version() == _NATIVE_EXTENSION_MLX_ABI
    )


def cpu_compatibility_backend() -> str:
    """Name the active CPU compatibility implementation for diagnostics."""

    return "native-reference" if native_extension_available() else "pure-mlx-fallback"


def native_extension_unavailable_reason() -> str | None:
    """Explain why strict CPU uses the supported pure-MLX fallback, if active."""

    if _rmsnorm_native is None:
        return _native_import_failure or "Optional native reference extension is not installed"
    runtime_version = _runtime_mlx_version()
    if runtime_version != _NATIVE_EXTENSION_MLX_ABI:
        return (
            f"Native reference extension requires MLX ABI {_NATIVE_EXTENSION_MLX_ABI}; "
            f"runtime MLX is {runtime_version}"
        )
    return None


def _pure_mlx_rope(states: mx.array, position_ids: mx.array) -> mx.array:
    if states.ndim != 4 or states.shape[-1] != 64:
        raise ValueError(
            "RoPE states must have [batch, sequence, heads, 64] shape; "
            f"got {states.shape}"
        )
    if position_ids.shape != states.shape[:2]:
        raise ValueError(
            f"RoPE positions {position_ids.shape} do not match states {states.shape[:2]}"
        )
    values = states.astype(mx.float32)
    half = values.shape[-1] // 2
    exponents = (2.0 / values.shape[-1]) * mx.arange(half, dtype=mx.float32)
    timescale = mx.power(mx.array(10_000.0, dtype=mx.float32), exponents)
    radians = position_ids.astype(mx.float32)[..., None] / timescale[None, None, :]
    sine = mx.sin(radians)[..., None, :]
    cosine = mx.cos(radians)[..., None, :]
    first, second = values[..., :half], values[..., half:]
    return mx.concatenate(
        (first * cosine - second * sine, second * cosine + first * sine),
        axis=-1,
    )


def _pure_mlx_rms_norm(
    input: mx.array,
    weight: mx.array,
    eps: float,
) -> mx.array:
    values = input.astype(mx.float32)
    variance = mx.mean(values * values, axis=-1, keepdims=True)
    normalized = values * mx.rsqrt(variance + mx.array(eps, dtype=mx.float32))
    return normalized * weight.astype(mx.float32)


def reference_rope(states: mx.array, position_ids: mx.array) -> mx.array:
    """Run the CPU RoPE arithmetic order used by the pinned PyTorch reference."""

    if native_extension_available():
        return _rmsnorm_native.reference_rope(states, position_ids)
    return _pure_mlx_rope(states, position_ids)


def reference_softmax(input: mx.array) -> mx.array:
    """Run the CPU softmax arithmetic used by the pinned PyTorch reference."""

    if native_extension_available():
        return _rmsnorm_native.reference_softmax(input)
    return mx.softmax(input.astype(mx.float32), axis=-1)


def reference_silu(input: mx.array) -> mx.array:
    """Run the CPU SiLU arithmetic used by the pinned PyTorch reference."""

    if native_extension_available():
        return _rmsnorm_native.reference_silu(input)
    return nn.silu(input.astype(mx.float32))


class ReferenceRMSNorm(nn.Module):
    """RMSNorm with PyTorch's CPU reduction order and MLX's GPU kernel."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        if width not in (720, 960):
            raise ValueError(f"ReferenceRMSNorm only supports audited widths 720 and 960, got {width}")
        self.width = width
        self.eps = eps
        self.weight = mx.ones((width,), dtype=mx.float32)

    def __call__(self, input: mx.array) -> mx.array:
        if mx.default_device() == mx.cpu:
            weight = self.weight.astype(mx.float32)
            if native_extension_available():
                return _rmsnorm_native.rms_norm(input, weight, self.eps)
            return _pure_mlx_rms_norm(input, weight, self.eps)
        return mx.fast.rms_norm(input, self.weight, self.eps)
