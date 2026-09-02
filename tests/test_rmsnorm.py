from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.numpy import load_file

from smolvla_mlx.rmsnorm import native_extension_available


requires_native_extension = pytest.mark.skipif(
    not native_extension_available(),
    reason="exact CPU parity requires the optional native reference extension",
)


@requires_native_extension
def test_cpu_reference_rmsnorm_matches_pytorch_cpu_exactly_on_real_prefix() -> None:
    """Fails if the runtime uses MLX's different CPU reduction order."""

    from smolvla_mlx.rmsnorm import ReferenceRMSNorm

    prefix = np.load("tests/golden/sample_000/prefix/embeddings.npy").astype(np.float32, copy=False)
    checkpoint = Path(".cache/smolvla_mlx/language-prefix-float32/model.float32.safetensors")
    weight = load_file(checkpoint)["language.layers.0.input_layernorm.weight"]
    expected = torch.rms_norm(
        torch.from_numpy(prefix.copy()),
        (prefix.shape[-1],),
        torch.from_numpy(weight.copy()),
        1e-5,
    ).numpy()

    with mx.stream(mx.cpu):
        normalizer = ReferenceRMSNorm(prefix.shape[-1], eps=1e-5)
        normalizer.weight = mx.array(weight)
        actual = normalizer(mx.array(prefix))
        mx.eval(actual)

    np.testing.assert_array_equal(np.array(actual), expected)


@requires_native_extension
def test_cpu_reference_rmsnorm_upcasts_bfloat16_storage_weight() -> None:
    """Fails if compact checkpoint storage reaches the float32 CPU primitive unchanged."""

    from smolvla_mlx.rmsnorm import ReferenceRMSNorm

    prefix = np.load("tests/golden/sample_000/prefix/embeddings.npy").astype(np.float32, copy=False)
    checkpoint = Path(".cache/smolvla_mlx/language-prefix-float32/model.float32.safetensors")
    source_weight = load_file(checkpoint)["language.layers.0.input_layernorm.weight"]
    expected_weight = torch.from_numpy(source_weight.copy()).to(torch.bfloat16).float()
    expected = torch.rms_norm(
        torch.from_numpy(prefix.copy()),
        (prefix.shape[-1],),
        expected_weight,
        1e-5,
    ).numpy()

    with mx.stream(mx.cpu):
        normalizer = ReferenceRMSNorm(prefix.shape[-1], eps=1e-5)
        normalizer.weight = mx.array(source_weight).astype(mx.bfloat16)
        actual = normalizer(mx.array(prefix))
        mx.eval(actual)

    np.testing.assert_array_equal(np.array(actual), expected)


@requires_native_extension
def test_cpu_reference_rmsnorm_matches_pytorch_cpu_on_expert_width() -> None:
    """The 720-wide action expert needs the same source CPU reduction contract."""

    from smolvla_mlx.rmsnorm import ReferenceRMSNorm

    suffix = np.load("tests/golden/sample_000/flow/step_00/suffix_embeddings.npy").astype(
        np.float32, copy=False
    )
    checkpoint = Path(".cache/smolvla_mlx/language-prefix-float32/model.float32.safetensors")
    weight = load_file(checkpoint)["expert.layers.0.input_layernorm.weight"]
    expected = torch.rms_norm(
        torch.from_numpy(suffix.copy()),
        (suffix.shape[-1],),
        torch.from_numpy(weight.copy()),
        1e-5,
    ).numpy()

    with mx.stream(mx.cpu):
        normalizer = ReferenceRMSNorm(suffix.shape[-1], eps=1e-5)
        normalizer.weight = mx.array(weight)
        actual = normalizer(mx.array(suffix))
        mx.eval(actual)

    np.testing.assert_array_equal(np.array(actual), expected)


def test_pure_mlx_fallback_covers_every_cpu_compatibility_primitive(monkeypatch) -> None:
    from smolvla_mlx import rmsnorm

    monkeypatch.setattr(rmsnorm, "_rmsnorm_native", None)
    assert rmsnorm.native_extension_available() is False
    assert rmsnorm.cpu_compatibility_backend() == "pure-mlx-fallback"

    values = mx.array(np.linspace(-1.0, 1.0, 720, dtype=np.float32).reshape(1, 1, 720))
    normalizer = rmsnorm.ReferenceRMSNorm(720, eps=1e-5)
    normalized = normalizer(values)
    expected = np.array(values) / np.sqrt(
        np.mean(np.array(values) ** 2, axis=-1, keepdims=True) + 1e-5
    )

    logits = mx.array([[[-1.0, 0.0, 1.0]]], dtype=mx.float32)
    softmax = rmsnorm.reference_softmax(logits)
    silu = rmsnorm.reference_silu(logits)

    states = mx.array(np.linspace(-0.5, 0.5, 128, dtype=np.float32).reshape(1, 2, 1, 64))
    positions = mx.array([[0, 1]], dtype=mx.int32)
    rotated = rmsnorm.reference_rope(states, positions)
    mx.eval(normalized, softmax, silu, rotated)

    np.testing.assert_allclose(np.array(normalized), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(softmax).sum(axis=-1), 1.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        np.array(silu),
        np.array(logits) / (1.0 + np.exp(-np.array(logits))),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(np.array(rotated[:, :1]), np.array(states[:, :1]))
    assert np.isfinite(np.array(rotated)).all()
