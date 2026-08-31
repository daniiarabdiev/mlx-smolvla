from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors.numpy import load_file


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
