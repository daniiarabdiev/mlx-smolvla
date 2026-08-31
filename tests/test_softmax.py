from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch


def test_cpu_reference_softmax_matches_pytorch_cpu_at_prefix_width() -> None:
    """Fails when CPU attention probabilities use a different exp/reduction path."""

    from smolvla_mlx.rmsnorm import reference_softmax

    scores = torch.linspace(-100.0, 16.0, steps=4 * 177, dtype=torch.float32).reshape(4, 177)
    expected = torch.nn.functional.softmax(scores, dim=-1).numpy()

    with mx.stream(mx.cpu):
        actual = reference_softmax(mx.array(scores.numpy()))
        mx.eval(actual)

    np.testing.assert_array_equal(np.array(actual), expected)


def test_cpu_reference_softmax_matches_pytorch_cpu_for_masked_attention_rows() -> None:
    """Fails when the vector sum uses a different four-lane reduction tree."""

    from smolvla_mlx.rmsnorm import reference_softmax

    scores = torch.randn(
        (1, 15, 177, 177),
        generator=torch.Generator().manual_seed(20260831),
        dtype=torch.float32,
    ) * 3.0
    scores[..., 130:, 94:] = torch.finfo(torch.float32).min
    expected = torch.nn.functional.softmax(scores, dim=-1).numpy()

    with mx.stream(mx.cpu):
        actual = reference_softmax(mx.array(scores.numpy()))
        mx.eval(actual)

    np.testing.assert_array_equal(np.array(actual), expected)
