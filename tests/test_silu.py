from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as torch_functional


def test_cpu_reference_silu_matches_pytorch_cpu_at_mlp_width() -> None:
    """Fails when the CPU SwiGLU activation uses non-PyTorch arithmetic."""

    from mlx_smolvla.rmsnorm import reference_silu

    inputs = torch.linspace(-18.0, 15.0, steps=3 * 2560, dtype=torch.float32).reshape(3, 2560)
    expected = torch_functional.silu(inputs).numpy()

    with mx.stream(mx.cpu):
        actual = reference_silu(mx.array(inputs.numpy()))
        mx.eval(actual)

    np.testing.assert_array_equal(np.array(actual), expected)
