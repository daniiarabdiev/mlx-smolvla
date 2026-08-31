from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch


def test_cpu_reference_rope_matches_pytorch_cpu_exactly_for_the_fixed_prefix() -> None:
    """Fails when CPU RoPE differs at the 177-token fixed-prefix boundary."""

    from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope

    from smolvla_mlx.language import _apply_reference_rope

    states = torch.linspace(
        -8.0,
        8.0,
        steps=177 * 15 * 64,
        dtype=torch.float32,
    ).reshape(1, 177, 15, 64)
    position_ids = torch.arange(177, dtype=torch.int64)[None, :]
    expected = apply_rope(states, position_ids).numpy()

    with mx.stream(mx.cpu):
        actual = _apply_reference_rope(
            mx.array(states.numpy()),
            mx.array(position_ids.numpy()).astype(mx.int32),
        )
        mx.eval(actual)

    np.testing.assert_array_equal(np.array(actual), expected)
