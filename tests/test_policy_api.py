"""Public native-policy loading and action-queue contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from smolvla_mlx.policy import SmolVLAMLX


@dataclass(frozen=True)
class _PolicyParts:
    policy: SmolVLAMLX
    dtype: str


@pytest.fixture(scope="module", params=("float32", "bfloat16"))
def native_policy(
    request: pytest.FixtureRequest,
    checkpoint_dir: Path,
    base_vlm_dir: Path,
) -> _PolicyParts:
    with mx.stream(mx.cpu):
        policy = SmolVLAMLX.from_pretrained(
            str(checkpoint_dir),
            cache_dir=Path(".cache/smolvla_mlx") / f"policy-{request.param}",
            dtype=request.param,
            tokenizer_dir=base_vlm_dir,
        )
    return _PolicyParts(policy=policy, dtype=request.param)


def test_from_pretrained_initializes_every_converted_checkpoint_parameter(native_policy: _PolicyParts) -> None:
    """Fails if a checkpoint tensor is ignored or attached to the wrong native module."""

    converted = mx.load(str(native_policy.policy.converted_weights_path))
    assert len(converted) == 500
    assert native_policy.policy.loaded_parameter_names == tuple(sorted(converted))


@pytest.mark.parametrize("golden", [0], indirect=True)
def test_select_action_uses_real_chunk_fifo_and_reset(golden, native_policy: _PolicyParts) -> None:
    """Queue behavior must use real model output, not a stubbed prediction path."""

    policy = native_policy.policy
    policy.reset()
    with mx.stream(mx.cpu):
        first = policy.select_action(golden.observation(), noise=golden.mx("noise", mx.float32))
        second = policy.select_action(golden.observation())

    assert first.shape == second.shape == (policy.config.action_dim,)
    assert policy.last_prefix_evaluations == 1
    assert policy.queued_actions == policy.config.n_action_steps - 2
    expected = golden.array("actions/unnormalized")[0]
    tolerance = 5e-3 if native_policy.dtype == "float32" else 5e-2
    np.testing.assert_allclose(first, expected[0], atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(second, expected[1], atol=tolerance, rtol=0.0)

    policy.reset()
    assert policy.queued_actions == 0
