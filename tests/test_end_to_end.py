"""Deterministic full-pipeline parity tests for the native policy."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tests.test_policy_api import native_policy


@pytest.mark.parametrize("golden", range(8), indirect=True)
def test_predict_action_chunk_matches_all_real_reference_traces(golden, native_policy) -> None:
    """Exercises preprocessing, vision, prefix cache reuse, expert, and action slicing together."""

    with mx.stream(mx.cpu):
        actual = native_policy.policy.predict_action_chunk(
            golden.observation(),
            noise=golden.mx("noise", mx.float32),
        )
        mx.eval(actual)

    expected = golden.array("actions/normalized").astype(np.float32, copy=False)
    actual_array = np.array(actual.astype(mx.float32))
    assert actual_array.shape == expected.shape == (1, 50, native_policy.policy.config.action_dim)
    max_abs = np.max(np.abs(actual_array - expected))
    assert max_abs <= (5e-3 if native_policy.dtype == "float32" else 5e-2), max_abs
    assert native_policy.policy.last_prefix_evaluations == 1
