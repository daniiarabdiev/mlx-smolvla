"""Golden tests for the native SmolVLA flow-matching sampler."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from smolvla_mlx.flow import euler_sample, euler_step, timestep_schedule
from tests.test_expert import _assert_error, expert_parts, prefix_cache


def test_schedule_and_single_euler_update_match_audited_reference() -> None:
    np.testing.assert_array_equal(
        np.array(timestep_schedule(10)),
        np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32),
    )
    with mx.stream(mx.cpu):
        updated = euler_step(
            mx.array([[1.0, -2.0]], dtype=mx.float32),
            mx.array([[3.0, -4.0]], dtype=mx.float32),
            mx.array(-0.1, dtype=mx.float32),
        )
        mx.eval(updated)
    np.testing.assert_array_equal(np.array(updated), np.array([[0.7, -1.6]], dtype=np.float32))


@pytest.mark.parametrize("golden", range(8), indirect=True)
def test_all_euler_steps_match_the_real_trace_and_final_normalized_actions(golden, expert_parts) -> None:
    """Exercises the frozen VLM cache across the complete 10-step denoising loop."""

    observed = []
    with mx.stream(mx.cpu):
        cache = prefix_cache(golden, expert_parts)

        def denoise(x_t: mx.array, timestep: mx.array) -> mx.array:
            result = expert_parts.expert.denoise(cache, x_t, timestep, collect_layer_outputs=True)
            observed.append((x_t, result))
            return result.velocity

        actions = euler_sample(
            denoise,
            golden.mx("noise", mx.float32),
            num_steps=10,
        )
        mx.eval(
            actions,
            *(
                value
                for x_t, result in observed
                for value in (x_t, result.hidden, result.velocity, *result.layer_outputs)
            ),
        )

    assert len(observed) == 10
    for step, (x_t, result) in enumerate(observed):
        _assert_error(x_t, golden.array(f"flow/step_{step:02d}/x_t"), dtype=expert_parts.dtype)
        assert len(result.layer_outputs) == 16
        for layer_index, layer_output in enumerate(result.layer_outputs):
            _assert_error(
                layer_output,
                golden.array(f"expert/step_{step:02d}/layer_{layer_index:02d}/output"),
                dtype=expert_parts.dtype,
            )
        _assert_error(result.hidden, golden.array(f"expert/step_{step:02d}/output"), dtype=expert_parts.dtype)
        _assert_error(result.velocity, golden.array(f"flow/step_{step:02d}/velocity"), dtype=expert_parts.dtype)

    expected = golden.array("actions/normalized")
    actual = np.array(actions[:, :, : expected.shape[-1]].astype(mx.float32))
    max_abs = np.max(np.abs(actual - expected.astype(np.float32, copy=False)))
    assert max_abs <= (5e-3 if expert_parts.dtype == "float32" else 5e-2), max_abs
