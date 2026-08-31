"""Training-only differentiable math and flow-objective contracts."""

from __future__ import annotations

import mlx.core as mx
import pytest


def test_differentiable_rms_norm_has_finite_input_and_weight_gradients() -> None:
    module = __import__("training.differentiable", fromlist=["differentiable_rms_norm"])
    function = module.differentiable_rms_norm
    inputs = mx.array([[1.0, -2.0, 3.0]], dtype=mx.float32)
    weight = mx.ones((3,), dtype=mx.float32)
    value_and_grad = mx.value_and_grad(
        lambda values, scale: mx.sum(function(values, scale, 1e-5)),
        argnums=(0, 1),
    )

    value, (input_gradient, weight_gradient) = value_and_grad(inputs, weight)
    mx.eval(value, input_gradient, weight_gradient)

    assert bool(mx.all(mx.isfinite(input_gradient)))
    assert bool(mx.all(mx.isfinite(weight_gradient)))
    assert input_gradient.shape == inputs.shape
    assert weight_gradient.shape == weight.shape


def test_flow_objective_ignores_padded_action_dimensions() -> None:
    module = __import__("training.objective", fromlist=["flow_matching_inputs"])
    actions = mx.zeros((1, 2, 4), dtype=mx.float32)
    noise = mx.ones((1, 2, 4), dtype=mx.float32)
    timesteps = mx.array([0.25], dtype=mx.float32)
    prediction = mx.array(
        [[[0.0, 1.0, 100.0, 100.0], [0.0, 1.0, 100.0, 100.0]]],
        dtype=mx.float32,
    )

    noisy_actions, target = module.flow_matching_inputs(actions, noise, timesteps)
    loss = module.masked_velocity_mse(prediction, target, action_dim=2)
    mx.eval(loss, noisy_actions, target)

    assert bool(mx.allclose(noisy_actions, mx.full(actions.shape, 0.25)))
    assert bool(mx.allclose(target, mx.ones(actions.shape)))
    assert float(loss) == 0.5


def test_flow_objective_ignores_temporally_padded_actions_in_numerator_and_denominator() -> None:
    module = __import__("training.objective", fromlist=["masked_velocity_mse"])
    target = mx.zeros((1, 2, 4), dtype=mx.float32)
    prediction = mx.array(
        [[[1.0, 3.0, 100.0, 100.0], [50.0, 70.0, 100.0, 100.0]]],
        dtype=mx.float32,
    )
    action_is_pad = mx.array([[False, True]], dtype=mx.bool_)

    loss = module.masked_velocity_mse(
        prediction,
        target,
        action_dim=2,
        action_is_pad=action_is_pad,
    )
    mx.eval(loss)

    assert float(loss) == 5.0


def test_flow_objective_all_padded_batch_uses_clamped_denominator() -> None:
    module = __import__("training.objective", fromlist=["masked_velocity_mse"])
    target = mx.zeros((1, 2, 4), dtype=mx.float32)
    prediction = mx.full((1, 2, 4), 100.0, dtype=mx.float32)

    loss = module.masked_velocity_mse(
        prediction,
        target,
        action_dim=2,
        action_is_pad=mx.ones((1, 2), dtype=mx.bool_),
    )
    mx.eval(loss)

    assert float(loss) == 0.0


def test_flow_inputs_reject_mismatched_action_and_noise_shapes() -> None:
    module = __import__("training.objective", fromlist=["flow_matching_inputs"])

    with pytest.raises(ValueError, match="actions and noise must have identical shapes"):
        module.flow_matching_inputs(
            mx.zeros((1, 2, 4)),
            mx.zeros((1, 3, 4)),
            mx.array([0.5]),
        )


def test_flow_inputs_require_a_timestep_for_each_batch_item() -> None:
    module = __import__("training.objective", fromlist=["flow_matching_inputs"])

    with pytest.raises(ValueError, match="timesteps must have shape"):
        module.flow_matching_inputs(
            mx.zeros((2, 3, 4)),
            mx.zeros((2, 3, 4)),
            mx.array([0.5]),
        )


def test_velocity_loss_rejects_action_width_outside_the_padded_tensor() -> None:
    module = __import__("training.objective", fromlist=["masked_velocity_mse"])
    velocity = mx.zeros((1, 2, 4))

    with pytest.raises(ValueError, match="action_dim must be in"):
        module.masked_velocity_mse(velocity, velocity, action_dim=5)


def test_velocity_loss_rejects_mismatched_prediction_and_target_shapes() -> None:
    module = __import__("training.objective", fromlist=["masked_velocity_mse"])

    with pytest.raises(ValueError, match="prediction and target velocity must have identical shapes"):
        module.masked_velocity_mse(
            mx.zeros((1, 2, 4)),
            mx.zeros((1, 3, 4)),
            action_dim=4,
        )


def test_velocity_loss_rejects_mismatched_temporal_mask_shape() -> None:
    module = __import__("training.objective", fromlist=["masked_velocity_mse"])
    velocity = mx.zeros((2, 3, 4))

    with pytest.raises(ValueError, match="action_is_pad must have shape"):
        module.masked_velocity_mse(
            velocity,
            velocity,
            action_dim=4,
            action_is_pad=mx.zeros((2, 2), dtype=mx.bool_),
        )
