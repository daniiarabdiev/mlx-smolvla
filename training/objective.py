"""Exact flow-matching input construction and physical-action loss."""

from __future__ import annotations

import mlx.core as mx


def flow_matching_inputs(
    actions: mx.array,
    noise: mx.array,
    timesteps: mx.array,
) -> tuple[mx.array, mx.array]:
    """Construct the reference's noisy action and target velocity tensors."""

    if actions.shape != noise.shape:
        raise ValueError(
            "actions and noise must have identical shapes; "
            f"got {actions.shape} and {noise.shape}"
        )
    expected_timesteps = (actions.shape[0],)
    if timesteps.shape != expected_timesteps:
        raise ValueError(
            f"timesteps must have shape {expected_timesteps}, got {timesteps.shape}"
        )

    action_values = actions.astype(mx.float32)
    noise_values = noise.astype(mx.float32)
    time = timesteps.astype(mx.float32)[:, None, None]
    noisy_actions = time * noise_values + (1.0 - time) * action_values
    target_velocity = noise_values - action_values
    return noisy_actions, target_velocity


def masked_velocity_mse(
    predicted_velocity: mx.array,
    target_velocity: mx.array,
    *,
    action_dim: int,
    action_is_pad: mx.array | None = None,
) -> mx.array:
    """Return LeRobot's MSE over valid timesteps and physical dimensions."""

    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError(
            "prediction and target velocity must have identical shapes; "
            f"got {predicted_velocity.shape} and {target_velocity.shape}"
        )
    padded_width = predicted_velocity.shape[-1]
    if not 1 <= action_dim <= padded_width:
        raise ValueError(f"action_dim must be in [1, {padded_width}], got {action_dim}")

    error = (
        predicted_velocity.astype(mx.float32)[:, :, :action_dim]
        - target_velocity.astype(mx.float32)[:, :, :action_dim]
    )
    squared_error = error * error
    if action_is_pad is None:
        return mx.mean(squared_error)

    expected_mask_shape = predicted_velocity.shape[:2]
    if action_is_pad.shape != expected_mask_shape:
        raise ValueError(
            f"action_is_pad must have shape {expected_mask_shape}, got {action_is_pad.shape}"
        )
    valid = mx.logical_not(action_is_pad.astype(mx.bool_)).astype(mx.float32)
    numerator = mx.sum(squared_error * valid[:, :, None])
    denominator = mx.sum(valid) * mx.array(action_dim, dtype=mx.float32)
    denominator = mx.maximum(denominator, mx.array(1.0, dtype=mx.float32))
    return numerator / denominator
