"""Audited flow-matching schedule and forward-Euler sampler for SmolVLA."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx


def timestep_schedule(num_steps: int) -> mx.array:
    """Return the reference's fp32 sampling times, from 1.0 down to 1 / steps."""

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    dt = -1.0 / num_steps
    return mx.array([1.0 + step * dt for step in range(num_steps)], dtype=mx.float32)


def euler_step(actions: mx.array, velocity: mx.array, dt: mx.array) -> mx.array:
    """Apply the source's single forward-Euler update: ``x_t + dt * v_t``."""

    if actions.shape != velocity.shape:
        raise ValueError(f"actions shape {actions.shape} does not match velocity shape {velocity.shape}")
    return actions + dt.astype(actions.dtype) * velocity


def euler_sample(
    denoise_fn: Callable[[mx.array, mx.array], mx.array],
    noise: mx.array,
    *,
    num_steps: int,
) -> mx.array:
    """Integrate a velocity field from the noise endpoint at t=1 to t=0."""

    if noise.ndim < 1:
        raise ValueError(f"noise must include a batch dimension, got {noise.shape}")
    schedule = timestep_schedule(num_steps)
    dt = mx.array(-1.0 / num_steps, dtype=noise.dtype)
    x_t = noise
    for step in range(num_steps):
        timestep = mx.broadcast_to(schedule[step], (noise.shape[0],))
        velocity = denoise_fn(x_t, timestep)
        x_t = euler_step(x_t, velocity, dt)
    return x_t
