"""Torch-free SmolVLA optimizer, clipping, and schedule semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map


@dataclass(frozen=True)
class SmolVLAOptimizerConfig:
    """Audited LeRobot SmolVLA optimizer and scheduler preset."""

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-10
    grad_clip_norm: float = 10.0
    warmup_steps: int = 1_000
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6
    training_horizon: int = 100_000

    def __post_init__(self) -> None:
        if not self.lr > 0:
            raise ValueError("optimizer learning rate must be positive")
        if len(self.betas) != 2 or not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError(f"optimizer betas must be in [0, 1), got {self.betas}")
        if self.eps < 0 or self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("optimizer epsilon/decay must be nonnegative and clip must be positive")
        if self.warmup_steps < 0 or self.decay_steps <= 0 or self.training_horizon <= 0:
            raise ValueError("scheduler step counts must be valid positive horizons")
        if not 0 <= self.decay_lr <= self.lr:
            raise ValueError("scheduler decay LR must be between zero and the peak LR")


def _actual_schedule_steps(config: SmolVLAOptimizerConfig) -> tuple[int, int]:
    actual_warmup_steps = config.warmup_steps
    actual_decay_steps = config.decay_steps
    if config.training_horizon < config.decay_steps:
        scale_factor = config.training_horizon / config.decay_steps
        actual_warmup_steps = int(config.warmup_steps * scale_factor)
        actual_decay_steps = config.training_horizon
    return actual_warmup_steps, actual_decay_steps


def cosine_decay_with_warmup_lr(
    current_step: int,
    config: SmolVLAOptimizerConfig,
) -> float:
    """Return LeRobot's exact optimizer LR before one zero-based update."""

    if current_step < 0:
        raise ValueError(f"scheduler step must be nonnegative, got {current_step}")
    actual_warmup_steps, actual_decay_steps = _actual_schedule_steps(config)
    if current_step < actual_warmup_steps:
        if current_step <= 0:
            multiplier = 1 / (actual_warmup_steps + 1)
        else:
            fraction = 1 - current_step / actual_warmup_steps
            multiplier = (1 / (actual_warmup_steps + 1) - 1) * fraction + 1
    else:
        step = min(current_step, actual_decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
        alpha = config.decay_lr / config.lr
        multiplier = (1 - alpha) * cosine_decay + alpha
    return config.lr * multiplier


@dataclass(frozen=True)
class GradientClipResult:
    """One globally clipped MLX gradient tree and its pre-clip norm."""

    gradients: dict
    total_norm: mx.array
    coefficient: mx.array


def clip_gradients_by_global_norm(
    gradients: dict,
    max_norm: float,
) -> GradientClipResult:
    """Match PyTorch's fp32 multi-tensor global L2 clipping order."""

    if max_norm <= 0:
        raise ValueError(f"maximum gradient norm must be positive, got {max_norm}")
    flat_gradients = tuple(tree_flatten(gradients))
    if not flat_gradients:
        raise ValueError("cannot clip an empty gradient tree")
    tensor_norms = [
        mx.linalg.norm(gradient.astype(mx.float32))
        for _, gradient in flat_gradients
    ]
    total_norm = mx.linalg.norm(mx.stack(tensor_norms).astype(mx.float32))
    coefficient = mx.minimum(
        mx.array(1.0, dtype=mx.float32),
        mx.array(max_norm, dtype=mx.float32)
        / (total_norm + mx.array(1e-6, dtype=mx.float32)),
    )
    clipped = tree_map(
        lambda gradient: gradient * coefficient.astype(gradient.dtype),
        gradients,
    )
    mx.eval(total_norm, coefficient)
    if not bool(mx.isfinite(total_norm)):
        raise RuntimeError("global gradient norm is non-finite")
    return GradientClipResult(
        gradients=clipped,
        total_norm=total_norm,
        coefficient=coefficient,
    )


class SmolVLAAdamW:
    """MLX AdamW with PyTorch bias correction and LeRobot LR ordering."""

    def __init__(self, config: SmolVLAOptimizerConfig | None = None) -> None:
        self.config = SmolVLAOptimizerConfig() if config is None else config
        self._optimizer = optim.AdamW(
            learning_rate=self.config.lr,
            betas=list(self.config.betas),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            bias_correction=True,
        )
        self._step_index = 0

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def state(self) -> dict:
        return self._optimizer.state

    def initialize(self, parameters: dict) -> None:
        """Materialize the expected optimizer-state schema without an update."""

        self._optimizer.init(parameters)
        mx.eval(self._optimizer.state)

    def validate_state_for(self, parameters: dict) -> None:
        """Reject optimizer state that cannot belong to the supplied parameters."""

        parameter_tensors = dict(tree_flatten(parameters))
        state_tensors = dict(tree_flatten(self._optimizer.state))
        expected_names = {"step", "learning_rate"}
        for name in parameter_tensors:
            expected_names.update({f"{name}.m", f"{name}.v"})
        if set(state_tensors) != expected_names:
            raise ValueError("optimizer state tensor names differ from the model schema")
        if (
            state_tensors["step"].shape != ()
            or state_tensors["step"].dtype != mx.uint64
            or state_tensors["learning_rate"].shape != ()
            or state_tensors["learning_rate"].dtype != mx.float32
        ):
            raise ValueError("optimizer state scalar shape/dtype differs from the schema")
        if int(state_tensors["step"]) != self._step_index:
            raise ValueError("optimizer state internal step differs from its schedule position")
        for name, parameter in parameter_tensors.items():
            for suffix in ("m", "v"):
                moment = state_tensors[f"{name}.{suffix}"]
                if moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                    raise ValueError(
                        "optimizer state moment shape/dtype differs from the model "
                        f"for {name}.{suffix}"
                    )

    def load_state(self, state: dict, *, step_index: int) -> None:
        """Restore a checkpointed MLX optimizer tree and schedule position."""

        if step_index < 0:
            raise ValueError("optimizer step index must be nonnegative")
        if "step" not in state:
            raise ValueError("optimizer checkpoint is missing its internal step")
        mx.eval(state)
        internal_step = int(state["step"])
        if internal_step != step_index:
            raise ValueError(
                f"optimizer step mismatch: state={internal_step}, checkpoint={step_index}"
            )
        self._optimizer.state = state
        self._step_index = step_index

    def update(self, model: nn.Module, gradients: dict) -> float:
        """Apply one scheduled update and return the LR used for that update."""

        learning_rate = cosine_decay_with_warmup_lr(self._step_index, self.config)
        self._optimizer.learning_rate = learning_rate
        self._optimizer.update(model, gradients)
        self._step_index += 1
        return learning_rate
