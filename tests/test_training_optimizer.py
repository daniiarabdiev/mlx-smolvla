"""Cross-framework contracts for the exact SmolVLA optimizer semantics."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch

from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


def test_smolvla_optimizer_defaults_are_the_audited_reference_preset() -> None:
    module = __import__("mlx_smolvla._lab.training.optimizer", fromlist=["SmolVLAOptimizerConfig"])

    config = module.SmolVLAOptimizerConfig()

    assert config.lr == 1e-4
    assert config.betas == (0.9, 0.95)
    assert config.eps == 1e-8
    assert config.weight_decay == 1e-10
    assert config.grad_clip_norm == 10.0
    assert config.warmup_steps == 1_000
    assert config.decay_steps == 30_000
    assert config.decay_lr == 2.5e-6
    assert config.training_horizon == 100_000


def test_first_25_learning_rates_equal_installed_lerobot_scheduler() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.optimizer",
        fromlist=["SmolVLAOptimizerConfig", "cosine_decay_with_warmup_lr"],
    )
    config = module.SmolVLAOptimizerConfig()
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=config.warmup_steps,
        num_decay_steps=config.decay_steps,
        peak_lr=config.lr,
        decay_lr=config.decay_lr,
    ).build(optimizer, config.training_horizon)

    reference_rates: list[float] = []
    for _ in range(25):
        reference_rates.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    mlx_rates = [module.cosine_decay_with_warmup_lr(step, config) for step in range(25)]
    np.testing.assert_allclose(mlx_rates, reference_rates, rtol=0, atol=1e-20)
    assert mlx_rates[0] == pytest.approx(9.990009990009991e-8, rel=0, abs=1e-20)
    assert mlx_rates[-1] == pytest.approx(2.4975024975025017e-6, rel=0, abs=1e-20)


def test_global_norm_clipping_matches_pytorch_multi_tensor_float32() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.optimizer",
        fromlist=["clip_gradients_by_global_norm"],
    )
    gradients = {
        "first": mx.array([3.0, 4.0], dtype=mx.float32),
        "nested": {"second": mx.array([[12.0]], dtype=mx.float32)},
    }
    first = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    second = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float32))
    first.grad = torch.tensor([3.0, 4.0], dtype=torch.float32)
    second.grad = torch.tensor([[12.0]], dtype=torch.float32)

    reference_norm = torch.nn.utils.clip_grad_norm_([first, second], 10.0)
    result = module.clip_gradients_by_global_norm(gradients, 10.0)
    mx.eval(result.total_norm, result.coefficient, result.gradients)

    assert float(result.total_norm) == pytest.approx(float(reference_norm), rel=1e-7)
    assert float(result.coefficient) == pytest.approx(10.0 / (13.0 + 1e-6), rel=1e-7)
    np.testing.assert_allclose(
        np.asarray(result.gradients["first"]),
        first.grad.numpy(),
        rtol=1e-7,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(result.gradients["nested"]["second"]),
        second.grad.numpy(),
        rtol=1e-7,
        atol=0,
    )


class TinyOptimizerModel(nn.Module):
    """Real MLX parameter tree used for direct optimizer evolution checks."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.array([[1.5, -2.0], [0.25, 4.0]], dtype=mx.float32)
        self.bias = mx.array([0.125, -0.75], dtype=mx.float32)


@pytest.mark.parametrize("step_count", [1, 25])
def test_adamw_parameter_evolution_matches_pytorch(step_count: int) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.optimizer",
        fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"],
    )
    config = module.SmolVLAOptimizerConfig(
        lr=1e-2,
        betas=(0.7, 0.8),
        eps=1e-4,
        weight_decay=0.2,
        grad_clip_norm=100.0,
        warmup_steps=2,
        decay_steps=8,
        decay_lr=1e-3,
        training_horizon=25,
    )
    mlx_model = TinyOptimizerModel()
    mlx_optimizer = module.SmolVLAAdamW(config)
    torch_weight = torch.nn.Parameter(
        torch.tensor([[1.5, -2.0], [0.25, 4.0]], dtype=torch.float32)
    )
    torch_bias = torch.nn.Parameter(torch.tensor([0.125, -0.75], dtype=torch.float32))
    torch_optimizer = torch.optim.AdamW(
        [torch_weight, torch_bias],
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    torch_scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=config.warmup_steps,
        num_decay_steps=config.decay_steps,
        peak_lr=config.lr,
        decay_lr=config.decay_lr,
    ).build(torch_optimizer, config.training_horizon)

    for step in range(step_count):
        weight_gradient = np.array(
            [[0.2 + step * 0.01, -0.4], [1e-5, 0.8 - step * 0.005]],
            dtype=np.float32,
        )
        bias_gradient = np.array([1e-6 * (step + 1), -0.3], dtype=np.float32)
        torch_weight.grad = torch.from_numpy(weight_gradient.copy())
        torch_bias.grad = torch.from_numpy(bias_gradient.copy())
        expected_lr = torch_optimizer.param_groups[0]["lr"]
        torch_optimizer.step()
        torch_optimizer.zero_grad()
        torch_scheduler.step()

        used_lr = mlx_optimizer.update(
            mlx_model,
            {
                "weight": mx.array(weight_gradient),
                "bias": mx.array(bias_gradient),
            },
        )
        mx.eval(mlx_model.parameters(), mlx_optimizer.state)

        assert used_lr == pytest.approx(expected_lr, rel=0, abs=1e-15)

    np.testing.assert_allclose(
        np.asarray(mlx_model.weight),
        torch_weight.detach().numpy(),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(mlx_model.bias),
        torch_bias.detach().numpy(),
        rtol=2e-6,
        atol=2e-7,
    )
    assert mlx_optimizer.step_index == step_count
    assert math.isfinite(float(mx.linalg.norm(mlx_model.weight)))
