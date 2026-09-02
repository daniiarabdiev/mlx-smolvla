"""Native MLX LoRA topology, precision, gradient, and merge contracts."""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest


def test_lora_linear_is_zero_initialized_and_uses_alpha_over_rank() -> None:
    module = __import__("training.lora", fromlist=["LoRALinear"])
    mx.random.seed(17)
    base = nn.Linear(5, 3, bias=True)
    expected = base(mx.array([[1.0, -2.0, 0.5, 3.0, -1.0]], dtype=mx.float32))

    adapter = module.LoRALinear(base, rank=2, alpha=6.0)
    actual = adapter(mx.array([[1.0, -2.0, 0.5, 3.0, -1.0]], dtype=mx.float32))
    mx.eval(expected, actual)

    assert adapter.rank == 2
    assert adapter.alpha == 6.0
    assert adapter.scale == 3.0
    assert adapter.lora_a.shape == (5, 2)
    assert adapter.lora_b.shape == (2, 3)
    assert adapter.lora_a.dtype == mx.float32
    assert adapter.lora_b.dtype == mx.float32
    assert float(mx.max(mx.abs(adapter.lora_a))) <= 1 / math.sqrt(5)
    assert bool(mx.all(adapter.lora_b == 0))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_lora_linear_merge_preserves_nonzero_adapter_output_and_bias() -> None:
    module = __import__("training.lora", fromlist=["LoRALinear"])
    base = nn.Linear(3, 2, bias=True)
    adapter = module.LoRALinear(base, rank=2, alpha=4.0)
    adapter.lora_a = mx.array(
        [[0.2, -0.3], [0.5, 0.7], [-0.4, 0.1]], dtype=mx.float32
    )
    adapter.lora_b = mx.array([[0.8, -0.6], [0.25, 0.9]], dtype=mx.float32)
    inputs = mx.array([[1.0, -2.0, 0.5], [-0.2, 0.4, 3.0]], dtype=mx.float32)

    expected = adapter(inputs)
    merged = adapter.merged_linear(dtype=mx.float32)
    actual = merged(inputs)
    mx.eval(expected, actual)

    assert isinstance(merged, nn.Linear)
    assert merged.bias is not None
    # Merging changes two fp32 matmuls into one and therefore changes Metal's
    # reduction association slightly; it must remain inside the protected
    # end-to-end inference tolerance.
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-3, atol=2e-3)


def test_install_lora_targets_exact_full_smolvla_topology() -> None:
    lora = __import__("training.lora", fromlist=["install_lora"])
    model_module = __import__("training.model", fromlist=["SmolVLATrainingModel"])
    mx.random.seed(20260901)
    model = model_module.SmolVLATrainingModel()
    model.set_dtype(mx.bfloat16)

    report = lora.install_lora(model, lora.LoRAConfig())
    trainable = tuple(tree_flatten(model.trainable_parameters()))

    assert report.adapter_count == 229
    assert len(report.target_names) == 229
    assert len(set(report.target_names)) == 229
    assert report.trainable_tensor_count == 458
    assert report.trainable_scalar_count > 0
    assert tuple(name for name, _ in trainable) == report.trainable_names
    assert all(name.endswith((".lora_a", ".lora_b")) for name, _ in trainable)
    assert all(value.dtype == mx.float32 for _, value in trainable)
    assert any(name.startswith("language.layers.0.self_attn.q_proj") for name in report.target_names)
    assert any(name.startswith("language.layers.15.mlp.down_proj") for name in report.target_names)
    assert any(name.startswith("expert.layers.0.self_attn.q_proj") for name in report.target_names)
    assert any(name.startswith("expert.layers.15.mlp.down_proj") for name in report.target_names)
    assert "expert.action_in_proj" in report.target_names
    assert "expert.action_out_proj" in report.target_names
    assert "expert.action_time_mlp_in" in report.target_names
    assert "expert.action_time_mlp_out" in report.target_names
    assert "state_proj" in report.target_names


def test_expert_only_lora_targets_only_expert_attention_and_mlp() -> None:
    lora = __import__("training.lora", fromlist=["install_lora", "merge_lora"])
    model_module = __import__("training.model", fromlist=["SmolVLATrainingModel"])
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    mx.random.seed(20260901)
    model = model_module.SmolVLATrainingModel()
    model.set_dtype(mx.bfloat16)

    report = lora.install_lora(
        model,
        lora.LoRAConfig(scope=lora.EXPERT_ONLY_SCOPE),
    )
    trainable = tuple(tree_flatten(model.trainable_parameters()))

    assert report.scope == "expert_only"
    assert report.adapter_count == 112
    assert report.trainable_tensor_count == 224
    assert report.trainable_scalar_count == 1_708_032
    assert tuple(name for name, _ in trainable) == report.trainable_names
    assert all(name.startswith("expert.layers.") for name in report.target_names)
    assert all(
        ".self_attn." in name or ".mlp." in name
        for name in report.target_names
    )
    assert not any(name.startswith("language.") for name in report.target_names)
    assert not any(name.startswith("vision.") for name in report.target_names)
    assert not any(name.startswith("state_proj") for name in report.target_names)
    assert not any("action_" in name for name in report.target_names)
    assert all(name.endswith((".lora_a", ".lora_b")) for name, _ in trainable)
    assert all(value.dtype == mx.float32 for _, value in trainable)
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=3000)
    )
    optimizer.initialize(model.trainable_parameters())
    optimizer.validate_state_for(model.trainable_parameters())

    merged = lora.merge_lora(model, dtype=mx.float32)
    assert merged.scope == "expert_only"
    assert merged.target_names == report.target_names
    assert merged.adapter_count == 112
    assert tuple(lora.iter_lora(model)) == ()
    assert tuple(tree_flatten(model.trainable_parameters())) == ()
    assert isinstance(model.language.layers[0].self_attn.q_proj, nn.Linear)
    assert isinstance(model.expert.layers[15].mlp.down_proj, nn.Linear)
    assert isinstance(model.state_proj, nn.Linear)


def test_lora_gradients_cover_every_adapter_tensor_on_full_training_path() -> None:
    lora = __import__("training.lora", fromlist=["install_lora"])
    model_module = __import__("training.model", fromlist=["SmolVLATrainingModel"])
    mx.random.seed(20260901)
    model = model_module.SmolVLATrainingModel()
    model.set_dtype(mx.bfloat16)
    report = lora.install_lora(model, lora.LoRAConfig())
    batch = model_module.make_random_audit_batch(seed=20260901)
    value_and_grad = nn.value_and_grad(
        model,
        lambda: model_module.training_loss(model, batch),
    )

    loss, gradients = value_and_grad()
    flat_gradients = tuple(tree_flatten(gradients))
    mx.eval(loss, *[gradient for _, gradient in flat_gradients])

    assert tuple(name for name, _ in flat_gradients) == report.trainable_names
    assert len(flat_gradients) == 458
    assert bool(mx.isfinite(loss))
    assert all(bool(mx.all(mx.isfinite(gradient))) for _, gradient in flat_gradients)
    # The standard zero-B initialization makes A's first gradient exactly zero.
    assert all(
        float(mx.linalg.norm(gradient.astype(mx.float32))) == 0.0
        for name, gradient in flat_gradients
        if name.endswith(".lora_a")
    )
    zero_b = [
        name
        for name, gradient in flat_gradients
        if name.endswith(".lora_b")
        and float(mx.linalg.norm(gradient.astype(mx.float32))) == 0.0
    ]
    # The terminal VLM layer exports K/V to the expert, but its query/output and
    # MLP result have no downstream consumer in SmolVLA's action loss.
    assert set(zero_b) == {
        "language.layers.15.self_attn.q_proj.lora_b",
        "language.layers.15.self_attn.o_proj.lora_b",
        "language.layers.15.mlp.gate_proj.lora_b",
        "language.layers.15.mlp.up_proj.lora_b",
        "language.layers.15.mlp.down_proj.lora_b",
    }


def test_merge_lora_replaces_every_wrapper_with_plain_linears() -> None:
    lora = __import__("training.lora", fromlist=["merge_lora", "iter_lora"])
    model_module = __import__("training.model", fromlist=["SmolVLATrainingModel"])
    mx.random.seed(20260901)
    model = model_module.SmolVLATrainingModel()
    model.set_dtype(mx.bfloat16)
    installed = lora.install_lora(model, lora.LoRAConfig(rank=4, alpha=8.0))
    assert len(tuple(lora.iter_lora(model))) == 229

    merged = lora.merge_lora(model, dtype=mx.float32)

    assert merged.target_names == installed.target_names
    assert merged.adapter_count == 229
    assert tuple(lora.iter_lora(model)) == ()
    assert tuple(tree_flatten(model.trainable_parameters())) == ()
    assert isinstance(model.state_proj, nn.Linear)
    assert isinstance(model.language.layers[0].self_attn.q_proj, nn.Linear)
    assert isinstance(model.expert.layers[15].mlp.down_proj, nn.Linear)


def test_lora_config_rejects_invalid_rank_alpha_and_dropout() -> None:
    module = __import__("training.lora", fromlist=["LoRAConfig"])

    for kwargs in (
        {"rank": 0},
        {"alpha": 0.0},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"scope": "language_only"},
    ):
        try:
            module.LoRAConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"LoRAConfig accepted invalid values: {kwargs}")


@pytest.mark.slow
def test_training_composition_loads_real_checkpoint_in_bfloat16() -> None:
    model_module = __import__("training.model", fromlist=["SmolVLATrainingModel"])

    model = model_module.SmolVLATrainingModel.from_pretrained(
        cache_dir=Path(".cache/mlx_smolvla/policy-float32"),
        dtype=mx.bfloat16,
    )
    parameters = tuple(tree_flatten(model.parameters()))

    assert len(parameters) == 500
    assert all(parameter.dtype == mx.bfloat16 for _, parameter in parameters)
    assert model.converted_weights_path is not None
    assert model.converted_weights_path.name == "model.bfloat16.safetensors"
