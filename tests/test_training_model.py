"""Composition contracts for the optional SmolVLA MLX training path."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SmallComponents(nn.Module):
    """Small real MLX modules exercising the same component ownership tree."""

    def __init__(self) -> None:
        super().__init__()
        self.vision = nn.Linear(2, 2)
        self.connector = nn.Linear(2, 2)
        self.language = nn.Linear(2, 2)
        self.state_proj = nn.Linear(2, 2)
        self.expert = nn.Linear(2, 2)


def test_reference_selection_trains_only_state_projection_and_expert() -> None:
    module = __import__("training.gradients", fromlist=["configure_reference_trainable"])
    model = SmallComponents()

    names = module.configure_reference_trainable(model)

    assert names
    assert all(name.startswith(("state_proj.", "expert.")) for name in names)
    assert any(name.startswith("state_proj.") for name in names)
    assert any(name.startswith("expert.") for name in names)


def test_canonical_parameter_names_keep_checkpoint_action_projections_at_root() -> None:
    module = __import__("training.gradients", fromlist=["canonical_parameter_name"])

    assert module.canonical_parameter_name("expert.action_in_proj.weight") == "action_in_proj.weight"
    assert module.canonical_parameter_name("expert.layers.0.mlp.up_proj.weight") == (
        "expert.layers.0.mlp.up_proj.weight"
    )
    assert module.canonical_parameter_name("state_proj.weight") == "state_proj.weight"


def test_random_audit_batch_has_the_audited_shapes_and_is_repeatable() -> None:
    module = __import__("training.model", fromlist=["make_random_audit_batch"])
    first = module.make_random_audit_batch(seed=0)
    second = module.make_random_audit_batch(seed=0)

    assert first.processed.pixel_values.shape == (2, 3, 512, 512)
    assert first.processed.pixel_attention_mask.shape == (2, 1)
    assert first.processed.input_ids.shape == (1, 48)
    assert first.processed.text_attention_mask.shape == (1, 48)
    assert first.processed.state.shape == (1, 6)
    assert first.actions.shape == (1, 50, 32)
    assert first.noise.shape == (1, 50, 32)
    assert first.timesteps.shape == (1,)
    assert first.action_dim == 6
    assert bool(mx.array_equal(first.processed.pixel_values, second.processed.pixel_values))
    assert bool(mx.array_equal(first.actions, second.actions))
    assert bool(mx.array_equal(first.noise, second.noise))


def test_full_random_weight_training_path_returns_a_finite_scalar_loss() -> None:
    model_module = __import__("training.model", fromlist=["SmolVLATrainingModel"])
    gradient_module = __import__("training.gradients", fromlist=["configure_reference_trainable"])
    mx.random.seed(0)
    model = model_module.SmolVLATrainingModel()
    model.set_dtype(mx.bfloat16)
    gradient_module.configure_reference_trainable(model)
    batch = model_module.make_random_audit_batch(seed=0)

    loss = model_module.training_loss(model, batch)
    mx.eval(loss)

    assert loss.shape == ()
    assert bool(mx.isfinite(loss))
    assert float(loss) > 0.0
