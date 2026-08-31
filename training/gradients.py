"""Trainable-parameter selection and canonical naming for SmolVLA training."""

from __future__ import annotations

import mlx.nn as nn
from mlx.utils import tree_flatten


def configure_reference_trainable(model: nn.Module) -> tuple[str, ...]:
    """Freeze the model, then enable the audited expert and state projection."""

    model.freeze()
    model.state_proj.unfreeze()
    model.expert.unfreeze()
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    if not names or not all(name.startswith(("state_proj.", "expert.")) for name in names):
        raise RuntimeError(f"unexpected reference trainable set: {names}")
    return names


def canonical_parameter_name(name: str) -> str:
    """Map the training container's action projections to checkpoint names."""

    if name.startswith("expert.action_"):
        return name.removeprefix("expert.")
    return name
