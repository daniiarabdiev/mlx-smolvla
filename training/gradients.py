"""Trainable-parameter selection and canonical naming for SmolVLA training."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np


@dataclass(frozen=True)
class GradientComparison:
    """Float64-accumulated metrics for one canonical gradient tensor."""

    name: str
    relative_l2: float
    cosine_similarity: float
    max_abs_difference: float
    reference_l2: float
    candidate_l2: float


def compare_gradient_arrays(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> GradientComparison:
    """Compare one reference/candidate pair without hiding zero or invalid data."""

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            f"gradient shapes differ for {name}: "
            f"{reference_array.shape} != {candidate_array.shape}"
        )
    reference_values = reference_array.reshape(-1).astype(np.float64)
    candidate_values = candidate_array.reshape(-1).astype(np.float64)
    if not np.all(np.isfinite(reference_values)) or not np.all(np.isfinite(candidate_values)):
        raise ValueError(f"gradient comparison contains non-finite values for {name}")

    reference_l2 = math.sqrt(float(np.dot(reference_values, reference_values)))
    if reference_l2 == 0.0:
        raise ValueError(f"gradient comparison has a zero-norm reference for {name}")
    candidate_l2 = math.sqrt(float(np.dot(candidate_values, candidate_values)))
    difference = candidate_values - reference_values
    difference_l2 = math.sqrt(float(np.dot(difference, difference)))
    cosine_similarity = (
        0.0
        if candidate_l2 == 0.0
        else float(np.dot(candidate_values, reference_values)) / (candidate_l2 * reference_l2)
    )
    return GradientComparison(
        name=name,
        relative_l2=difference_l2 / reference_l2,
        cosine_similarity=cosine_similarity,
        max_abs_difference=float(np.max(np.abs(difference))),
        reference_l2=reference_l2,
        candidate_l2=candidate_l2,
    )


def relative_loss_difference(reference: float, candidate: float) -> float:
    """Return absolute relative error against one finite, nonzero reference loss."""

    reference_value = float(reference)
    candidate_value = float(candidate)
    if not math.isfinite(reference_value) or not math.isfinite(candidate_value):
        raise ValueError("loss comparison contains a non-finite value")
    if reference_value == 0.0:
        raise ValueError("loss comparison requires a nonzero reference")
    return abs(candidate_value - reference_value) / abs(reference_value)


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
