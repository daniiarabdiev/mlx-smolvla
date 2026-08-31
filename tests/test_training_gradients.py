"""Numerical contracts for framework-to-framework gradient comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest


def test_gradient_comparison_matches_float64_relative_l2_and_cosine() -> None:
    module = __import__("training.gradients", fromlist=["compare_gradient_arrays"])
    reference = np.array([3.0, 4.0], dtype=np.float32)
    candidate = np.array([3.0, 5.0], dtype=np.float32)

    result = module.compare_gradient_arrays("weight", reference, candidate)

    assert result.name == "weight"
    assert result.reference_l2 == 5.0
    assert result.candidate_l2 == math.sqrt(34.0)
    assert result.relative_l2 == 0.2
    assert result.cosine_similarity == 29.0 / (5.0 * math.sqrt(34.0))
    assert result.max_abs_difference == 1.0


def test_zero_candidate_gradient_is_a_gateable_comparison() -> None:
    module = __import__("training.gradients", fromlist=["compare_gradient_arrays"])

    result = module.compare_gradient_arrays(
        "weight",
        np.array([1.0, -2.0], dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    )

    assert result.relative_l2 == 1.0
    assert result.cosine_similarity == 0.0


def test_relative_loss_difference_uses_the_nonzero_reference_denominator() -> None:
    module = __import__("training.gradients", fromlist=["relative_loss_difference"])

    assert module.relative_loss_difference(2.0, 2.0002) == pytest.approx(1e-4)


@pytest.mark.parametrize(
    ("reference", "candidate", "message"),
    (
        (np.ones((2,), dtype=np.float32), np.ones((3,), dtype=np.float32), "shapes differ"),
        (np.array([1.0, np.nan]), np.ones((2,), dtype=np.float32), "non-finite"),
        (np.ones((2,), dtype=np.float32), np.array([1.0, np.inf]), "non-finite"),
        (np.zeros((2,), dtype=np.float32), np.ones((2,), dtype=np.float32), "zero-norm reference"),
    ),
)
def test_gradient_comparison_rejects_invalid_inputs(
    reference: np.ndarray,
    candidate: np.ndarray,
    message: str,
) -> None:
    module = __import__("training.gradients", fromlist=["compare_gradient_arrays"])

    with pytest.raises(ValueError, match=message):
        module.compare_gradient_arrays("weight", reference, candidate)
