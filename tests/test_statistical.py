"""Statistical accuracy gate for the native SmolVLA policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from smolvla_mlx.statistical import StatisticalResult


@pytest.fixture(scope="session")
def statistical_result() -> StatisticalResult:
    return StatisticalResult.from_json(Path(".cache/statistical.json"))


@pytest.mark.slow
def test_mlx_is_not_worse_than_reference_on_fifty_real_frames(statistical_result: StatisticalResult) -> None:
    assert statistical_result.sample_count >= 50
    assert statistical_result.mlx_fp32_mae <= 1.05 * statistical_result.torch_fp32_mae
    assert statistical_result.mlx_bf16_mae <= 1.05 * statistical_result.torch_fp32_mae
