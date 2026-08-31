"""Gradient accumulation, budget, metrics, and run-state contracts for T3."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 1, bias=False)
        self.proj.weight = mx.array([[0.25, -0.5]], dtype=mx.float32)


def _tiny_loss(model: TinyRegressor, batch: tuple[mx.array, mx.array]) -> mx.array:
    inputs, targets = batch
    error = model.proj(inputs) - targets
    return mx.mean(error * error)


def test_accumulation_is_the_mean_of_eight_distinct_microbatch_gradients() -> None:
    module = __import__("training.finetune", fromlist=["accumulate_gradients"])
    model = TinyRegressor()
    batches = tuple(
        (
            mx.array([[float(index), 1.0]], dtype=mx.float32),
            mx.array([[float(index) * 0.4]], dtype=mx.float32),
        )
        for index in range(1, 9)
    )

    result = module.accumulate_gradients(model, batches, _tiny_loss)
    mx.eval(result.mean_loss, result.gradients)

    individual_losses = []
    individual_gradients = []
    for batch in batches:
        loss, gradients = nn.value_and_grad(model, lambda: _tiny_loss(model, batch))()
        individual_losses.append(float(loss))
        individual_gradients.append(dict(tree_flatten(gradients))["proj.weight"])
    expected_gradient = sum(individual_gradients) / 8.0
    mx.eval(expected_gradient)
    assert result.microbatch_count == 8
    assert math.isclose(float(result.mean_loss), sum(individual_losses) / 8.0, rel_tol=0, abs_tol=1e-6)
    np.testing.assert_allclose(
        np.asarray(dict(tree_flatten(result.gradients))["proj.weight"]),
        np.asarray(expected_gradient),
        rtol=0,
        atol=1e-6,
    )


def test_step_budget_caps_nominal_run_and_reserves_export_time() -> None:
    module = __import__("training.finetune", fromlist=["select_step_budget"])

    assert module.select_step_budget(1.0, nominal_steps=3000, training_seconds=6900) == 3000
    assert module.select_step_budget(3.0, nominal_steps=3000, training_seconds=6900) == 2300
    assert module.select_step_budget(7.0, nominal_steps=3000, training_seconds=6900) == 985
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        try:
            module.select_step_budget(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid median update time was accepted: {invalid}")


def test_flow_draws_are_seeded_finite_and_follow_smolvla_support() -> None:
    module = __import__("training.finetune", fromlist=["sample_flow_draws"])
    mx.random.seed(20260901)
    first = module.sample_flow_draws((1, 50, 32))
    mx.eval(first.noise, first.timesteps)
    mx.random.seed(20260901)
    second = module.sample_flow_draws((1, 50, 32))
    mx.eval(second.noise, second.timesteps)

    assert first.noise.shape == (1, 50, 32)
    assert first.noise.dtype == mx.float32
    assert first.timesteps.shape == (1,)
    assert first.timesteps.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(first.noise)))
    assert bool(mx.all(first.timesteps >= 0.001))
    assert bool(mx.all(first.timesteps <= 1.0))
    assert bool(mx.array_equal(first.noise, second.noise))
    assert bool(mx.array_equal(first.timesteps, second.timesteps))


def test_metrics_csv_has_one_durable_complete_row_per_update(tmp_path: Path) -> None:
    module = __import__("training.finetune", fromlist=["MetricsWriter"])
    path = tmp_path / "metrics.csv"
    writer = module.MetricsWriter(path)
    writer.write(
        step=1,
        loss=1.25,
        smoothed_loss=1.25,
        learning_rate=1e-6,
        gradient_norm=2.0,
        clip_coefficient=1.0,
        elapsed_seconds=0.5,
        updates_per_second=2.0,
        peak_memory_bytes=1234,
    )
    writer.write(
        step=2,
        loss=1.0,
        smoothed_loss=1.245,
        learning_rate=2e-6,
        gradient_norm=3.0,
        clip_coefficient=1.0,
        elapsed_seconds=1.0,
        updates_per_second=2.0,
        peak_memory_bytes=2345,
    )
    writer.close()

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert tuple(rows[0]) == module.METRICS_FIELDS
    assert rows[0]["step"] == "1"
    assert rows[1]["step"] == "2"
    assert rows[1]["peak_memory_bytes"] == "2345"


def test_atomic_run_state_replaces_complete_json(tmp_path: Path) -> None:
    module = __import__("training.finetune", fromlist=["write_run_state"])
    path = tmp_path / "run.json"

    first_hash = module.write_run_state(path, {"status": "running", "step": 0})
    second_hash = module.write_run_state(path, {"status": "complete", "step": 7})

    assert first_hash != second_hash
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "complete", "step": 7}
    assert len(second_hash) == 64
