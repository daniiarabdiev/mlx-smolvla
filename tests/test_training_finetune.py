"""Gradient accumulation, budget, metrics, and run-state contracts for T3."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 1, bias=False)
        self.proj.weight = mx.array([[0.25, -0.5]], dtype=mx.float32)


class WiderTinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 1, bias=False)
        self.proj.weight = mx.array([[0.25, -0.5, 0.75]], dtype=mx.float32)


def _tiny_loss(model: TinyRegressor, batch: tuple[mx.array, mx.array]) -> mx.array:
    inputs, targets = batch
    error = model.proj(inputs) - targets
    return mx.mean(error * error)


def _tiny_checkpoint_state(module, optimizer, *, step: int, selected_steps: int = 10):
    return module.CheckpointState(
        completed_step=step,
        selected_steps=selected_steps,
        smoothed_loss=1.0 / step,
        elapsed_training_seconds=float(step),
        peak_memory_bytes=1234,
        samples_consumed=step * 8,
        flow_draw_count=step * 8,
        last_update=module.UpdateResult(
            loss=1.0 / step,
            learning_rate=float(optimizer.state["learning_rate"]),
            gradient_norm=1.0,
            clip_coefficient=1.0,
            seconds=1.0,
        ),
        run_config_sha256="a" * 64,
    )


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


def test_metrics_resume_archives_uncheckpointed_tail_before_appending(tmp_path: Path) -> None:
    module = __import__("training.finetune", fromlist=["MetricsWriter"])
    path = tmp_path / "metrics.csv"
    with module.MetricsWriter(path) as writer:
        for step in range(1, 4):
            writer.write(
                step=step,
                loss=float(step),
                smoothed_loss=float(step),
                learning_rate=1e-4,
                gradient_norm=2.0,
                clip_coefficient=1.0,
                elapsed_seconds=float(step),
                updates_per_second=1.0,
                peak_memory_bytes=1234,
            )

    with module.MetricsWriter(path, resume_from_step=2) as writer:
        assert writer.recovery_path == tmp_path / "metrics.recovery-000001.csv"
        writer.write(
            step=3,
            loss=30.0,
            smoothed_loss=12.0,
            learning_rate=1e-4,
            gradient_norm=2.0,
            clip_coefficient=1.0,
            elapsed_seconds=4.0,
            updates_per_second=0.75,
            peak_memory_bytes=2345,
        )

    with path.open(newline="", encoding="utf-8") as handle:
        active_rows = list(csv.DictReader(handle))
    with writer.recovery_path.open(newline="", encoding="utf-8") as handle:
        recovered_rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in active_rows] == [1, 2, 3]
    assert [float(row["loss"]) for row in active_rows] == [1.0, 2.0, 30.0]
    assert [int(row["step"]) for row in recovered_rows] == [1, 2, 3]
    assert [float(row["loss"]) for row in recovered_rows] == [1.0, 2.0, 3.0]


def test_metrics_resume_preserves_and_discards_a_torn_trailing_row(tmp_path: Path) -> None:
    module = __import__("training.finetune", fromlist=["MetricsWriter"])
    path = tmp_path / "metrics.csv"
    with module.MetricsWriter(path) as writer:
        for step in (1, 2):
            writer.write(
                step=step,
                loss=float(step),
                smoothed_loss=float(step),
                learning_rate=1e-4,
                gradient_norm=2.0,
                clip_coefficient=1.0,
                elapsed_seconds=float(step),
                updates_per_second=1.0,
                peak_memory_bytes=1234,
            )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("broken")
        handle.flush()

    with module.MetricsWriter(path, resume_from_step=2) as writer:
        recovery_path = writer.recovery_path
        writer.write(
            step=3,
            loss=3.0,
            smoothed_loss=3.0,
            learning_rate=1e-4,
            gradient_norm=2.0,
            clip_coefficient=1.0,
            elapsed_seconds=3.0,
            updates_per_second=1.0,
            peak_memory_bytes=1234,
        )

    assert recovery_path is not None
    assert recovery_path.read_text(encoding="utf-8").endswith("broken")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [1, 2, 3]


def test_metrics_resume_rejects_a_checkpoint_boundary_mismatch(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["CheckpointState", "MetricsWriter", "UpdateResult"],
    )
    path = tmp_path / "metrics.csv"
    with module.MetricsWriter(path) as writer:
        writer.write(
            step=1,
            loss=1.0,
            smoothed_loss=1.0,
            learning_rate=1e-4,
            gradient_norm=2.0,
            clip_coefficient=1.0,
            elapsed_seconds=1.0,
            updates_per_second=1.0,
            peak_memory_bytes=1234,
        )
    checkpoint_state = module.CheckpointState(
        completed_step=1,
        selected_steps=2,
        smoothed_loss=1.0,
        elapsed_training_seconds=1.0,
        peak_memory_bytes=1234,
        samples_consumed=8,
        flow_draw_count=8,
        last_update=module.UpdateResult(
            loss=9.0,
            learning_rate=1e-4,
            gradient_norm=2.0,
            clip_coefficient=1.0,
            seconds=1.0,
        ),
        run_config_sha256="a" * 64,
    )

    try:
        module.MetricsWriter(
            path,
            resume_from_step=1,
            checkpoint_state=checkpoint_state,
        )
    except ValueError as error:
        assert "boundary" in str(error)
    else:
        raise AssertionError("metrics/checkpoint boundary mismatch was accepted")


def test_metrics_resume_rejects_incorrect_boundary_throughput(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["CheckpointState", "MetricsWriter", "UpdateResult"],
    )
    path = tmp_path / "metrics.csv"
    with module.MetricsWriter(path) as writer:
        writer.write(
            step=1,
            loss=1.0,
            smoothed_loss=1.0,
            learning_rate=1e-4,
            gradient_norm=2.0,
            clip_coefficient=1.0,
            elapsed_seconds=2.0,
            updates_per_second=999.0,
            peak_memory_bytes=1234,
        )
    checkpoint_state = module.CheckpointState(
        completed_step=1,
        selected_steps=2,
        smoothed_loss=1.0,
        elapsed_training_seconds=2.0,
        peak_memory_bytes=1234,
        samples_consumed=8,
        flow_draw_count=8,
        last_update=module.UpdateResult(
            loss=1.0,
            learning_rate=1e-4,
            gradient_norm=2.0,
            clip_coefficient=1.0,
            seconds=1.0,
        ),
        run_config_sha256="a" * 64,
    )

    try:
        module.MetricsWriter(
            path,
            resume_from_step=1,
            checkpoint_state=checkpoint_state,
        )
    except ValueError as error:
        assert "boundary" in str(error)
    else:
        raise AssertionError("incorrect checkpoint-boundary throughput was accepted")


def test_flow_rng_fast_forward_reproduces_the_next_draw_exactly() -> None:
    module = __import__("training.finetune", fromlist=["advance_flow_random_state"])
    shape = (1, 4, 3)
    mx.random.seed(20260901)
    for _ in range(5):
        module.sample_flow_draws(shape)
    expected = module.sample_flow_draws(shape)
    mx.eval(expected.noise, expected.timesteps)

    mx.random.seed(20260901)
    module.advance_flow_random_state(draw_count=5, shape=shape)
    actual = module.sample_flow_draws(shape)
    mx.eval(actual.noise, actual.timesteps)

    np.testing.assert_array_equal(np.asarray(actual.noise), np.asarray(expected.noise))
    np.testing.assert_array_equal(np.asarray(actual.timesteps), np.asarray(expected.timesteps))


def test_atomic_checkpoint_restores_model_and_optimizer_for_exact_continuation(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "CheckpointState",
            "UpdateResult",
            "load_latest_training_checkpoint",
            "save_training_checkpoint",
        ],
    )
    optimizer_module = __import__(
        "training.optimizer",
        fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"],
    )
    config = optimizer_module.SmolVLAOptimizerConfig(training_horizon=6)
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(config)
    gradients = (
        mx.array([[0.1, -0.2]], dtype=mx.float32),
        mx.array([[0.3, 0.4]], dtype=mx.float32),
        mx.array([[-0.5, 0.6]], dtype=mx.float32),
        mx.array([[0.7, -0.8]], dtype=mx.float32),
    )
    for gradient in gradients[:3]:
        optimizer.update(model, {"proj": {"weight": gradient}})
        mx.eval(model.parameters(), optimizer.state)

    state = module.CheckpointState(
        completed_step=3,
        selected_steps=6,
        smoothed_loss=0.25,
        elapsed_training_seconds=4.0,
        peak_memory_bytes=1234,
        samples_consumed=24,
        flow_draw_count=24,
        last_update=module.UpdateResult(
            loss=0.2,
            learning_rate=optimizer_module.cosine_decay_with_warmup_lr(2, config),
            gradient_norm=1.0,
            clip_coefficient=1.0,
            seconds=1.0,
        ),
        run_config_sha256="a" * 64,
    )
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    saved = module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=tmp_path / "checkpoints",
        state=state,
        trainable_names=names,
    )
    assert saved.path.name == "step-000003"
    assert (tmp_path / "checkpoints" / "latest.json").is_file()

    optimizer.update(model, {"proj": {"weight": gradients[3]}})
    mx.eval(model.parameters(), optimizer.state)
    uninterrupted_weight = np.asarray(model.proj.weight).copy()
    uninterrupted_state = {
        name: np.asarray(value).copy() for name, value in tree_flatten(optimizer.state)
    }

    resumed_model = TinyRegressor()
    resumed_optimizer = optimizer_module.SmolVLAAdamW(config)
    loaded = module.load_latest_training_checkpoint(
        model=resumed_model,
        optimizer=resumed_optimizer,
        checkpoint_root=tmp_path / "checkpoints",
        trainable_names=names,
        expected_run_config_sha256="a" * 64,
    )
    assert loaded.state == state
    resumed_optimizer.update(resumed_model, {"proj": {"weight": gradients[3]}})
    mx.eval(resumed_model.parameters(), resumed_optimizer.state)

    np.testing.assert_array_equal(np.asarray(resumed_model.proj.weight), uninterrupted_weight)
    resumed_state = dict(tree_flatten(resumed_optimizer.state))
    assert tuple(resumed_state) == tuple(uninterrupted_state)
    for name, expected in uninterrupted_state.items():
        np.testing.assert_array_equal(np.asarray(resumed_state[name]), expected, err_msg=name)


def test_checkpoint_save_rejects_optimizer_state_for_a_different_model_schema(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint"],
    )
    optimizer_module = __import__(
        "training.optimizer",
        fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"],
    )
    model = TinyRegressor()
    wider_model = WiderTinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    optimizer.update(
        wider_model,
        {"proj": {"weight": mx.array([[0.1, -0.2, 0.3]], dtype=mx.float32)}},
    )
    mx.eval(wider_model.parameters(), optimizer.state)
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))

    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=tmp_path / "checkpoints",
            state=_tiny_checkpoint_state(module, optimizer, step=1),
            trainable_names=names,
        )
    except ValueError as error:
        assert "optimizer state" in str(error)
        assert "shape/dtype" in str(error)
    else:
        raise AssertionError("optimizer state for a different model schema was checkpointed")


def test_checkpoint_cadence_is_enabled_and_validated() -> None:
    module = __import__("training.finetune", fromlist=["FineTuneConfig"])

    assert module.FineTuneConfig().checkpoint_interval == 100
    try:
        module.FineTuneConfig(checkpoint_interval=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero checkpoint interval was accepted")


def test_run_config_digest_locks_trajectory_but_not_the_resume_switch() -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "training_run_config_sha256"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAOptimizerConfig"]
    )
    arguments = {
        "selected_steps": 3000,
        "train_statistics_sha256": "b" * 64,
        "train_episodes": (0, 1),
        "holdout_episodes": (2,),
        "base_artifact": {
            "model_sha256": "c" * 64,
            "name_map_sha256": "d" * 64,
        },
        "optimizer_config": optimizer_module.SmolVLAOptimizerConfig(
            training_horizon=3000
        ),
    }

    fresh = module.training_run_config_sha256(module.FineTuneConfig(), **arguments)
    resumed = module.training_run_config_sha256(
        module.FineTuneConfig(resume=True), **arguments
    )
    changed_rank = module.training_run_config_sha256(
        module.FineTuneConfig(rank=4), **arguments
    )
    changed_cadence = module.training_run_config_sha256(
        module.FineTuneConfig(checkpoint_interval=50), **arguments
    )
    changed_base = module.training_run_config_sha256(
        module.FineTuneConfig(),
        **{
            **arguments,
            "base_artifact": {
                **arguments["base_artifact"],
                "model_sha256": "e" * 64,
            },
        },
    )
    changed_optimizer = module.training_run_config_sha256(
        module.FineTuneConfig(),
        **{
            **arguments,
            "optimizer_config": optimizer_module.SmolVLAOptimizerConfig(
                lr=2e-4, training_horizon=3000
            ),
        },
    )

    assert len(fresh) == 64
    assert fresh == resumed
    assert fresh != changed_rank
    assert fresh != changed_cadence
    assert fresh != changed_base
    assert fresh != changed_optimizer


def test_checkpoint_retention_keeps_only_three_valid_newest_steps(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune", fromlist=["save_training_checkpoint"]
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "step-999999").mkdir()
    (root / ".step-000006.partial").mkdir()
    (root / "notes").mkdir()
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    removed = []
    for step in range(1, 6):
        optimizer.update(
            model,
            {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
        )
        mx.eval(model.parameters(), optimizer.state)
        saved = module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=step),
            trainable_names=names,
            keep_last=3,
        )
        removed.extend(saved.pruned_checkpoints)

    assert removed == ["step-000001", "step-000002"]
    assert sorted(path.name for path in root.iterdir()) == [
        ".step-000006.partial",
        "latest.json",
        "notes",
        "step-000003",
        "step-000004",
        "step-000005",
        "step-999999",
    ]


def test_checkpoint_retention_never_counts_a_valid_different_run(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["prune_training_checkpoints", "save_training_checkpoint"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    root = tmp_path / "checkpoints"
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    for step in range(1, 10):
        optimizer.update(
            model,
            {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
        )
        mx.eval(model.parameters(), optimizer.state)
        if step in {1, 2, 3, 9}:
            state = _tiny_checkpoint_state(module, optimizer, step=step)
            if step == 9:
                state = module.CheckpointState(
                    **{
                        **state.__dict__,
                        "run_config_sha256": "b" * 64,
                    }
                )
            module.save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_root=root,
                state=state,
                trainable_names=names,
                keep_last=10,
            )

    removed = module.prune_training_checkpoints(
        root,
        keep_last=2,
        expected_run_config_sha256="a" * 64,
        trainable_names=names,
        expected_model_tensors=dict(tree_flatten(model.trainable_parameters())),
        expected_optimizer_tensors=dict(tree_flatten(optimizer.state)),
    )

    assert removed == ("step-000001",)
    assert sorted(path.name for path in root.iterdir() if path.name.startswith("step-")) == [
        "step-000002",
        "step-000003",
        "step-000009",
    ]


def test_checkpoint_discovery_repairs_a_stale_latest_pointer(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "load_latest_training_checkpoint",
            "save_training_checkpoint",
            "write_run_state",
        ],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    root = tmp_path / "checkpoints"
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    saved = []
    for step in (1, 2):
        optimizer.update(
            model,
            {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
        )
        mx.eval(model.parameters(), optimizer.state)
        saved.append(
            module.save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_root=root,
                state=_tiny_checkpoint_state(module, optimizer, step=step),
                trainable_names=names,
                keep_last=3,
            )
        )
    module.write_run_state(
        root / "latest.json",
        {
            "format_version": 1,
            "checkpoint": saved[0].path.name,
            "completed_step": 1,
            "metadata_sha256": saved[0].metadata_sha256,
        },
    )

    resumed_model = TinyRegressor()
    resumed_optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    loaded = module.load_latest_training_checkpoint(
        model=resumed_model,
        optimizer=resumed_optimizer,
        checkpoint_root=root,
        trainable_names=names,
        expected_run_config_sha256="a" * 64,
    )

    assert loaded.state.completed_step == 2
    assert json.loads((root / "latest.json").read_text(encoding="utf-8"))[
        "completed_step"
    ] == 2

    (root / "latest.json").unlink()
    repaired_model = TinyRegressor()
    repaired_optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    repaired = module.load_latest_training_checkpoint(
        model=repaired_model,
        optimizer=repaired_optimizer,
        checkpoint_root=root,
        trainable_names=names,
        expected_run_config_sha256="a" * 64,
    )
    assert repaired.state.completed_step == 2
    assert (root / "latest.json").is_file()


def test_atomic_run_state_replaces_complete_json(tmp_path: Path) -> None:
    module = __import__("training.finetune", fromlist=["write_run_state"])
    path = tmp_path / "run.json"

    first_hash = module.write_run_state(path, {"status": "running", "step": 0})
    second_hash = module.write_run_state(path, {"status": "complete", "step": 7})

    assert first_hash != second_hash
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "complete", "step": 7}
    assert len(second_hash) == 64


def test_adapter_checkpoint_uses_a_real_safetensors_temporary_suffix(
    tmp_path: Path,
) -> None:
    module = __import__("training.finetune", fromlist=["_save_adapter_checkpoint"])
    model = TinyRegressor()
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    report = SimpleNamespace(
        rank=1,
        alpha=1.0,
        dropout=0.0,
        adapter_count=1,
        trainable_names=names,
        trainable_scalar_count=2,
    )
    path = tmp_path / "adapter.safetensors"

    digest = module._save_adapter_checkpoint(model, path, lora_report=report)

    assert path.is_file()
    assert set(mx.load(str(path))) == {"proj.weight"}
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))[
        "sha256"
    ] == digest
    assert list(tmp_path.glob(".adapter*")) == []
