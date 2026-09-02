"""Gradient accumulation, budget, metrics, and run-state contracts for T3."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import py_compile
import shutil
import stat
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _validated_t3b_runtime_provenance(monkeypatch):
    """Unit tests exercise T3B internals without bypassing the public CLI gate."""

    module = __import__("training.finetune", fromlist=["_require_t3b_runtime_provenance"])
    evidence = {
        "format_version": 1,
        "frozen": True,
        "native_dependency_scope": (
            "direct-extension-origin-bound; "
            "transitive-dyld-images-inventory-hashed-only"
        ),
        "modules": {
            "training.finetune": [
                {
                    "origin": str(Path(module.__file__).resolve()),
                    "kind": "source",
                    "file_sha256": "a" * 64,
                    "code_sha256": "b" * 64,
                }
            ]
        },
    }
    monkeypatch.setattr(
        module,
        "_require_t3b_runtime_provenance",
        lambda **_kwargs: evidence,
    )


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


def _t3b_bridge_evidence(*, dataset_digest: str = "8" * 64) -> dict[str, object]:
    components = {
        name: (dataset_digest if name == "dataset" else "9" * 64)
        for name in (
            "bridge",
            "config",
            "metadata",
            "dataset",
            "sampler",
            "preprocessor",
            "loader",
        )
    }
    return {
        "format_version": 1,
        "sha256": hashlib.sha256(
            json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "components": components,
    }


def _t3b_frozen_inputs(
    *,
    model_sha256: str = "b" * 64,
    name_map_sha256: str = "c" * 64,
) -> dict[str, object]:
    contract = __import__(
        "training.t3_contract",
        fromlist=[
            "FROZEN_BASE_REPORT_SHA256",
            "FROZEN_CHECKPOINT_REVISION_TREE_SHA256",
            "FROZEN_DATASET_REVISION_TREE_SHA256",
            "FROZEN_EVALUATION_MANIFEST_SHA256",
            "FROZEN_EVALUATION_METADATA_SHA256",
            "FROZEN_TRAIN_STATISTICS_SHA256",
            "FROZEN_TOKENIZER_REVISION_TREE_SHA256",
        ],
    )
    discovery = __import__(
        "reference.discovery", fromlist=["DATASET_REVISION"]
    )

    def tree(files: dict[str, str]) -> dict[str, object]:
        canonical = {"files": dict(sorted(files.items())), "links": {}}
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {"tree_sha256": digest, **canonical}

    checkpoint_names = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    source_model_sha256 = "1" * 64
    checkpoint_files = {
        name: source_model_sha256 if name == "model.safetensors" else "2" * 64
        for name in checkpoint_names
    }
    tokenizer_names = {
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    evaluation_files = {
        f"cases/{ordinal:03d}/{name}.npy": "3" * 64
        for ordinal in range(56)
        for name in ("camera1", "camera2", "noise", "state", "target_action")
    }
    evaluation_files.update(
        {
            "manifest.json": contract.FROZEN_EVALUATION_MANIFEST_SHA256,
            "metadata.json": contract.FROZEN_EVALUATION_METADATA_SHA256,
        }
    )
    dataset_files = {
        "data/chunk-000/file-000.parquet": "4" * 64,
        "meta/episodes/chunk-000/file-000.parquet": "a" * 64,
        "meta/info.json": "5" * 64,
        "meta/stats.json": "6" * 64,
        "meta/tasks.parquet": "7" * 64,
        "videos/observation.images.side/chunk-000/file-000.mp4": "b" * 64,
        "videos/observation.images.up/chunk-000/file-000.mp4": "c" * 64,
        f"revision/{discovery.DATASET_REVISION}.json": (
            contract.FROZEN_DATASET_REVISION_TREE_SHA256
        ),
    }
    return {
        "format_version": 1,
        "revision_trees": {
            "checkpoint_sha256": contract.FROZEN_CHECKPOINT_REVISION_TREE_SHA256,
            "dataset_sha256": contract.FROZEN_DATASET_REVISION_TREE_SHA256,
            "tokenizer_sha256": contract.FROZEN_TOKENIZER_REVISION_TREE_SHA256,
        },
        "train_statistics_sha256": contract.FROZEN_TRAIN_STATISTICS_SHA256,
        "processor_statistics_sha256": "8" * 64,
        "source_checkpoint": tree(checkpoint_files),
        "native_checkpoint": tree(checkpoint_files),
        "native_conversion": {
            "model_file": "model.bfloat16.safetensors",
            "model_sha256": model_sha256,
            "name_map_file": "name_map.json",
            "name_map_sha256": name_map_sha256,
            "source_model_sha256": source_model_sha256,
            "tensor_count": 500,
            "parameter_count": 450_046_176,
            "dtype": "bfloat16",
        },
        "pinned_dataset": tree(dataset_files),
        "tokenizer_snapshot": tree(
            {name: "9" * 64 for name in tokenizer_names}
        ),
        "native_tokenizer_snapshot": tree(
            {
                name: "9" * 64
                for name in ("tokenizer.json", "tokenizer_config.json")
            }
        ),
        "evaluation_artifact": tree(evaluation_files),
        "base_report": {
            "file": "t3-base-evaluation.json",
            "sha256": contract.FROZEN_BASE_REPORT_SHA256,
        },
    }


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


def _install_minimal_fresh_run_fakes(module, config, monkeypatch, inject) -> list[object]:
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(
            training_horizon=config.nominal_steps
        )
    )
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    report = SimpleNamespace(
        scope=config.lora_scope,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        adapter_count=1,
        trainable_names=names,
        trainable_tensor_count=len(names),
        trainable_scalar_count=sum(value.size for _, value in tree_flatten(model.parameters())),
    )
    split = SimpleNamespace(
        train_episodes=(0, 1),
        holdout_episodes=(2,),
        holdout_fraction=1 / 3,
    )
    stats = SimpleNamespace(sha256="a" * 64, processor_stats={})
    bridge = SimpleNamespace(state_dict=lambda: {"samples_consumed": 0, "num_samples": 2})

    def build(_config, *, training_horizon):
        assert training_horizon == config.nominal_steps
        inject()
        return split, stats, model, report, bridge, optimizer

    monkeypatch.setattr(module, "_MINIMUM_FREE_BYTES", 0)
    monkeypatch.setattr(module, "_build_training_components", build)
    monkeypatch.setattr(
        module,
        "training_base_artifact_identity",
        lambda _model: {
            "model_file": "model.safetensors",
            "model_sha256": "b" * 64,
            "name_map_file": "name_map.json",
            "name_map_sha256": "c" * 64,
        },
    )
    updates: list[object] = []
    monkeypatch.setattr(
        module,
        "_optimizer_update",
        lambda **_kwargs: updates.append(object()),
    )
    return updates


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


def test_fixed_step_budget_commits_all_3000_updates_without_timings(
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "fixed_step_budget"],
    )
    config = module.FineTuneConfig(
        budget_mode=module.FIXED_BUDGET_MODE,
        lora_scope="expert_only",
    )

    budget = module.fixed_step_budget(config)

    assert budget == {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-fixed-step-budget",
        "mode": "fixed_steps",
        "timing_measurements": False,
        "selected_steps": 3000,
        "nominal_steps": 3000,
        "effective_batch_size": 8,
    }
    assert module.validate_fixed_step_budget(budget, config=config) == budget
    monkeypatch.setattr(
        module,
        "benchmark_lora_updates",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("fixed budget attempted a timing benchmark")
        ),
    )
    resolved = module.resolve_training_budget(config)
    assert resolved.selected_steps == 3000
    assert resolved.artifact_name == "budget.json"
    assert resolved.artifact == budget
    assert resolved.benchmark is None
    assert module.resolve_training_budget(config, persisted=budget) == resolved
    changed = {**budget, "selected_steps": 2999}
    try:
        module.validate_fixed_step_budget(changed, config=config)
    except ValueError as error:
        assert "fixed step budget" in str(error)
    else:
        raise AssertionError("a reduced fixed step budget was accepted")


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


def test_metrics_writer_rejects_named_child_replacement(tmp_path: Path) -> None:
    module = __import__("training.finetune", fromlist=["MetricsWriter"])
    path = tmp_path / "metrics.csv"
    original = tmp_path / "original-metrics.csv"
    writer = module.MetricsWriter(path)
    path.rename(original)
    path.write_text("competitor\n", encoding="utf-8")

    try:
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
    except RuntimeError as error:
        assert "metrics file changed" in str(error)
    else:
        raise AssertionError("replaced named metrics file was accepted")
    assert path.read_text(encoding="utf-8") == "competitor\n"
    # Close only the underlying handle: the public close must continue to reject.
    writer._handle.close()
    writer.close()


def test_metrics_resume_rejects_recovery_child_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__("training.finetune", fromlist=["MetricsWriter"])
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
    real_read = module.MetricsWriter._read_checkpoint_prefix
    swapped = False

    def replace_recovery_before_read(self, source, resume_from_step, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            source.rename(source.with_name("original-recovery.csv"))
            source.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return real_read(self, source, resume_from_step, **kwargs)

    monkeypatch.setattr(
        module.MetricsWriter,
        "_read_checkpoint_prefix",
        replace_recovery_before_read,
    )
    try:
        module.MetricsWriter(path, resume_from_step=1)
    except RuntimeError as error:
        assert "metrics recovery changed" in str(error)
    else:
        raise AssertionError("replaced metrics recovery file was accepted")


def test_metrics_resume_preserves_a_destination_replaced_before_prefix_publish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__("training.finetune", fromlist=["MetricsWriter"])
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
    detached = tmp_path / "metrics-before-race.csv"
    real_replace = module.MetricsWriter._replace_with_prefix
    swapped = False

    def replace_destination_before_prefix(self, rows, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            path.rename(detached)
            path.write_text("competitor\n", encoding="utf-8")
        return real_replace(self, rows, **kwargs)

    monkeypatch.setattr(
        module.MetricsWriter,
        "_replace_with_prefix",
        replace_destination_before_prefix,
    )
    try:
        module.MetricsWriter(path, resume_from_step=1)
    except RuntimeError as error:
        assert "destination changed" in str(error)
    else:
        raise AssertionError("replaced metrics destination was displaced")

    assert path.read_text(encoding="utf-8") == "competitor\n"
    assert detached.read_text(encoding="utf-8").startswith("step,")


def test_no_clobber_publication_quarantines_a_source_replaced_during_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_rename_entry_no_clobber_at"],
    )
    stage = tmp_path / ".state.stage"
    stage.write_text("expected\n", encoding="utf-8")
    expected = stage.stat()
    original = tmp_path / ".state.original"
    target = tmp_path / "state.json"
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_rename = module._renameatx_np
    swapped = False

    def replace_source_during_rename(**kwargs):
        nonlocal swapped
        if not swapped and kwargs["source_name"] == stage.name:
            swapped = True
            stage.rename(original)
            stage.write_text("competitor\n", encoding="utf-8")
        return real_rename(**kwargs)

    monkeypatch.setattr(module, "_renameatx_np", replace_source_during_rename)
    try:
        module._rename_entry_no_clobber_at(
            source_descriptor=descriptor,
            source_name=stage.name,
            destination_descriptor=descriptor,
            destination_name=target.name,
            expected_device=expected.st_dev,
            expected_inode=expected.st_ino,
            expected_directory=False,
        )
    except RuntimeError as error:
        assert "changed during publication" in str(error)
    else:
        raise AssertionError("replacement staging inode became public")

    assert not target.exists()
    failures = tuple(tmp_path.glob(".state.json.publication-failed-*"))
    assert len(failures) == 1
    assert failures[0].read_text(encoding="utf-8") == "competitor\n"
    assert original.read_text(encoding="utf-8") == "expected\n"

    original.rename(stage)
    module._rename_entry_no_clobber_at(
        source_descriptor=descriptor,
        source_name=stage.name,
        destination_descriptor=descriptor,
        destination_name=target.name,
        expected_device=expected.st_dev,
        expected_inode=expected.st_ino,
        expected_directory=False,
    )
    os.close(descriptor)
    assert target.read_text(encoding="utf-8") == "expected\n"


def test_cas_publication_restores_a_destination_replaced_during_rotation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_publish_staged_file_at"],
    )
    stage = tmp_path / ".run.json.stage"
    stage.write_text("new\n", encoding="utf-8")
    staged = stage.stat()
    target = tmp_path / "run.json"
    target.write_text("old\n", encoding="utf-8")
    expected = target.stat()
    displaced_old = tmp_path / "old-run.json"
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_rename = module._renameatx_np
    swapped = False

    def replace_destination_during_rotation(**kwargs):
        nonlocal swapped
        if (
            not swapped
            and kwargs["source_name"] == target.name
            and ".previous-" in kwargs["destination_name"]
        ):
            swapped = True
            target.rename(displaced_old)
            target.write_text("competitor\n", encoding="utf-8")
        return real_rename(**kwargs)

    monkeypatch.setattr(module, "_renameatx_np", replace_destination_during_rotation)
    try:
        module._publish_staged_file_at(
            parent_descriptor=descriptor,
            staged_name=stage.name,
            destination_name=target.name,
            staged_device=staged.st_dev,
            staged_inode=staged.st_ino,
            expected_destination=expected,
        )
    except RuntimeError as error:
        assert "destination changed" in str(error)
    else:
        raise AssertionError("CAS publication displaced a replacement destination")
    finally:
        os.close(descriptor)

    assert target.read_text(encoding="utf-8") == "competitor\n"
    assert stage.read_text(encoding="utf-8") == "new\n"
    assert displaced_old.read_text(encoding="utf-8") == "old\n"
    assert not tuple(tmp_path.glob(".run.json.previous-*"))


def test_cas_publication_rolls_back_a_destination_replaced_after_publish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_publish_staged_file_at"],
    )
    stage = tmp_path / ".run.json.stage"
    stage.write_text("new\n", encoding="utf-8")
    staged = stage.stat()
    target = tmp_path / "run.json"
    target.write_text("old\n", encoding="utf-8")
    expected = target.stat()
    published_new = tmp_path / "detached-new-run.json"
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_publish = module._rename_entry_no_clobber_at
    swapped = False

    def replace_destination_after_publish(**kwargs):
        nonlocal swapped
        result = real_publish(**kwargs)
        if not swapped and kwargs["destination_name"] == target.name:
            swapped = True
            target.rename(published_new)
            target.write_text("competitor\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        module,
        "_rename_entry_no_clobber_at",
        replace_destination_after_publish,
    )
    try:
        module._publish_staged_file_at(
            parent_descriptor=descriptor,
            staged_name=stage.name,
            destination_name=target.name,
            staged_device=staged.st_dev,
            staged_inode=staged.st_ino,
            expected_destination=expected,
        )
    except RuntimeError as error:
        assert "published state file changed" in str(error)
    else:
        raise AssertionError("post-publication replacement destroyed rollback evidence")
    finally:
        os.close(descriptor)

    assert target.read_text(encoding="utf-8") == "old\n"
    assert published_new.read_text(encoding="utf-8") == "new\n"
    failures = tuple(tmp_path.glob(".run.json.publication-failed-*"))
    assert len(failures) == 1
    assert failures[0].read_text(encoding="utf-8") == "competitor\n"
    assert not tuple(tmp_path.glob(".run.json.previous-*"))


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


def test_final_metrics_binding_detects_replacement_after_writer_close(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "MetricsWriter",
            "_acquire_t3b_training_lock",
            "_bind_t3b_metrics_file",
            "_release_t3b_training_lock",
            "_revalidate_t3b_metrics_file",
        ],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    output = tmp_path / "t3b"
    output.mkdir()
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    checkpoint_state = _tiny_checkpoint_state(module, optimizer, step=1)
    with module.MetricsWriter(output / "metrics.csv") as writer:
        writer.write(
            step=1,
            loss=checkpoint_state.last_update.loss,
            smoothed_loss=checkpoint_state.smoothed_loss,
            learning_rate=checkpoint_state.last_update.learning_rate,
            gradient_norm=checkpoint_state.last_update.gradient_norm,
            clip_coefficient=checkpoint_state.last_update.clip_coefficient,
            elapsed_seconds=checkpoint_state.elapsed_training_seconds,
            updates_per_second=(
                checkpoint_state.completed_step
                / checkpoint_state.elapsed_training_seconds
            ),
            peak_memory_bytes=checkpoint_state.peak_memory_bytes,
        )
    lease = module._acquire_t3b_training_lock(output)
    try:
        evidence = module._bind_t3b_metrics_file(
            lease,
            checkpoint_state=checkpoint_state,
        )
        assert evidence["file"] == "metrics.csv"
        assert evidence["row_count"] == 1
        assert len(evidence["sha256"]) == 64
        detached = output / "metrics.original.csv"
        (output / "metrics.csv").rename(detached)
        (output / "metrics.csv").write_bytes(detached.read_bytes())
        try:
            module._revalidate_t3b_metrics_file(lease, verify_bytes=True)
        except RuntimeError as error:
            assert "metrics" in str(error)
        else:
            raise AssertionError("post-close metrics replacement escaped binding")
    finally:
        module._release_t3b_training_lock(lease)


def test_exporting_resume_schema_requires_exact_final_metrics_evidence() -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_validate_t3b_resume_run_document"],
    )
    immutable = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-lora-run",
        "selected_steps": 3000,
    }
    checkpoint_count = 1 + len(tuple(range(100, 3001, 100)))
    document = {
        **immutable,
        "status": "exporting",
        "checkpoint_count": checkpoint_count,
        "resume_count": 0,
        "metrics_recoveries": [],
        "checkpoint_recoveries": [],
        "startup_recoveries": [],
        "disk_free_before_bytes": 1,
        "process": {"identity": "validated separately"},
        "last_completed_step": 3000,
        "last_checkpoint": {
            "step": 3000,
            "path": "/private/run/checkpoints/step-003000",
            "metadata_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "optimizer_sha256": "c" * 64,
        },
        "metrics": {
            "file": "metrics.csv",
            "sha256": "d" * 64,
            "size_bytes": 123,
            "row_count": 3000,
        },
    }

    module._validate_t3b_resume_run_document(
        document,
        expected_immutable=immutable,
        selected_steps=3000,
        checkpoint_interval=100,
    )

    invalid_documents = []
    missing = json.loads(json.dumps(document))
    missing.pop("metrics")
    invalid_documents.append(missing)
    for name, value in (
        ("file", "../metrics.csv"),
        ("sha256", "D" * 64),
        ("size_bytes", 0),
        ("row_count", 2999),
    ):
        changed = json.loads(json.dumps(document))
        changed["metrics"][name] = value
        invalid_documents.append(changed)
    extra = json.loads(json.dumps(document))
    extra["metrics"]["extra"] = True
    invalid_documents.append(extra)

    for candidate in invalid_documents:
        try:
            module._validate_t3b_resume_run_document(
                candidate,
                expected_immutable=immutable,
                selected_steps=3000,
                checkpoint_interval=100,
            )
        except ValueError as error:
            assert "metrics" in str(error)
        else:
            raise AssertionError(
                f"invalid exporting metrics evidence was accepted: {candidate}"
            )


def test_exporting_resume_rejects_self_consistent_replacement_metrics(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "MetricsWriter",
            "_final_metrics_evidence",
            "_snapshot_regular_file",
            "_validate_t3b_resume_metrics_evidence",
        ],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    checkpoint_state = _tiny_checkpoint_state(module, optimizer, step=2)
    path = tmp_path / "metrics.csv"

    def write_metrics(first_loss: float) -> None:
        with module.MetricsWriter(path) as writer:
            writer.write(
                step=1,
                loss=first_loss,
                smoothed_loss=first_loss,
                learning_rate=checkpoint_state.last_update.learning_rate,
                gradient_norm=checkpoint_state.last_update.gradient_norm,
                clip_coefficient=checkpoint_state.last_update.clip_coefficient,
                elapsed_seconds=1.0,
                updates_per_second=1.0,
                peak_memory_bytes=checkpoint_state.peak_memory_bytes,
            )
            writer.write(
                step=2,
                loss=checkpoint_state.last_update.loss,
                smoothed_loss=checkpoint_state.smoothed_loss,
                learning_rate=checkpoint_state.last_update.learning_rate,
                gradient_norm=checkpoint_state.last_update.gradient_norm,
                clip_coefficient=checkpoint_state.last_update.clip_coefficient,
                elapsed_seconds=checkpoint_state.elapsed_training_seconds,
                updates_per_second=(
                    checkpoint_state.completed_step
                    / checkpoint_state.elapsed_training_seconds
                ),
                peak_memory_bytes=checkpoint_state.peak_memory_bytes,
            )

    write_metrics(9.0)
    committed_snapshot = module._snapshot_regular_file(
        path,
        label="committed final metrics",
        capture_payload=True,
    )
    document = {
        "metrics": module._final_metrics_evidence(
            committed_snapshot,
            checkpoint_state=checkpoint_state,
        )
    }
    module._validate_t3b_resume_metrics_evidence(
        document,
        snapshot=committed_snapshot,
        checkpoint_state=checkpoint_state,
    )

    path.unlink()
    write_metrics(8.0)
    replacement_snapshot = module._snapshot_regular_file(
        path,
        label="replacement final metrics",
        capture_payload=True,
    )
    try:
        module._validate_t3b_resume_metrics_evidence(
            document,
            snapshot=replacement_snapshot,
            checkpoint_state=checkpoint_state,
        )
    except ValueError as error:
        assert "committed evidence" in str(error)
    else:
        raise AssertionError("self-consistent replacement final metrics were accepted")


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
    expert_only = module.training_run_config_sha256(
        module.FineTuneConfig(lora_scope="expert_only"),
        **arguments,
    )
    fixed_budget = module.training_run_config_sha256(
        module.FineTuneConfig(budget_mode=module.FIXED_BUDGET_MODE),
        **arguments,
    )
    fixed_resumed = module.training_run_config_sha256(
        module.FineTuneConfig(
            budget_mode=module.FIXED_BUDGET_MODE,
            resume=True,
        ),
        **arguments,
    )

    assert len(fresh) == 64
    assert fresh == resumed
    assert fresh != changed_rank
    assert fresh != changed_cadence
    assert fresh != changed_base
    assert fresh != changed_optimizer
    assert fresh != expert_only
    assert fresh != fixed_budget
    assert fixed_budget == fixed_resumed


def test_finetune_config_rejects_unknown_scope_and_budget_mode() -> None:
    module = __import__("training.finetune", fromlist=["FineTuneConfig"])
    for kwargs in (
        {"lora_scope": "vision"},
        {"budget_mode": "post_hoc"},
    ):
        try:
            module.FineTuneConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"fine-tune config accepted {kwargs}")

    nonfrozen = module.FineTuneConfig(
        nominal_steps=2_999,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    try:
        module._validate_t3b_frozen_config(nonfrozen)
    except ValueError as error:
        assert "frozen plan" in str(error)
    else:
        raise AssertionError("a nonfrozen T3B step count was accepted")


def test_t3b_train_statistics_must_match_the_frozen_population() -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_validate_t3b_train_statistics_sha256"],
    )
    contract = __import__(
        "training.t3_contract", fromlist=["FROZEN_TRAIN_STATISTICS_SHA256"]
    )
    expected = contract.FROZEN_TRAIN_STATISTICS_SHA256

    assert module._validate_t3b_train_statistics_sha256(expected) == expected
    try:
        module._validate_t3b_train_statistics_sha256("a" * 64)
    except ValueError as error:
        assert "frozen training statistics" in str(error)
    else:
        raise AssertionError("arbitrary live training statistics were accepted")


def test_pretraining_launch_config_is_self_hashed_and_no_clobber(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "FineTuneConfig",
            "assemble_finetune_launch_config",
            "write_finetune_launch_config",
        ],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAOptimizerConfig"]
    )
    config = module.FineTuneConfig(
        output_dir=tmp_path / "t3b",
        budget_mode=module.FIXED_BUDGET_MODE,
        lora_scope="expert_only",
    )
    target_names = tuple(f"expert.layers.target_{index}" for index in range(112))
    report = SimpleNamespace(
        scope="expert_only",
        rank=8,
        alpha=16.0,
        dropout=0.0,
        adapter_count=112,
        target_names=target_names,
        trainable_names=tuple(
            f"{name}.{suffix}"
            for name in target_names
            for suffix in ("lora_a", "lora_b")
        ),
        trainable_tensor_count=224,
        trainable_scalar_count=1_708_032,
    )
    frozen_inputs = _t3b_frozen_inputs()
    document = module.assemble_finetune_launch_config(
        config=config,
        budget=module.fixed_step_budget(config),
        train_statistics_sha256=frozen_inputs["train_statistics_sha256"],
        train_episodes=(0, 1),
        holdout_episodes=(2,),
        base_artifact={
            "model_file": "model.bfloat16.safetensors",
            "model_sha256": "b" * 64,
            "name_map_file": "name_map.json",
            "name_map_sha256": "c" * 64,
        },
        optimizer_config=optimizer_module.SmolVLAOptimizerConfig(
            training_horizon=3000
        ),
        lora_report=report,
        reference_freeze_policy={
            "lerobot_version": "0.6.1",
            "freeze_vision_encoder": True,
            "train_expert_only": True,
            "train_state_proj": True,
            "configuration_source_sha256": "d" * 64,
            "implementation_source_sha256": "e" * 64,
        },
        implementation_sha256={"training/finetune.py": "f" * 64},
        frozen_inputs=frozen_inputs,
        training_bridge=_t3b_bridge_evidence(),
        created_at_ns=1_788_264_000_000_000_000,
    )

    validated = module.validate_finetune_launch_config(document)
    assert validated == document
    assert document["budget"]["timing_measurements"] is False
    assert document["training"]["lora_scope"] == "expert_only"
    assert document["lora_topology"]["adapter_count"] == 112
    assert document["lora_topology"]["trainable_scalar_count"] == 1_708_032
    assert document["export_audit"]["run_config_sha256"] == document["run_config_sha256"]
    assert len(document["export_audit"]["evaluation_manifest_sha256"]) == 64
    assert len(document["export_audit"]["evaluation_metadata_sha256"]) == 64
    assert len(document["export_audit"]["base_report_sha256"]) == 64
    assert len(document["run_config_sha256"]) == 64
    assert len(document["configuration_sha256"]) == 64

    path = config.output_dir / "launch.json"
    digest = module.write_finetune_launch_config(path, document)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    read_document, read_digest = module._read_finetune_launch_config(path)
    assert read_document == document
    assert read_digest == digest
    try:
        module.write_finetune_launch_config(path, document)
    except FileExistsError:
        pass
    else:
        raise AssertionError("launch configuration was overwritten")

    bound_parent = tmp_path / "bound-launch"
    bound_parent.mkdir()
    bound_target = bound_parent / "launch.json"
    detached_parent = tmp_path / "detached-bound-launch"
    real_create = module._create_staged_file_at
    swapped_parent = False

    def swap_parent_before_staging(*args, **kwargs):
        nonlocal swapped_parent
        if not swapped_parent:
            swapped_parent = True
            bound_parent.rename(detached_parent)
            bound_parent.mkdir()
        return real_create(*args, **kwargs)

    monkeypatch.setattr(module, "_create_staged_file_at", swap_parent_before_staging)
    try:
        module.write_finetune_launch_config(bound_target, document)
    except RuntimeError as error:
        assert "launch configuration directory" in str(error)
    else:
        raise AssertionError("launch publication accepted a replaced parent path")
    assert tuple(bound_parent.iterdir()) == ()
    assert (detached_parent / "launch.json").is_file()
    monkeypatch.setattr(module, "_create_staged_file_at", real_create)

    staged_parent = tmp_path / "staged-launch"
    staged_parent.mkdir()
    staged_target = staged_parent / "launch.json"
    real_rename = module._rename_entry_no_clobber_at
    replaced_stage: Path | None = None

    def replace_stage_before_publish(**kwargs):
        nonlocal replaced_stage
        if replaced_stage is None and kwargs["destination_name"] == "launch.json":
            stage = staged_parent / kwargs["source_name"]
            replaced_stage = staged_parent / f"{stage.name}.original"
            stage.rename(replaced_stage)
            stage.write_text("competitor\n", encoding="utf-8")
        return real_rename(**kwargs)

    monkeypatch.setattr(
        module,
        "_rename_entry_no_clobber_at",
        replace_stage_before_publish,
    )
    try:
        module.write_finetune_launch_config(staged_target, document)
    except RuntimeError as error:
        assert "staged entry changed" in str(error)
    else:
        raise AssertionError("replaced launch staging inode was published")
    assert replaced_stage is not None and replaced_stage.is_file()
    competitors = [
        entry for entry in staged_parent.iterdir() if entry != replaced_stage
    ]
    assert len(competitors) == 1
    assert competitors[0].read_text(encoding="utf-8") == "competitor\n"
    assert not staged_target.exists()
    monkeypatch.setattr(module, "_rename_entry_no_clobber_at", real_rename)

    changed = json.loads(json.dumps(document))
    changed["training"]["lora_scope"] = "legacy_full"
    try:
        module.validate_finetune_launch_config(changed)
    except ValueError as error:
        assert "configuration digest" in str(error)
    else:
        raise AssertionError("mutated launch configuration was accepted")

    symlink = tmp_path / "launch-link.json"
    symlink.symlink_to(path)
    try:
        module._read_finetune_launch_config(symlink)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("symlinked launch configuration was accepted")

    tampered = tmp_path / "launch-tampered.json"
    changed["configuration_sha256"] = document["configuration_sha256"]
    tampered.write_text(json.dumps(changed), encoding="utf-8")
    try:
        module._read_finetune_launch_config(tampered)
    except ValueError as error:
        assert "configuration digest" in str(error)
    else:
        raise AssertionError("tampered launch configuration was accepted")


def test_runtime_launch_binding_reconstructs_every_frozen_input(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "FineTuneConfig",
            "assemble_finetune_launch_config",
            "validate_finetune_launch_runtime_binding",
        ],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAOptimizerConfig"]
    )
    config = module.FineTuneConfig(
        output_dir=tmp_path / "t3b",
        budget_mode=module.FIXED_BUDGET_MODE,
        lora_scope="expert_only",
    )
    target_names = tuple(f"expert.layers.target_{index}" for index in range(112))
    report = SimpleNamespace(
        scope="expert_only",
        rank=8,
        alpha=16.0,
        dropout=0.0,
        adapter_count=112,
        target_names=target_names,
        trainable_names=tuple(
            f"{name}.{suffix}"
            for name in target_names
            for suffix in ("lora_a", "lora_b")
        ),
        trainable_tensor_count=224,
        trainable_scalar_count=1_708_032,
    )
    base_artifact = {
        "model_file": "model.bfloat16.safetensors",
        "model_sha256": "b" * 64,
        "name_map_file": "name_map.json",
        "name_map_sha256": "c" * 64,
    }
    reference_policy = {
        "lerobot_version": "0.6.1",
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "configuration_source_sha256": "d" * 64,
        "implementation_source_sha256": "e" * 64,
    }
    implementation = {"training/finetune.py": "f" * 64}
    frozen_inputs = _t3b_frozen_inputs()
    optimizer = optimizer_module.SmolVLAOptimizerConfig(training_horizon=3000)
    document = module.assemble_finetune_launch_config(
        config=config,
        budget=module.fixed_step_budget(config),
        train_statistics_sha256=frozen_inputs["train_statistics_sha256"],
        train_episodes=(0, 1),
        holdout_episodes=(2,),
        base_artifact=base_artifact,
        optimizer_config=optimizer,
        lora_report=report,
        reference_freeze_policy=reference_policy,
        implementation_sha256=implementation,
        frozen_inputs=frozen_inputs,
        training_bridge=_t3b_bridge_evidence(),
        created_at_ns=1_788_264_000_000_000_000,
    )

    assert module.validate_finetune_launch_runtime_binding(
        document,
        config=config,
        budget=module.fixed_step_budget(config),
        train_statistics_sha256=frozen_inputs["train_statistics_sha256"],
        train_episodes=(0, 1),
        holdout_episodes=(2,),
        base_artifact=base_artifact,
        optimizer_config=optimizer,
        lora_report=report,
        reference_freeze_policy=reference_policy,
        implementation_sha256=implementation,
        frozen_inputs=frozen_inputs,
        training_bridge=_t3b_bridge_evidence(),
    ) == document

    changed_implementation = {"training/finetune.py": "0" * 64}
    try:
        module.validate_finetune_launch_runtime_binding(
            document,
            config=config,
            budget=module.fixed_step_budget(config),
            train_statistics_sha256=frozen_inputs["train_statistics_sha256"],
            train_episodes=(0, 1),
            holdout_episodes=(2,),
            base_artifact=base_artifact,
            optimizer_config=optimizer,
            lora_report=report,
            reference_freeze_policy=reference_policy,
            implementation_sha256=changed_implementation,
            frozen_inputs=frozen_inputs,
            training_bridge=_t3b_bridge_evidence(),
        )
    except ValueError as error:
        assert "runtime inputs" in str(error)
    else:
        raise AssertionError("changed implementation was accepted at launch")

    try:
        module.validate_finetune_launch_runtime_binding(
            document,
            config=config,
            budget=module.fixed_step_budget(config),
            train_statistics_sha256=frozen_inputs["train_statistics_sha256"],
            train_episodes=(0, 1),
            holdout_episodes=(2,),
            base_artifact=base_artifact,
            optimizer_config=optimizer,
            lora_report=report,
            reference_freeze_policy=reference_policy,
            implementation_sha256=implementation,
            frozen_inputs=frozen_inputs,
            training_bridge=_t3b_bridge_evidence(dataset_digest="0" * 64),
        )
    except ValueError as error:
        assert "runtime inputs" in str(error)
    else:
        raise AssertionError("post-prepare training bridge drift was accepted")


def test_prepared_t3b_output_rejects_uncommitted_entries(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_validate_prepared_t3b_output"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    for name in ("launch.json", "training.log", "training.pid"):
        (output / name).write_text(name, encoding="utf-8")

    module._validate_prepared_t3b_output(output)

    unexpected = output / "notes.txt"
    unexpected.write_text("not frozen", encoding="utf-8")
    try:
        module._validate_prepared_t3b_output(output)
    except FileExistsError as error:
        assert "unexpected" in str(error)
    else:
        raise AssertionError("unexpected fresh-run entry was accepted")
    unexpected.unlink()

    log = output / "training.log"
    log.unlink()
    log.symlink_to(output / "launch.json")
    try:
        module._validate_prepared_t3b_output(output)
    except FileExistsError as error:
        assert "unsafe" in str(error)
    else:
        raise AssertionError("symlinked training log was accepted")


def test_t3b_prestart_reconciles_budget_and_atomic_write_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "run_lora_finetune"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    (output / "launch.json").write_text("{}\n", encoding="utf-8")
    config = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    expected_budget = module.fixed_step_budget(config)
    module.write_run_state(output / "budget.json", expected_budget)
    (output / ".budget.json.crashed").write_text(
        '{"partial":true}\n', encoding="utf-8"
    )
    (output / ".run.json.crashed").write_text(
        '{"status":"running"}\n', encoding="utf-8"
    )
    entered = []
    sentinel = object()

    def enter_training(*_args, **kwargs):
        entered.append(tuple(kwargs["startup_recoveries"]))
        return sentinel

    monkeypatch.setattr(module, "_run_lora_finetune_impl", enter_training)
    result = module.run_lora_finetune(
        config,
        training_log_path=output / "training.log",
    )

    assert result is sentinel
    assert json.loads((output / "budget.json").read_text(encoding="utf-8")) == (
        expected_budget
    )
    assert not tuple(output.glob(".budget.json.*"))
    assert not tuple(output.glob(".run.json.*"))
    recovered = sorted(
        path.name for path in (output / "startup-recoveries").iterdir()
    )
    assert recovered == [
        "budget-json-partial-000001",
        "run-json-partial-000001",
    ]
    assert entered == [
        (
            "startup-recoveries/budget-json-partial-000001",
            "startup-recoveries/run-json-partial-000001",
        )
    ]


def test_t3b_training_lock_allows_only_one_live_owner(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_acquire_t3b_training_lock", "_release_t3b_training_lock"],
    )
    output = tmp_path / "t3b"
    output.mkdir()

    first = module._acquire_t3b_training_lock(output)
    try:
        try:
            module._acquire_t3b_training_lock(output)
        except BlockingIOError as error:
            assert "already owned" in str(error)
        else:
            raise AssertionError("a second training process acquired the same run")
    finally:
        module._release_t3b_training_lock(first)

    second = module._acquire_t3b_training_lock(output)
    module._release_t3b_training_lock(second)


def test_t3b_launcher_opens_the_training_log_exclusively_without_following(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "run_lora_finetune"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    (output / "launch.json").write_text("{}\n", encoding="utf-8")
    log_path = output / "training.log"
    outside = tmp_path / "outside.log"
    outside.write_text("outside\n", encoding="utf-8")
    config = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    calls = []

    def enter_training(*_args, **kwargs):
        calls.append(kwargs["training_log_identity"])
        print("inside training", flush=True)
        return "finished"

    monkeypatch.setattr(module, "_run_lora_finetune_impl", enter_training)

    log_path.symlink_to(outside)
    try:
        module.run_lora_finetune(config, training_log_path=log_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("launcher followed a symlinked training log")
    assert outside.read_text(encoding="utf-8") == "outside\n"
    assert calls == []

    log_path.unlink()
    log_path.write_text("competitor\n", encoding="utf-8")
    try:
        module.run_lora_finetune(config, training_log_path=log_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("launcher appended to a competing fresh-run log")
    assert log_path.read_text(encoding="utf-8") == "competitor\n"
    assert calls == []

    log_path.unlink()
    result = module.run_lora_finetune(config, training_log_path=log_path)

    assert result == "finished"
    assert len(calls) == 1
    assert calls[0]["file"] == "training.log"
    identity = log_path.stat()
    assert calls[0]["device"] == identity.st_dev
    assert calls[0]["inode"] == identity.st_ino
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["artifact_type"] == "smolvla-mlx-training-log"
    assert lines[1:] == ["inside training"]

    # A crash after the bound log is published but before run.json exists is
    # restartable by preserving that inode under startup recoveries and creating
    # a fresh, exclusively published log.
    result = module.run_lora_finetune(config, training_log_path=log_path)
    assert result == "finished"
    assert calls[1] != calls[0]
    assert log_path.read_text(encoding="utf-8").splitlines()[1:] == ["inside training"]
    recovered_logs = tuple(
        (output / "startup-recoveries").glob("training-log-prestart-*")
    )
    assert len(recovered_logs) == 1
    assert recovered_logs[0].read_text(encoding="utf-8").splitlines()[1:] == [
        "inside training"
    ]


def test_t3b_launcher_rejects_live_log_replacement_before_training_body(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "run_lora_finetune"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    (output / "launch.json").write_text("{}\n", encoding="utf-8")
    log_path = output / "training.log"
    detached_log = tmp_path / "detached-training.log"
    config = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    entered = []
    real_reconcile = module._reconcile_t3b_prestart_output

    def reconcile_then_replace(*args, **kwargs):
        result = real_reconcile(*args, **kwargs)
        log_path.rename(detached_log)
        log_path.write_text("competitor\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        module,
        "_reconcile_t3b_prestart_output",
        reconcile_then_replace,
    )
    monkeypatch.setattr(
        module,
        "_run_lora_finetune_impl",
        lambda *_args, **_kwargs: entered.append(True),
    )
    try:
        module.run_lora_finetune(config, training_log_path=log_path)
    except RuntimeError as error:
        assert "training log changed" in str(error)
    else:
        raise AssertionError("launcher accepted a replaced live training log")

    assert entered == []
    assert log_path.read_text(encoding="utf-8") == "competitor\n"
    assert json.loads(detached_log.read_text(encoding="utf-8").splitlines()[0])[
        "artifact_type"
    ] == "smolvla-mlx-training-log"


def test_t3b_resume_log_must_match_the_prior_run_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "run_lora_finetune"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    (output / "launch.json").write_text("{}\n", encoding="utf-8")
    log_path = output / "training.log"
    identities = []

    def enter_training(*_args, **kwargs):
        identities.append(kwargs["training_log_identity"])
        return "finished"

    monkeypatch.setattr(module, "_run_lora_finetune_impl", enter_training)
    fresh = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    module.run_lora_finetune(fresh, training_log_path=log_path)
    module.write_run_state(
        output / "run.json",
        {"process": {"training_log": identities[0]}},
    )
    resume = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
        resume=True,
    )
    module.run_lora_finetune(resume, training_log_path=log_path)
    assert identities[1] == identities[0]

    run_document = json.loads((output / "run.json").read_text(encoding="utf-8"))
    run_document["process"]["training_log"]["inode"] += 1
    module.write_run_state(output / "run.json", run_document)
    try:
        module.run_lora_finetune(resume, training_log_path=log_path)
    except ValueError as error:
        assert "differs from run metadata" in str(error)
    else:
        raise AssertionError("resume appended to a log with a different identity")
    assert len(identities) == 2


def test_t3b_training_lock_rejects_symlink_and_ignores_stale_pid(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_acquire_t3b_training_lock", "_release_t3b_training_lock"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    outside = tmp_path / "outside.lock"
    outside.write_text("outside", encoding="utf-8")
    (output / "training.lock").symlink_to(outside)
    try:
        module._acquire_t3b_training_lock(output)
    except FileExistsError as error:
        assert "unsafe" in str(error)
    else:
        raise AssertionError("symlinked training lock was accepted")

    (output / "training.lock").unlink()
    (output / "training.pid").write_text('{"pid": 1}\n', encoding="utf-8")
    descriptor = module._acquire_t3b_training_lock(output)
    try:
        module.write_run_state(output / "training.pid", {"pid": 999})
    finally:
        module._release_t3b_training_lock(descriptor)
    assert json.loads((output / "training.pid").read_text()) == {"pid": 999}


def test_t3b_training_lock_rejects_named_inode_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_acquire_t3b_training_lock"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    lock_path = output / "training.lock"
    real_flock = module.fcntl.flock
    swapped = False

    def swap_after_lock(descriptor, operation):
        nonlocal swapped
        result = real_flock(descriptor, operation)
        if not swapped and operation & module.fcntl.LOCK_EX:
            swapped = True
            lock_path.unlink()
            lock_path.write_text("replacement\n", encoding="utf-8")
        return result

    monkeypatch.setattr(module.fcntl, "flock", swap_after_lock)
    try:
        module._acquire_t3b_training_lock(output)
    except RuntimeError as error:
        assert "changed while it was acquired" in str(error)
    else:
        raise AssertionError("replaced lock pathname was accepted")


def test_t3b_checkpoint_root_binding_rejects_replacement_without_writes(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_bind_t3b_checkpoint_root",
            "_release_t3b_training_lock",
            "_revalidate_t3b_checkpoint_root",
        ],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    lease = module._acquire_t3b_training_lock(output)
    detached = tmp_path / "detached-checkpoints"
    try:
        module._bind_t3b_checkpoint_root(lease, allow_existing=False)
        (output / "checkpoints").rename(detached)
        replacement = output / "checkpoints"
        replacement.mkdir()
        (replacement / "owner.txt").write_text("competitor\n", encoding="utf-8")
        try:
            module._revalidate_t3b_checkpoint_root(lease)
        except RuntimeError as error:
            assert "changed while bound" in str(error)
        else:
            raise AssertionError("replacement checkpoint root retained run authority")
    finally:
        module._release_t3b_training_lock(lease)

    assert (output / "checkpoints" / "owner.txt").read_text(encoding="utf-8") == (
        "competitor\n"
    )
    assert tuple(detached.iterdir()) == ()


def test_fresh_checkpoint_root_binding_never_accepts_a_late_competitor(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_bind_t3b_checkpoint_root",
            "_release_t3b_training_lock",
        ],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    lease = module._acquire_t3b_training_lock(output)
    competitor = output / "checkpoints"
    competitor.mkdir()
    (competitor / "owner.txt").write_text("competitor\n", encoding="utf-8")
    try:
        try:
            module._bind_t3b_checkpoint_root(lease, allow_existing=False)
        except FileExistsError as error:
            assert "fresh-run binding" in str(error)
        else:
            raise AssertionError("fresh checkpoint binding accepted a competitor")
    finally:
        module._release_t3b_training_lock(lease)

    assert (competitor / "owner.txt").read_text(encoding="utf-8") == "competitor\n"


def test_t3b_training_lock_rejects_output_directory_replacement_before_body(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "run_lora_finetune"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    (output / "launch.json").write_text("{}\n", encoding="utf-8")
    config = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    real_acquire = module._acquire_t3b_training_lock
    detached = tmp_path / "detached-t3b"

    def acquire_then_replace(path):
        lease = real_acquire(path)
        output.rename(detached)
        output.mkdir()
        return lease

    entered = []
    monkeypatch.setattr(module, "_acquire_t3b_training_lock", acquire_then_replace)
    monkeypatch.setattr(
        module,
        "_run_lora_finetune_impl",
        lambda *_args, **_kwargs: entered.append(True),
    )
    try:
        module.run_lora_finetune(config)
    except RuntimeError as error:
        assert "output directory changed" in str(error)
    else:
        raise AssertionError("post-lock output-directory replacement was accepted")
    assert entered == []


def test_t3b_live_owner_blocks_resume_before_training_body(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "run_lora_finetune"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    (output / "run.json").write_text('{"status":"interrupted"}\n', encoding="utf-8")
    config = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
        resume=True,
    )
    calls = []
    monkeypatch.setattr(
        module,
        "_run_lora_finetune_impl",
        lambda *_args, **_kwargs: calls.append(True),
    )
    descriptor = module._acquire_t3b_training_lock(output)
    try:
        try:
            module.run_lora_finetune(config)
        except BlockingIOError as error:
            assert "already owned" in str(error)
        else:
            raise AssertionError("live training owner did not block a resume")
    finally:
        module._release_t3b_training_lock(descriptor)
    assert calls == []


def test_stable_path_guards_reject_file_directory_and_ancestor_swaps(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_private_stable_file",
            "_revalidate_directory_snapshot",
            "_snapshot_directory",
        ],
    )
    source = tmp_path / "source.json"
    source.write_text('{"value":1}\n', encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    try:
        with module._private_stable_file(source, label="test source") as (private, _):
            assert private.path.read_bytes() == source.read_bytes()
            replacement.write_text('{"value":2}\n', encoding="utf-8")
            replacement.replace(source)
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("file replacement during private use was accepted")

    final = tmp_path / "final"
    final.mkdir()
    final_snapshot = module._snapshot_directory(final, label="final")
    moved = tmp_path / "final-real"
    final.rename(moved)
    final.symlink_to(moved, target_is_directory=True)
    try:
        module._revalidate_directory_snapshot(final_snapshot, label="final")
    except FileNotFoundError as error:
        assert "symlink" in str(error) or "unsafe" in str(error)
    else:
        raise AssertionError("final-directory symlink swap was accepted")
    final.unlink()
    moved.rename(final)

    ancestor = tmp_path / "ancestor"
    nested = ancestor / "nested"
    nested.mkdir(parents=True)
    nested_snapshot = module._snapshot_directory(nested, label="nested")
    ancestor_real = tmp_path / "ancestor-real"
    ancestor.rename(ancestor_real)
    ancestor.symlink_to(ancestor_real, target_is_directory=True)
    try:
        module._revalidate_directory_snapshot(nested_snapshot, label="nested")
    except FileNotFoundError as error:
        assert "symlink" in str(error) or "unsafe" in str(error)
    else:
        raise AssertionError("ancestor-directory symlink swap was accepted")
    ancestor.unlink()
    ancestor_real.rename(ancestor)


def test_stable_copy_cleanup_preserves_a_replacement_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_copy_stable_file_no_clobber"],
    )
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured source bytes\n")
    destination = tmp_path / "private.bin"
    original = tmp_path / "private.original.bin"
    real_fsync = module.os.fsync
    swapped = False

    def replace_destination_before_sync(descriptor):
        nonlocal swapped
        if not swapped:
            swapped = True
            destination.rename(original)
            destination.write_bytes(b"competitor bytes\n")
            raise RuntimeError("injected sync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", replace_destination_before_sync)
    try:
        module._copy_stable_file_no_clobber(
            source,
            destination,
            label="test source",
        )
    except RuntimeError as error:
        assert "injected sync failure" in str(error)
    else:
        raise AssertionError("injected stable-copy failure was accepted")

    assert destination.read_bytes() == b"competitor bytes\n"
    assert original.read_bytes() == source.read_bytes()


def test_private_stable_file_cleanup_preserves_a_replacement_file(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_private_stable_file"],
    )
    source = tmp_path / "source.json"
    source.write_text('{"value":1}\n', encoding="utf-8")
    replacement: Path | None = None
    original: Path | None = None

    try:
        with module._private_stable_file(source, label="test source") as (private, _):
            replacement = private.path
            original = replacement.with_name(f"{replacement.name}.original")
            replacement.rename(original)
            replacement.write_text("competitor\n", encoding="utf-8")
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("private-file replacement was accepted")

    assert replacement is not None and original is not None
    assert replacement.read_text(encoding="utf-8") == "competitor\n"
    assert original.read_bytes() == source.read_bytes()


def test_private_stable_tensor_load_uses_retained_inode_and_rejects_replacement(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_private_stable_file"],
    )
    source = tmp_path / "source.safetensors"
    mx.save_safetensors(source, {"value": mx.array([1.0], dtype=mx.float32)})
    replacement: Path | None = None
    original: Path | None = None

    try:
        with module._private_stable_file(source, label="test tensor") as (
            private,
            _,
        ):
            replacement = private.path
            original = replacement.with_name(f"{replacement.name}.original")
            replacement.rename(original)
            mx.save_safetensors(
                replacement,
                {"value": mx.array([9.0], dtype=mx.float32)},
            )
            with private.open_reader() as handle:
                loaded = mx.load(handle, format="safetensors")
                mx.eval(loaded)
            assert loaded["value"].tolist() == [1.0]
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("private tensor replacement was accepted")

    assert replacement is not None and original is not None
    assert mx.load(replacement)["value"].tolist() == [9.0]
    assert mx.load(original, format="safetensors")["value"].tolist() == [1.0]


def test_bound_temporary_directory_cleanup_preserves_a_replacement(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_bound_temporary_directory"],
    )
    replacement: Path | None = None
    original: Path | None = None

    try:
        with module._bound_temporary_directory(
            tmp_path,
            prefix=".validation-",
            label="test validation directory",
        ) as temporary:
            replacement = temporary.path
            original = temporary.with_name(f"{temporary.name}.original")
            replacement.rename(original)
            replacement.mkdir()
            (replacement / "owner.txt").write_text("competitor\n", encoding="utf-8")
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("temporary-directory replacement was accepted")

    assert replacement is not None and original is not None
    assert (replacement / "owner.txt").read_text(encoding="utf-8") == "competitor\n"
    assert original.is_dir()


def test_t3b_tree_snapshot_rejects_escaping_links_and_output_overlap(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_reject_t3b_output_input_overlap", "_snapshot_tree_evidence"],
    )
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = root / "file.bin"
    link.symlink_to(outside)
    try:
        module._snapshot_tree_evidence(
            root=root,
            paths={"file.bin": link},
            label="test tree",
            allowed_symlink_root=root,
        )
    except ValueError as error:
        assert "escapes" in str(error)
    else:
        raise AssertionError("escaping input symlink was accepted")

    input_root = tmp_path / "inputs"
    input_root.mkdir()
    try:
        module._reject_t3b_output_input_overlap(
            input_root / "training",
            {"dataset": input_root},
        )
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("training output nested in an input was accepted")

    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "physical-alias"
    alias.symlink_to(physical, target_is_directory=True)
    try:
        module._reject_t3b_output_input_overlap(
            alias / "training",
            {"dataset": physical},
        )
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("physical overlap hidden by a symlink alias was accepted")


def test_t3b_entry_points_reject_symlinked_output_ancestry_before_work(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "prepare_lora_finetune_launch", "run_lora_finetune"],
    )
    physical_parent = tmp_path / "physical"
    physical_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(physical_parent, target_is_directory=True)
    output = alias_parent / "t3b"
    config = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    calls = []
    monkeypatch.setattr(
        module,
        "_build_training_components",
        lambda *_args, **_kwargs: calls.append("build"),
    )
    try:
        module.prepare_lora_finetune_launch(config)
    except FileNotFoundError as error:
        assert "unsafe" in str(error) or "symlink" in str(error)
    else:
        raise AssertionError("prepare accepted a symlinked output ancestor")
    assert calls == []

    (physical_parent / "t3b").mkdir()
    (physical_parent / "t3b" / "launch.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_run_lora_finetune_impl",
        lambda *_args, **_kwargs: calls.append("run"),
    )
    try:
        module.run_lora_finetune(config)
    except FileNotFoundError as error:
        assert "unsafe" in str(error) or "symlink" in str(error)
    else:
        raise AssertionError("run accepted a symlinked output ancestor")
    assert calls == []


def test_frozen_tree_files_must_match_the_independent_revision_records(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_snapshot_tree_evidence", "_validate_tree_files_against_revision"],
    )
    root = tmp_path / "snapshot"
    root.mkdir()
    lfs_path = root / "large.bin"
    blob_path = root / "small.json"
    lfs_path.write_bytes(b"lfs payload")
    blob_path.write_bytes(b'{"value":1}\n')
    paths = {"large.bin": lfs_path, "small.json": blob_path}
    evidence = module._snapshot_tree_evidence(
        root=root,
        paths=paths,
        label="test revision tree",
    )
    blob_payload = blob_path.read_bytes()
    revision = {
        "format_version": 1,
        "files": {
            "large.bin": {
                "size": lfs_path.stat().st_size,
                "lfs_sha256": hashlib.sha256(lfs_path.read_bytes()).hexdigest(),
            },
            "small.json": {
                "size": len(blob_payload),
                "blob_id": hashlib.sha1(
                    f"blob {len(blob_payload)}\0".encode("ascii") + blob_payload
                ).hexdigest(),
            },
        },
    }
    revision_path = tmp_path / "revision.json"
    revision_path.write_text(json.dumps(revision), encoding="utf-8")
    revision_sha256 = hashlib.sha256(revision_path.read_bytes()).hexdigest()

    assert module._validate_tree_files_against_revision(
        evidence=evidence,
        paths=paths,
        revision_tree_path=revision_path,
        expected_revision_sha256=revision_sha256,
        label="test tree",
    ) == revision_sha256

    lfs_path.write_bytes(b"bad payload")
    changed_evidence = module._snapshot_tree_evidence(
        root=root,
        paths=paths,
        label="test revision tree",
    )
    try:
        module._validate_tree_files_against_revision(
            evidence=changed_evidence,
            paths=paths,
            revision_tree_path=revision_path,
            expected_revision_sha256=revision_sha256,
            label="test tree",
        )
    except ValueError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("corrupted local input was blessed by its revision tree")


def test_t3b_dataset_inventory_rejects_extra_behavior_shards(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_validate_t3b_dataset_inventory"],
    )
    root = tmp_path / "dataset"
    for name in module._T3B_DATASET_FILES:
        if name.startswith("revision/"):
            continue
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    module._validate_t3b_dataset_inventory(root)

    for extra_name in (
        "data/chunk-001/file-001.parquet",
        "meta/episodes/chunk-001/file-001.parquet",
    ):
        extra = root / extra_name
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(b"extra shard")
        try:
            module._validate_t3b_dataset_inventory(root)
        except ValueError as error:
            assert "inventory" in str(error)
            assert extra_name in str(error)
        else:
            raise AssertionError(f"extra behavior shard was accepted: {extra_name}")
        extra.unlink()


def test_t3b_input_capture_requires_identical_before_and_after_views() -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_require_unchanged_t3b_inputs"],
    )
    before = _t3b_frozen_inputs()
    module._require_unchanged_t3b_inputs(before, before, context="test capture")
    after = json.loads(json.dumps(before))
    after["processor_statistics_sha256"] = "0" * 64
    try:
        module._require_unchanged_t3b_inputs(before, after, context="test capture")
    except RuntimeError as error:
        assert "changed during test capture" in str(error)
    else:
        raise AssertionError("different T3B input views were accepted")


def test_private_t3b_export_source_is_a_byte_copy_not_a_hard_link(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "_private_t3b_source_checkpoint"],
    )
    source = tmp_path / "source"
    source.mkdir()
    for ordinal, name in enumerate(sorted(module._T3B_CHECKPOINT_FILES)):
        (source / name).write_bytes(f"source-{ordinal}\n".encode())
    source_evidence = module._snapshot_tree_evidence(
        root=source,
        paths=module._checkpoint_input_paths(source),
        label="test source checkpoint",
    )
    frozen_inputs = _t3b_frozen_inputs()
    frozen_inputs["source_checkpoint"] = source_evidence
    frozen_inputs["native_checkpoint"] = source_evidence
    frozen_inputs["native_conversion"]["source_model_sha256"] = source_evidence[
        "files"
    ]["model.safetensors"]
    output = tmp_path / "run"
    output.mkdir()
    config = module.FineTuneConfig(output_dir=output)
    monkeypatch.setattr(module, "resolve_base_checkpoint", lambda _cache: source)
    original = (source / "config.json").read_bytes()

    try:
        with module._private_t3b_source_checkpoint(
            config=config,
            expected_evidence=frozen_inputs,
        ) as private:
            (source / "config.json").write_bytes(b"changed in place\n")
            assert (private / "config.json").read_bytes() == original
    except RuntimeError as error:
        assert "changed during export" in str(error)
    else:
        raise AssertionError("source mutation during private export was accepted")


def test_private_t3b_source_cleanup_preserves_a_replacement_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "_private_t3b_source_checkpoint"],
    )
    source = tmp_path / "source"
    source.mkdir()
    for ordinal, name in enumerate(sorted(module._T3B_CHECKPOINT_FILES)):
        (source / name).write_bytes(f"source-{ordinal}\n".encode())
    source_evidence = module._snapshot_tree_evidence(
        root=source,
        paths=module._checkpoint_input_paths(source),
        label="test source checkpoint",
    )
    frozen_inputs = _t3b_frozen_inputs()
    frozen_inputs["source_checkpoint"] = source_evidence
    frozen_inputs["native_checkpoint"] = source_evidence
    frozen_inputs["native_conversion"]["source_model_sha256"] = source_evidence[
        "files"
    ]["model.safetensors"]
    output = tmp_path / "run"
    output.mkdir()
    config = module.FineTuneConfig(output_dir=output)
    monkeypatch.setattr(module, "resolve_base_checkpoint", lambda _cache: source)
    replacement: Path | None = None
    original: Path | None = None

    try:
        with module._private_t3b_source_checkpoint(
            config=config,
            expected_evidence=frozen_inputs,
        ) as private:
            replacement = private.path
            original = replacement.with_name(f"{replacement.name}.original")
            replacement.rename(original)
            replacement.mkdir()
            (replacement / "owner.txt").write_text("competitor\n", encoding="utf-8")
    except (FileNotFoundError, RuntimeError):
        pass
    else:
        raise AssertionError("private source replacement was accepted")

    assert replacement is not None and original is not None
    assert (replacement / "owner.txt").read_text(encoding="utf-8") == "competitor\n"
    assert (original / "config.json").is_file()


def test_runtime_model_must_match_the_committed_converted_bytes(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_validate_runtime_model_matches_converted_checkpoint"],
    )
    model = TinyRegressor()
    path = tmp_path / "model.bfloat16.safetensors"
    model.proj.weight = model.proj.weight.astype(mx.bfloat16)
    mx.save_safetensors(
        str(path),
        {"proj.weight": model.proj.weight},
    )

    module._validate_runtime_model_matches_converted_checkpoint(model, path)

    model.proj.weight = model.proj.weight + mx.array(1, dtype=mx.bfloat16)
    try:
        module._validate_runtime_model_matches_converted_checkpoint(model, path)
    except RuntimeError as error:
        assert "live model tensor differs" in str(error)
    else:
        raise AssertionError("runtime model drift from converted bytes was accepted")


def test_conversion_semantic_validator_can_only_mutate_private_byte_copies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_validate_t3b_conversion_from_stable_hardlinks"],
    )
    convert_module = __import__(
        "mlx_smolvla.convert",
        fromlist=["validate_converted_checkpoint"],
    )
    source_root = tmp_path / "source"
    converted_root = tmp_path / "converted"
    source_root.mkdir()
    converted_root.mkdir()
    source = source_root / "model.safetensors"
    converted = converted_root / "model.bfloat16.safetensors"
    name_map = converted_root / "name_map.json"
    source.write_bytes(b"canonical source\n")
    converted.write_bytes(b"canonical converted\n")
    name_map.write_bytes(b'{"canonical":true}\n')
    canonical = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, converted, name_map)
    }

    def mutate_private(source_dir, converted_path, name_map_path, **_kwargs):
        (Path(source_dir) / "model.safetensors").write_bytes(b"mutated private source")
        Path(converted_path).write_bytes(b"mutated private converted")
        Path(name_map_path).write_bytes(b"mutated private map")
        return SimpleNamespace(
            tensor_count=500,
            parameter_count=450_046_176,
            dtype="bfloat16",
            source_model_sha256=canonical[source],
            converted_model_sha256=canonical[converted],
            name_map_sha256=canonical[name_map],
        )

    monkeypatch.setattr(convert_module, "validate_converted_checkpoint", mutate_private)
    report = module._validate_t3b_conversion_from_stable_hardlinks(
        source_model_path=source,
        converted_model_path=converted,
        name_map_path=name_map,
    )

    assert report["source_model_sha256"] == canonical[source]
    for path, digest in canonical.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_runtime_validation_loader_can_only_mutate_a_private_model_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_validate_runtime_model_matches_converted_checkpoint"],
    )
    model = TinyRegressor()
    model.proj.weight = model.proj.weight.astype(mx.bfloat16)
    path = tmp_path / "model.bfloat16.safetensors"
    mx.save_safetensors(str(path), {"proj.weight": model.proj.weight})
    canonical = path.read_bytes()
    real_load = module.mx.load

    def mutate_loaded_path(private_file, *args, **kwargs):
        loaded = real_load(private_file, *args, **kwargs)
        module.mx.eval(loaded)
        module._descriptor_path(private_file.fileno()).write_bytes(
            b"mutated private model"
        )
        return loaded

    monkeypatch.setattr(module.mx, "load", mutate_loaded_path)
    try:
        module._validate_runtime_model_matches_converted_checkpoint(model, path)
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("mutation of the bound private model was accepted")

    assert path.read_bytes() == canonical


def test_t3b_frozen_input_collector_validates_the_real_pinned_population() -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "collect_t3b_frozen_input_evidence"],
    )
    model_module = __import__(
        "training.model",
        fromlist=["SmolVLATrainingModel"],
    )
    lora_module = __import__("training.lora", fromlist=["LoRAConfig", "install_lora"])
    contract = __import__(
        "training.t3_contract",
        fromlist=[
            "FROZEN_BASE_REPORT_SHA256",
            "FROZEN_TRAIN_STATISTICS_SHA256",
        ],
    )
    config = module.FineTuneConfig(
        cache_dir=Path(".cache/hf"),
        native_cache=Path(".cache/mlx_smolvla/policy-float32"),
        output_dir=Path(".cache/training/t3b"),
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    model = model_module.SmolVLATrainingModel.from_pretrained(
        cache_dir=config.native_cache,
        dtype=mx.bfloat16,
    )
    lora_module.install_lora(
        model,
        lora_module.LoRAConfig(scope="expert_only"),
    )

    evidence = module.collect_t3b_frozen_input_evidence(config, model)

    assert evidence["train_statistics_sha256"] == (
        contract.FROZEN_TRAIN_STATISTICS_SHA256
    )
    assert evidence["base_report"]["sha256"] == contract.FROZEN_BASE_REPORT_SHA256
    assert set(evidence["pinned_dataset"]["files"]) == module._T3B_DATASET_FILES
    assert set(evidence["evaluation_artifact"]["files"]) == (
        module._T3B_EVALUATION_FILES
    )
    assert evidence["source_checkpoint"]["files"] == (
        evidence["native_checkpoint"]["files"]
    )
    assert evidence["native_conversion"]["tensor_count"] == 500
    assert evidence["native_conversion"]["parameter_count"] == 450_046_176
    assert evidence["native_conversion"]["dtype"] == "bfloat16"


def test_finetune_cli_prepare_only_does_not_enter_training(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    script = __import__("scripts.finetune_lora", fromlist=["main"])
    calls = []
    output = tmp_path / "t3b"
    launch = output / "launch.json"

    def prepare(config, *, output_path):
        calls.append((config, output_path))
        return (
            {
                "configuration_sha256": "a" * 64,
                "run_config_sha256": "b" * 64,
                "lora_topology": {
                    "adapter_count": 112,
                    "trainable_tensor_count": 224,
                    "trainable_scalar_count": 1_708_032,
                },
            },
            "c" * 64,
        )

    monkeypatch.setattr(script, "prepare_lora_finetune_launch", prepare)
    monkeypatch.setattr(
        script,
        "run_lora_finetune",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepare-only entered training")
        ),
    )

    script.main(
        [
            "--prepare-only",
            "--output",
            str(output),
            "--launch-config",
            str(launch),
            "--lora-scope",
            "expert_only",
            "--budget-mode",
            "fixed_steps",
        ]
    )

    assert len(calls) == 1
    assert calls[0][0].output_dir == output
    assert calls[0][0].lora_scope == "expert_only"
    assert calls[0][0].budget_mode == "fixed_steps"
    assert calls[0][1] == launch
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "launch_config": str(launch.resolve()),
        "launch_file_sha256": "c" * 64,
        "configuration_sha256": "a" * 64,
        "run_config_sha256": "b" * 64,
        "adapter_count": 112,
        "trainable_tensor_count": 224,
        "trainable_scalar_count": 1_708_032,
    }


def test_finetune_cli_requires_and_forwards_the_bound_t3b_log(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    script = __import__("scripts.finetune_lora", fromlist=["main"])
    output = tmp_path / "t3b"
    launch = output / "launch.json"
    log = output / "training.log"
    common = [
        "--output",
        str(output),
        "--launch-config",
        str(launch),
        "--lora-scope",
        "expert_only",
        "--budget-mode",
        "fixed_steps",
    ]
    try:
        script.main(common)
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("fixed T3B CLI launch accepted no bound log")
    assert "--log-file" in capsys.readouterr().err

    calls = []

    def run(config, **kwargs):
        calls.append((config, kwargs))
        return SimpleNamespace(
            selected_steps=3000,
            training_seconds=1.0,
            final_loss=0.5,
            final_smoothed_loss=0.6,
            peak_memory_bytes=123,
            adapter_sha256="a" * 64,
            export_dir=output / "export",
            run_state_sha256="b" * 64,
        )

    monkeypatch.setattr(script, "run_lora_finetune", run)
    script.main([*common, "--log-file", str(log)])

    assert len(calls) == 1
    assert calls[0][1]["launch_config_path"] == launch
    assert calls[0][1]["training_log_path"] == log


def test_core_t3b_api_requires_the_bound_log_for_fresh_and_resume(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "run_lora_finetune"],
    )
    for resume in (False, True):
        output = tmp_path / ("resume" if resume else "fresh")
        output.mkdir()
        config = module.FineTuneConfig(
            output_dir=output,
            lora_scope="expert_only",
            budget_mode=module.FIXED_BUDGET_MODE,
            resume=resume,
        )
        try:
            module.run_lora_finetune(config)
        except ValueError as error:
            assert "training.log" in str(error)
        else:
            raise AssertionError("core T3B API accepted a missing bound log")


def test_prepared_t3b_directory_is_started_fresh_and_only_then_resumable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "run_lora_finetune"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    launch = output / "launch.json"
    launch.write_text("{}\n", encoding="utf-8")
    log = output / "training.log"
    calls = []
    expected = object()

    def execute(config, **kwargs):
        calls.append((config, kwargs))
        return expected

    monkeypatch.setattr(module, "_run_lora_finetune_impl", execute)
    fresh = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
        resume=False,
    )

    result = module.run_lora_finetune(
        fresh,
        launch_config_path=launch,
        training_log_path=log,
    )

    assert result is expected
    assert len(calls) == 1
    assert calls[0][0].resume is False
    assert calls[0][1]["launch_config_path"] == launch
    assert not (output / "run.json").exists()

    resume = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
        resume=True,
    )
    with pytest.raises(FileNotFoundError, match="no resumable metadata"):
        module.run_lora_finetune(
            resume,
            launch_config_path=launch,
            training_log_path=log,
        )
    assert len(calls) == 1


def test_t3b_run_revalidates_launch_immediately_before_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "run_lora_finetune"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAOptimizerConfig"]
    )
    output = tmp_path / "t3b"
    config = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    target_names = tuple(f"expert.layers.target_{index}" for index in range(112))
    report = SimpleNamespace(
        scope="expert_only",
        rank=8,
        alpha=16.0,
        dropout=0.0,
        adapter_count=112,
        target_names=target_names,
        trainable_names=tuple(
            f"{name}.{suffix}"
            for name in target_names
            for suffix in ("lora_a", "lora_b")
        ),
        trainable_tensor_count=224,
        trainable_scalar_count=1_708_032,
    )
    split = SimpleNamespace(
        train_episodes=(0, 1),
        holdout_episodes=(2,),
        holdout_fraction=1 / 3,
    )
    frozen_inputs = _t3b_frozen_inputs()
    stats = SimpleNamespace(
        sha256=frozen_inputs["train_statistics_sha256"],
        processor_stats={"observation.state": {"mean": [0.0]}},
    )
    base_artifact = {
        "model_file": "model.bfloat16.safetensors",
        "model_sha256": "b" * 64,
        "name_map_file": "name_map.json",
        "name_map_sha256": "c" * 64,
    }
    optimizer = SimpleNamespace(
        config=optimizer_module.SmolVLAOptimizerConfig(training_horizon=3000)
    )
    bridge = SimpleNamespace(
        state_dict=lambda: {
            "samples_consumed": config.effective_batch_size,
            "num_samples": len(split.train_episodes),
        }
    )
    bridge_evidence = _t3b_bridge_evidence()
    reference_policy = {
        "lerobot_version": "0.6.1",
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "configuration_source_sha256": "d" * 64,
        "implementation_source_sha256": "e" * 64,
    }
    launch = module.assemble_finetune_launch_config(
        config=config,
        budget=module.fixed_step_budget(config),
        train_statistics_sha256=stats.sha256,
        train_episodes=split.train_episodes,
        holdout_episodes=split.holdout_episodes,
        base_artifact=base_artifact,
        optimizer_config=optimizer.config,
        lora_report=report,
        reference_freeze_policy=reference_policy,
        implementation_sha256={"training/finetune.py": "f" * 64},
        frozen_inputs=frozen_inputs,
        training_bridge=bridge_evidence,
        created_at_ns=1_788_264_000_000_000_000,
    )
    module.write_finetune_launch_config(output / "launch.json", launch)
    monkeypatch.setattr(
        module,
        "_build_training_components",
        lambda _config, training_horizon: (
            split,
            stats,
            object(),
            report,
            bridge,
            optimizer,
        ),
    )
    monkeypatch.setattr(
        module, "training_base_artifact_identity", lambda _model: base_artifact
    )
    monkeypatch.setattr(
        module,
        "_validate_t3b_training_bridge_semantics",
        lambda **_kwargs: bridge_evidence,
    )
    frozen_input_calls = []
    mutate_before_first_update = True

    def collect_frozen_inputs(
        _config,
        _model,
        *,
        runtime_statistics=None,
        validate_runtime_model=True,
    ):
        frozen_input_calls.append(
            (
                _model is None,
                None if runtime_statistics is None else runtime_statistics.sha256,
                validate_runtime_model,
            )
        )
        if not mutate_before_first_update or len(frozen_input_calls) <= 2:
            return frozen_inputs
        changed = json.loads(json.dumps(frozen_inputs))
        changed["processor_statistics_sha256"] = "0" * 64
        return changed

    monkeypatch.setattr(
        module,
        "collect_t3b_frozen_input_evidence",
        collect_frozen_inputs,
    )
    monkeypatch.setattr(
        module, "reference_freeze_policy_evidence", lambda: reference_policy
    )
    implementation_calls = []

    def implementation_hashes():
        implementation_calls.append(len(implementation_calls) + 1)
        return {"training/finetune.py": "f" * 64}

    update_calls = []
    monkeypatch.setattr(module, "finetune_implementation_hashes", implementation_hashes)
    monkeypatch.setattr(
        module,
        "_optimizer_update",
        lambda **_kwargs: update_calls.append(True),
    )

    try:
        module.run_lora_finetune(
            config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except ValueError as error:
        assert "runtime inputs" in str(error)
    else:
        raise AssertionError("frozen input mutation was not rejected before update 1")

    assert implementation_calls == [1, 2]
    assert frozen_input_calls == [
        (True, None, False),
        (False, stats.sha256, True),
        (False, stats.sha256, True),
    ]
    assert update_calls == []
    run_document = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run_document["status"] == "interrupted"
    assert run_document["last_completed_step"] == 0
    assert run_document["launch_configuration"] == {
        "file": "launch.json",
        "file_sha256": hashlib.sha256((output / "launch.json").read_bytes()).hexdigest(),
        "configuration_sha256": launch["configuration_sha256"],
        "run_config_sha256": launch["run_config_sha256"],
    }
    assert run_document["training_bridge"] == bridge_evidence

    mutate_before_first_update = False
    frozen_input_calls.clear()
    update_calls.clear()

    def stop_after_zero_step_resume(**_kwargs):
        update_calls.append(True)
        raise RuntimeError("zero-step resume reached optimizer update")

    monkeypatch.setattr(module, "_optimizer_update", stop_after_zero_step_resume)
    resume_config = module.FineTuneConfig(
        output_dir=output,
        lora_scope="expert_only",
        budget_mode=module.FIXED_BUDGET_MODE,
        resume=True,
    )
    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except RuntimeError as error:
        assert "zero-step resume reached" in str(error)
    else:
        raise AssertionError("zero-step resume did not reach the optimizer loop")
    assert update_calls == [True]
    resumed = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert resumed["status"] == "interrupted"
    assert resumed["resume_count"] == 1
    assert resumed["resumed_from_step"] == 0

    stale_running = dict(resumed)
    stale_running["status"] = "running"
    stale_running.pop("last_completed_step", None)
    module.write_run_state(output / "run.json", stale_running)
    update_calls.clear()

    def stop_after_stale_running_resume(**_kwargs):
        update_calls.append(True)
        raise RuntimeError("stale-running resume reached optimizer update")

    monkeypatch.setattr(module, "_optimizer_update", stop_after_stale_running_resume)
    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except RuntimeError as error:
        assert "stale-running resume reached" in str(error)
    else:
        raise AssertionError("stale running step-0 run was not replayed")
    assert update_calls == [True]
    replayed = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert replayed["status"] == "interrupted"
    assert replayed["resume_count"] == 2
    assert replayed["resumed_from_step"] == 0

    for metrics_path in output.glob("metrics*.csv"):
        metrics_path.unlink()
    stale_without_metrics = dict(replayed)
    stale_without_metrics["status"] = "running"
    stale_without_metrics.pop("last_completed_step", None)
    stale_without_metrics["metrics_recoveries"] = []
    module.write_run_state(output / "run.json", stale_without_metrics)
    update_calls.clear()

    def stop_after_missing_metrics_resume(**_kwargs):
        update_calls.append(True)
        raise RuntimeError("missing-metrics resume reached optimizer update")

    monkeypatch.setattr(module, "_optimizer_update", stop_after_missing_metrics_resume)
    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except RuntimeError as error:
        assert "missing-metrics resume reached" in str(error)
    else:
        raise AssertionError("stale step-0 run without metrics was not replayed")
    assert update_calls == [True]
    assert (output / "metrics.csv").is_file()
    replayed_without_metrics = json.loads(
        (output / "run.json").read_text(encoding="utf-8")
    )
    assert replayed_without_metrics["status"] == "interrupted"
    assert replayed_without_metrics["resume_count"] == 3
    assert replayed_without_metrics["resumed_from_step"] == 0

    for label, payload in (
        ("empty", b""),
        ("torn", b"step,loss\n1,"),
    ):
        stale_metrics = json.loads(
            (output / "run.json").read_text(encoding="utf-8")
        )
        stale_metrics["status"] = "running"
        stale_metrics.pop("last_completed_step", None)
        module.write_run_state(output / "run.json", stale_metrics)
        (output / "metrics.csv").write_bytes(payload)
        update_calls.clear()

        def stop_after_stale_metrics_resume(_label=label, **_kwargs):
            update_calls.append(True)
            raise RuntimeError(f"{_label}-metrics resume reached optimizer update")

        monkeypatch.setattr(
            module,
            "_optimizer_update",
            stop_after_stale_metrics_resume,
        )
        try:
            module.run_lora_finetune(
                resume_config,
                launch_config_path=output / "launch.json",
                training_log_path=output / "training.log",
            )
        except RuntimeError as error:
            assert f"{label}-metrics resume reached" in str(error)
        else:
            raise AssertionError(f"stale {label} step-0 metrics were not replayed")
        assert update_calls == [True]
        recovery = max(output.glob("metrics.recovery-*.csv"))
        assert recovery.read_bytes() == payload

    stale_before_first_checkpoint = json.loads(
        (output / "run.json").read_text(encoding="utf-8")
    )
    stale_before_first_checkpoint["status"] = "running"
    stale_before_first_checkpoint.pop("last_completed_step", None)
    module.write_run_state(output / "run.json", stale_before_first_checkpoint)
    original_save_training_checkpoint = module.save_training_checkpoint

    monkeypatch.setattr(
        module,
        "_optimizer_update",
        lambda **_kwargs: module.UpdateResult(
            loss=1.0,
            learning_rate=1e-4,
            gradient_norm=1.0,
            clip_coefficient=1.0,
            seconds=0.01,
        ),
    )

    def fail_after_checkpoint_root_creation(**kwargs):
        assert Path(kwargs["checkpoint_root"]).is_dir()
        raise RuntimeError("first checkpoint write failed after root creation")

    monkeypatch.setattr(
        module,
        "save_training_checkpoint",
        fail_after_checkpoint_root_creation,
    )
    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except RuntimeError as error:
        assert "first checkpoint write failed" in str(error)
    else:
        raise AssertionError("first-checkpoint failure was not exercised")
    assert (output / "checkpoints").is_dir()
    assert not tuple((output / "checkpoints").iterdir())
    failed_first_checkpoint = json.loads(
        (output / "run.json").read_text(encoding="utf-8")
    )
    assert failed_first_checkpoint["status"] == "interrupted"
    assert failed_first_checkpoint["last_completed_step"] == 1
    assert failed_first_checkpoint["checkpoint_count"] == 0
    assert "last_checkpoint" not in failed_first_checkpoint

    monkeypatch.setattr(
        module,
        "save_training_checkpoint",
        original_save_training_checkpoint,
    )
    update_calls.clear()

    def stop_after_empty_checkpoint_root_resume(**_kwargs):
        update_calls.append(True)
        raise RuntimeError("empty-checkpoint-root resume reached optimizer update")

    monkeypatch.setattr(
        module,
        "_optimizer_update",
        stop_after_empty_checkpoint_root_resume,
    )
    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except RuntimeError as error:
        assert "empty-checkpoint-root resume reached" in str(error)
    else:
        raise AssertionError("empty first-checkpoint root was not replayed")
    assert update_calls == [True]

    partial = output / "checkpoints" / (".step-000001." + "a" * 24)
    partial.mkdir()
    (partial / "model.safetensors").write_bytes(b"partial checkpoint")
    stale_with_partial_checkpoint = json.loads(
        (output / "run.json").read_text(encoding="utf-8")
    )
    stale_with_partial_checkpoint["status"] = "running"
    stale_with_partial_checkpoint["last_completed_step"] = 1
    module.write_run_state(output / "run.json", stale_with_partial_checkpoint)
    update_calls.clear()

    # Simulate SIGKILL after the staging tree is moved but before run.json records it.
    unrecorded_recoveries = module._prepare_zero_step_checkpoint_replay(
        output / "checkpoints",
        output_dir=output,
    )
    assert unrecorded_recoveries is not None
    assert len(unrecorded_recoveries) == 1
    assert not partial.exists()

    def stop_after_partial_checkpoint_resume(**_kwargs):
        update_calls.append(True)
        raise RuntimeError("partial-checkpoint resume reached optimizer update")

    monkeypatch.setattr(
        module,
        "_optimizer_update",
        stop_after_partial_checkpoint_resume,
    )
    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except RuntimeError as error:
        assert "partial-checkpoint resume reached" in str(error)
    else:
        raise AssertionError("partial first-checkpoint staging tree was not replayed")
    assert update_calls == [True]
    recoveries = tuple((output / "checkpoint-recoveries").iterdir())
    assert len(recoveries) == 1
    assert recoveries[0].name.startswith("step-000001-partial-")
    assert (recoveries[0] / "model.safetensors").read_bytes() == b"partial checkpoint"
    replayed_partial_checkpoint = json.loads(
        (output / "run.json").read_text(encoding="utf-8")
    )
    assert replayed_partial_checkpoint["checkpoint_recoveries"] == [
        f"checkpoint-recoveries/{recoveries[0].name}"
    ]

    # Simulate SIGKILL after the first checkpoint and latest pointer publish but
    # before run.json records the checkpoint. Both published entries must move
    # together so the fixed trajectory can replay step 1 from step zero.
    published_checkpoint = output / "checkpoints" / "step-000001"
    published_checkpoint.mkdir()
    (published_checkpoint / "model.safetensors").write_bytes(
        b"unrecorded published checkpoint"
    )
    (output / "checkpoints" / "latest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint": "step-000001",
                "completed_step": 1,
                "metadata_sha256": "a" * 64,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stale_after_first_pointer = json.loads(
        (output / "run.json").read_text(encoding="utf-8")
    )
    stale_after_first_pointer["status"] = "running"
    stale_after_first_pointer["last_completed_step"] = 1
    stale_after_first_pointer["checkpoint_count"] = 0
    stale_after_first_pointer.pop("last_checkpoint", None)
    module.write_run_state(output / "run.json", stale_after_first_pointer)
    update_calls.clear()

    def stop_after_published_checkpoint_replay(**_kwargs):
        update_calls.append(True)
        raise RuntimeError("published-checkpoint resume reached optimizer update")

    monkeypatch.setattr(
        module,
        "_optimizer_update",
        stop_after_published_checkpoint_replay,
    )
    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except RuntimeError as error:
        assert "published-checkpoint resume reached" in str(error)
    else:
        raise AssertionError("published first-checkpoint crash was not replayed")
    assert update_calls == [True]
    assert not published_checkpoint.exists()
    assert not (output / "checkpoints" / "latest.json").exists()
    published_recoveries = tuple(
        path.name for path in (output / "checkpoint-recoveries").iterdir()
    )
    assert any(
        name.startswith("step-000001-published-")
        for name in published_recoveries
    )
    assert any(
        name.startswith("latest-pointer-published-")
        for name in published_recoveries
    )

    orphan_metrics_name = "metrics.recovery-999999.csv"
    (output / orphan_metrics_name).write_bytes(b"orphaned before run publication\n")
    update_calls.clear()

    def stop_after_orphan_metrics_adoption(**_kwargs):
        update_calls.append(True)
        raise RuntimeError("orphan-metrics resume reached optimizer update")

    monkeypatch.setattr(
        module,
        "_optimizer_update",
        stop_after_orphan_metrics_adoption,
    )
    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except RuntimeError as error:
        assert "orphan-metrics resume reached" in str(error)
    else:
        raise AssertionError("orphaned metrics recovery was not adopted")
    adopted_metrics = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert orphan_metrics_name in adopted_metrics["metrics_recoveries"]

    previous_run_name = f".run.json.previous-{'c' * 24}"
    (output / "run.json").rename(output / previous_run_name)
    update_calls.clear()

    def stop_after_previous_run_restoration(**_kwargs):
        update_calls.append(True)
        raise RuntimeError("restored-run resume reached optimizer update")

    monkeypatch.setattr(
        module,
        "_optimizer_update",
        stop_after_previous_run_restoration,
    )
    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except RuntimeError as error:
        assert "restored-run resume reached" in str(error)
    else:
        raise AssertionError("public resume did not restore the previous run generation")
    assert (output / "run.json").is_file()
    assert not (output / previous_run_name).exists()

    valid_run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    lease = module._acquire_t3b_training_lock(output)
    try:
        prior_pid = module._snapshot_regular_file_at(
            lease.output_descriptor,
            "training.pid",
            label="valid process identity",
            capture_payload=True,
        )
        module._backup_resume_process_identity(lease, prior_pid)
        module.write_run_state(
            output / "training.pid",
            {"uncommitted": True},
            parent_descriptor=lease.output_descriptor,
            expected_parent_snapshot=lease.output_snapshot,
        )
    finally:
        module._release_t3b_training_lock(lease)
    invalid_run = json.loads(json.dumps(valid_run))
    invalid_run["seed"] = int(invalid_run["seed"]) + 1
    module.write_run_state(output / "run.json", invalid_run)
    pid_before_invalid_resume = (output / "training.pid").read_bytes()
    process_recovery_root = output / "process-identity-recoveries"
    process_recoveries_before = {
        path.name: (path.read_bytes(), path.stat().st_ino)
        for path in process_recovery_root.iterdir()
    }

    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except ValueError as error:
        assert "immutable field differs" in str(error)
    else:
        raise AssertionError("self-rehashed contradictory run metadata was accepted")
    assert (output / "training.pid").read_bytes() == pid_before_invalid_resume
    assert {
        path.name: (path.read_bytes(), path.stat().st_ino)
        for path in process_recovery_root.iterdir()
    } == process_recoveries_before

    module.write_run_state(output / "run.json", valid_run)
    lease = module._acquire_t3b_training_lock(output)
    try:
        valid_run_snapshot = module._snapshot_regular_file_at(
            lease.output_descriptor,
            "run.json",
            label="restored valid run metadata",
            capture_payload=True,
        )
        module._reconcile_resume_process_identity(lease, valid_run_snapshot)
    finally:
        module._release_t3b_training_lock(lease)

    detached_output = tmp_path / "detached-after-update"

    def replace_output_during_update(**_kwargs):
        output.rename(detached_output)
        output.mkdir()
        return module.UpdateResult(
            loss=1.0,
            learning_rate=1e-4,
            gradient_norm=1.0,
            clip_coefficient=1.0,
            seconds=0.01,
        )

    monkeypatch.setattr(module, "_optimizer_update", replace_output_during_update)
    try:
        module.run_lora_finetune(
            resume_config,
            launch_config_path=output / "launch.json",
            training_log_path=output / "training.log",
        )
    except RuntimeError as error:
        assert "output directory changed" in str(error)
    else:
        raise AssertionError("mid-update output-directory replacement was accepted")
    assert tuple(output.iterdir()) == ()
    assert (detached_output / "run.json").is_file()


def test_prepare_t3b_launch_freezes_actual_components_before_updates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "prepare_lora_finetune_launch"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    config = module.FineTuneConfig(
        output_dir=tmp_path / "t3b",
        budget_mode=module.FIXED_BUDGET_MODE,
        lora_scope="expert_only",
    )
    target_names = tuple(f"expert.layers.target_{index}" for index in range(112))
    report = SimpleNamespace(
        scope="expert_only",
        rank=8,
        alpha=16.0,
        dropout=0.0,
        adapter_count=112,
        target_names=target_names,
        trainable_names=tuple(
            f"{name}.{suffix}"
            for name in target_names
            for suffix in ("lora_a", "lora_b")
        ),
        trainable_tensor_count=224,
        trainable_scalar_count=1_708_032,
    )
    split = SimpleNamespace(train_episodes=(0, 1), holdout_episodes=(2,))
    frozen_inputs = _t3b_frozen_inputs()
    stats = SimpleNamespace(
        sha256=frozen_inputs["train_statistics_sha256"],
        processor_stats={"observation.state": {"mean": [0.0]}},
    )
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=3000)
    )
    model_marker = object()
    bridge_marker = object()
    bridge_evidence = _t3b_bridge_evidence()
    monkeypatch.setattr(
        module,
        "_build_training_components",
        lambda _config, training_horizon: (
            split,
            stats,
            model_marker,
            report,
            bridge_marker,
            optimizer,
        ),
    )
    monkeypatch.setattr(
        module,
        "_validate_t3b_training_bridge_semantics",
        lambda **_kwargs: bridge_evidence,
    )
    monkeypatch.setattr(
        module,
        "training_base_artifact_identity",
        lambda _model: {
            "model_file": "model.bfloat16.safetensors",
            "model_sha256": "b" * 64,
            "name_map_file": "name_map.json",
            "name_map_sha256": "c" * 64,
        },
    )
    frozen_input_calls = []

    def collect_frozen_inputs(
        _config,
        model,
        *,
        runtime_statistics=None,
        validate_runtime_model=True,
    ):
        frozen_input_calls.append(
            (
                model,
                runtime_statistics,
                validate_runtime_model,
            )
        )
        return frozen_inputs

    monkeypatch.setattr(
        module,
        "collect_t3b_frozen_input_evidence",
        collect_frozen_inputs,
    )
    monkeypatch.setattr(
        module,
        "reference_freeze_policy_evidence",
        lambda: {
            "lerobot_version": "0.6.1",
            "freeze_vision_encoder": True,
            "train_expert_only": True,
            "train_state_proj": True,
            "configuration_source_sha256": "d" * 64,
            "implementation_source_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(
        module,
        "finetune_implementation_hashes",
        lambda: {"training/finetune.py": "f" * 64},
    )
    monkeypatch.setattr(module.time, "time_ns", lambda: 1_788_264_000_000_000_000)

    document, digest = module.prepare_lora_finetune_launch(config)

    path = config.output_dir / "launch.json"
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert document["configuration_sha256"] == (
        module.validate_finetune_launch_config(document)["configuration_sha256"]
    )
    assert document["run_config_sha256"] == module.training_run_config_sha256(
        config,
        selected_steps=3000,
        train_statistics_sha256=frozen_inputs["train_statistics_sha256"],
        train_episodes=(0, 1),
        holdout_episodes=(2,),
        base_artifact={
            "model_file": "model.bfloat16.safetensors",
            "model_sha256": "b" * 64,
            "name_map_file": "name_map.json",
            "name_map_sha256": "c" * 64,
        },
        optimizer_config=optimizer.config,
    )
    assert document["training_bridge"] == bridge_evidence
    assert frozen_input_calls == [
        (None, None, False),
        (model_marker, stats, True),
    ]

    swapped_config = module.FineTuneConfig(
        output_dir=tmp_path / "t3b-swapped",
        budget_mode=module.FIXED_BUDGET_MODE,
        lora_scope="expert_only",
    )
    detached_output = tmp_path / "detached-t3b-swapped"
    real_write_launch = module.write_finetune_launch_config
    swapped = False

    def replace_output_before_writer(path, value, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            swapped_config.output_dir.rename(detached_output)
            swapped_config.output_dir.mkdir()
        return real_write_launch(path, value, **kwargs)

    monkeypatch.setattr(
        module,
        "write_finetune_launch_config",
        replace_output_before_writer,
    )
    try:
        module.prepare_lora_finetune_launch(swapped_config)
    except RuntimeError as error:
        assert "launch configuration directory changed" in str(error)
    else:
        raise AssertionError("prepare wrote launch.json into a replacement output")
    assert tuple(swapped_config.output_dir.iterdir()) == ()
    assert (detached_output / "launch.json").is_file()

    monkeypatch.setattr(module, "write_finetune_launch_config", real_write_launch)
    parent = tmp_path / "prepared-parent"
    parent.mkdir()
    parent_swap_config = module.FineTuneConfig(
        output_dir=parent / "t3b",
        budget_mode=module.FIXED_BUDGET_MODE,
        lora_scope="expert_only",
    )
    detached_parent = tmp_path / "detached-prepared-parent"
    real_mkdir = module.os.mkdir
    parent_swapped = False

    def swap_parent_during_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal parent_swapped
        if not parent_swapped and path == parent_swap_config.output_dir.name:
            parent_swapped = True
            parent.rename(detached_parent)
            real_mkdir(parent, 0o700)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "mkdir", swap_parent_during_mkdir)
    try:
        module.prepare_lora_finetune_launch(parent_swap_config)
    except RuntimeError as error:
        assert "parent changed" in str(error)
    else:
        raise AssertionError("prepare created output below a replacement parent")
    assert tuple(parent.iterdir()) == ()
    assert (detached_parent / "t3b").is_dir()
    assert not (detached_parent / "t3b" / "launch.json").exists()


def test_checkpoint_retention_keeps_only_three_valid_newest_steps(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune", fromlist=["save_training_checkpoint"]
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
        "latest.json",
        "step-000003",
        "step-000004",
        "step-000005",
    ]


def test_checkpoint_retention_rejects_a_valid_checkpoint_from_a_different_run(
    tmp_path: Path,
) -> None:
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
        if step in {1, 2, 3}:
            state = _tiny_checkpoint_state(module, optimizer, step=step)
            module.save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_root=root,
                state=state,
                trainable_names=names,
                keep_last=10,
            )
    foreign_root = tmp_path / "foreign-checkpoints"
    foreign_state = _tiny_checkpoint_state(module, optimizer, step=9)
    foreign_state = module.CheckpointState(
        **{
            **foreign_state.__dict__,
            "run_config_sha256": "b" * 64,
        }
    )
    foreign = module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=foreign_root,
        state=foreign_state,
        trainable_names=names,
        keep_last=10,
    )
    shutil.move(str(foreign.path), root / foreign.path.name)

    with pytest.raises(ValueError, match="invalid candidate"):
        module.prune_training_checkpoints(
            root,
            keep_last=2,
            expected_run_config_sha256="a" * 64,
            trainable_names=names,
            expected_model_tensors=dict(tree_flatten(model.trainable_parameters())),
            expected_optimizer_tensors=dict(tree_flatten(optimizer.state)),
        )

    assert sorted(path.name for path in root.iterdir() if path.name.startswith("step-")) == [
        "step-000001",
        "step-000002",
        "step-000003",
        "step-000009",
    ]


def test_checkpoint_pruning_rejects_selected_path_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    for step in range(1, 5):
        optimizer.update(
            model,
            {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
        )
        mx.eval(model.parameters(), optimizer.state)
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=step),
            trainable_names=names,
            keep_last=10,
        )
    real_move = module._move_entry_to_unique_recovery_at
    original = root / "original-step-000001"
    swapped = False

    def replace_selected_checkpoint(**kwargs):
        nonlocal swapped
        if not swapped and kwargs["source_name"] == "step-000001":
            swapped = True
            (root / "step-000001").rename(original)
            (root / "step-000001").mkdir()
            (root / "step-000001" / "owner.txt").write_text(
                "replacement\n", encoding="utf-8"
            )
        return real_move(**kwargs)

    monkeypatch.setattr(
        module,
        "_move_entry_to_unique_recovery_at",
        replace_selected_checkpoint,
    )
    try:
        module.prune_training_checkpoints(
            root,
            keep_last=3,
            expected_run_config_sha256="a" * 64,
            trainable_names=names,
            expected_model_tensors=dict(tree_flatten(model.trainable_parameters())),
            expected_optimizer_tensors=dict(tree_flatten(optimizer.state)),
        )
    except RuntimeError as error:
        assert "staged entry changed" in str(error)
    else:
        raise AssertionError("replaced checkpoint path was pruned")
    assert (root / "step-000001" / "owner.txt").read_text() == "replacement\n"
    assert (original / "metadata.json").is_file()


def test_checkpoint_publication_never_clobbers_an_inserted_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint"],
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    real_rename = module._rename_entry_no_clobber_at
    inserted = False

    def insert_target_before_publish(**kwargs):
        nonlocal inserted
        if not inserted and kwargs["destination_name"] == "step-000001":
            inserted = True
            target = root / "step-000001"
            target.mkdir()
            (target / "owner.txt").write_text("competitor\n", encoding="utf-8")
        return real_rename(**kwargs)

    monkeypatch.setattr(
        module,
        "_rename_entry_no_clobber_at",
        insert_target_before_publish,
    )
    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=1),
            trainable_names=names,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("checkpoint publication clobbered an inserted target")
    assert (root / "step-000001" / "owner.txt").read_text() == "competitor\n"
    assert not (root / "latest.json").exists()


def test_checkpoint_publication_rejects_staged_directory_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint"],
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    real_rename = module._rename_entry_no_clobber_at
    original_staging: Path | None = None

    def replace_staging_before_publish(**kwargs):
        nonlocal original_staging
        if original_staging is None and kwargs["destination_name"] == "step-000001":
            staging = root / kwargs["source_name"]
            original_staging = root / f"{staging.name}.original"
            staging.rename(original_staging)
            staging.mkdir()
            (staging / "model.safetensors").write_bytes(b"replacement")
        return real_rename(**kwargs)

    monkeypatch.setattr(
        module,
        "_rename_entry_no_clobber_at",
        replace_staging_before_publish,
    )
    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=1),
            trainable_names=names,
        )
    except RuntimeError as error:
        assert "staged entry changed" in str(error)
    else:
        raise AssertionError("replaced staged checkpoint directory was published")
    assert original_staging is not None
    assert (original_staging / "metadata.json").is_file()
    assert not (root / "step-000001").exists()
    assert not (root / "latest.json").exists()


def test_checkpoint_staging_uses_its_bound_inode_for_first_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    root = tmp_path / "checkpoints"
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    real_save = module.mx.save_safetensors
    replacement: Path | None = None
    original: Path | None = None
    swapped = False

    def replace_stage_before_first_write(path, tensors):
        nonlocal replacement, original, swapped
        if not swapped:
            swapped = True
            replacement = next(root.glob(".step-000001.*"))
            original = replacement.with_name(f"{replacement.name}.original")
            replacement.rename(original)
            replacement.mkdir()
            (replacement / "owner.txt").write_text("competitor\n", encoding="utf-8")
        return real_save(path, tensors)

    monkeypatch.setattr(module.mx, "save_safetensors", replace_stage_before_first_write)
    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=1),
            trainable_names=names,
        )
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("renamed checkpoint stage was accepted")

    assert replacement is not None and original is not None
    assert tuple(path.name for path in replacement.iterdir()) == ("owner.txt",)
    assert (original / "model.safetensors").is_file()
    assert not (root / "step-000001").exists()
    assert not (root / "latest.json").exists()


def test_checkpoint_first_child_write_never_follows_an_inserted_symlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    root = tmp_path / "checkpoints"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"guard\n")
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    real_create = module._create_exclusive_child_file_at
    inserted = False

    def insert_symlink(parent_descriptor, name, **kwargs):
        nonlocal inserted
        if not inserted and name == "model.safetensors":
            inserted = True
            os.symlink(outside, name, dir_fd=parent_descriptor)
        return real_create(parent_descriptor, name, **kwargs)

    monkeypatch.setattr(module, "_create_exclusive_child_file_at", insert_symlink)
    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=1),
            trainable_names=names,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("checkpoint serializer followed an inserted child symlink")

    assert outside.read_bytes() == b"guard\n"
    assert not (root / "step-000001").exists()


def test_checkpoint_publication_revalidates_staged_child_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint"],
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    real_rename = module._rename_entry_no_clobber_at
    corrupted = False

    def corrupt_child_before_publish(**kwargs):
        nonlocal corrupted
        if not corrupted and kwargs["destination_name"] == "step-000001":
            corrupted = True
            model_path = root / kwargs["source_name"] / "model.safetensors"
            model_path.write_bytes(b"corrupt checkpoint bytes")
        return real_rename(**kwargs)

    monkeypatch.setattr(
        module,
        "_rename_entry_no_clobber_at",
        corrupt_child_before_publish,
    )
    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=1),
            trainable_names=names,
        )
    except (ValueError, RuntimeError) as error:
        assert "checkpoint" in str(error) or "safetensor" in str(error).lower()
    else:
        raise AssertionError("checkpoint with mutated staged bytes was accepted")
    assert (root / "step-000001").is_dir()
    assert not (root / "latest.json").exists()


def test_checkpoint_save_retains_child_authority_through_pointer_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint"],
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    real_pointer = module._write_latest_checkpoint_pointer
    replaced = False
    detached: Path | None = None

    def replace_child_before_pointer(checkpoint_root, checkpoint, **kwargs):
        nonlocal replaced, detached
        if not replaced:
            replaced = True
            model_path = checkpoint.path / "model.safetensors"
            detached = checkpoint.path / "model.original.safetensors"
            model_path.rename(detached)
            shutil.copyfile(detached, model_path)
        return real_pointer(checkpoint_root, checkpoint, **kwargs)

    monkeypatch.setattr(
        module,
        "_write_latest_checkpoint_pointer",
        replace_child_before_pointer,
    )
    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=1),
            trainable_names=names,
        )
    except RuntimeError as error:
        assert "checkpoint" in str(error) and "bound" in str(error)
    else:
        raise AssertionError("checkpoint child replacement escaped save authority")

    assert detached is not None and detached.is_file()
    assert (root / "step-000001" / "model.safetensors").is_file()


def test_checkpoint_binding_is_retained_through_run_state_publication(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_persist_run_state_with_checkpoint_binding",
            "save_training_checkpoint",
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    checkpoint = module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=root,
        state=_tiny_checkpoint_state(module, optimizer, step=1),
        trainable_names=names,
        keep_last=10,
    )
    root_snapshot = module._snapshot_directory(root, label="test checkpoint root")
    root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    run_path = tmp_path / "run.json"

    def corrupt_checkpoint_during_publication(path: Path, document: object) -> str:
        (checkpoint.path / "model.safetensors").write_bytes(
            b"corrupted during run-state publication"
        )
        return module.write_run_state(path, document)

    try:
        try:
            module._persist_run_state_with_checkpoint_binding(
                checkpoint=checkpoint,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=root_snapshot,
                persist=corrupt_checkpoint_during_publication,
                path=run_path,
                value={"last_checkpoint": checkpoint.path.name},
            )
        except RuntimeError as error:
            assert "checkpoint" in str(error) and "changed" in str(error)
        else:
            raise AssertionError("checkpoint mutation escaped run-state binding")
    finally:
        os.close(root_descriptor)

    assert run_path.is_file()


def test_retained_checkpoint_replacement_is_rejected_before_run_state_publication(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_persist_run_state_with_checkpoint_binding",
            "save_training_checkpoint",
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
    checkpoint = None
    for step in range(1, 4):
        optimizer.update(
            model,
            {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
        )
        mx.eval(model.parameters(), optimizer.state)
        checkpoint = module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=step),
            trainable_names=names,
            keep_last=3,
        )
    assert checkpoint is not None

    older = root / "step-000001"
    detached = tmp_path / "step-000001.original"
    older.rename(detached)
    shutil.copytree(detached, older)
    root_snapshot = module._snapshot_directory(root, label="test checkpoint root")
    root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    run_path = tmp_path / "run.json"
    persist_called = False

    def persist(path: Path, document: object) -> str:
        nonlocal persist_called
        persist_called = True
        return module.write_run_state(path, document)

    try:
        with pytest.raises(RuntimeError, match="retained checkpoint changed after save"):
            module._persist_run_state_with_checkpoint_binding(
                checkpoint=checkpoint,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=root_snapshot,
                persist=persist,
                path=run_path,
                value={"last_checkpoint": checkpoint.path.name},
            )
    finally:
        os.close(root_descriptor)

    assert not persist_called
    assert not run_path.exists()


def test_terminal_checkpoint_binding_survives_cadence_cas_until_final_state(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_bind_t3b_checkpoint_root",
            "_bind_t3b_final_checkpoint_namespace",
            "_persist_run_state_with_checkpoint_binding",
            "_release_t3b_training_lock",
            "_revalidate_t3b_final_checkpoint_namespace",
            "save_training_checkpoint",
        ],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    output = tmp_path / "run"
    output.mkdir()
    root = output / "checkpoints"
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    checkpoint = None
    for step in range(1, 4):
        optimizer.update(
            model,
            {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
        )
        mx.eval(model.parameters(), optimizer.state)
        checkpoint = module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=step),
            trainable_names=names,
            keep_last=3,
        )
    assert checkpoint is not None

    lease = module._acquire_t3b_training_lock(output)
    try:
        module._bind_t3b_checkpoint_root(lease, allow_existing=True)
        assert lease.checkpoint_root_descriptor is not None
        assert lease.checkpoint_root_snapshot is not None
        module._persist_run_state_with_checkpoint_binding(
            checkpoint=checkpoint,
            checkpoint_root_descriptor=lease.checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=lease.checkpoint_root_snapshot,
            persist=lambda path, value: module.write_run_state(path, value),
            path=output / "run.json",
            value={"last_checkpoint": checkpoint.path.name},
        )
        module._bind_t3b_final_checkpoint_namespace(lease, checkpoint=checkpoint)
        module._revalidate_t3b_final_checkpoint_namespace(lease, verify_bytes=True)

        retained_path = root / "step-000001"
        model_path = retained_path / "model.safetensors"
        detached = retained_path / "model.original.safetensors"
        model_path.rename(detached)
        shutil.copyfile(detached, model_path)
        try:
            module._revalidate_t3b_final_checkpoint_namespace(
                lease,
                verify_bytes=True,
            )
        except RuntimeError as error:
            assert "checkpoint" in str(error) and "changed" in str(error)
        else:
            raise AssertionError(
                "post-cadence-CAS retained checkpoint replacement escaped binding"
            )
    finally:
        module._release_t3b_training_lock(lease)


def test_checkpoint_save_rejects_latest_pointer_swap_at_publication_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint", "write_run_state"],
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=root,
        state=_tiny_checkpoint_state(module, optimizer, step=1),
        trainable_names=names,
        keep_last=10,
    )
    original_pointer = root / "latest.original.json"
    real_pointer = module._write_latest_checkpoint_pointer
    swapped = False

    def swap_pointer_before_publication(checkpoint_root, checkpoint, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            (root / "latest.json").rename(original_pointer)
            module.write_run_state(
                root / "latest.json",
                {
                    "format_version": 1,
                    "checkpoint": "competitor",
                    "completed_step": 999,
                    "metadata_sha256": "f" * 64,
                },
            )
        return real_pointer(checkpoint_root, checkpoint, **kwargs)

    monkeypatch.setattr(
        module,
        "_write_latest_checkpoint_pointer",
        swap_pointer_before_publication,
    )
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=2),
            trainable_names=names,
            keep_last=10,
        )
    except RuntimeError as error:
        assert "destination changed" in str(error) or "namespace changed" in str(error)
    else:
        raise AssertionError("replacement latest pointer was silently overwritten")

    assert swapped
    assert original_pointer.is_file()
    assert json.loads((root / "latest.json").read_text(encoding="utf-8"))[
        "checkpoint"
    ] == "competitor"


def test_checkpoint_save_rejects_a_contaminated_existing_namespace_prepublication(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )

    def two_checkpoint_root(name: str):
        root = tmp_path / name
        model = TinyRegressor()
        optimizer = optimizer_module.SmolVLAAdamW(
            optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
        )
        names = tuple(
            tensor_name
            for tensor_name, _ in tree_flatten(model.trainable_parameters())
        )
        for step in (1, 2):
            optimizer.update(
                model,
                {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
            )
            mx.eval(model.parameters(), optimizer.state)
            module.save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_root=root,
                state=_tiny_checkpoint_state(module, optimizer, step=step),
                trainable_names=names,
                keep_last=10,
            )
        optimizer.update(
            model,
            {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
        )
        mx.eval(model.parameters(), optimizer.state)
        return root, model, optimizer, names

    extra_root, extra_model, extra_optimizer, extra_names = two_checkpoint_root(
        "extra"
    )
    extra_pointer = (extra_root / "latest.json").read_bytes()
    extra_inventory = sorted(path.name for path in extra_root.iterdir())
    try:
        module.save_training_checkpoint(
            model=extra_model,
            optimizer=extra_optimizer,
            checkpoint_root=extra_root,
            state=_tiny_checkpoint_state(module, extra_optimizer, step=3),
            trainable_names=extra_names,
            keep_last=10,
            expected_existing_checkpoint_steps={1},
        )
    except ValueError as error:
        assert "retained trajectory" in str(error)
    else:
        raise AssertionError("extra valid checkpoint was accepted before live save")
    assert (extra_root / "latest.json").read_bytes() == extra_pointer
    assert sorted(path.name for path in extra_root.iterdir()) == extra_inventory

    invalid_root, invalid_model, invalid_optimizer, invalid_names = two_checkpoint_root(
        "invalid"
    )
    (invalid_root / "step-000002" / "model.safetensors").write_bytes(
        b"invalid existing checkpoint"
    )
    invalid_pointer = (invalid_root / "latest.json").read_bytes()
    invalid_inventory = sorted(path.name for path in invalid_root.iterdir())
    try:
        module.save_training_checkpoint(
            model=invalid_model,
            optimizer=invalid_optimizer,
            checkpoint_root=invalid_root,
            state=_tiny_checkpoint_state(module, invalid_optimizer, step=3),
            trainable_names=invalid_names,
            keep_last=10,
            expected_existing_checkpoint_steps={1, 2},
        )
    except ValueError as error:
        assert "invalid candidate" in str(error)
    else:
        raise AssertionError("invalid checkpoint was accepted before live save")
    assert (invalid_root / "latest.json").read_bytes() == invalid_pointer
    assert sorted(path.name for path in invalid_root.iterdir()) == invalid_inventory


def test_checkpoint_save_rejects_namespace_insertion_after_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint"],
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=root,
        state=_tiny_checkpoint_state(module, optimizer, step=1),
        trainable_names=names,
        keep_last=10,
    )
    pointer_before = (root / "latest.json").read_bytes()
    real_create_stage = module._create_staged_directory_at
    inserted = False

    def insert_foreign_entry_after_preflight(*args, **kwargs):
        nonlocal inserted
        if not inserted:
            inserted = True
            (root / "foreign-owner.txt").write_bytes(b"concurrent namespace owner")
        return real_create_stage(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "_create_staged_directory_at",
        insert_foreign_entry_after_preflight,
    )
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=2),
            trainable_names=names,
            keep_last=10,
            expected_existing_checkpoint_steps={1},
        )
    except RuntimeError as error:
        assert "namespace changed" in str(error)
    else:
        raise AssertionError("post-preflight namespace insertion was accepted")

    assert inserted
    assert (root / "foreign-owner.txt").read_bytes() == b"concurrent namespace owner"
    assert not (root / "step-000002").exists()
    assert (root / "latest.json").read_bytes() == pointer_before


def test_checkpoint_save_rejects_same_schema_value_corruption_before_hashing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["save_training_checkpoint"],
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    real_save = module.mx.save_safetensors
    corrupted = False

    def corrupt_model_after_save(destination, tensors):
        nonlocal corrupted
        real_save(destination, tensors)
        if not corrupted:
            corrupted = True
            destination.flush()
            destination.seek(0)
            rewritten = module.mx.load(destination, format="safetensors")
            name = sorted(rewritten)[0]
            rewritten[name] = rewritten[name] + module.mx.ones_like(rewritten[name])
            module.mx.eval(rewritten)
            destination.seek(0)
            destination.truncate(0)
            real_save(destination, rewritten)

    monkeypatch.setattr(module.mx, "save_safetensors", corrupt_model_after_save)
    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=1),
            trainable_names=names,
        )
    except RuntimeError as error:
        assert "checkpoint model tensor value changed" in str(error)
    else:
        raise AssertionError("same-schema checkpoint value corruption was accepted")

    assert corrupted
    assert not (root / "step-000001").exists()
    assert not (root / "latest.json").exists()


def test_checkpoint_reader_requires_the_exact_three_file_inventory(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_read_checkpoint_directory", "save_training_checkpoint"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    saved = module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=tmp_path / "checkpoints",
        state=_tiny_checkpoint_state(module, optimizer, step=1),
        trainable_names=names,
    )
    (saved.path / "unexpected.txt").write_text("competitor\n", encoding="utf-8")

    try:
        module._read_checkpoint_directory(
            saved.path,
            expected_run_config_sha256="a" * 64,
            trainable_names=names,
            expected_model_tensors=dict(tree_flatten(model.trainable_parameters())),
            expected_optimizer_tensors=dict(tree_flatten(optimizer.state)),
        )
    except ValueError as error:
        assert "inventory differs" in str(error)
    else:
        raise AssertionError("checkpoint with an extra child was accepted")


def test_checkpoint_reader_keeps_the_original_candidate_directory_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_read_checkpoint_directory", "save_training_checkpoint"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    root = tmp_path / "checkpoints"
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    saved = module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=root,
        state=_tiny_checkpoint_state(module, optimizer, step=1),
        trainable_names=names,
    )
    expected_model = dict(tree_flatten(model.trainable_parameters()))
    expected_optimizer = dict(tree_flatten(optimizer.state))
    root_snapshot = module._snapshot_directory(root, label="test checkpoint root")
    root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    detached = root / "original-step-000001"
    replacement = saved.path
    real_private_file = module._private_stable_file
    tensor_file_count = 0

    @contextmanager
    def swap_candidate_around_tensor_reads(*args, **kwargs):
        nonlocal tensor_file_count
        is_bound_tensor = kwargs.get("source_parent_descriptor") is not None
        if is_bound_tensor:
            tensor_file_count += 1
            if tensor_file_count == 1:
                replacement.rename(detached)
                replacement.mkdir()
                (replacement / "owner.txt").write_text(
                    "replacement\n", encoding="utf-8"
                )
        try:
            with real_private_file(*args, **kwargs) as value:
                yield value
        finally:
            if is_bound_tensor and tensor_file_count == 2 and replacement.exists():
                shutil.rmtree(replacement)
                detached.rename(replacement)

    monkeypatch.setattr(module, "_private_stable_file", swap_candidate_around_tensor_reads)
    try:
        checkpoint, loaded_model, loaded_optimizer = module._read_checkpoint_directory(
            saved.path,
            expected_run_config_sha256="a" * 64,
            trainable_names=names,
            expected_model_tensors=expected_model,
            expected_optimizer_tensors=expected_optimizer,
            checkpoint_root_descriptor=root_descriptor,
            expected_checkpoint_root_snapshot=root_snapshot,
        )
    finally:
        os.close(root_descriptor)

    assert tensor_file_count == 2
    assert checkpoint.path == saved.path
    for name, expected in expected_model.items():
        np.testing.assert_array_equal(np.asarray(loaded_model[name]), np.asarray(expected))
    for name, expected in expected_optimizer.items():
        np.testing.assert_array_equal(
            np.asarray(loaded_optimizer[name]), np.asarray(expected)
        )


def test_checkpoint_restore_rejects_candidate_replacement_during_pointer_repair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["load_latest_training_checkpoint", "save_training_checkpoint"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    root = tmp_path / "checkpoints"
    saved_model = TinyRegressor()
    saved_optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    names = tuple(
        name for name, _ in tree_flatten(saved_model.trainable_parameters())
    )
    saved_optimizer.update(
        saved_model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(saved_model.parameters(), saved_optimizer.state)
    saved = module.save_training_checkpoint(
        model=saved_model,
        optimizer=saved_optimizer,
        checkpoint_root=root,
        state=_tiny_checkpoint_state(module, saved_optimizer, step=1),
        trainable_names=names,
    )
    resumed_model = TinyRegressor()
    model_before = np.asarray(resumed_model.proj.weight).copy()
    resumed_optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    optimizer_before = {
        name: np.asarray(value).copy()
        for name, value in tree_flatten(resumed_optimizer.state)
    }
    detached = root / "detached-step-000001"
    real_write_pointer = module._write_latest_checkpoint_pointer
    swapped = False

    def replace_candidate_before_pointer(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            saved.path.rename(detached)
            shutil.copytree(detached, saved.path)
        return real_write_pointer(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "_write_latest_checkpoint_pointer",
        replace_candidate_before_pointer,
    )
    try:
        module.load_latest_training_checkpoint(
            model=resumed_model,
            optimizer=resumed_optimizer,
            checkpoint_root=root,
            trainable_names=names,
            expected_run_config_sha256="a" * 64,
        )
    except RuntimeError as error:
        assert "selected checkpoint changed" in str(error)
    else:
        raise AssertionError("replaced selected checkpoint was restored")

    assert swapped
    np.testing.assert_array_equal(np.asarray(resumed_model.proj.weight), model_before)
    optimizer_after = dict(tree_flatten(resumed_optimizer.state))
    assert set(optimizer_after) == set(optimizer_before)
    for name, expected in optimizer_before.items():
        np.testing.assert_array_equal(np.asarray(optimizer_after[name]), expected)


def test_checkpoint_discovery_rejects_out_of_cadence_candidates_before_mutation(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["load_latest_training_checkpoint", "save_training_checkpoint", "write_run_state"],
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
    step_one_weight: np.ndarray | None = None
    for step in (1, 2):
        optimizer.update(
            model,
            {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
        )
        mx.eval(model.parameters(), optimizer.state)
        if step == 1:
            step_one_weight = np.asarray(model.proj.weight).copy()
        saved.append(
            module.save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_root=root,
                state=_tiny_checkpoint_state(module, optimizer, step=step),
                trainable_names=names,
                keep_last=10,
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
    inventory_before = sorted(path.name for path in root.iterdir())
    resumed_model = TinyRegressor()
    resumed_optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    optimizer_before = {
        name: np.asarray(value).copy()
        for name, value in tree_flatten(resumed_optimizer.state)
    }

    try:
        module.load_latest_training_checkpoint(
            model=resumed_model,
            optimizer=resumed_optimizer,
            checkpoint_root=root,
            trainable_names=names,
            expected_run_config_sha256="a" * 64,
            expected_selected_steps=10,
            expected_effective_batch_size=8,
            expected_checkpoint_interval=100,
        )
    except ValueError as error:
        assert "checkpoint namespace" in str(error)
    else:
        raise AssertionError("out-of-cadence checkpoint namespace was accepted")

    assert step_one_weight is not None
    np.testing.assert_array_equal(
        np.asarray(resumed_model.proj.weight),
        np.asarray(TinyRegressor().proj.weight),
    )
    assert resumed_optimizer.step_index == 0
    assert sorted(path.name for path in root.iterdir()) == inventory_before
    assert json.loads((root / "latest.json").read_text(encoding="utf-8"))[
        "completed_step"
    ] == 1


def test_checkpoint_discovery_rejects_noncanonical_debris_before_mutation(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["load_latest_training_checkpoint", "save_training_checkpoint"],
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    saved = module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=root,
        state=_tiny_checkpoint_state(module, optimizer, step=1),
        trainable_names=names,
        keep_last=10,
    )
    module.write_run_state(
        root / "latest.json",
        {
            "format_version": 1,
            "checkpoint": saved.path.name,
            "completed_step": 0,
            "metadata_sha256": "0" * 64,
        },
    )
    (root / "unexpected-owner.txt").write_bytes(b"foreign checkpoint debris")
    inventory_before = {
        path.name: path.read_bytes() if path.is_file() else None
        for path in root.iterdir()
    }
    resumed_model = TinyRegressor()
    model_before = np.asarray(resumed_model.proj.weight).copy()
    resumed_optimizer = optimizer_module.SmolVLAAdamW(optimizer.config)

    try:
        module.load_latest_training_checkpoint(
            model=resumed_model,
            optimizer=resumed_optimizer,
            checkpoint_root=root,
            trainable_names=names,
            expected_run_config_sha256="a" * 64,
            expected_selected_steps=10,
            expected_effective_batch_size=8,
            expected_checkpoint_interval=1,
        )
    except ValueError as error:
        assert "invalid candidates" in str(error)
    else:
        raise AssertionError("noncanonical checkpoint debris was accepted")

    assert {
        path.name: path.read_bytes() if path.is_file() else None
        for path in root.iterdir()
    } == inventory_before
    np.testing.assert_array_equal(np.asarray(resumed_model.proj.weight), model_before)
    assert resumed_optimizer.step_index == 0


def test_t3b_checkpoint_discovery_rejects_an_extra_valid_retained_step(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["load_latest_training_checkpoint", "save_training_checkpoint"],
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
    for step in range(1, 5):
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
                keep_last=10,
            )
        )
    last = saved[-1]
    recorded = {
        "step": 4,
        "path": str(last.path),
        "metadata_sha256": last.metadata_sha256,
        "model_sha256": last.model_sha256,
        "optimizer_sha256": last.optimizer_sha256,
    }
    inventory_before = sorted(path.name for path in root.iterdir())
    resumed_model = TinyRegressor()
    resumed_optimizer = optimizer_module.SmolVLAAdamW(optimizer.config)

    try:
        module.load_latest_training_checkpoint(
            model=resumed_model,
            optimizer=resumed_optimizer,
            checkpoint_root=root,
            trainable_names=names,
            expected_run_config_sha256="a" * 64,
            expected_selected_steps=10,
            expected_effective_batch_size=8,
            expected_checkpoint_interval=1,
            expected_last_checkpoint=recorded,
            allowed_uncommitted_step=5,
        )
    except ValueError as error:
        assert "retained trajectory" in str(error)
    else:
        raise AssertionError("extra retained checkpoint escaped exact resume binding")

    assert resumed_optimizer.step_index == 0
    assert sorted(path.name for path in root.iterdir()) == inventory_before


def test_resume_quarantines_only_the_invalid_uncommitted_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["load_latest_training_checkpoint", "save_training_checkpoint"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    output = tmp_path / "run"
    output.mkdir()
    root = output / "checkpoints"
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    committed = module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=root,
        state=_tiny_checkpoint_state(module, optimizer, step=1),
        trainable_names=names,
        keep_last=10,
    )
    recorded = {
        "step": 1,
        "path": str(committed.path),
        "metadata_sha256": committed.metadata_sha256,
        "model_sha256": committed.model_sha256,
        "optimizer_sha256": committed.optimizer_sha256,
    }
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    real_rename = module._rename_entry_no_clobber_at
    corrupted = False

    def corrupt_uncommitted_child(**kwargs):
        nonlocal corrupted
        if not corrupted and kwargs["destination_name"] == "step-000002":
            corrupted = True
            staged_model = root / kwargs["source_name"] / "model.safetensors"
            staged_model.write_bytes(b"interrupted invalid candidate")
        return real_rename(**kwargs)

    monkeypatch.setattr(
        module,
        "_rename_entry_no_clobber_at",
        corrupt_uncommitted_child,
    )
    try:
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=2),
            trainable_names=names,
            keep_last=10,
        )
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("invalid uncommitted checkpoint was unexpectedly saved")
    monkeypatch.setattr(module, "_rename_entry_no_clobber_at", real_rename)
    assert (root / "step-000002").is_dir()

    resumed_model = TinyRegressor()
    resumed_optimizer = optimizer_module.SmolVLAAdamW(optimizer.config)
    loaded = module.load_latest_training_checkpoint(
        model=resumed_model,
        optimizer=resumed_optimizer,
        checkpoint_root=root,
        trainable_names=names,
        expected_run_config_sha256="a" * 64,
        expected_selected_steps=10,
        expected_effective_batch_size=8,
        expected_checkpoint_interval=1,
        expected_last_checkpoint=recorded,
        allowed_uncommitted_step=2,
    )
    assert loaded.state.completed_step == 1
    assert not (root / "step-000002").exists()
    hidden = tuple(root.glob(".recovery-step-000002-*"))
    assert len(hidden) == 1

    recoveries = module._prepare_zero_step_checkpoint_replay(
        root,
        output_dir=output,
        expected_staging_step=2,
        allow_published_entries=True,
        allowed_published_steps={1, 2},
    )
    assert recoveries is not None
    assert len(recoveries) == 1
    recovered_path = output / recoveries[0]
    assert recovered_path.is_dir()
    assert (recovered_path / "model.safetensors").read_bytes() == (
        b"interrupted invalid candidate"
    )


def test_invalid_uncommitted_checkpoint_is_not_quarantined_until_namespace_validates(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["load_latest_training_checkpoint", "save_training_checkpoint"],
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
                keep_last=10,
            )
        )
    committed = saved[0]
    module.write_run_state(
        root / "latest.json",
        {
            "format_version": 1,
            "checkpoint": committed.path.name,
            "completed_step": 1,
            "metadata_sha256": committed.metadata_sha256,
        },
    )
    (root / "step-000002" / "model.safetensors").write_bytes(
        b"invalid allowed crash-window checkpoint"
    )
    extra = root / "step-999999"
    extra.mkdir()
    (extra / "unexpected.bin").write_bytes(b"second invalid candidate")

    def namespace_bytes() -> tuple[tuple[str, str, bytes | None], ...]:
        inventory = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            identity = path.lstat()
            if stat.S_ISREG(identity.st_mode):
                inventory.append((relative, "file", path.read_bytes()))
            elif stat.S_ISDIR(identity.st_mode):
                inventory.append((relative, "directory", None))
            else:
                inventory.append((relative, "other", None))
        return tuple(inventory)

    before = namespace_bytes()
    resumed_model = TinyRegressor()
    resumed_optimizer = optimizer_module.SmolVLAAdamW(optimizer.config)
    recorded = {
        "step": 1,
        "path": str(committed.path),
        "metadata_sha256": committed.metadata_sha256,
        "model_sha256": committed.model_sha256,
        "optimizer_sha256": committed.optimizer_sha256,
    }
    try:
        module.load_latest_training_checkpoint(
            model=resumed_model,
            optimizer=resumed_optimizer,
            checkpoint_root=root,
            trainable_names=names,
            expected_run_config_sha256="a" * 64,
            expected_selected_steps=10,
            expected_effective_batch_size=8,
            expected_checkpoint_interval=1,
            expected_last_checkpoint=recorded,
            allowed_uncommitted_step=2,
        )
    except ValueError as error:
        assert "invalid candidates" in str(error)
    else:
        raise AssertionError("multiply-invalid checkpoint namespace was accepted")

    assert namespace_bytes() == before


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


def test_checkpoint_resume_stays_bound_to_the_original_output_directory(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_release_t3b_training_lock",
            "load_latest_training_checkpoint",
            "save_training_checkpoint",
        ],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    output = tmp_path / "t3b"
    output.mkdir()
    root = output / "checkpoints"
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    for step in range(1, 5):
        optimizer.update(
            model,
            {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
        )
        mx.eval(model.parameters(), optimizer.state)
        module.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=root,
            state=_tiny_checkpoint_state(module, optimizer, step=step),
            trainable_names=names,
            keep_last=10,
        )

    lease = module._acquire_t3b_training_lock(output)
    detached = tmp_path / "detached-t3b"
    output.rename(detached)
    output.mkdir()
    replacement_root = output / "checkpoints"
    replacement_root.mkdir()
    (replacement_root / "owner.txt").write_text("competitor\n", encoding="utf-8")
    try:
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
            checkpoint_parent_descriptor=lease.output_descriptor,
            expected_checkpoint_parent_snapshot=lease.output_snapshot,
        )
    finally:
        module._release_t3b_training_lock(lease)

    assert loaded.state.completed_step == 4
    assert (detached / "checkpoints" / "latest.json").is_file()
    assert not (detached / "checkpoints" / "step-000001").exists()
    assert (replacement_root / "owner.txt").read_text(encoding="utf-8") == (
        "competitor\n"
    )
    assert tuple(path.name for path in replacement_root.iterdir()) == ("owner.txt",)


def test_zero_step_checkpoint_recovery_stays_bound_to_original_output(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_prepare_zero_step_checkpoint_replay",
            "_release_t3b_training_lock",
        ],
    )
    output = tmp_path / "t3b"
    partial = output / "checkpoints" / (".step-000001." + "a" * 24)
    partial.mkdir(parents=True)
    (partial / "owner.txt").write_text("original\n", encoding="utf-8")
    lease = module._acquire_t3b_training_lock(output)
    detached = tmp_path / "detached-t3b"
    output.rename(detached)
    replacement_partial = output / "checkpoints" / (".step-000001." + "b" * 24)
    replacement_partial.mkdir(parents=True)
    (replacement_partial / "owner.txt").write_text("competitor\n", encoding="utf-8")
    try:
        recovered = module._prepare_zero_step_checkpoint_replay(
            output / "checkpoints",
            output_dir=output,
            output_descriptor=lease.output_descriptor,
            expected_output_snapshot=lease.output_snapshot,
        )
    finally:
        module._release_t3b_training_lock(lease)

    assert recovered is not None and len(recovered) == 1
    recovered_path = detached / recovered[0]
    assert (recovered_path / "owner.txt").read_text(encoding="utf-8") == "original\n"
    assert not (detached / "checkpoints" / partial.name).exists()
    assert (replacement_partial / "owner.txt").read_text(encoding="utf-8") == (
        "competitor\n"
    )
    assert not (output / "checkpoint-recoveries").exists()


def test_later_cadence_checkpoint_staging_is_quarantined_and_retry_safe(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_bind_t3b_checkpoint_root",
            "_prepare_zero_step_checkpoint_replay",
            "_release_t3b_training_lock",
        ],
    )
    output = tmp_path / "t3b"
    checkpoint_root = output / "checkpoints"
    (checkpoint_root / "step-000001").mkdir(parents=True)
    (checkpoint_root / "latest.json").write_text("{}\n", encoding="utf-8")
    partial = checkpoint_root / (".step-000100." + "a" * 24)
    partial.mkdir()
    (partial / "model.safetensors").write_bytes(b"partial model\n")
    lease = module._acquire_t3b_training_lock(output)
    try:
        module._bind_t3b_checkpoint_root(lease, allow_existing=True)
        first = module._prepare_zero_step_checkpoint_replay(
            checkpoint_root,
            output_dir=output,
            output_descriptor=lease.output_descriptor,
            expected_output_snapshot=lease.output_snapshot,
            checkpoint_root_descriptor=lease.checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=lease.checkpoint_root_snapshot,
            expected_staging_step=100,
            allow_published_entries=True,
        )
        second = module._prepare_zero_step_checkpoint_replay(
            checkpoint_root,
            output_dir=output,
            output_descriptor=lease.output_descriptor,
            expected_output_snapshot=lease.output_snapshot,
            checkpoint_root_descriptor=lease.checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=lease.checkpoint_root_snapshot,
            expected_staging_step=100,
            allow_published_entries=True,
        )
    finally:
        module._release_t3b_training_lock(lease)

    assert first == second
    assert first is not None and len(first) == 1
    recovered = output / first[0]
    assert recovered.name.startswith("step-000100-partial-")
    assert (recovered / "model.safetensors").read_bytes() == b"partial model\n"
    assert not partial.exists()
    assert (checkpoint_root / "step-000001").is_dir()
    assert (checkpoint_root / "latest.json").is_file()


def test_checkpoint_transaction_debris_is_quarantined_and_retry_safe(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_bind_t3b_checkpoint_root",
            "_prepare_zero_step_checkpoint_replay",
            "_release_t3b_training_lock",
        ],
    )
    output = tmp_path / "t3b"
    checkpoint_root = output / "checkpoints"
    (checkpoint_root / "step-000001").mkdir(parents=True)
    (checkpoint_root / "latest.json").write_text("{}\n", encoding="utf-8")
    staged = {
        ".step-000100." + "a" * 24: ("partial", b"partial\n"),
        ".discarded-step-000100-000001": ("discarded", b"discarded\n"),
        ".recovery-step-000100-000001": ("replaced", b"replaced\n"),
        ".pruned-step-000001-000001": ("pruned", b"pruned\n"),
    }
    for name, (_, payload) in staged.items():
        candidate = checkpoint_root / name
        candidate.mkdir()
        (candidate / "owner.bin").write_bytes(payload)
    pointer_stage = checkpoint_root / (".latest.json." + "b" * 24)
    pointer_stage.write_bytes(b'{"step":100}\n')
    failed_publish = checkpoint_root / (
        ".step-000100.publication-failed-" + "c" * 24
    )
    failed_publish.mkdir()
    (failed_publish / "owner.bin").write_bytes(b"failed publish\n")
    pointer_previous = checkpoint_root / (
        ".latest.json.previous-" + "d" * 24
    )
    pointer_previous.write_bytes(b'{"step":1}\n')

    lease = module._acquire_t3b_training_lock(output)
    try:
        module._bind_t3b_checkpoint_root(lease, allow_existing=True)
        first = module._prepare_zero_step_checkpoint_replay(
            checkpoint_root,
            output_dir=output,
            output_descriptor=lease.output_descriptor,
            expected_output_snapshot=lease.output_snapshot,
            checkpoint_root_descriptor=lease.checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=lease.checkpoint_root_snapshot,
            expected_staging_step=100,
            allow_published_entries=True,
        )
        second = module._prepare_zero_step_checkpoint_replay(
            checkpoint_root,
            output_dir=output,
            output_descriptor=lease.output_descriptor,
            expected_output_snapshot=lease.output_snapshot,
            checkpoint_root_descriptor=lease.checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=lease.checkpoint_root_snapshot,
            expected_staging_step=100,
            allow_published_entries=True,
        )
    finally:
        module._release_t3b_training_lock(lease)

    assert first == second
    assert first is not None and len(first) == 7
    recovered = {Path(name).name: output / name for name in first}
    for original, (kind, payload) in staged.items():
        assert not (checkpoint_root / original).exists()
        matching = [
            path
            for name, path in recovered.items()
            if name.startswith("step-") and f"-{kind}-" in name
        ]
        assert len(matching) == 1
        assert (matching[0] / "owner.bin").read_bytes() == payload
    pointer_matches = [
        path for name, path in recovered.items() if name.startswith("latest-pointer-partial-")
    ]
    assert len(pointer_matches) == 1
    assert pointer_matches[0].read_bytes() == b'{"step":100}\n'
    assert not pointer_stage.exists()
    failed_matches = [
        path
        for name, path in recovered.items()
        if name.startswith("step-000100-publication-failed-")
    ]
    assert len(failed_matches) == 1
    assert (failed_matches[0] / "owner.bin").read_bytes() == b"failed publish\n"
    previous_matches = [
        path
        for name, path in recovered.items()
        if name.startswith("latest-pointer-previous-")
    ]
    assert len(previous_matches) == 1
    assert previous_matches[0].read_bytes() == b'{"step":1}\n'


def test_resume_output_staging_is_quarantined_and_retry_safe(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_reconcile_t3b_resume_output_staging",
            "_release_t3b_training_lock",
        ],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    source_stage = output / (".source-checkpoint-" + "a" * 24)
    export_stage = output / (".export." + "b" * 24)
    adapter_stage = output / (".adapter-stage-" + "c" * 24)
    tokenizer_stage = output / (".tokenizer-snapshot-" + "d" * 24)
    for path, payload in (
        (source_stage, b"source\n"),
        (export_stage, b"export\n"),
        (adapter_stage, b"adapter\n"),
        (tokenizer_stage, b"tokenizer\n"),
    ):
        path.mkdir()
        (path / "owner.bin").write_bytes(payload)
    run_stage = output / (".run.json." + "e" * 24)
    run_stage.write_bytes(b'{"partial":true}\n')
    metrics_stage = output / (".metrics.csv." + "f" * 24)
    metrics_stage.write_bytes(b"step,loss\n")
    pid_stage = output / (".training.pid." + "0" * 24)
    pid_stage.write_bytes(b'{"pid":123}\n')
    run_previous = output / (".run.json.previous-" + "1" * 24)
    run_previous.write_bytes(b'{"status":"running"}\n')
    adapter_failed = output / (
        ".adapter.safetensors.publication-failed-" + "2" * 24
    )
    adapter_failed.write_bytes(b"competitor adapter\n")

    lease = module._acquire_t3b_training_lock(output)
    try:
        first = module._reconcile_t3b_resume_output_staging(lease)
        second = module._reconcile_t3b_resume_output_staging(lease)
    finally:
        module._release_t3b_training_lock(lease)

    assert first == second
    assert len(first) == 9
    for path in (
        source_stage,
        export_stage,
        adapter_stage,
        tokenizer_stage,
        run_stage,
        metrics_stage,
        pid_stage,
        run_previous,
        adapter_failed,
    ):
        assert not path.exists()
    recovered = {Path(name).name: output / name for name in first}
    expected_payloads = {
        "source-checkpoint": b"source\n",
        "export-stage": b"export\n",
        "adapter-stage": b"adapter\n",
        "tokenizer-snapshot": b"tokenizer\n",
    }
    for prefix, payload in expected_payloads.items():
        matches = [path for name, path in recovered.items() if name.startswith(prefix)]
        assert len(matches) == 1
        assert (matches[0] / "owner.bin").read_bytes() == payload
    run_matches = [
        path for name, path in recovered.items() if name.startswith("run-json-partial-")
    ]
    assert len(run_matches) == 1
    assert run_matches[0].read_bytes() == b'{"partial":true}\n'
    metrics_matches = [
        path for name, path in recovered.items() if name.startswith("metrics-csv-partial-")
    ]
    pid_matches = [
        path for name, path in recovered.items() if name.startswith("training-pid-partial-")
    ]
    assert len(metrics_matches) == len(pid_matches) == 1
    assert metrics_matches[0].read_bytes() == b"step,loss\n"
    assert pid_matches[0].read_bytes() == b'{"pid":123}\n'
    run_previous_matches = [
        path for name, path in recovered.items() if name.startswith("run-json-previous-")
    ]
    adapter_failed_matches = [
        path
        for name, path in recovered.items()
        if name.startswith("adapter-safetensors-publication-failed-")
    ]
    assert len(run_previous_matches) == len(adapter_failed_matches) == 1
    assert run_previous_matches[0].read_bytes() == b'{"status":"running"}\n'
    assert adapter_failed_matches[0].read_bytes() == b"competitor adapter\n"


def test_loaded_end_checkpoint_reconciles_stale_run_metadata(tmp_path: Path) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_reconcile_loaded_checkpoint_run_document"],
    )
    checkpoint = SimpleNamespace(
        path=tmp_path / "checkpoints" / "step-003000",
        state=SimpleNamespace(completed_step=3000),
        metadata_sha256="a" * 64,
        model_sha256="b" * 64,
        optimizer_sha256="c" * 64,
    )
    stale = {
        "status": "running",
        "checkpoint_count": 30,
        "last_checkpoint": {"step": 2900},
    }

    reconciled = module._reconcile_loaded_checkpoint_run_document(
        stale,
        checkpoint=checkpoint,
        selected_steps=3000,
        checkpoint_interval=100,
    )

    assert reconciled["checkpoint_count"] == 31
    assert reconciled["last_checkpoint"] == {
        "step": 3000,
        "path": str(checkpoint.path),
        "metadata_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "optimizer_sha256": "c" * 64,
    }


def test_atomic_run_state_replaces_complete_json(tmp_path: Path) -> None:
    module = __import__("training.finetune", fromlist=["write_run_state"])
    path = tmp_path / "run.json"

    first_hash = module.write_run_state(path, {"status": "running", "step": 0})
    second_hash = module.write_run_state(path, {"status": "complete", "step": 7})

    assert first_hash != second_hash
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "complete", "step": 7}
    assert len(second_hash) == 64


def test_bound_run_state_compare_and_swap_preserves_a_replacement(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_write_run_state_with_binding"],
    )
    path = tmp_path / "run.json"
    _, binding = module._write_run_state_with_binding(
        path,
        {"status": "running", "step": 1},
        expected_destination_snapshot=None,
    )
    original = tmp_path / "run.original.json"
    path.rename(original)
    competitor = b'{"owner":"competitor"}\n'
    path.write_bytes(competitor)
    competitor_inode = path.stat().st_ino

    try:
        module._write_run_state_with_binding(
            path,
            {"status": "running", "step": 2},
            expected_destination_snapshot=binding,
        )
    except RuntimeError as error:
        assert "destination changed" in str(error)
    else:
        raise AssertionError("run-state CAS overwrote a replacement")

    assert path.stat().st_ino == competitor_inode
    assert path.read_bytes() == competitor
    assert json.loads(original.read_text(encoding="utf-8"))["step"] == 1


def test_fresh_run_never_overwrites_a_late_run_document(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "_run_lora_finetune_impl"],
    )
    output = tmp_path / "fresh"
    config = module.FineTuneConfig(
        output_dir=output,
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    competitor = b'{"owner":"competitor"}\n'

    def inject() -> None:
        (output / "run.json").write_bytes(competitor)

    updates = _install_minimal_fresh_run_fakes(
        module,
        config,
        monkeypatch,
        inject,
    )
    try:
        module._run_lora_finetune_impl(config)
    except (RuntimeError, FileExistsError) as error:
        assert "state appeared unexpectedly" in str(error) or (
            "destination appeared" in str(error)
        )
    else:
        raise AssertionError("fresh run overwrote a late run.json competitor")

    assert updates == []
    assert (output / "run.json").read_bytes() == competitor


def test_fresh_run_never_overwrites_a_late_budget_document(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "_run_lora_finetune_impl"],
    )
    output = tmp_path / "fresh"
    config = module.FineTuneConfig(
        output_dir=output,
        budget_mode=module.FIXED_BUDGET_MODE,
    )
    competitor = b'{"owner":"competitor"}\n'

    def inject() -> None:
        (output / "budget.json").write_bytes(competitor)

    updates = _install_minimal_fresh_run_fakes(
        module,
        config,
        monkeypatch,
        inject,
    )
    try:
        module._run_lora_finetune_impl(config)
    except FileExistsError as error:
        assert "budget artifact" in str(error)
    else:
        raise AssertionError("fresh run overwrote a late budget.json competitor")

    assert updates == []
    assert (output / "budget.json").read_bytes() == competitor
    assert not (output / "run.json").exists()


def test_atomic_run_state_uses_bound_parent_after_path_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_acquire_t3b_training_lock", "write_run_state"],
    )
    output = tmp_path / "t3b"
    output.mkdir()
    lease = module._acquire_t3b_training_lock(output)
    detached = tmp_path / "detached-t3b"
    real_create = module._create_staged_file_at
    swapped = False

    def replace_parent_before_staging(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            output.rename(detached)
            output.mkdir()
        return real_create(*args, **kwargs)

    monkeypatch.setattr(module, "_create_staged_file_at", replace_parent_before_staging)
    try:
        module.write_run_state(
            output / "run.json",
            {"status": "running"},
            parent_descriptor=lease.output_descriptor,
            expected_parent_snapshot=lease.output_snapshot,
        )
    finally:
        module._release_t3b_training_lock(lease)

    assert tuple(output.iterdir()) == ()
    assert json.loads((detached / "run.json").read_text(encoding="utf-8")) == {
        "status": "running"
    }


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
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["sha256"] == digest
    assert "scope" not in metadata
    assert list(tmp_path.glob(".adapter*")) == []


def test_expert_adapter_checkpoint_records_nonlegacy_scope(tmp_path: Path) -> None:
    module = __import__("training.finetune", fromlist=["_save_adapter_checkpoint"])
    model = TinyRegressor()
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    report = SimpleNamespace(
        scope="expert_only",
        rank=1,
        alpha=1.0,
        dropout=0.0,
        adapter_count=1,
        trainable_names=names,
        trainable_scalar_count=2,
    )
    path = tmp_path / "adapter.safetensors"

    module._save_adapter_checkpoint(model, path, lora_report=report)

    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["scope"] == "expert_only"


def test_adapter_checkpoint_exact_retry_reuses_the_published_pair(
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

    first = module._save_adapter_checkpoint(model, path, lora_report=report)
    adapter_identity = path.stat()
    metadata_identity = path.with_suffix(".json").stat()
    second = module._save_adapter_checkpoint(model, path, lora_report=report)

    assert second == first
    assert path.stat().st_ino == adapter_identity.st_ino
    assert path.with_suffix(".json").stat().st_ino == metadata_identity.st_ino


def test_adapter_checkpoint_never_clobbers_an_inserted_target(
    tmp_path: Path,
    monkeypatch,
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
    real_rename = module._rename_entry_no_clobber_at
    inserted = False

    def insert_target_before_publish(**kwargs):
        nonlocal inserted
        if not inserted and kwargs["destination_name"] == path.name:
            inserted = True
            path.write_bytes(b"competitor adapter")
        return real_rename(**kwargs)

    monkeypatch.setattr(
        module,
        "_rename_entry_no_clobber_at",
        insert_target_before_publish,
    )
    try:
        module._save_adapter_checkpoint(model, path, lora_report=report)
    except FileExistsError:
        pass
    else:
        raise AssertionError("adapter publication clobbered an inserted target")

    assert path.read_bytes() == b"competitor adapter"
    assert not path.with_suffix(".json").exists()
    assert not tuple(tmp_path.glob(".adapter-stage-*"))


def test_adapter_staging_uses_its_bound_inode_for_first_write(
    tmp_path: Path,
    monkeypatch,
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
    real_save = module.mx.save_safetensors
    replacement: Path | None = None
    original: Path | None = None
    swapped = False

    def replace_stage_before_first_write(target, tensors):
        nonlocal replacement, original, swapped
        if not swapped:
            swapped = True
            replacement = next(tmp_path.glob(".adapter-stage-*"))
            original = replacement.with_name(f"{replacement.name}.original")
            replacement.rename(original)
            replacement.mkdir()
            (replacement / "owner.txt").write_text("competitor\n", encoding="utf-8")
        return real_save(target, tensors)

    monkeypatch.setattr(module.mx, "save_safetensors", replace_stage_before_first_write)
    try:
        module._save_adapter_checkpoint(model, path, lora_report=report)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("renamed adapter stage was accepted")

    assert replacement is not None and original is not None
    assert tuple(item.name for item in replacement.iterdir()) == ("owner.txt",)
    assert (original / path.name).is_file()
    assert not path.exists()


def test_adapter_first_child_write_never_follows_an_inserted_symlink(
    tmp_path: Path,
    monkeypatch,
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
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"guard\n")
    real_create = module._create_exclusive_child_file_at
    inserted = False

    def insert_symlink(parent_descriptor, name, **kwargs):
        nonlocal inserted
        if not inserted and name == path.name:
            inserted = True
            os.symlink(outside, name, dir_fd=parent_descriptor)
        return real_create(parent_descriptor, name, **kwargs)

    monkeypatch.setattr(module, "_create_exclusive_child_file_at", insert_symlink)
    try:
        module._save_adapter_checkpoint(model, path, lora_report=report)
    except FileExistsError:
        pass
    else:
        raise AssertionError("adapter serializer followed an inserted child symlink")

    assert outside.read_bytes() == b"guard\n"
    assert not path.exists()


def test_finetune_provenance_covers_repo_and_installed_runtime_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["finetune_implementation_hashes"],
    )
    hashes = module.finetune_implementation_hashes()
    assert {
        "mlx_smolvla/__init__.py",
        "training/evaluation.py",
        "training/reference_export.py",
        "installed/lerobot/datasets/dataset_reader.py",
        "installed/lerobot/datasets/dataset_metadata.py",
        "installed/lerobot/datasets/factory.py",
        "installed/lerobot/datasets/lerobot_dataset.py",
        "installed/lerobot/datasets/sampler.py",
        "installed/lerobot/utils/collate.py",
        "distribution/lerobot/RECORD",
        "distribution/datasets/RECORD",
        "distribution/pyarrow/RECORD",
        "distribution/torch/RECORD",
        "distribution/transformers/RECORD",
        "distribution/tokenizers/RECORD",
        "distribution/av/RECORD",
    } <= set(hashes)

    runtime_source = tmp_path / "runtime.py"
    runtime_source.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_t3b_installed_runtime_inputs",
        lambda: {"installed/test/runtime.py": runtime_source},
    )
    monkeypatch.setattr(module, "_T3B_RUNTIME_DISTRIBUTIONS", ())
    before = module.finetune_implementation_hashes()
    runtime_source.write_text("VERSION = 2\n", encoding="utf-8")
    after = module.finetune_implementation_hashes()
    assert before["installed/test/runtime.py"] != after["installed/test/runtime.py"]


def test_distribution_payload_hash_rejects_recorded_file_mutation(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_hash_distribution_recorded_files"],
    )
    site_packages = tmp_path / "venv" / "lib" / "site-packages"
    package = site_packages / "example_runtime"
    dist_info = site_packages / "example_runtime-1.0.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir()
    runtime = package / "runtime.py"
    runtime.write_bytes(b"VALUE = 1\n")
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(runtime.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    record = dist_info / "RECORD"
    record.write_text(
        "example_runtime/runtime.py,sha256="
        f"{digest},{runtime.stat().st_size}\n"
        "example_runtime-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    before = module._hash_distribution_recorded_files(
        record,
        install_root=site_packages,
        allowed_root=tmp_path / "venv",
    )
    assert len(before) == 64

    runtime.write_bytes(b"VALUE = 2\n")
    try:
        module._hash_distribution_recorded_files(
            record,
            install_root=site_packages,
            allowed_root=tmp_path / "venv",
        )
    except RuntimeError as error:
        assert "differs from RECORD" in str(error)
    else:
        raise AssertionError("mutated RECORD-listed runtime bytes were accepted")


def test_distribution_payload_hash_rejects_a_recorded_symlink(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_hash_distribution_recorded_files"],
    )
    environment = tmp_path / "venv"
    site_packages = environment / "lib" / "site-packages"
    package = site_packages / "example_runtime"
    dist_info = site_packages / "example_runtime-1.0.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir()
    payload = b"VALUE = 1\n"
    target = environment / "same-bytes.py"
    target.write_bytes(payload)
    runtime = package / "runtime.py"
    runtime.symlink_to(target)
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode(
        "ascii"
    ).rstrip("=")
    record = dist_info / "RECORD"
    record.write_text(
        "example_runtime/runtime.py,sha256="
        f"{digest},{len(payload)}\n"
        "example_runtime-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )

    try:
        module._hash_distribution_recorded_files(
            record,
            install_root=site_packages,
            allowed_root=environment,
        )
    except FileNotFoundError as error:
        assert "symlink" in str(error) or "unsafe" in str(error)
    else:
        raise AssertionError("symlinked RECORD-listed runtime source was accepted")


def test_distribution_inventory_rejects_unrecorded_import_precedence_file(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_hash_distribution_package_inventory"],
    )
    site_packages = tmp_path / "venv" / "lib" / "site-packages"
    package = site_packages / "example_runtime"
    dist_info = site_packages / "example_runtime-1.0.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir()
    source = package / "__init__.py"
    source.write_bytes(b"VALUE = 1\n")
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(source.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    record = dist_info / "RECORD"
    record.write_text(
        "example_runtime/__init__.py,sha256="
        f"{digest},{source.stat().st_size}\n"
        "example_runtime-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    before = module._hash_distribution_package_inventory(
        record,
        install_root=site_packages,
        allowed_root=tmp_path / "venv",
    )
    assert len(before) == 64
    py_compile.compile(str(source), doraise=True)
    after_lazy_bytecode = module._hash_distribution_package_inventory(
        record,
        install_root=site_packages,
        allowed_root=tmp_path / "venv",
    )
    assert after_lazy_bytecode == before

    unrecorded = package / "__init__.cpython-312-darwin.so"
    unrecorded.write_bytes(b"unrecorded native loader")
    try:
        module._hash_distribution_package_inventory(
            record,
            install_root=site_packages,
            allowed_root=tmp_path / "venv",
        )
    except RuntimeError as error:
        assert "unrecorded file" in str(error)
    else:
        raise AssertionError("unrecorded import-precedence file was accepted")


def test_t3b_input_resolution_rejects_symlinked_configured_cache_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "_resolve_t3b_input_paths"],
    )
    real_cache = tmp_path / "real-cache"
    real_native = tmp_path / "real-native"
    real_cache.mkdir()
    real_native.mkdir()
    cache_alias = tmp_path / "cache-alias"
    native_alias = tmp_path / "native-alias"
    cache_alias.symlink_to(real_cache, target_is_directory=True)
    native_alias.symlink_to(real_native, target_is_directory=True)
    resolver_calls: list[Path] = []
    monkeypatch.setattr(
        module,
        "resolve_base_checkpoint",
        lambda path: resolver_calls.append(Path(path)),
    )
    config = module.FineTuneConfig(
        output_dir=tmp_path / "run",
        cache_dir=cache_alias,
        native_cache=native_alias,
    )

    try:
        module._resolve_t3b_input_paths(config, None)
    except FileNotFoundError as error:
        assert "cache" in str(error).lower() or "ancestry" in str(error).lower()
    else:
        raise AssertionError("symlinked configured cache root was accepted")
    assert resolver_calls == []


def test_resume_process_identity_requires_the_exact_bound_pid_document(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_snapshot_regular_file",
            "_validate_resume_process_identity",
            "write_run_state",
        ],
    )
    identity = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-training-process",
        "pid": 123,
        "parent_pid": 1,
        "started_at_ns": 1_788_264_000_000_000_000,
        "working_directory": "/tmp/work",
        "executable": "/usr/bin/python3",
        "launch_config": "/tmp/run/launch.json",
        "launch_file_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "run_config_sha256": "c" * 64,
        "training_log": {
            "format_version": 1,
            "artifact_type": "smolvla-mlx-training-log",
            "file": "training.log",
            "device": 1,
            "inode": 2,
            "created_at_ns": 1_788_264_000_000_000_000,
            "header_sha256": "d" * 64,
        },
    }
    path = tmp_path / "training.pid"
    digest = module.write_run_state(path, identity)
    snapshot = module._snapshot_regular_file(
        path,
        label="test process identity",
        capture_payload=True,
    )
    run_document = {
        "process": {
            **identity,
            "identity_file": "training.pid",
            "identity_sha256": digest,
        }
    }
    module._validate_resume_process_identity(run_document, snapshot)

    try:
        module._validate_resume_process_identity(run_document, None)
    except FileNotFoundError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing bound PID document was accepted")

    module.write_run_state(path, {**identity, "pid": 999})
    tampered = module._snapshot_regular_file(
        path,
        label="tampered process identity",
        capture_payload=True,
    )
    try:
        module._validate_resume_process_identity(run_document, tampered)
    except ValueError as error:
        assert "digest" in str(error) or "content" in str(error)
    else:
        raise AssertionError("tampered bound PID document was accepted")


def test_resume_pid_rotation_recovers_a_crash_before_run_publication(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_backup_resume_process_identity",
            "_reconcile_resume_process_identity",
            "_release_t3b_training_lock",
            "_snapshot_regular_file_at",
            "write_run_state",
        ],
    )
    output = tmp_path / "run"
    output.mkdir()
    old_identity = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-training-process",
        "pid": 123,
        "parent_pid": 1,
        "started_at_ns": 1_788_264_000_000_000_000,
        "working_directory": "/tmp/work",
        "executable": "/usr/bin/python3",
        "launch_config": str(output / "launch.json"),
        "launch_file_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "run_config_sha256": "c" * 64,
    }
    old_digest = module.write_run_state(output / "training.pid", old_identity)
    module.write_run_state(
        output / "run.json",
        {
            "process": {
                **old_identity,
                "identity_file": "training.pid",
                "identity_sha256": old_digest,
            }
        },
    )
    lease = module._acquire_t3b_training_lock(output)
    try:
        old_snapshot = module._snapshot_regular_file_at(
            lease.output_descriptor,
            "training.pid",
            label="old process identity",
            capture_payload=True,
        )
        previous_name = module._backup_resume_process_identity(lease, old_snapshot)
        previous_path = output / "process-identity-recoveries" / previous_name
        assert previous_path.is_file()
        assert hashlib.sha256(previous_path.read_bytes()).hexdigest() == old_digest
        module.write_run_state(
            output / "training.pid",
            {**old_identity, "pid": 999, "started_at_ns": old_identity["started_at_ns"] + 1},
            parent_descriptor=lease.output_descriptor,
            expected_parent_snapshot=lease.output_snapshot,
        )
    finally:
        module._release_t3b_training_lock(lease)

    lease = module._acquire_t3b_training_lock(output)
    try:
        run_snapshot = module._snapshot_regular_file_at(
            lease.output_descriptor,
            "run.json",
            label="unchanged run metadata",
            capture_payload=True,
        )
        recovered = module._reconcile_resume_process_identity(lease, run_snapshot)
        restored = module._snapshot_regular_file_at(
            lease.output_descriptor,
            "training.pid",
            label="restored process identity",
            capture_payload=True,
        )
        assert restored.sha256 == old_digest
        assert recovered
        assert all((output / path).is_file() for path in recovered)
        recovery_root = output / "process-identity-recoveries"
        uncommitted = tuple(recovery_root.glob("training-pid-uncommitted-*"))
        assert len(uncommitted) == 1
        assert recovered == (
            f"process-identity-recoveries/{uncommitted[0].name}",
        )
        assert previous_path.name not in {Path(path).name for path in recovered}
    finally:
        module._release_t3b_training_lock(lease)


def test_resume_restores_uniquely_bound_previous_state_generations(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_release_t3b_training_lock",
            "_restore_missing_t3b_previous_generations",
            "write_run_state",
        ],
    )
    output = tmp_path / "run"
    output.mkdir()
    expected: dict[str, tuple[bytes, tuple[int, int]]] = {}
    for name, value in (
        ("run.json", {"kind": "run"}),
        ("training.pid", {"kind": "pid"}),
        ("budget.json", {"kind": "budget"}),
        ("launch.json", {"kind": "launch"}),
    ):
        module.write_run_state(output / name, value)
        identity = (output / name).stat()
        expected[name] = (
            (output / name).read_bytes(),
            (identity.st_dev, identity.st_ino),
        )
        (output / name).rename(output / f".{name}.previous-{'a' * 24}")
    (output / "training.log").write_bytes(b"bound training log\n")
    log_identity = (output / "training.log").stat()
    expected["training.log"] = (
        (output / "training.log").read_bytes(),
        (log_identity.st_dev, log_identity.st_ino),
    )
    (output / "training.log").rename(
        output / f".training.log.previous-{'b' * 24}"
    )

    lease = module._acquire_t3b_training_lock(output)
    try:
        restored = module._restore_missing_t3b_previous_generations(lease)
    finally:
        module._release_t3b_training_lock(lease)

    assert set(restored) == set(expected)
    for name, (payload, identity) in expected.items():
        restored_path = output / name
        assert restored_path.read_bytes() == payload
        restored_identity = restored_path.stat()
        assert (restored_identity.st_dev, restored_identity.st_ino) == identity
    assert not tuple(output.glob(".*.previous-*"))


def test_metrics_recovery_inventory_adopts_only_exact_regular_files(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_metrics_recovery_inventory",
            "_release_t3b_training_lock",
        ],
    )
    output = tmp_path / "run"
    output.mkdir()
    (output / "metrics.recovery-000002.csv").write_bytes(b"second\n")
    (output / "metrics.recovery-000001.csv").write_bytes(b"first\n")
    lease = module._acquire_t3b_training_lock(output)
    try:
        assert module._metrics_recovery_inventory(lease) == (
            "metrics.recovery-000001.csv",
            "metrics.recovery-000002.csv",
        )
    finally:
        module._release_t3b_training_lock(lease)

    (output / "metrics.recovery-not-numbered.csv").write_bytes(b"unsafe\n")
    lease = module._acquire_t3b_training_lock(output)
    try:
        try:
            module._metrics_recovery_inventory(lease)
        except FileExistsError as error:
            assert "metrics recovery" in str(error)
        else:
            raise AssertionError("unsafe metrics recovery name was accepted")
    finally:
        module._release_t3b_training_lock(lease)

def test_checkpoint_state_decoder_rejects_noncanonical_json_scalars() -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_checkpoint_state_from_dict"],
    )
    valid = {
        "completed_step": 1,
        "selected_steps": 10,
        "smoothed_loss": 1.0,
        "elapsed_training_seconds": 1.0,
        "peak_memory_bytes": 1234,
        "samples_consumed": 8,
        "flow_draw_count": 8,
        "last_update": {
            "loss": 1.0,
            "learning_rate": 1e-4,
            "gradient_norm": 1.0,
            "clip_coefficient": 1.0,
            "seconds": 1.0,
        },
        "run_config_sha256": "a" * 64,
    }
    malformed = []
    for field, value in (
        ("completed_step", True),
        ("selected_steps", "10"),
        ("peak_memory_bytes", -1),
        ("run_config_sha256", "A" * 64),
        ("smoothed_loss", 1),
    ):
        candidate = json.loads(json.dumps(valid))
        candidate[field] = value
        malformed.append(candidate)
    extra_update = json.loads(json.dumps(valid))
    extra_update["last_update"]["extra"] = 1.0
    malformed.append(extra_update)
    invalid_clip = json.loads(json.dumps(valid))
    invalid_clip["last_update"]["clip_coefficient"] = 1.1
    malformed.append(invalid_clip)

    for candidate in malformed:
        try:
            module._checkpoint_state_from_dict(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"noncanonical checkpoint state was accepted: {candidate}")


def test_self_rehashed_checkpoint_rejects_noncanonical_state_metadata(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_read_checkpoint_directory", "save_training_checkpoint"],
    )
    optimizer_module = __import__(
        "training.optimizer", fromlist=["SmolVLAAdamW", "SmolVLAOptimizerConfig"]
    )
    model = TinyRegressor()
    optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    saved = module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=tmp_path / "checkpoints",
        state=_tiny_checkpoint_state(module, optimizer, step=1),
        trainable_names=names,
    )
    metadata_path = saved.path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["state"]["completed_step"] = True
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        module._read_checkpoint_directory(
            saved.path,
            expected_run_config_sha256="a" * 64,
            trainable_names=names,
            expected_model_tensors=dict(tree_flatten(model.trainable_parameters())),
            expected_optimizer_tensors=dict(tree_flatten(optimizer.state)),
        )
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("self-rehashed noncanonical checkpoint was accepted")


def test_resume_rejects_a_self_rehashed_committed_checkpoint_binding(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["load_latest_training_checkpoint", "save_training_checkpoint"],
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    saved = module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=root,
        state=_tiny_checkpoint_state(module, optimizer, step=1),
        trainable_names=names,
    )
    committed = {
        "step": 1,
        "path": str(saved.path),
        "metadata_sha256": saved.metadata_sha256,
        "model_sha256": saved.model_sha256,
        "optimizer_sha256": saved.optimizer_sha256,
    }
    model_path = saved.path / "model.safetensors"
    altered = mx.load(str(model_path))
    mx.eval(altered)
    first_name = next(iter(altered))
    altered[first_name] = altered[first_name] + mx.array(
        1.0,
        dtype=altered[first_name].dtype,
    )
    mx.eval(altered)
    mx.save_safetensors(str(model_path), altered)
    metadata_path = saved.path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model"]["sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latest_before = (root / "latest.json").read_bytes()
    resumed_model = TinyRegressor()
    resumed_optimizer = optimizer_module.SmolVLAAdamW(optimizer.config)
    model_before = np.asarray(resumed_model.proj.weight).copy()

    try:
        module.load_latest_training_checkpoint(
            model=resumed_model,
            optimizer=resumed_optimizer,
            checkpoint_root=root,
            trainable_names=names,
            expected_run_config_sha256="a" * 64,
            expected_selected_steps=10,
            expected_effective_batch_size=8,
            expected_checkpoint_interval=1,
            expected_last_checkpoint=committed,
            allowed_uncommitted_step=2,
        )
    except ValueError as error:
        assert "recorded checkpoint" in str(error)
    else:
        raise AssertionError("self-rehashed committed checkpoint was restored")

    np.testing.assert_array_equal(np.asarray(resumed_model.proj.weight), model_before)
    assert resumed_optimizer.step_index == 0
    assert (root / "latest.json").read_bytes() == latest_before


def test_forged_checkpoint_metrics_boundary_causes_zero_resume_mutations(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "MetricsWriter",
            "_snapshot_regular_file_at",
            "load_latest_training_checkpoint",
            "save_training_checkpoint",
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
    optimizer.update(
        model,
        {"proj": {"weight": mx.array([[0.1, -0.2]], dtype=mx.float32)}},
    )
    mx.eval(model.parameters(), optimizer.state)
    state = _tiny_checkpoint_state(module, optimizer, step=1)
    saved = module.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=root,
        state=state,
        trainable_names=names,
        keep_last=10,
    )
    metrics_path = tmp_path / "metrics.csv"
    with module.MetricsWriter(metrics_path) as writer:
        writer.write(
            step=1,
            loss=state.last_update.loss,
            smoothed_loss=state.smoothed_loss,
            learning_rate=state.last_update.learning_rate,
            gradient_norm=state.last_update.gradient_norm,
            clip_coefficient=state.last_update.clip_coefficient,
            elapsed_seconds=state.elapsed_training_seconds,
            updates_per_second=1.0,
            peak_memory_bytes=state.peak_memory_bytes,
        )
    metadata_path = saved.path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["state"]["last_update"]["loss"] = 9.0
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latest_before = (root / "latest.json").read_bytes()
    inventory_before = sorted(path.name for path in root.iterdir())
    resumed_model = TinyRegressor()
    model_before = np.asarray(resumed_model.proj.weight).copy()
    resumed_optimizer = optimizer_module.SmolVLAAdamW(
        optimizer_module.SmolVLAOptimizerConfig(training_horizon=10)
    )
    optimizer_before = {
        name: np.asarray(value).copy()
        for name, value in tree_flatten(resumed_optimizer.state)
    }
    output_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        metrics_snapshot = module._snapshot_regular_file_at(
            output_descriptor,
            "metrics.csv",
            label="test metrics",
            capture_payload=True,
        )
        try:
            module.load_latest_training_checkpoint(
                model=resumed_model,
                optimizer=resumed_optimizer,
                checkpoint_root=root,
                trainable_names=names,
                expected_run_config_sha256="a" * 64,
                expected_selected_steps=10,
                expected_effective_batch_size=8,
                expected_checkpoint_interval=100,
                metrics_parent_descriptor=output_descriptor,
                expected_metrics_snapshot=metrics_snapshot,
            )
        except ValueError as error:
            assert "checkpoint namespace" in str(error)
        else:
            raise AssertionError("forged checkpoint/metrics boundary was accepted")
    finally:
        os.close(output_descriptor)

    np.testing.assert_array_equal(np.asarray(resumed_model.proj.weight), model_before)
    optimizer_after = dict(tree_flatten(resumed_optimizer.state))
    assert set(optimizer_after) == set(optimizer_before)
    for name, expected in optimizer_before.items():
        np.testing.assert_array_equal(np.asarray(optimizer_after[name]), expected)
    assert (root / "latest.json").read_bytes() == latest_before
    assert sorted(path.name for path in root.iterdir()) == inventory_before


def test_t3b_training_bridge_reconstruction_rejects_materialized_aba(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "_validate_t3b_training_bridge_semantics"],
    )
    frozen_inputs = _t3b_frozen_inputs()
    live_evidence = {
        "format_version": 1,
        "sha256": "1" * 64,
        "components": {"tokenizer": "2" * 64},
    }
    audit_evidence = {
        "format_version": 1,
        "sha256": "3" * 64,
        "components": {"tokenizer": "4" * 64},
    }
    live_bridge = SimpleNamespace(
        episodes=(0, 1),
        semantic_evidence=lambda: live_evidence,
        state_dict=lambda: {"samples_consumed": 0},
        load_state_dict=lambda _state: None,
    )
    audit_bridge = SimpleNamespace(semantic_evidence=lambda: audit_evidence)
    monkeypatch.setattr(module, "TrainingDataBridge", lambda **_kwargs: audit_bridge)
    monkeypatch.setattr(
        module,
        "collect_t3b_frozen_input_evidence",
        lambda *_args, **_kwargs: frozen_inputs,
    )

    try:
        module._validate_t3b_training_bridge_semantics(
            config=module.FineTuneConfig(output_dir=tmp_path / "run"),
            bridge=live_bridge,
            stats=SimpleNamespace(processor_stats={}),
            expected_frozen_inputs=frozen_inputs,
        )
    except RuntimeError as error:
        assert "clean reconstruction" in str(error)
    else:
        raise AssertionError("materialized bridge ABA mismatch was accepted")


def test_t3b_training_bridge_reconstruction_materializes_disposable_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["FineTuneConfig", "_validate_t3b_training_bridge_semantics"],
    )
    frozen_inputs = _t3b_frozen_inputs()
    bridge_evidence = {
        "format_version": 1,
        "sha256": "1" * 64,
        "components": {"tokenizer": "2" * 64},
    }
    events: list[str] = []

    def live_semantic_evidence():
        events.append("live-evidence")
        return bridge_evidence

    live_bridge = SimpleNamespace(
        episodes=(0, 1),
        semantic_evidence=live_semantic_evidence,
        state_dict=lambda: {"samples_consumed": 0},
        load_state_dict=lambda _state: events.append("live-reset"),
        next_batch=lambda: (_ for _ in ()).throw(
            AssertionError("semantic validation consumed the live bridge")
        ),
    )
    audit_batch = object()
    audit_bridge = SimpleNamespace(
        semantic_evidence=lambda: bridge_evidence,
        next_batch=lambda: events.append("audit-decode") or audit_batch,
    )
    monkeypatch.setattr(module, "TrainingDataBridge", lambda **_kwargs: audit_bridge)
    monkeypatch.setattr(
        module,
        "collect_t3b_frozen_input_evidence",
        lambda *_args, **_kwargs: frozen_inputs,
    )

    result = module._validate_t3b_training_bridge_semantics(
        config=module.FineTuneConfig(output_dir=tmp_path / "run"),
        bridge=live_bridge,
        stats=SimpleNamespace(processor_stats={}),
        expected_frozen_inputs=frozen_inputs,
    )

    assert result == bridge_evidence
    assert events == [
        "live-evidence",
        "audit-decode",
        "live-reset",
        "live-evidence",
    ]


def test_t3b_resume_run_rejects_self_rehashed_immutable_contradictions() -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_validate_t3b_resume_run_document"],
    )
    immutable = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-lora-run",
        "seed": 20260901,
        "selected_steps": 3000,
        "effective_batch_size": 8,
        "lora": {"scope": "expert_only", "rank": 8},
    }
    document = {
        **immutable,
        "status": "interrupted",
        "checkpoint_count": 0,
        "resume_count": 0,
        "metrics_recoveries": [],
        "checkpoint_recoveries": [],
        "startup_recoveries": [],
        "disk_free_before_bytes": 1,
        "process": {"identity": "validated separately"},
        "last_completed_step": 0,
        "interruption": {"type": "RuntimeError", "message": "stopped"},
    }
    module._validate_t3b_resume_run_document(
        document,
        expected_immutable=immutable,
        selected_steps=3000,
        checkpoint_interval=100,
    )

    for name, value in (
        ("seed", 20260902),
        ("selected_steps", True),
        ("lora", {"scope": "expert_only", "rank": 4}),
    ):
        changed = json.loads(json.dumps(document))
        changed[name] = value
        try:
            module._validate_t3b_resume_run_document(
                changed,
                expected_immutable=immutable,
                selected_steps=3000,
                checkpoint_interval=100,
            )
        except ValueError as error:
            assert "immutable field" in str(error)
        else:
            raise AssertionError(f"contradictory resumable field was accepted: {name}")


def test_recorded_recovery_paths_require_safe_live_inventory_before_resume(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_release_t3b_training_lock",
            "_validate_recorded_t3b_recovery_inventories",
        ],
    )
    output = tmp_path / "run"
    output.mkdir()
    recovery_files = {
        "checkpoint-recoveries/latest-pointer-previous-000001": b"pointer\n",
        "startup-recoveries/budget-json-partial-000001": b"budget\n",
        "resume-recoveries/run-json-partial-000001": b"run\n",
        "process-identity-recoveries/training-pid-previous-000001": b"pid\n",
    }
    for relative, payload in recovery_files.items():
        path = output / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
    document = {
        "checkpoint_recoveries": [
            "checkpoint-recoveries/latest-pointer-previous-000001"
        ],
        "startup_recoveries": [
            "startup-recoveries/budget-json-partial-000001",
            "resume-recoveries/run-json-partial-000001",
            "process-identity-recoveries/training-pid-previous-000001",
        ],
    }
    lease = module._acquire_t3b_training_lock(output)
    try:
        module._validate_recorded_t3b_recovery_inventories(
            document,
            output_dir=output,
            lease=lease,
        )

        traversing = json.loads(json.dumps(document))
        traversing["checkpoint_recoveries"][0] = (
            "checkpoint-recoveries/../run.json"
        )
        try:
            module._validate_recorded_t3b_recovery_inventories(
                traversing,
                output_dir=output,
                lease=lease,
            )
        except ValueError as error:
            assert "path is invalid" in str(error)
        else:
            raise AssertionError("traversing checkpoint recovery path was accepted")

        missing = json.loads(json.dumps(document))
        missing["startup_recoveries"][0] = (
            "startup-recoveries/budget-json-partial-999999"
        )
        try:
            module._validate_recorded_t3b_recovery_inventories(
                missing,
                output_dir=output,
                lease=lease,
            )
        except FileNotFoundError as error:
            assert "recorded startup recoveries are missing" in str(error)
        else:
            raise AssertionError("missing startup recovery path was accepted")

        process_recovery = output / document["startup_recoveries"][2]
        outside = tmp_path / "outside-pid"
        outside.write_bytes(b"replacement\n")
        process_recovery.unlink()
        process_recovery.symlink_to(outside)
        try:
            module._validate_recorded_t3b_recovery_inventories(
                document,
                output_dir=output,
                lease=lease,
            )
        except FileExistsError as error:
            assert "inventory is unsafe" in str(error)
        else:
            raise AssertionError("symlink-tampered recovery artifact was accepted")
    finally:
        module._release_t3b_training_lock(lease)


def test_startup_recovery_inventory_reads_the_retained_tree_during_aba(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_release_t3b_training_lock",
            "_startup_recovery_inventory",
        ],
    )
    output = tmp_path / "run"
    recovery = output / "startup-recoveries"
    recovery.mkdir(parents=True)
    artifact_name = "budget-json-partial-000001"
    (recovery / artifact_name).write_bytes(b"retained recovery\n")
    detached = tmp_path / "detached-startup-recoveries"
    competitor = tmp_path / "competitor-startup-recoveries"
    restored_competitor = tmp_path / "restored-competitor-startup-recoveries"
    real_snapshot = module._snapshot_regular_file_at
    swapped = False

    def swap_and_restore(parent_descriptor, name, **kwargs):
        nonlocal swapped
        if not swapped and kwargs.get("label") == "T3B startup recovery artifact":
            swapped = True
            recovery.rename(detached)
            competitor.mkdir()
            (competitor / artifact_name).write_bytes(b"competitor recovery\n")
            competitor.rename(recovery)
            try:
                return real_snapshot(parent_descriptor, name, **kwargs)
            finally:
                recovery.rename(restored_competitor)
                detached.rename(recovery)
        return real_snapshot(parent_descriptor, name, **kwargs)

    monkeypatch.setattr(module, "_snapshot_regular_file_at", swap_and_restore)
    lease = module._acquire_t3b_training_lock(output)
    try:
        inventory = module._startup_recovery_inventory(
            output,
            output_descriptor=lease.output_descriptor,
            expected_output_snapshot=lease.output_snapshot,
        )
    finally:
        module._release_t3b_training_lock(lease)

    assert swapped
    assert inventory == (f"startup-recoveries/{artifact_name}",)
    assert (recovery / artifact_name).read_bytes() == b"retained recovery\n"
    assert (restored_competitor / artifact_name).read_bytes() == (
        b"competitor recovery\n"
    )


def test_t3b_final_artifact_bindings_reject_child_replacement(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=[
            "_acquire_t3b_training_lock",
            "_bind_t3b_adapter_files",
            "_bind_t3b_export_files",
            "_bind_t3b_export_root",
            "_release_t3b_training_lock",
            "_revalidate_t3b_adapter_files",
            "_revalidate_t3b_export_root",
        ],
    )
    output = tmp_path / "run"
    output.mkdir()
    lease = module._acquire_t3b_training_lock(output)
    try:
        adapter_payload = b"adapter bytes\n"
        adapter_sha256 = hashlib.sha256(adapter_payload).hexdigest()
        (output / "adapter.safetensors").write_bytes(adapter_payload)
        adapter_metadata = {
            "format_version": 1,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "adapter_count": 112,
            "tensor_count": 224,
            "scalar_count": 1_708_032,
            "sha256": adapter_sha256,
            "scope": "expert_only",
        }
        (output / "adapter.json").write_text(
            json.dumps(adapter_metadata, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lora_report = SimpleNamespace(
            scope="expert_only",
            rank=8,
            alpha=16.0,
            dropout=0.0,
            adapter_count=112,
            trainable_names=tuple(f"tensor-{index}" for index in range(224)),
            trainable_scalar_count=1_708_032,
        )
        module._bind_t3b_adapter_files(
            lease,
            expected_sha256=adapter_sha256,
            lora_report=lora_report,
        )

        export = output / "export"
        export.mkdir()
        file_sha256 = {}
        for name in sorted(module._T3B_CHECKPOINT_FILES):
            payload = f"{name}\n".encode()
            (export / name).write_bytes(payload)
            file_sha256[name] = hashlib.sha256(payload).hexdigest()
        metadata = {"adapter_sha256": adapter_sha256}
        manifest = {
            "format_version": 1,
            "artifact_type": "smolvla-mlx-merged-training-checkpoint",
            "dtype": "float32",
            "tensor_count": 500,
            "parameter_count": 450_046_176,
            "source_checkpoint": {
                "repo_id": module.CHECKPOINT_ID,
                "revision": module.CHECKPOINT_REVISION,
            },
            "metadata": metadata,
            "file_sha256": file_sha256,
        }
        (export / "training_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        module._bind_t3b_export_root(lease)
        module._bind_t3b_export_files(
            lease,
            expected_report=SimpleNamespace(file_sha256=file_sha256),
            expected_metadata=metadata,
        )

        (export / "model.safetensors").unlink()
        (export / "model.safetensors").write_bytes(b"replacement model\n")
        try:
            module._revalidate_t3b_export_root(lease, verify_bytes=True)
        except RuntimeError as error:
            assert "export file changed" in str(error)
        else:
            raise AssertionError("replaced export model remained bound")

        (output / "adapter.json").unlink()
        (output / "adapter.json").write_text(
            '{"sha256":"competitor"}\n', encoding="utf-8"
        )
        try:
            module._revalidate_t3b_adapter_files(lease, verify_bytes=True)
        except RuntimeError as error:
            assert "adapter file changed" in str(error)
        else:
            raise AssertionError("replaced adapter metadata remained bound")
    finally:
        module._release_t3b_training_lock(lease)


def test_bound_file_revalidation_rechecks_the_public_name_after_hashing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = __import__(
        "training.finetune",
        fromlist=["_open_bound_regular_file_at", "_revalidate_bound_regular_file_at"],
    )
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"canonical bytes\n")
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    snapshot, descriptor = module._open_bound_regular_file_at(
        parent_descriptor,
        target.name,
        label="test bound file",
    )
    detached = tmp_path / "detached.bin"
    real_read = module.os.read
    swapped = False

    def replace_public_name(fd, count):
        nonlocal swapped
        payload = real_read(fd, count)
        if fd == descriptor and payload and not swapped:
            swapped = True
            target.rename(detached)
            target.write_bytes(b"replacement bytes\n")
        return payload

    monkeypatch.setattr(module.os, "read", replace_public_name)
    try:
        try:
            module._revalidate_bound_regular_file_at(
                parent_descriptor,
                target.name,
                expected=snapshot,
                descriptor=descriptor,
                label="test bound file",
                verify_bytes=True,
            )
        except RuntimeError as error:
            assert "changed while bound" in str(error)
        else:
            raise AssertionError("public-name replacement during hashing was accepted")
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)

    assert swapped
    assert target.read_bytes() == b"replacement bytes\n"
    assert detached.read_bytes() == b"canonical bytes\n"
