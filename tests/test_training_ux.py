"""Stage T4 public training UX and exact-resume evidence contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import pytest


class _TinyFullModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = nn.Linear(2, 2)
        self.connector = nn.Linear(2, 2)
        self.language = nn.Linear(2, 2)
        self.state_proj = nn.Linear(2, 2)
        self.expert = nn.Linear(2, 2)


def test_training_modes_are_explicit_and_validate_public_controls(tmp_path: Path) -> None:
    from training.ux import FullTrainingConfig, LoRATrainingConfig

    common = {
        "dataset": "owner/dataset",
        "steps": 100,
        "batch_size": 2,
        "learning_rate": 3e-5,
        "output_dir": tmp_path / "run",
        "checkpoint_interval": 25,
    }
    full = FullTrainingConfig(**common)
    lora = LoRATrainingConfig(**common, rank=4, alpha=8.0)

    assert full.mode == "full"
    assert lora.mode == "lora"
    assert full.resume is False
    assert lora.rank == 4
    assert lora.alpha == 8.0
    assert full.dtype == "bfloat16"
    assert FullTrainingConfig(**common, dtype="float32").dtype == "float32"

    for field, value in (("steps", 0), ("batch_size", 0), ("learning_rate", 0.0)):
        invalid = {**common, field: value}
        try:
            FullTrainingConfig(**invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {field} was accepted")
    with pytest.raises(ValueError, match="dtype"):
        FullTrainingConfig(**common, dtype="float16")


def test_dataset_source_accepts_local_path_and_materializes_repo_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lerobot.datasets.dataset_metadata as metadata_module
    import training.ux as module

    local = tmp_path / "local-data"
    (local / "meta").mkdir(parents=True)
    (local / "meta" / "info.json").write_text(
        '{"total_episodes": 3}', encoding="utf-8"
    )
    local_source = module._dataset_source(
        module.FullTrainingConfig(dataset=local, output_dir=tmp_path / "out")
    )
    assert local_source["repo_id"] == "local/local-data"
    assert local_source["root"] == str(local.resolve())
    assert local_source["total_episodes"] == 3

    downloads = []

    class Metadata:
        total_episodes = 4
        revision = "resolved-revision"

        def __init__(self, repo_id, *, root, revision):
            assert repo_id == "owner/remote-data"
            assert revision is None
            self.root = Path(root)

    def download(repo_id, *, repo_type, revision, local_dir):
        downloads.append((repo_id, repo_type, revision, Path(local_dir)))
        data = Path(local_dir) / "data" / "chunk-000"
        data.mkdir(parents=True)
        (data / "file-000.parquet").touch()
        return str(local_dir)

    monkeypatch.setattr(metadata_module, "LeRobotDatasetMetadata", Metadata)
    monkeypatch.setattr(module, "snapshot_download", download)
    remote = module._dataset_source(
        module.FullTrainingConfig(
            dataset="owner/remote-data",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "remote-out",
        )
    )
    assert remote["repo_id"] == "owner/remote-data"
    assert remote["revision"] == "resolved-revision"
    assert remote["total_episodes"] == 4
    assert downloads == [
        (
            "owner/remote-data",
            "dataset",
            None,
            (tmp_path / "cache/datasets/owner__remote-data").resolve(),
        )
    ]


def test_full_mode_matches_reference_trainable_policy_and_optimizer_coverage() -> None:
    from training.optimizer import SmolVLAAdamW, SmolVLAOptimizerConfig
    from training.ux import configure_full_training, optimizer_coverage_evidence

    model = _TinyFullModel()
    report = configure_full_training(model)

    assert report.mode == "full"
    assert report.trainable_names == (
        "state_proj.weight",
        "state_proj.bias",
        "expert.weight",
        "expert.bias",
    )
    assert set(name for name, _ in tree_flatten(model.trainable_parameters())) == set(
        report.trainable_names
    )
    assert all(
        name.startswith(("state_proj.", "expert.")) for name in report.trainable_names
    )
    assert all(
        value.dtype == mx.float32
        for _, value in tree_flatten(model.trainable_parameters())
    )

    optimizer = SmolVLAAdamW(SmolVLAOptimizerConfig(training_horizon=100))
    evidence = optimizer_coverage_evidence(model, optimizer, report)
    assert evidence["covered"] is True
    assert evidence["parameter_tensor_count"] == 4
    assert evidence["moment_tensor_count"] == 8


def test_trajectory_hash_excludes_measurement_time_but_binds_continuation_state() -> None:
    from training.ux import trajectory_state_sha256

    state = {
        "completed_step": 50,
        "selected_steps": 100,
        "smoothed_loss": 0.25,
        "samples_consumed": 100,
        "flow_draw_count": 100,
        "last_loss": 0.2,
        "learning_rate": 1e-4,
        "gradient_norm": 1.5,
        "clip_coefficient": 1.0,
        "elapsed_training_seconds": 99.0,
        "peak_memory_bytes": 123,
    }
    changed_time = {**state, "elapsed_training_seconds": 1.0, "peak_memory_bytes": 999}
    changed_draws = {**state, "flow_draw_count": 101}

    assert trajectory_state_sha256(state) == trajectory_state_sha256(changed_time)
    assert trajectory_state_sha256(state) != trajectory_state_sha256(changed_draws)


def _write_resume_fixture(
    root: Path,
    *,
    parameter_delta: float = 0.0,
    loss_delta: float = 0.0,
) -> None:
    checkpoint = root / "checkpoints" / "step-000100"
    checkpoint.mkdir(parents=True)
    mx.save_safetensors(
        checkpoint / "model.safetensors",
        {"expert.weight": mx.array([[1.0 + parameter_delta]], dtype=mx.float32)},
    )
    mx.save_safetensors(
        checkpoint / "optimizer.safetensors",
        {
            "step": mx.array(100, dtype=mx.uint64),
            "expert.weight.m": mx.array([[0.1]], dtype=mx.float32),
            "expert.weight.v": mx.array([[0.2]], dtype=mx.float32),
        },
    )
    with (root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step",
                "loss",
                "smoothed_loss",
                "learning_rate",
                "gradient_norm",
                "clip_coefficient",
                "elapsed_seconds",
                "peak_memory_bytes",
                "draw_sha256",
            ),
        )
        writer.writeheader()
        for step in range(1, 101):
            writer.writerow(
                {
                    "step": step,
                    "loss": 0.25 + (loss_delta if step == 100 else 0.0),
                    "smoothed_loss": 0.3,
                    "learning_rate": 1e-4,
                    "gradient_norm": 1.0,
                    "clip_coefficient": 1.0,
                    "elapsed_seconds": 12.0,
                    "peak_memory_bytes": 123,
                    "draw_sha256": "d" * 64,
                }
            )
    sampler = {"samples_consumed": 100, "epoch": 0, "start_index": 100}
    sampler_sha = hashlib.sha256(
        json.dumps(sampler, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "run.json").write_text(
        json.dumps(
            {
                "status": "trained_and_exported",
                "mode": "full",
                "selected_steps": 100,
                "run_config_sha256": "a" * 64,
                "final_evidence": {
                    "draw_chain_sha256": "d" * 64,
                    "sampler_state_sha256": sampler_sha,
                    "trajectory_state_sha256": "e" * 64,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_resume_evaluator_enforces_fixed_parameter_loss_and_state_gates(
    tmp_path: Path,
) -> None:
    from training.ux import evaluate_resume_exactness

    direct = tmp_path / "direct"
    resumed = tmp_path / "resumed"
    _write_resume_fixture(direct)
    _write_resume_fixture(resumed, parameter_delta=5e-7, loss_delta=5e-8)

    report = evaluate_resume_exactness(direct, resumed)
    assert report["passed"] is True
    assert report["gates"] == {
        "parameter_max_abs": 1e-6,
        "loss_max_abs": 1e-7,
    }
    assert report["parameter_max_abs"] <= 1e-6
    assert report["loss_max_abs"] <= 1e-7
    assert report["optimizer_exact"] is True
    assert report["draw_chain_exact"] is True
    assert report["sampler_state_exact"] is True
    assert report["trajectory_state_exact"] is True

    outside = tmp_path / "outside"
    _write_resume_fixture(outside, parameter_delta=2e-6)
    failed = evaluate_resume_exactness(direct, outside)
    assert failed["passed"] is False
    assert failed["parameter_max_abs"] > 1e-6


def test_cli_train_surface_selects_exactly_one_mode() -> None:
    from smolvla_mlx.cli import _parser

    parser = _parser()
    full = parser.parse_args(
        [
            "train",
            "owner/dataset",
            "--full",
            "--steps",
            "100",
            "--batch-size",
            "2",
            "--lr",
            "0.00003",
            "--output",
            "run",
            "--checkpoint-every",
            "25",
            "--dtype",
            "float32",
        ]
    )
    assert full.training_mode == "full"
    assert full.dataset == "owner/dataset"
    assert full.steps == 100
    assert full.batch_size == 2
    assert full.learning_rate == 3e-5
    assert full.checkpoint_interval == 25
    assert full.dtype == "float32"

    for invalid in (["train", "owner/dataset"], ["train", "owner/dataset", "--lora", "--full"]):
        try:
            parser.parse_args(invalid)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"invalid train mode was accepted: {invalid}")


class _TinyBridge:
    def __init__(self) -> None:
        self.samples_consumed = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "samples_consumed": self.samples_consumed,
            "num_samples": 100,
            "epoch": self.samples_consumed // 100,
            "start_index": self.samples_consumed % 100,
            "sampler_seed": 20_260_901,
            "episodes": [0],
        }

    def load_state_dict(self, state) -> None:
        self.samples_consumed = int(state["samples_consumed"])


def test_public_runner_retains_three_checkpoints_and_resumes_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import training.ux as module
    from training.finetune import UpdateResult
    from training.optimizer import SmolVLAAdamW, SmolVLAOptimizerConfig

    def prepare(config):
        mx.random.seed(config.seed)
        model = _TinyFullModel()
        model.update(
            {
                "vision": {"weight": mx.zeros((2, 2)), "bias": mx.zeros((2,))},
                "connector": {"weight": mx.zeros((2, 2)), "bias": mx.zeros((2,))},
                "language": {"weight": mx.zeros((2, 2)), "bias": mx.zeros((2,))},
                "state_proj": {"weight": mx.ones((2, 2)), "bias": mx.zeros((2,))},
                "expert": {"weight": mx.ones((2, 2)), "bias": mx.zeros((2,))},
            }
        )
        topology = module.configure_full_training(model)
        optimizer = SmolVLAAdamW(
            SmolVLAOptimizerConfig(
                lr=config.learning_rate,
                decay_lr=min(2.5e-6, config.learning_rate),
                training_horizon=config.steps,
            )
        )
        coverage = module.optimizer_coverage_evidence(model, optimizer, topology)
        return module._TrainingComponents(
            dataset={
                "requested": str(config.dataset),
                "repo_id": "owner/data",
                "root": "/dataset",
                "revision": "r1",
                "total_episodes": 2,
            },
            train_episodes=(0,),
            holdout_episodes=(1,),
            train_statistics_sha256="b" * 64,
            processor_stats={},
            model=model,
            bridge=_TinyBridge(),
            optimizer=optimizer,
            topology=topology,
            optimizer_coverage=coverage,
            base_artifact={"model_sha256": "c" * 64, "name_map_sha256": "d" * 64},
            lora_report=None,
        )

    def update(*, model, bridge, optimizer, batch_size, draw_chain_sha256):
        del batch_size
        bridge.samples_consumed += 1
        gradients = tree_map(
            lambda value: mx.full(value.shape, 0.01, dtype=value.dtype),
            model.trainable_parameters(),
        )
        learning_rate = optimizer.update(model, gradients)
        mx.eval(model.trainable_parameters(), optimizer.state)
        loss = 1.0 / bridge.samples_consumed
        chain = hashlib.sha256(
            bytes.fromhex(draw_chain_sha256)
            + str(bridge.samples_consumed).encode("ascii")
        ).hexdigest()
        return (
            UpdateResult(
                loss=loss,
                learning_rate=learning_rate,
                gradient_norm=0.01,
                clip_coefficient=1.0,
                    seconds=0.001,
            ),
            chain,
            None,
        )

    def finalize(*, output_dir, **kwargs):
        del kwargs
        export = output_dir / "export"
        export.mkdir()
        return {
            "path": str(export),
            "tensor_count": 10,
            "parameter_count": 10,
            "file_sha256": {},
            "action_validation": {"finite": True},
        }

    monkeypatch.setattr(module, "_prepare_training", prepare)
    monkeypatch.setattr(module, "_perform_update", update)
    monkeypatch.setattr(module, "_finalize_export", finalize)

    direct_dir = tmp_path / "direct"
    resumed_dir = tmp_path / "resumed"
    direct = module.run_training(
        module.FullTrainingConfig(
            dataset="owner/data",
            steps=4,
            output_dir=direct_dir,
            checkpoint_interval=1,
        )
    )
    assert direct.final_loss == 0.25
    assert sorted(path.name for path in (direct_dir / "checkpoints").iterdir()) == [
        "latest.json",
        "step-000002",
        "step-000003",
        "step-000004",
    ]

    def interrupt(step, total, update):
        del total, update
        if step == 2:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.run_training(
            module.FullTrainingConfig(
                dataset="owner/data",
                steps=4,
                output_dir=resumed_dir,
                checkpoint_interval=1,
            ),
            progress=interrupt,
        )
    assert json.loads((resumed_dir / "run.json").read_text())["status"] == "interrupted"

    resumed = module.run_training(
        module.FullTrainingConfig(
            dataset="owner/data",
            steps=4,
            output_dir=resumed_dir,
            checkpoint_interval=1,
            resume=True,
        )
    )
    assert resumed.final_loss == direct.final_loss
    report = module.evaluate_resume_exactness(direct_dir, resumed_dir)
    assert report["passed"] is True
    assert report["parameter_max_abs"] == 0.0
    assert report["loss_max_abs"] == 0.0
