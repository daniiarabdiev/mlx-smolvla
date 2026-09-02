"""Frozen held-out evidence and immutable Stage T3 outcome-gate contracts."""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from training.data import TrainingArtifact


_EVALUATION_DIR = Path(".cache/training/t3-evaluation")
_BASE_REPORT = Path(".cache/training/t3-base-evaluation.json")


def test_t3_outcome_thresholds_are_immutable() -> None:
    module = __import__("training.evaluation", fromlist=["MAE_IMPROVEMENT_RATIO_MAXIMUM"])

    assert module.MAE_IMPROVEMENT_RATIO_MAXIMUM == 0.9
    assert module.TORCH_MLX_MAE_RATIO_MINIMUM == 0.95
    assert module.TORCH_MLX_MAE_RATIO_MAXIMUM == 1.05
    assert module.IMAGE_PREPROCESSING_MAX_ABSOLUTE_TOLERANCE == 1e-5
    assert module.STATE_PREPROCESSING_MAX_ABSOLUTE_TOLERANCE == 1e-6
    assert module.INFERENCE_MAX_ABSOLUTE_TOLERANCE == 5e-3
    assert module.EVALUATION_MINIMUM_FREE_BYTES == 40 * 1024**3
    assert module.EVALUATION_SAMPLE_COUNT == 56


def test_stats_active_parity_applies_the_separate_preprocessing_limits() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_stats_active_parity_passed"],
    )

    assert module._stats_active_parity_passed(
        image_preprocessing_max_abs=1e-5,
        state_preprocessing_max_abs=1e-6,
        normalized_action_max_abs=5e-3,
        physical_action_max_abs=5e-3,
        physical_action_standardized_max_abs=5e-3,
    )
    assert not module._stats_active_parity_passed(
        image_preprocessing_max_abs=1.00001e-5,
        state_preprocessing_max_abs=0.0,
        normalized_action_max_abs=0.0,
        physical_action_max_abs=0.0,
        physical_action_standardized_max_abs=0.0,
    )
    assert not module._stats_active_parity_passed(
        image_preprocessing_max_abs=0.0,
        state_preprocessing_max_abs=1.00001e-6,
        normalized_action_max_abs=0.0,
        physical_action_max_abs=0.0,
        physical_action_standardized_max_abs=0.0,
    )


def test_evaluation_free_space_floor_is_enforced_before_loading() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_require_minimum_free_bytes"],
    )

    module._require_minimum_free_bytes(module.EVALUATION_MINIMUM_FREE_BYTES)
    with pytest.raises(RuntimeError, match="40 GiB"):
        module._require_minimum_free_bytes(module.EVALUATION_MINIMUM_FREE_BYTES - 1)


def test_frozen_evaluation_artifact_is_complete_integral_and_unseen() -> None:
    module = __import__("training.evaluation", fromlist=["load_evaluation_cases"])
    artifact = TrainingArtifact(_EVALUATION_DIR)
    names = artifact.verify_all()
    metadata = artifact.metadata
    cases = module.load_evaluation_cases(_EVALUATION_DIR)

    assert metadata["artifact_type"] == "smolvla-lora-heldout-evaluation"
    assert metadata["sample_count"] == 56
    assert metadata["noise_seed"] == 20260902
    assert metadata["heldout_episodes"] == [2, 7, 21, 28, 31, 34, 35, 41]
    assert metadata["train_statistics_sha256"] == (
        "5aa5ab85e0c71c0adee97782be37907b0918050a8539bb3aab88fe392953948e"
    )
    assert len(names) == 280
    assert len(cases) == 56
    assert len({case.absolute_index for case in cases}) == 56
    assert {case.episode for case in cases} == {2, 7, 21, 28, 31, 34, 35, 41}
    assert all(case.camera1.dtype == np.uint8 for case in cases)
    assert all(case.camera2.dtype == np.uint8 for case in cases)
    assert all(case.state.shape == (6,) for case in cases)
    assert all(case.target_action.shape == (6,) for case in cases)
    assert all(case.noise.shape == (1, 50, 32) for case in cases)
    assert all(np.isfinite(case.noise).all() for case in cases)


def test_frozen_evaluation_rejects_metadata_changed_after_capture(
    tmp_path: Path,
) -> None:
    module = __import__("training.evaluation", fromlist=["load_evaluation_cases"])
    altered = tmp_path / "evaluation"
    altered.mkdir()
    (altered / "manifest.json").write_bytes(
        (_EVALUATION_DIR / "manifest.json").read_bytes()
    )
    metadata = json.loads(
        (_EVALUATION_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    metadata["cases"][0]["task"] = "a task changed after the baseline was frozen"
    (altered / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation metadata SHA-256"):
        module.load_evaluation_cases(altered)


def test_evaluation_metadata_is_reconstructed_from_the_pinned_dataset() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validate_evaluation_metadata_against_dataset"],
    )
    metadata = json.loads(
        (_EVALUATION_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    dataset_root = Path(".cache/hf/datasets/svla_so101_pickplace")

    module._validate_evaluation_metadata_against_dataset(metadata, dataset_root)

    changed = {**metadata, "cases": [dict(case) for case in metadata["cases"]]}
    changed["cases"][0]["task"] = "retrospectively changed task\n"
    with pytest.raises(ValueError, match="canonical pinned-dataset reconstruction"):
        module._validate_evaluation_metadata_against_dataset(changed, dataset_root)


def test_frozen_evaluation_rejects_a_symlinked_ancestor(tmp_path: Path) -> None:
    module = __import__("training.evaluation", fromlist=["load_evaluation_cases"])
    alias = tmp_path / "training-cache"
    alias.symlink_to(_EVALUATION_DIR.resolve().parent, target_is_directory=True)

    with pytest.raises(FileNotFoundError, match="missing or unsafe"):
        module.load_evaluation_cases(alias / _EVALUATION_DIR.name)


def test_json_evidence_rejects_a_symlinked_ancestor(tmp_path: Path) -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_read_json_document"],
    )
    alias = tmp_path / "training-cache"
    alias.symlink_to(_BASE_REPORT.resolve().parent, target_is_directory=True)

    with pytest.raises(FileNotFoundError, match="missing or unsafe"):
        module._read_json_document(
            alias / _BASE_REPORT.name,
            label="base held-out evaluation",
        )


def test_outcome_paths_must_stay_in_the_repository_cache(tmp_path: Path) -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_require_repository_cache_path"],
    )

    with pytest.raises(ValueError, match="repository-local .cache"):
        module._require_repository_cache_path(
            tmp_path / "outcome.json",
            label="fine-tune outcome report",
        )


def test_pinned_dataset_root_rejects_a_nested_symlink(tmp_path: Path) -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_require_pinned_dataset_root"],
    )
    cache_dir = tmp_path / "hf"
    (cache_dir / "datasets").mkdir(parents=True)
    (cache_dir / "datasets" / "svla_so101_pickplace").symlink_to(
        Path(".cache/hf/datasets/svla_so101_pickplace").resolve(),
        target_is_directory=True,
    )

    with pytest.raises(FileNotFoundError, match="missing or unsafe"):
        module._require_pinned_dataset_root(cache_dir)


def test_pinned_dataset_root_verifies_files_against_the_revision_tree(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_require_pinned_dataset_root"],
    )
    source = Path(".cache/hf/datasets/svla_so101_pickplace")
    cache_dir = tmp_path / "hf"
    target = cache_dir / "datasets" / "svla_so101_pickplace"
    revision_tree = (
        Path(".cache/huggingface/trees")
        / "f641879e22172be7e8161d5e6c1503c2d2feb657.json"
    )
    relative_files = (
        Path("data/chunk-000/file-000.parquet"),
        Path("meta/info.json"),
        Path("meta/stats.json"),
        Path("meta/tasks.parquet"),
        revision_tree,
    )
    for relative in relative_files:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    data_path = target / "data/chunk-000/file-000.parquet"
    data_path.write_bytes(data_path.read_bytes() + b"changed after download")

    with pytest.raises(ValueError, match="pinned dataset file differs from revision"):
        module._require_pinned_dataset_root(cache_dir)


def test_absolute_error_aggregates_physical_action_mae_without_rounding() -> None:
    module = __import__("training.evaluation", fromlist=["absolute_error"])
    prediction = np.array([1.0, 3.0, -2.0], dtype=np.float32)
    target = np.array([0.0, 5.0, 2.0], dtype=np.float32)

    error_sum, count = module.absolute_error(prediction, target)

    assert error_sum == 7.0
    assert count == 3


def test_outcome_gate_requires_all_three_independent_results() -> None:
    module = __import__("training.evaluation", fromlist=["evaluate_outcome_gates"])

    passing = module.evaluate_outcome_gates(
        base_mlx_mae=10.0,
        fine_mlx_mae=9.0,
        torch_mae=9.45,
        parity_max_abs=0.005,
    )
    failed_improvement = module.evaluate_outcome_gates(
        base_mlx_mae=10.0,
        fine_mlx_mae=9.00001,
        torch_mae=9.0,
        parity_max_abs=0.0,
    )
    failed_roundtrip = module.evaluate_outcome_gates(
        base_mlx_mae=10.0,
        fine_mlx_mae=8.0,
        torch_mae=8.401,
        parity_max_abs=0.0,
    )
    failed_parity = module.evaluate_outcome_gates(
        base_mlx_mae=10.0,
        fine_mlx_mae=8.0,
        torch_mae=8.0,
        parity_max_abs=0.005001,
    )

    assert passing.passed
    assert passing.improvement_passed
    assert passing.roundtrip_passed
    assert passing.parity_passed
    assert passing.fine_to_base_ratio == 0.9
    assert math.isclose(passing.torch_to_mlx_ratio, 1.05, rel_tol=0, abs_tol=1e-15)
    assert not failed_improvement.passed
    assert not failed_improvement.improvement_passed
    assert not failed_roundtrip.passed
    assert not failed_roundtrip.roundtrip_passed
    assert not failed_parity.passed
    assert not failed_parity.parity_passed


def test_base_report_is_bound_to_frozen_manifest_before_training() -> None:
    artifact = TrainingArtifact(_EVALUATION_DIR)
    report = json.loads(_BASE_REPORT.read_text(encoding="utf-8"))

    assert report["artifact_type"] == "smolvla-lora-base-heldout-evaluation"
    assert report["evaluation_manifest_sha256"] == artifact.metadata["manifest_sha256"]
    assert report["sample_count"] == 56
    assert report["element_count"] == 336
    assert report["mlx_mae"] > 0
    assert len(report["samples"]) == 56
    assert report["device"] == "Device(cpu, 0)"
    assert report["dtype"] == "float32"


def _mae_evaluation(module, *, framework: str, mae: float):
    cases = json.loads(
        (_EVALUATION_DIR / "metadata.json").read_text(encoding="utf-8")
    )["cases"]
    return module.MAEEvaluation(
        framework=framework,
        device="cpu",
        dtype="float32",
        sample_count=56,
        element_count=336,
        mae=mae,
        absolute_error_sum=mae * 336,
        samples=tuple(
            {
                "ordinal": case["ordinal"],
                "episode": case["episode"],
                "frame_index": case["frame_index"],
                "absolute_index": case["absolute_index"],
                "absolute_error_sum": mae * 6,
                "element_count": 6,
            }
            for case in cases
        ),
    )


def _actual_outcome_inputs(module) -> dict[str, object]:
    training_run = json.loads(
        Path(".cache/training/t3/run.json").read_text(encoding="utf-8")
    )
    export_manifest = json.loads(
        Path(".cache/training/t3/export/training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "training_run": training_run,
        "training_run_sha256": (
            "c7c3b86361c0872e26f2088cbd33ada865cf450b6711a9b737ece933c1868c82"
        ),
        "metrics_sha256": (
            "7f3a8c070f8102d7edc0afe5a9f4e5088321d1cdd21548fc21e9c772dbfafc2c"
        ),
        "training_artifact_sha256": {
            "adapter": (
                "814e6f4b2a78a46b609aa7b48a28b4509f709d3e851e588dcd9a4bd2ca1408dc"
            ),
            "adapter_metadata": (
                "0f088f4d42655d0cd5915dca1a80819f1c14d6f5dd44236a16ff644520c50b98"
            ),
            "final_checkpoint_metadata": (
                "5d912a1e94bb1809c8fe72570450dc1c6d5c6c8973e8bcc00b1045eb465431ef"
            ),
            "final_checkpoint_model": (
                "814e6f4b2a78a46b609aa7b48a28b4509f709d3e851e588dcd9a4bd2ca1408dc"
            ),
            "final_checkpoint_optimizer": (
                "c9440be75315e04c1812ba18da0e0daccd2990fb0f6fdb1841d40ef7b01ffb5a"
            ),
        },
        "native_conversion_sha256": {
            "native_conversion_model": (
                "76d893b95c739cbd2a02598025e360a596edf3a7f90a8c4b1cb63d23ae54b42a"
            ),
            "native_conversion_name_map": (
                "1664c39008363b98587a8c8fc54ed3af5e899b7fd092944d146bbfe4efc17902"
            ),
        },
        "evaluation_manifest": TrainingArtifact(_EVALUATION_DIR).metadata,
        "evaluation_metadata_sha256": (
            "f49ee54aead7ce3ede7b94d5638864afd2e12ef57ae2622eb6574333820cd107"
        ),
        "base_report": json.loads(_BASE_REPORT.read_text(encoding="utf-8")),
        "base_report_sha256": (
            "211d6778b0530208ca2e81abe6f4002cc683e24d496a09ddbe39c100ebd4f7ce"
        ),
        "export_manifest": export_manifest,
        "export_manifest_sha256": (
            "55ad6834cbb3acb9dd565a57296a274d78e7cdc863aa81c3e6ef25da8b66ba03"
        ),
        "fine_mlx": _mae_evaluation(module, framework="mlx", mae=4.0),
        "torch_result": _mae_evaluation(module, framework="torch", mae=4.0),
        "parity": module.StatsActiveParity(
            sample_count=56,
            image_preprocessing_max_abs=0.0,
            state_preprocessing_max_abs=0.0,
            preprocessing_max_abs=0.0,
            normalized_action_max_abs=0.0,
            physical_action_max_abs=0.0,
            physical_action_standardized_max_abs=0.0,
            gate_max_abs=0.0,
            passed=True,
            samples=tuple(
                {
                    "ordinal": case.ordinal,
                    "episode": case.episode,
                    "frame_index": case.frame_index,
                    "absolute_index": case.absolute_index,
                    "image_preprocessing_max_abs": 0.0,
                    "state_preprocessing_max_abs": 0.0,
                    "preprocessing_max_abs": 0.0,
                    "normalized_action_max_abs": 0.0,
                    "physical_action_max_abs": 0.0,
                    "physical_action_standardized_max_abs": 0.0,
                }
                for case in module.load_evaluation_cases(_EVALUATION_DIR)
            ),
        ),
    }


def test_outcome_report_records_the_complete_frozen_evaluation_identity() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["assemble_finetune_outcome_report"],
    )

    report = module.assemble_finetune_outcome_report(
        **_actual_outcome_inputs(module)
    )

    assert report["source_sha256"]["evaluation_manifest"] == (
        "9cabca6cd21e8658a94e42980af3e91ecd8ff5ed5daca5f75eb7a1ebd1d261a3"
    )
    assert report["source_sha256"]["evaluation_metadata"] == (
        "f49ee54aead7ce3ede7b94d5638864afd2e12ef57ae2622eb6574333820cd107"
    )
    assert report["source_sha256"]["final_checkpoint_optimizer"] == (
        "c9440be75315e04c1812ba18da0e0daccd2990fb0f6fdb1841d40ef7b01ffb5a"
    )
    assert report["source_sha256"]["native_conversion_model"] == (
        "76d893b95c739cbd2a02598025e360a596edf3a7f90a8c4b1cb63d23ae54b42a"
    )
    assert report["source_sha256"]["dataset_revision_tree"] == (
        "09c0f368ed112082c8a53fa6c83b286834bd855f2f817a7f281c9bb2ad7d3ee4"
    )


def test_outcome_report_rejects_parity_that_skips_frozen_cases() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["assemble_finetune_outcome_report"],
    )

    inputs = _actual_outcome_inputs(module)
    incomplete = module.StatsActiveParity(
        sample_count=8,
        image_preprocessing_max_abs=0.0,
        state_preprocessing_max_abs=0.0,
        preprocessing_max_abs=0.0,
        normalized_action_max_abs=0.0,
        physical_action_max_abs=0.0,
        physical_action_standardized_max_abs=0.0,
        gate_max_abs=0.0,
        passed=True,
        samples=(),
    )

    with pytest.raises(ValueError, match="every frozen held-out case"):
        module.assemble_finetune_outcome_report(
            **{**inputs, "parity": incomplete}
        )


def test_outcome_report_requires_raw_physical_error_in_the_parity_gate() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["assemble_finetune_outcome_report"],
    )
    inputs = _actual_outcome_inputs(module)
    original = inputs["parity"]
    records = [dict(record) for record in original.samples]
    records[0]["normalized_action_max_abs"] = 0.001
    records[0]["physical_action_max_abs"] = 0.006
    records[0]["physical_action_standardized_max_abs"] = 0.001
    weakened = module.StatsActiveParity(
        sample_count=56,
        image_preprocessing_max_abs=0.0,
        state_preprocessing_max_abs=0.0,
        preprocessing_max_abs=0.0,
        normalized_action_max_abs=0.001,
        physical_action_max_abs=0.006,
        physical_action_standardized_max_abs=0.001,
        gate_max_abs=0.001,
        passed=True,
        samples=tuple(records),
    )

    with pytest.raises(ValueError, match="parity gate maximum"):
        module.assemble_finetune_outcome_report(
            **{**inputs, "parity": weakened}
        )


def test_outcome_report_rejects_missing_fine_tuned_sample_evidence() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["assemble_finetune_outcome_report"],
    )
    inputs = _actual_outcome_inputs(module)
    incomplete = module.MAEEvaluation(
        framework="mlx",
        device="cpu",
        dtype="float32",
        sample_count=56,
        element_count=336,
        mae=4.0,
        absolute_error_sum=1344.0,
        samples=(),
    )

    with pytest.raises(ValueError, match="fine-tuned MAE evidence"):
        module.assemble_finetune_outcome_report(
            **{**inputs, "fine_mlx": incomplete}
        )


def _write_metrics(path: Path, module, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _complete_training_summary(*, steps: int = 1) -> dict[str, object]:
    return {
        "selected_steps": steps,
        "actual_training_seconds": 2.0,
        "peak_memory_bytes": 100,
        "final_loss": 1.0,
        "final_smoothed_loss": 1.0,
        "optimizer": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 1e-10,
            "grad_clip_norm": 10.0,
            "warmup_steps": 1000,
            "decay_steps": 30000,
            "decay_lr": 2.5e-6,
            "training_horizon": steps,
        },
    }


def test_metrics_validation_rejects_negative_training_semantics(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validated_metrics_sha256"],
    )
    path = tmp_path / "metrics.csv"
    _write_metrics(
        path,
        module,
        [
            {
                "step": 1,
                "loss": -1.0,
                "smoothed_loss": -1.0,
                "learning_rate": -1e-4,
                "gradient_norm": -2.0,
                "clip_coefficient": -1.0,
                "elapsed_seconds": -1.0,
                "updates_per_second": -1.0,
                "peak_memory_bytes": 100,
            }
        ],
    )

    with pytest.raises(ValueError, match="training metrics row 1 is invalid"):
        module._validated_metrics_sha256(
            path,
            expected_steps=1,
            training_run=_complete_training_summary(),
        )


def test_metrics_validation_reconciles_the_final_row_to_the_run(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validated_metrics_sha256"],
    )
    path = tmp_path / "metrics.csv"
    _write_metrics(
        path,
        module,
        [
            {
                "step": 1,
                "loss": 1.0,
                "smoothed_loss": 1.0,
                "learning_rate": 1e-4,
                "gradient_norm": 2.0,
                "clip_coefficient": 1.0,
                "elapsed_seconds": 1.0,
                "updates_per_second": 1.0,
                "peak_memory_bytes": 100,
            }
        ],
    )
    changed_run = {**_complete_training_summary(), "final_loss": 2.0}

    with pytest.raises(ValueError, match="final training metrics differ"):
        module._validated_metrics_sha256(
            path,
            expected_steps=1,
            training_run=changed_run,
        )


def test_metrics_validation_rejects_a_positive_but_wrong_learning_rate(
    tmp_path: Path,
) -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validated_metrics_sha256"],
    )
    path = tmp_path / "metrics.csv"
    _write_metrics(
        path,
        module,
        [
            {
                "step": 1,
                "loss": 1.0,
                "smoothed_loss": 1.0,
                "learning_rate": 5e-5,
                "gradient_norm": 2.0,
                "clip_coefficient": 1.0,
                "elapsed_seconds": 1.0,
                "updates_per_second": 1.0,
                "peak_memory_bytes": 100,
            }
        ],
    )

    with pytest.raises(ValueError, match="training metrics row 1 is invalid"):
        module._validated_metrics_sha256(
            path,
            expected_steps=1,
            training_run=_complete_training_summary(),
        )


def test_run_configuration_digest_is_recomputed_from_the_run_document() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validated_training_run_config_sha256"],
    )
    run = json.loads(
        Path(".cache/training/t3/run.json").read_text(encoding="utf-8")
    )

    assert module._validated_training_run_config_sha256(run) == (
        "a13ccee79e82ccbbd4717f8245d005cb2c856f05ab42a3ca6de90fbcf30050c3"
    )

    changed = {**run, "optimizer": {**run["optimizer"], "lr": 5e-5}}
    with pytest.raises(ValueError, match="recomputed training run configuration"):
        module._validated_training_run_config_sha256(changed)


def test_t3b_run_configuration_digest_reconstructs_fixed_expert_only_scope() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validated_training_run_config_sha256"],
    )
    run = json.loads(
        Path(".cache/training/t3b/run.json").read_text(encoding="utf-8")
    )

    assert module._validated_training_run_config_sha256(run) == (
        "09895b216aff79ea3e26294aa4ef0484e5d316ee88eef7733782f95a9da62350"
    )

    changed = {**run, "lora": {**run["lora"], "scope": "legacy_full"}}
    with pytest.raises(ValueError, match="recomputed training run configuration"):
        module._validated_training_run_config_sha256(changed)


def test_completed_t3b_artifacts_bind_expert_only_adapter_metadata() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validate_completed_training_artifacts"],
    )
    run = json.loads(
        Path(".cache/training/t3b/run.json").read_text(encoding="utf-8")
    )

    evidence, checkpoint = module._validate_completed_training_artifacts(
        Path(".cache/training/t3b"),
        run,
    )

    assert evidence["adapter"] == run["adapter_sha256"]
    assert checkpoint.completed_step == 3000


def test_t3b_expected_export_metadata_includes_lora_scope() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_expected_export_metadata"],
    )
    run = json.loads(
        Path(".cache/training/t3b/run.json").read_text(encoding="utf-8")
    )

    metadata = module._expected_export_metadata(run)

    assert metadata["lora_scope"] == "expert_only"
    assert metadata["support_file_sha256"] == {
        name: digest
        for name, digest in run["export"]["file_sha256"].items()
        if name != "model.safetensors"
    }


def test_completed_metrics_trace_is_validated_and_hashed_from_the_same_bytes() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validated_metrics_sha256"],
    )
    run = json.loads(
        Path(".cache/training/t3/run.json").read_text(encoding="utf-8")
    )
    _, checkpoint_state = module._validate_completed_training_artifacts(
        Path(".cache/training/t3"),
        run,
    )

    digest = module._validated_metrics_sha256(
        Path(".cache/training/t3/metrics.csv"),
        expected_steps=3000,
        training_run=run,
        checkpoint_state=checkpoint_state,
    )

    assert digest == (
        "7f3a8c070f8102d7edc0afe5a9f4e5088321d1cdd21548fc21e9c772dbfafc2c"
    )

    changed_update = replace(
        checkpoint_state.last_update,
        gradient_norm=checkpoint_state.last_update.gradient_norm + 1.0,
    )
    changed_checkpoint = replace(checkpoint_state, last_update=changed_update)
    with pytest.raises(ValueError, match="final checkpoint"):
        module._validated_metrics_sha256(
            Path(".cache/training/t3/metrics.csv"),
            expected_steps=3000,
            training_run=run,
            checkpoint_state=changed_checkpoint,
        )


def test_expected_export_metadata_binds_run_and_frozen_evaluation_sources() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_expected_export_metadata"],
    )
    run = json.loads(
        Path(".cache/training/t3/run.json").read_text(encoding="utf-8")
    )

    metadata = module._expected_export_metadata(run)

    assert metadata["run_config_sha256"] == (
        "a13ccee79e82ccbbd4717f8245d005cb2c856f05ab42a3ca6de90fbcf30050c3"
    )
    assert metadata["evaluation_manifest_sha256"] == (
        "9cabca6cd21e8658a94e42980af3e91ecd8ff5ed5daca5f75eb7a1ebd1d261a3"
    )
    assert metadata["evaluation_metadata_sha256"] == (
        "f49ee54aead7ce3ede7b94d5638864afd2e12ef57ae2622eb6574333820cd107"
    )
    assert metadata["base_report_sha256"] == (
        "211d6778b0530208ca2e81abe6f4002cc683e24d496a09ddbe39c100ebd4f7ce"
    )
    assert metadata["dataset"] == {
        "id": "lerobot/svla_so101_pickplace",
        "revision": "f641879e22172be7e8161d5e6c1503c2d2feb657",
    }


def test_native_conversion_is_cryptographically_bound_to_the_validated_export() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validate_native_conversion_for_export"],
    )
    converted_path = Path(
        ".cache/smolvla_mlx/policy-float32/converted/"
        "b607b2937c1abf79/float32/model.float32.safetensors"
    )

    evidence = module._validate_native_conversion_for_export(
        Path(".cache/training/t3/export"),
        converted_path,
        expected_source_sha256=(
            "1053c1a622b837032dbfe9933833415b358d010d746d4e17779c546a8549d179"
        ),
    )

    assert evidence == {
        "native_conversion_model": (
            "76d893b95c739cbd2a02598025e360a596edf3a7f90a8c4b1cb63d23ae54b42a"
        ),
        "native_conversion_name_map": (
            "1664c39008363b98587a8c8fc54ed3af5e899b7fd092944d146bbfe4efc17902"
        ),
    }


def test_outcome_report_rejects_incomplete_export_audit_metadata() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["assemble_finetune_outcome_report"],
    )
    inputs = _actual_outcome_inputs(module)
    export_manifest = dict(inputs["export_manifest"])
    metadata = dict(export_manifest["metadata"])
    del metadata["evaluation_metadata_sha256"]
    export_manifest["metadata"] = metadata

    with pytest.raises(ValueError, match="export audit metadata"):
        module.assemble_finetune_outcome_report(
            **{**inputs, "export_manifest": export_manifest}
        )


def test_completed_adapter_and_final_checkpoint_are_hash_validated() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validate_completed_training_artifacts"],
    )
    run = json.loads(
        Path(".cache/training/t3/run.json").read_text(encoding="utf-8")
    )

    evidence, checkpoint_state = module._validate_completed_training_artifacts(
        Path(".cache/training/t3"),
        run,
    )

    assert evidence == {
        "adapter": "814e6f4b2a78a46b609aa7b48a28b4509f709d3e851e588dcd9a4bd2ca1408dc",
        "adapter_metadata": (
            "0f088f4d42655d0cd5915dca1a80819f1c14d6f5dd44236a16ff644520c50b98"
        ),
        "final_checkpoint_metadata": (
            "5d912a1e94bb1809c8fe72570450dc1c6d5c6c8973e8bcc00b1045eb465431ef"
        ),
        "final_checkpoint_model": (
            "814e6f4b2a78a46b609aa7b48a28b4509f709d3e851e588dcd9a4bd2ca1408dc"
        ),
        "final_checkpoint_optimizer": (
            "c9440be75315e04c1812ba18da0e0daccd2990fb0f6fdb1841d40ef7b01ffb5a"
        ),
    }
    assert checkpoint_state.completed_step == 3000
    assert checkpoint_state.samples_consumed == 24_000
    assert checkpoint_state.flow_draw_count == 24_000


def test_exported_processor_statistics_derive_from_the_frozen_train_rows() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["_validate_export_statistics"],
    )
    run = json.loads(
        Path(".cache/training/t3/run.json").read_text(encoding="utf-8")
    )

    digest = module._validate_export_statistics(
        export_dir=Path(".cache/training/t3/export"),
        cache_dir=Path(".cache/hf"),
        train_episodes=tuple(run["split"]["train_episodes"]),
        expected_sha256=run["train_statistics_sha256"],
    )

    assert digest == (
        "5aa5ab85e0c71c0adee97782be37907b0918050a8539bb3aab88fe392953948e"
    )


def test_outcome_report_rejects_a_changed_baseline_with_its_new_hash() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["assemble_finetune_outcome_report"],
    )
    inputs = _actual_outcome_inputs(module)
    changed = {**inputs["base_report"], "mlx_mae": 40.0}
    changed_payload = (json.dumps(changed, indent=2, sort_keys=True) + "\n").encode()

    with pytest.raises(ValueError, match="pre-training frozen base report"):
        module.assemble_finetune_outcome_report(
            **{
                **inputs,
                "base_report": changed,
                "base_report_sha256": hashlib.sha256(changed_payload).hexdigest(),
            }
        )


def test_outcome_report_rejects_internally_inconsistent_base_evidence() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["assemble_finetune_outcome_report"],
    )
    inputs = _actual_outcome_inputs(module)
    changed = {
        **inputs["base_report"],
        "absolute_error_sum": inputs["base_report"]["absolute_error_sum"] + 1.0,
    }

    with pytest.raises(ValueError, match="base evaluation evidence is inconsistent"):
        module.assemble_finetune_outcome_report(
            **{**inputs, "base_report": changed}
        )


def test_outcome_report_binds_the_complete_run_export_and_frozen_population() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["StatsActiveParity", "assemble_finetune_outcome_report"],
    )
    report = module.assemble_finetune_outcome_report(
        **_actual_outcome_inputs(module)
    )

    assert report["artifact_type"] == "smolvla-lora-finetune-outcome"
    assert report["source_sha256"]["evaluation_manifest"] == (
        "9cabca6cd21e8658a94e42980af3e91ecd8ff5ed5daca5f75eb7a1ebd1d261a3"
    )
    assert report["gates"]["passed"]
    assert report["gates"]["fine_to_base_ratio"] < 0.9
    assert report["gates"]["torch_to_mlx_ratio"] == 1.0
    assert report["training"]["selected_steps"] == 3000


def test_outcome_report_rejects_an_incomplete_training_run() -> None:
    module = __import__(
        "training.evaluation",
        fromlist=["StatsActiveParity", "assemble_finetune_outcome_report"],
    )
    parity = module.StatsActiveParity(
        sample_count=8,
        image_preprocessing_max_abs=0.0,
        state_preprocessing_max_abs=0.0,
        preprocessing_max_abs=0.0,
        normalized_action_max_abs=0.0,
        physical_action_max_abs=0.0,
        physical_action_standardized_max_abs=0.0,
        gate_max_abs=0.0,
        passed=True,
        samples=(),
    )

    try:
        module.assemble_finetune_outcome_report(
            training_run={"artifact_type": "smolvla-mlx-lora-run", "status": "running"},
            training_run_sha256="a" * 64,
            metrics_sha256="b" * 64,
            training_artifact_sha256={},
            native_conversion_sha256={},
            evaluation_manifest={},
            evaluation_metadata_sha256="e" * 64,
            base_report={},
            base_report_sha256="c" * 64,
            export_manifest={},
            export_manifest_sha256="d" * 64,
            fine_mlx=_mae_evaluation(module, framework="mlx", mae=8.0),
            torch_result=_mae_evaluation(module, framework="torch", mae=8.0),
            parity=parity,
        )
    except ValueError as error:
        assert "complete" in str(error)
    else:
        raise AssertionError("an incomplete training run was evaluated")
