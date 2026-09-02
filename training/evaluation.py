"""Frozen held-out MAE, Torch round trip, and stats-active parity for T3."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Mapping

import mlx.core as mx
import numpy as np

from reference.discovery import DATASET_ID, DATASET_REVISION
from smolvla_mlx.convert import validate_converted_checkpoint
from training.data import TrainingArtifact, TrainingArtifactWriter
from training.dataset import (
    SPLIT_SEED,
    TrainingDataBridge,
    compute_train_statistics,
    make_episode_split,
    make_heldout_case_specs,
)
from training.export import resolve_base_checkpoint, validate_merged_checkpoint_export
from training.finetune import (
    ADAPTIVE_BUDGET_MODE,
    FIXED_BUDGET_MODE,
    METRICS_FIELDS,
    CheckpointState,
    FineTuneConfig,
    _read_checkpoint_directory,
    training_run_config_sha256,
    write_run_state,
)
from training.optimizer import SmolVLAOptimizerConfig, cosine_decay_with_warmup_lr
from training.preprocessing import (
    StatsAwareSmolVLAPreprocessor,
    load_stats_aware_policy,
)
from training.t3_contract import (
    FROZEN_BASE_REPORT_SHA256,
    FROZEN_DATASET_REVISION_TREE_SHA256,
    FROZEN_EVALUATION_MANIFEST_SHA256,
    FROZEN_EVALUATION_METADATA_SHA256,
    FROZEN_TRAIN_STATISTICS_SHA256,
    frozen_export_audit_metadata,
)


MAE_IMPROVEMENT_RATIO_MAXIMUM = 0.9
TORCH_MLX_MAE_RATIO_MINIMUM = 0.95
TORCH_MLX_MAE_RATIO_MAXIMUM = 1.05
IMAGE_PREPROCESSING_MAX_ABSOLUTE_TOLERANCE = 1e-5
STATE_PREPROCESSING_MAX_ABSOLUTE_TOLERANCE = 1e-6
INFERENCE_MAX_ABSOLUTE_TOLERANCE = 5e-3
EVALUATION_MINIMUM_FREE_BYTES = 40 * 1024**3
EVALUATION_SAMPLE_COUNT = 56
EVALUATION_NOISE_SEED = 20_260_902
_SAMPLES_PER_EPISODE = 7
_EXPECTED_TENSORS_PER_CASE = 5
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_CACHE = _REPOSITORY_ROOT / ".cache"
_EXPORT_SUPPORT_FILE_NAMES = {
    "config.json",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
}


def _require_repository_cache_path(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    if not absolute.is_relative_to(_REPOSITORY_CACHE):
        raise ValueError(f"{label} must stay under the repository-local .cache directory")
    return absolute


def _require_real_directory(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}") from error
    if resolved != absolute or not absolute.is_dir():
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    return absolute


def _require_real_file(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}") from error
    if resolved != absolute or not absolute.is_file():
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    return absolute


def _require_minimum_free_bytes(free_bytes: int) -> None:
    if free_bytes < EVALUATION_MINIMUM_FREE_BYTES:
        raise RuntimeError(
            "Stage T3 evaluation requires at least 40 GiB free; "
            f"only {free_bytes} bytes are available"
        )


def _require_pinned_dataset_root(cache_dir: Path) -> Path:
    """Reject symlinks at every pinned-dataset path consumed by evaluation."""

    cache_dir = _require_real_directory(cache_dir, label="Hugging Face cache directory")
    dataset_root = _require_real_directory(
        cache_dir / "datasets" / "svla_so101_pickplace",
        label="pinned dataset directory",
    )
    required_files = {
        "pinned dataset rows": dataset_root / "data" / "chunk-000" / "file-000.parquet",
        "pinned dataset information": dataset_root / "meta" / "info.json",
        "pinned dataset statistics": dataset_root / "meta" / "stats.json",
        "pinned dataset tasks": dataset_root / "meta" / "tasks.parquet",
    }
    for label, path in required_files.items():
        _require_real_file(path, label=label)
    revision_tree_path = _require_real_file(
        dataset_root
        / ".cache"
        / "huggingface"
        / "trees"
        / f"{DATASET_REVISION}.json",
        label="pinned dataset revision tree",
    )
    revision_tree_payload = revision_tree_path.read_bytes()
    if (
        hashlib.sha256(revision_tree_payload).hexdigest()
        != FROZEN_DATASET_REVISION_TREE_SHA256
    ):
        raise ValueError("pinned dataset revision tree differs from the audited revision")
    revision_tree = json.loads(revision_tree_payload)
    tree_files = revision_tree.get("files") if isinstance(revision_tree, Mapping) else None
    if (
        not isinstance(revision_tree, Mapping)
        or revision_tree.get("format_version") != 1
        or not isinstance(tree_files, Mapping)
    ):
        raise ValueError("pinned dataset revision tree is invalid")
    relative_paths = (
        Path("data/chunk-000/file-000.parquet"),
        Path("meta/info.json"),
        Path("meta/stats.json"),
        Path("meta/tasks.parquet"),
    )
    for relative_path in relative_paths:
        path = dataset_root / relative_path
        record = tree_files.get(relative_path.as_posix())
        if not isinstance(record, Mapping):
            raise ValueError("pinned dataset revision tree is incomplete")
        payload = path.read_bytes()
        expected_size = record.get("size")
        lfs_sha256 = record.get("lfs_sha256")
        blob_id = record.get("blob_id")
        if isinstance(lfs_sha256, str):
            actual_digest = hashlib.sha256(payload).hexdigest()
            expected_digest = lfs_sha256
        elif isinstance(blob_id, str):
            header = f"blob {len(payload)}\0".encode("ascii")
            actual_digest = hashlib.sha1(header + payload).hexdigest()
            expected_digest = blob_id
        else:
            raise ValueError("pinned dataset revision tree has no file digest")
        if expected_size != len(payload) or actual_digest != expected_digest:
            raise ValueError(
                f"pinned dataset file differs from revision: {relative_path}"
            )
    return dataset_root


def _validate_evaluation_metadata_against_dataset(
    metadata: Mapping[str, object],
    dataset_root: Path,
) -> None:
    """Reconstruct the captured metadata from the pinned dataset and frozen algorithm."""

    import pyarrow.parquet as parquet

    dataset_root = _require_real_directory(
        dataset_root,
        label="pinned dataset directory",
    )
    data_path = _require_real_file(
        dataset_root / "data" / "chunk-000" / "file-000.parquet",
        label="pinned dataset rows",
    )
    task_path = _require_real_file(
        dataset_root / "meta" / "tasks.parquet",
        label="pinned dataset tasks",
    )
    _require_real_file(
        dataset_root / "meta" / "stats.json",
        label="pinned dataset statistics",
    )
    split = make_episode_split(num_episodes=50, seed=SPLIT_SEED)
    statistics = compute_train_statistics(dataset_root, split.train_episodes)
    specs = make_heldout_case_specs(
        dataset_root,
        split.holdout_episodes,
        samples_per_episode=_SAMPLES_PER_EPISODE,
        chunk_size=50,
    )
    rows = parquet.read_table(
        data_path,
        columns=["episode_index", "frame_index", "index", "task_index"],
    )
    tasks = parquet.read_table(task_path)
    task_name_columns = [name for name in tasks.column_names if name != "task_index"]
    if task_name_columns != ["__index_level_0__"]:
        raise ValueError("pinned dataset task schema differs from the captured source")
    task_indices = np.asarray(tasks["task_index"], dtype=np.int64)
    task_names = tasks[task_name_columns[0]].to_pylist()
    if len(task_indices) != len(task_names) or len(set(task_indices.tolist())) != len(
        task_indices
    ):
        raise ValueError("pinned dataset task table is invalid")
    task_by_index = {
        int(task_index): str(task_name)
        for task_index, task_name in zip(task_indices, task_names, strict=True)
    }
    absolute_indices = np.asarray(rows["index"], dtype=np.int64)
    if len(set(absolute_indices.tolist())) != len(absolute_indices):
        raise ValueError("pinned dataset absolute indices are not unique")
    row_by_index = {
        int(absolute_index): position
        for position, absolute_index in enumerate(absolute_indices)
    }
    episode_indices = np.asarray(rows["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(rows["frame_index"], dtype=np.int64)
    row_task_indices = np.asarray(rows["task_index"], dtype=np.int64)
    cases: list[dict[str, object]] = []
    for ordinal, spec in enumerate(specs):
        try:
            position = row_by_index[spec.absolute_index]
            task = task_by_index[int(row_task_indices[position])]
        except KeyError as error:
            raise ValueError("pinned dataset case identity is incomplete") from error
        if (
            int(episode_indices[position]) != spec.episode
            or int(frame_indices[position]) != spec.frame_index
        ):
            raise ValueError("pinned dataset case identity is inconsistent")
        cases.append(
            {
                "ordinal": ordinal,
                "episode": spec.episode,
                "frame_index": spec.frame_index,
                "absolute_index": spec.absolute_index,
                # The precommitted capture path ran SmolVLA's newline task step.
                "task": task if task.endswith("\n") else f"{task}\n",
            }
        )
    expected = {
        "format_version": 1,
        "artifact_type": "smolvla-lora-heldout-evaluation",
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "split_seed": SPLIT_SEED,
        "noise_seed": EVALUATION_NOISE_SEED,
        "sample_count": EVALUATION_SAMPLE_COUNT,
        "samples_per_episode": _SAMPLES_PER_EPISODE,
        "train_episodes": list(split.train_episodes),
        "heldout_episodes": list(split.holdout_episodes),
        "train_statistics_sha256": statistics.sha256,
        "metric": "physical first-action MAE over six action dimensions",
        "cases": cases,
        "manifest_sha256": FROZEN_EVALUATION_MANIFEST_SHA256,
        "tensor_count": EVALUATION_SAMPLE_COUNT * _EXPECTED_TENSORS_PER_CASE,
    }
    if dict(metadata) != expected:
        raise ValueError(
            "evaluation metadata differs from its canonical pinned-dataset reconstruction"
        )


def _frozen_evaluation_artifact(
    root: str | Path,
    *,
    dataset_root: Path | None = None,
) -> tuple[TrainingArtifact, str]:
    root = _require_real_directory(Path(root), label="held-out evaluation directory")
    metadata_path = _require_real_file(
        root / "metadata.json",
        label="held-out evaluation metadata",
    )
    _require_real_file(
        root / "manifest.json",
        label="held-out evaluation tensor manifest",
    )
    metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if metadata_sha256 != FROZEN_EVALUATION_METADATA_SHA256:
        raise ValueError(
            "evaluation metadata SHA-256 differs from the pre-training frozen artifact"
        )
    artifact = TrainingArtifact(root)
    if artifact.metadata.get("manifest_sha256") != FROZEN_EVALUATION_MANIFEST_SHA256:
        raise ValueError(
            "evaluation tensor-manifest SHA-256 differs from the pre-training frozen artifact"
        )
    if dataset_root is None:
        dataset_root = _require_pinned_dataset_root(_REPOSITORY_CACHE / "hf")
    _validate_evaluation_metadata_against_dataset(artifact.metadata, dataset_root)
    for record in artifact.manifest.values():
        _require_real_file(
            root / str(record["path"]),
            label="held-out evaluation tensor",
        )
    return artifact, metadata_sha256


@dataclass(frozen=True)
class EvaluationCase:
    """One immutable raw held-out observation, target, and flow draw."""

    ordinal: int
    episode: int
    frame_index: int
    absolute_index: int
    task: str
    camera1: np.ndarray
    camera2: np.ndarray
    state: np.ndarray
    target_action: np.ndarray
    noise: np.ndarray

    @property
    def observation(self) -> dict[str, np.ndarray | str]:
        return {
            "observation.images.camera1": self.camera1,
            "observation.images.camera2": self.camera2,
            "observation.state": self.state,
            "task": self.task,
        }


def capture_evaluation_artifact(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    split_seed: int = SPLIT_SEED,
    noise_seed: int = EVALUATION_NOISE_SEED,
) -> dict[str, object]:
    """Capture 56 unseen observations/targets/noise arrays before training."""

    cache_dir = Path(cache_dir)
    split = make_episode_split(num_episodes=50, seed=split_seed)
    dataset_root = cache_dir / "datasets" / "svla_so101_pickplace"
    stats = compute_train_statistics(dataset_root, split.train_episodes)
    specs = make_heldout_case_specs(
        dataset_root,
        split.holdout_episodes,
        samples_per_episode=_SAMPLES_PER_EPISODE,
        chunk_size=50,
    )
    if len(specs) != EVALUATION_SAMPLE_COUNT:
        raise RuntimeError(f"held-out case count changed: {len(specs)}")
    bridge = TrainingDataBridge(
        cache_dir=cache_dir,
        episodes=split.holdout_episodes,
        sampler_seed=split_seed,
        stats=stats.processor_stats,
    )
    generator = np.random.default_rng(noise_seed)
    writer = TrainingArtifactWriter(Path(output_dir))
    case_metadata = []
    for ordinal, spec in enumerate(specs):
        batch = bridge.frame(episode=spec.episode, frame_index=spec.frame_index)
        if batch.absolute_index != spec.absolute_index:
            raise RuntimeError("evaluation bridge identity differs from frozen specification")
        camera1 = np.asarray(batch.observation["observation.images.camera1"])
        camera2 = np.asarray(batch.observation["observation.images.camera2"])
        state = np.asarray(batch.observation["observation.state"], dtype=np.float32)
        target = np.asarray(batch.raw_actions[0, 0], dtype=np.float32)
        noise = generator.standard_normal((1, 50, 32), dtype=np.float32)
        prefix = f"cases/{ordinal:03d}"
        writer.add(f"{prefix}/camera1", camera1)
        writer.add(f"{prefix}/camera2", camera2)
        writer.add(f"{prefix}/state", state)
        writer.add(f"{prefix}/target_action", target)
        writer.add(f"{prefix}/noise", noise)
        case_metadata.append(
            {
                "ordinal": ordinal,
                "episode": spec.episode,
                "frame_index": spec.frame_index,
                "absolute_index": spec.absolute_index,
                "task": batch.task,
            }
        )
    return writer.finalize(
        {
            "format_version": 1,
            "artifact_type": "smolvla-lora-heldout-evaluation",
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "split_seed": split_seed,
            "noise_seed": noise_seed,
            "sample_count": len(specs),
            "samples_per_episode": _SAMPLES_PER_EPISODE,
            "train_episodes": list(split.train_episodes),
            "heldout_episodes": list(split.holdout_episodes),
            "train_statistics_sha256": stats.sha256,
            "metric": "physical first-action MAE over six action dimensions",
            "cases": case_metadata,
        }
    )


def load_evaluation_cases(
    root: str | Path,
    *,
    dataset_root: Path | None = None,
) -> tuple[EvaluationCase, ...]:
    """Strictly verify and load every frozen held-out case."""

    artifact, _ = _frozen_evaluation_artifact(root, dataset_root=dataset_root)
    names = artifact.verify_all()
    metadata = artifact.metadata
    required = {
        "artifact_type": "smolvla-lora-heldout-evaluation",
        "sample_count": EVALUATION_SAMPLE_COUNT,
        "samples_per_episode": _SAMPLES_PER_EPISODE,
        "noise_seed": EVALUATION_NOISE_SEED,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"evaluation artifact {key} changed: {metadata.get(key)!r}")
    if len(names) != EVALUATION_SAMPLE_COUNT * _EXPECTED_TENSORS_PER_CASE:
        raise ValueError("evaluation artifact tensor count changed")
    raw_cases = metadata.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != EVALUATION_SAMPLE_COUNT:
        raise ValueError("evaluation artifact case metadata is incomplete")
    cases = []
    for ordinal, raw in enumerate(raw_cases):
        if not isinstance(raw, dict) or raw.get("ordinal") != ordinal:
            raise ValueError(f"evaluation case metadata changed at ordinal {ordinal}")
        task = raw.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError(f"evaluation task is invalid at ordinal {ordinal}")
        prefix = f"cases/{ordinal:03d}"
        case = EvaluationCase(
            ordinal=ordinal,
            episode=int(raw["episode"]),
            frame_index=int(raw["frame_index"]),
            absolute_index=int(raw["absolute_index"]),
            task=task,
            camera1=artifact.load(f"{prefix}/camera1"),
            camera2=artifact.load(f"{prefix}/camera2"),
            state=artifact.load(f"{prefix}/state").astype(np.float32, copy=False),
            target_action=artifact.load(f"{prefix}/target_action").astype(np.float32, copy=False),
            noise=artifact.load(f"{prefix}/noise").astype(np.float32, copy=False),
        )
        if case.camera1.dtype != np.uint8 or case.camera2.dtype != np.uint8:
            raise ValueError(f"evaluation cameras must remain uint8 at ordinal {ordinal}")
        if case.state.shape != (6,) or case.target_action.shape != (6,):
            raise ValueError(f"evaluation physical vector shape changed at ordinal {ordinal}")
        if case.noise.shape != (1, 50, 32) or not np.isfinite(case.noise).all():
            raise ValueError(f"evaluation noise is invalid at ordinal {ordinal}")
        cases.append(case)
    if len({case.absolute_index for case in cases}) != EVALUATION_SAMPLE_COUNT:
        raise ValueError("evaluation case identities are not unique")
    return tuple(cases)


def absolute_error(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """Return float64-accumulated absolute-error sum and element count."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shapes differ: {prediction.shape} != {target.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("MAE inputs contain non-finite values")
    error = np.abs(prediction - target)
    return float(np.sum(error, dtype=np.float64)), int(error.size)


@dataclass(frozen=True)
class MAEEvaluation:
    """Complete per-case physical-action MAE evidence for one policy."""

    framework: str
    device: str
    dtype: str
    sample_count: int
    element_count: int
    mae: float
    absolute_error_sum: float
    samples: tuple[Mapping[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["samples"] = [dict(sample) for sample in self.samples]
        return value


def load_base_stats_policy(
    *,
    cache_dir: str | Path,
    native_cache: str | Path,
):
    """Load the immutable fp32 base with the frozen train-only statistics."""

    from smolvla_mlx.policy import SmolVLAMLX

    cache_dir = Path(cache_dir)
    split = make_episode_split(num_episodes=50, seed=SPLIT_SEED)
    stats = compute_train_statistics(
        cache_dir / "datasets" / "svla_so101_pickplace",
        split.train_episodes,
    )
    checkpoint = resolve_base_checkpoint(cache_dir)
    with mx.stream(mx.cpu):
        policy = SmolVLAMLX.from_pretrained(
            checkpoint,
            cache_dir=native_cache,
            dtype=mx.float32,
            execution_mode="strict",
        )
        policy.preprocessor = StatsAwareSmolVLAPreprocessor(
            base=policy.preprocessor,
            state_mean=stats.stats["observation.state"]["mean"],
            state_std=stats.stats["observation.state"]["std"],
            action_mean=stats.stats["action"]["mean"],
            action_std=stats.stats["action"]["std"],
        )
    return policy


def evaluate_mlx_policy(
    policy,
    cases: tuple[EvaluationCase, ...],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> MAEEvaluation:
    """Score a native fp32 policy on identical physical first-action targets."""

    error_sum = 0.0
    element_count = 0
    records = []
    with mx.stream(mx.cpu):
        if mx.default_device() != mx.cpu:
            raise RuntimeError("held-out MLX evaluation must run on the fp32 CPU compatibility lane")
        for index, case in enumerate(cases):
            normalized = policy.predict_action_chunk(
                case.observation,
                noise=mx.array(case.noise).astype(mx.float32),
            )
            physical = policy.preprocessor.unnormalize_actions(normalized)
            mx.eval(physical)
            prediction = np.asarray(physical.astype(mx.float32))[0, 0]
            case_sum, case_count = absolute_error(prediction, case.target_action)
            error_sum += case_sum
            element_count += case_count
            records.append(
                {
                    "ordinal": case.ordinal,
                    "episode": case.episode,
                    "frame_index": case.frame_index,
                    "absolute_index": case.absolute_index,
                    "absolute_error_sum": case_sum,
                    "element_count": case_count,
                }
            )
            if progress is not None:
                progress(index + 1, len(cases))
    if element_count == 0:
        raise RuntimeError("held-out MLX evaluation produced no metric elements")
    return MAEEvaluation(
        framework="mlx",
        device="Device(cpu, 0)",
        dtype="float32",
        sample_count=len(cases),
        element_count=element_count,
        mae=error_sum / element_count,
        absolute_error_sum=error_sum,
        samples=tuple(records),
    )


def _torch_observation(case: EvaluationCase):
    import torch

    return {
        "observation.images.camera1": torch.from_numpy(case.camera1.copy()).float() / 255.0,
        "observation.images.camera2": torch.from_numpy(case.camera2.copy()).float() / 255.0,
        "observation.state": torch.from_numpy(case.state.copy()).float(),
        "task": case.task,
    }


def evaluate_torch_export(
    reference,
    cases: tuple[EvaluationCase, ...],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> MAEEvaluation:
    """Score the strict Torch export on the exact same cases and flow noise."""

    import torch

    error_sum = 0.0
    element_count = 0
    records = []
    for index, case in enumerate(cases):
        batch = reference.preprocessor(_torch_observation(case))
        noise = torch.from_numpy(case.noise.copy()).float()
        with torch.inference_mode():
            normalized = reference.policy.predict_action_chunk(batch, noise=noise)
            physical = reference.postprocessor(normalized)
        prediction = physical.detach().cpu().float().numpy()[0, 0]
        case_sum, case_count = absolute_error(prediction, case.target_action)
        error_sum += case_sum
        element_count += case_count
        records.append(
            {
                "ordinal": case.ordinal,
                "episode": case.episode,
                "frame_index": case.frame_index,
                "absolute_index": case.absolute_index,
                "absolute_error_sum": case_sum,
                "element_count": case_count,
            }
        )
        if progress is not None:
            progress(index + 1, len(cases))
    return MAEEvaluation(
        framework="torch",
        device=str(reference.device),
        dtype=str(reference.dtype).removeprefix("torch."),
        sample_count=len(cases),
        element_count=element_count,
        mae=error_sum / element_count,
        absolute_error_sum=error_sum,
        samples=tuple(records),
    )


@dataclass(frozen=True)
class StatsActiveParity:
    """P0 preprocessing, P1 normalized-flow, and P2 stats-active action parity."""

    sample_count: int
    image_preprocessing_max_abs: float
    state_preprocessing_max_abs: float
    preprocessing_max_abs: float
    normalized_action_max_abs: float
    physical_action_max_abs: float
    physical_action_standardized_max_abs: float
    gate_max_abs: float
    passed: bool
    samples: tuple[Mapping[str, object], ...]


def _stats_active_parity_passed(
    *,
    image_preprocessing_max_abs: float,
    state_preprocessing_max_abs: float,
    normalized_action_max_abs: float,
    physical_action_max_abs: float,
    physical_action_standardized_max_abs: float,
) -> bool:
    values = (
        image_preprocessing_max_abs,
        state_preprocessing_max_abs,
        normalized_action_max_abs,
        physical_action_max_abs,
        physical_action_standardized_max_abs,
    )
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("stats-active parity metrics must be finite and nonnegative")
    return (
        image_preprocessing_max_abs
        <= IMAGE_PREPROCESSING_MAX_ABSOLUTE_TOLERANCE
        and state_preprocessing_max_abs
        <= STATE_PREPROCESSING_MAX_ABSOLUTE_TOLERANCE
        and normalized_action_max_abs <= INFERENCE_MAX_ABSOLUTE_TOLERANCE
        and physical_action_max_abs <= INFERENCE_MAX_ABSOLUTE_TOLERANCE
        and physical_action_standardized_max_abs
        <= INFERENCE_MAX_ABSOLUTE_TOLERANCE
    )


def run_stats_active_parity(
    mlx_policy,
    reference,
    cases: tuple[EvaluationCase, ...],
) -> StatsActiveParity:
    """Run every pre-training frozen case through the three parity levels."""

    import torch
    from lerobot.policies.common.vla_utils import resize_with_pad

    if len(cases) != EVALUATION_SAMPLE_COUNT:
        raise ValueError("stats-active parity requires every frozen held-out case")
    records = []
    p0_image_max = 0.0
    p0_state_max = 0.0
    p1_max = 0.0
    p2_raw_max = 0.0
    p2_standardized_max = 0.0
    action_std = np.asarray(mlx_policy.preprocessor.action_std, dtype=np.float32)
    with mx.stream(mx.cpu):
        for case in cases:
            observation = _torch_observation(case)
            torch_batch = reference.preprocessor(observation)
            images = []
            masks = []
            for key in reference.policy.config.image_features:
                if key not in torch_batch:
                    continue
                image = torch_batch[key]
                image = image[:, -1] if image.ndim == 5 else image
                width, height = reference.policy.config.resize_imgs_with_padding
                image = resize_with_pad(image, height, width, pad_value=0)
                images.append(image * 2.0 - 1.0)
                masks.append(torch.ones(image.shape[0], dtype=torch.bool))
            torch_pixels = torch.cat(images, dim=0).cpu().float().numpy()
            torch_masks = torch.cat([mask.reshape(-1, 1) for mask in masks], dim=0).numpy()
            torch_state = torch_batch["observation.state"].cpu().float().numpy()
            torch_ids = torch_batch["observation.language.tokens"].cpu().numpy()
            torch_text_mask = torch_batch["observation.language.attention_mask"].cpu().numpy()

            native = mlx_policy.preprocessor(case.observation)
            mx.eval(
                native.pixel_values,
                native.pixel_attention_mask,
                native.input_ids,
                native.text_attention_mask,
                native.state,
            )
            if not np.array_equal(np.asarray(native.pixel_attention_mask), torch_masks):
                raise RuntimeError("stats-active parity image masks differ")
            if not np.array_equal(np.asarray(native.input_ids), torch_ids):
                raise RuntimeError("stats-active parity token IDs differ")
            if not np.array_equal(np.asarray(native.text_attention_mask), torch_text_mask):
                raise RuntimeError("stats-active parity token masks differ")
            image_preprocessing_max = float(
                np.max(np.abs(np.asarray(native.pixel_values) - torch_pixels))
            )
            state_preprocessing_max = float(
                np.max(np.abs(np.asarray(native.state) - torch_state))
            )
            preprocessing_max = max(
                image_preprocessing_max,
                state_preprocessing_max,
            )

            noise = torch.from_numpy(case.noise.copy()).float()
            with torch.inference_mode():
                torch_normalized = reference.policy.predict_action_chunk(torch_batch, noise=noise)
                torch_physical = reference.postprocessor(torch_normalized)
            mlx_normalized = mlx_policy.predict_action_chunk(
                case.observation,
                noise=mx.array(case.noise).astype(mx.float32),
            )
            mlx_physical = mlx_policy.preprocessor.unnormalize_actions(mlx_normalized)
            mx.eval(mlx_normalized, mlx_physical)
            torch_normalized_array = torch_normalized.cpu().float().numpy()
            torch_physical_array = torch_physical.cpu().float().numpy()
            mlx_normalized_array = np.asarray(mlx_normalized.astype(mx.float32))
            mlx_physical_array = np.asarray(mlx_physical.astype(mx.float32))
            normalized_max = float(
                np.max(np.abs(mlx_normalized_array - torch_normalized_array))
            )
            physical_difference = np.abs(mlx_physical_array - torch_physical_array)
            physical_max = float(np.max(physical_difference))
            standardized_max = float(np.max(physical_difference / action_std))
            p0_image_max = max(p0_image_max, image_preprocessing_max)
            p0_state_max = max(p0_state_max, state_preprocessing_max)
            p1_max = max(p1_max, normalized_max)
            p2_raw_max = max(p2_raw_max, physical_max)
            p2_standardized_max = max(p2_standardized_max, standardized_max)
            records.append(
                {
                    "ordinal": case.ordinal,
                    "episode": case.episode,
                    "frame_index": case.frame_index,
                    "absolute_index": case.absolute_index,
                    "image_preprocessing_max_abs": image_preprocessing_max,
                    "state_preprocessing_max_abs": state_preprocessing_max,
                    "preprocessing_max_abs": preprocessing_max,
                    "normalized_action_max_abs": normalized_max,
                    "physical_action_max_abs": physical_max,
                    "physical_action_standardized_max_abs": standardized_max,
                }
            )
    preprocessing_max = max(p0_image_max, p0_state_max)
    gate_max = max(
        preprocessing_max,
        p1_max,
        p2_raw_max,
        p2_standardized_max,
    )
    return StatsActiveParity(
        sample_count=len(cases),
        image_preprocessing_max_abs=p0_image_max,
        state_preprocessing_max_abs=p0_state_max,
        preprocessing_max_abs=preprocessing_max,
        normalized_action_max_abs=p1_max,
        physical_action_max_abs=p2_raw_max,
        physical_action_standardized_max_abs=p2_standardized_max,
        gate_max_abs=gate_max,
        passed=_stats_active_parity_passed(
            image_preprocessing_max_abs=p0_image_max,
            state_preprocessing_max_abs=p0_state_max,
            normalized_action_max_abs=p1_max,
            physical_action_max_abs=p2_raw_max,
            physical_action_standardized_max_abs=p2_standardized_max,
        ),
        samples=tuple(records),
    )


@dataclass(frozen=True)
class OutcomeGates:
    """All three independent immutable T3 outcome decisions."""

    passed: bool
    improvement_passed: bool
    roundtrip_passed: bool
    parity_passed: bool
    fine_to_base_ratio: float
    torch_to_mlx_ratio: float
    parity_max_abs: float


def evaluate_outcome_gates(
    *,
    base_mlx_mae: float,
    fine_mlx_mae: float,
    torch_mae: float,
    parity_max_abs: float,
    image_preprocessing_max_abs: float = 0.0,
    state_preprocessing_max_abs: float = 0.0,
) -> OutcomeGates:
    """Apply the brief's immutable thresholds without rounding or adjustment."""

    values = (
        base_mlx_mae,
        fine_mlx_mae,
        torch_mae,
        parity_max_abs,
        image_preprocessing_max_abs,
        state_preprocessing_max_abs,
    )
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("outcome metrics must be finite and nonnegative")
    if base_mlx_mae == 0 or fine_mlx_mae == 0:
        raise ValueError("outcome MAE ratios require nonzero MLX denominators")
    fine_to_base = fine_mlx_mae / base_mlx_mae
    torch_to_mlx = torch_mae / fine_mlx_mae
    improvement = fine_to_base <= MAE_IMPROVEMENT_RATIO_MAXIMUM
    roundtrip = TORCH_MLX_MAE_RATIO_MINIMUM <= torch_to_mlx <= TORCH_MLX_MAE_RATIO_MAXIMUM
    parity = _stats_active_parity_passed(
        image_preprocessing_max_abs=image_preprocessing_max_abs,
        state_preprocessing_max_abs=state_preprocessing_max_abs,
        normalized_action_max_abs=parity_max_abs,
        physical_action_max_abs=parity_max_abs,
        physical_action_standardized_max_abs=parity_max_abs,
    )
    return OutcomeGates(
        passed=improvement and roundtrip and parity,
        improvement_passed=improvement,
        roundtrip_passed=roundtrip,
        parity_passed=parity,
        fine_to_base_ratio=fine_to_base,
        torch_to_mlx_ratio=torch_to_mlx,
        parity_max_abs=parity_max_abs,
    )


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    return value


def _optimizer_config_from_training_run(
    training_run: Mapping[str, object],
) -> SmolVLAOptimizerConfig:
    value = training_run.get("optimizer")
    if not isinstance(value, Mapping):
        raise ValueError("training run is missing optimizer configuration")
    expected_fields = {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "grad_clip_norm",
        "warmup_steps",
        "decay_steps",
        "decay_lr",
        "training_horizon",
    }
    if set(value) != expected_fields:
        raise ValueError("training run optimizer fields differ from the frozen schema")
    try:
        betas = tuple(float(beta) for beta in value["betas"])
        return SmolVLAOptimizerConfig(
            lr=float(value["lr"]),
            betas=betas,
            eps=float(value["eps"]),
            weight_decay=float(value["weight_decay"]),
            grad_clip_norm=float(value["grad_clip_norm"]),
            warmup_steps=int(value["warmup_steps"]),
            decay_steps=int(value["decay_steps"]),
            decay_lr=float(value["decay_lr"]),
            training_horizon=int(value["training_horizon"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("training run optimizer configuration is invalid") from error


def _validated_training_run_config_sha256(
    training_run: Mapping[str, object],
) -> str:
    """Recompute every trajectory-affecting run field before trusting its digest."""

    recorded = _require_sha256(
        "training run configuration",
        training_run.get("run_config_sha256"),
    )
    benchmark = training_run.get("benchmark")
    split = training_run.get("split")
    lora = training_run.get("lora")
    base_artifact = training_run.get("base_artifact")
    if not all(isinstance(value, Mapping) for value in (split, lora, base_artifact)):
        raise ValueError("training run configuration cannot be reconstructed")
    budget_mode = training_run.get("budget_mode", ADAPTIVE_BUDGET_MODE)
    if budget_mode == FIXED_BUDGET_MODE:
        if benchmark is not None:
            raise ValueError("training run configuration cannot be reconstructed")
        benchmark_warmup_updates = FineTuneConfig.benchmark_warmup_updates
        benchmark_measured_updates = FineTuneConfig.benchmark_measured_updates
    elif budget_mode == ADAPTIVE_BUDGET_MODE and isinstance(benchmark, Mapping):
        benchmark_warmup_updates = int(benchmark["warmup_updates"])
        benchmark_measured_updates = int(benchmark["measured_updates"])
    else:
        raise ValueError("training run configuration cannot be reconstructed")
    if training_run.get("dataset") != {
        "id": DATASET_ID,
        "revision": DATASET_REVISION,
    }:
        raise ValueError("training run dataset differs from the pinned revision")
    if (
        training_run.get("base_dtype") != "bfloat16"
        or training_run.get("adapter_dtype") != "float32"
    ):
        raise ValueError("training run dtype configuration changed")
    try:
        config = FineTuneConfig(
            seed=int(training_run["seed"]),
            sampler_seed=int(training_run["sampler_seed"]),
            nominal_steps=int(training_run["nominal_steps"]),
            effective_batch_size=int(training_run["effective_batch_size"]),
            training_seconds=float(training_run["training_seconds_budget"]),
            benchmark_warmup_updates=benchmark_warmup_updates,
            benchmark_measured_updates=benchmark_measured_updates,
            rank=int(lora["rank"]),
            alpha=float(lora["alpha"]),
            dropout=float(lora["dropout"]),
            lora_scope=str(lora.get("scope", FineTuneConfig.lora_scope)),
            budget_mode=str(budget_mode),
            checkpoint_interval=int(training_run["checkpoint_interval"]),
        )
        optimizer_config = _optimizer_config_from_training_run(training_run)
        selected_steps = int(training_run["selected_steps"])
        recomputed = training_run_config_sha256(
            config,
            selected_steps=selected_steps,
            train_statistics_sha256=_require_sha256(
                "training statistics",
                training_run.get("train_statistics_sha256"),
            ),
            train_episodes=tuple(int(value) for value in split["train_episodes"]),
            holdout_episodes=tuple(int(value) for value in split["holdout_episodes"]),
            base_artifact={str(key): str(value) for key, value in base_artifact.items()},
            optimizer_config=optimizer_config,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("training run configuration cannot be reconstructed") from error
    if optimizer_config.training_horizon != selected_steps or recomputed != recorded:
        raise ValueError(
            "recomputed training run configuration differs from its recorded digest"
        )
    return recomputed


def _validate_base_evaluation_evidence(
    base_report: Mapping[str, object],
    evaluation_manifest: Mapping[str, object],
) -> None:
    samples = base_report.get("samples")
    cases = evaluation_manifest.get("cases")
    if (
        not isinstance(samples, list)
        or len(samples) != EVALUATION_SAMPLE_COUNT
        or not isinstance(cases, list)
        or len(cases) != EVALUATION_SAMPLE_COUNT
    ):
        raise ValueError("base evaluation evidence is inconsistent")
    error_sums: list[float] = []
    element_count = 0
    identity_fields = ("ordinal", "episode", "frame_index", "absolute_index")
    for ordinal, (sample, case) in enumerate(zip(samples, cases, strict=True)):
        if not isinstance(sample, Mapping) or not isinstance(case, Mapping):
            raise ValueError("base evaluation evidence is inconsistent")
        try:
            error_sum = float(sample["absolute_error_sum"])
            sample_elements = int(sample["element_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("base evaluation evidence is inconsistent") from error
        if (
            not math.isfinite(error_sum)
            or error_sum < 0
            or sample_elements != 6
            or sample.get("ordinal") != ordinal
            or any(sample.get(field) != case.get(field) for field in identity_fields)
        ):
            raise ValueError("base evaluation evidence is inconsistent")
        error_sums.append(error_sum)
        element_count += sample_elements
    total_error = math.fsum(error_sums)
    try:
        recorded_total = float(base_report["absolute_error_sum"])
        recorded_mae = float(base_report["mlx_mae"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("base evaluation evidence is inconsistent") from error
    if (
        element_count != EVALUATION_SAMPLE_COUNT * 6
        or total_error != recorded_total
        or recorded_mae != total_error / element_count
        or not math.isfinite(recorded_mae)
        or recorded_mae <= 0
    ):
        raise ValueError("base evaluation evidence is inconsistent")


def _validate_mae_evaluation_evidence(
    evaluation: MAEEvaluation,
    evaluation_manifest: Mapping[str, object],
    *,
    label: str,
) -> None:
    cases = evaluation_manifest.get("cases")
    if (
        len(evaluation.samples) != EVALUATION_SAMPLE_COUNT
        or not isinstance(cases, list)
        or len(cases) != EVALUATION_SAMPLE_COUNT
    ):
        raise ValueError(f"{label} MAE evidence is incomplete")
    identity_fields = ("ordinal", "episode", "frame_index", "absolute_index")
    error_sums: list[float] = []
    element_count = 0
    for ordinal, (sample, case) in enumerate(
        zip(evaluation.samples, cases, strict=True)
    ):
        if not isinstance(sample, Mapping) or not isinstance(case, Mapping):
            raise ValueError(f"{label} MAE evidence is inconsistent")
        try:
            error_sum = float(sample["absolute_error_sum"])
            sample_elements = int(sample["element_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{label} MAE evidence is inconsistent") from error
        if (
            not math.isfinite(error_sum)
            or error_sum < 0
            or sample_elements != 6
            or sample.get("ordinal") != ordinal
            or any(sample.get(field) != case.get(field) for field in identity_fields)
        ):
            raise ValueError(f"{label} MAE evidence is inconsistent")
        error_sums.append(error_sum)
        element_count += sample_elements
    total_error = math.fsum(error_sums)
    if (
        element_count != evaluation.element_count
        or total_error != evaluation.absolute_error_sum
        or evaluation.mae != total_error / element_count
        or not math.isfinite(evaluation.mae)
        or evaluation.mae < 0
    ):
        raise ValueError(f"{label} MAE evidence is inconsistent")


def assemble_finetune_outcome_report(
    *,
    training_run: Mapping[str, object],
    training_run_sha256: str,
    metrics_sha256: str,
    training_artifact_sha256: Mapping[str, str],
    native_conversion_sha256: Mapping[str, str],
    evaluation_manifest: Mapping[str, object],
    evaluation_metadata_sha256: str,
    base_report: Mapping[str, object],
    base_report_sha256: str,
    export_manifest: Mapping[str, object],
    export_manifest_sha256: str,
    fine_mlx: MAEEvaluation,
    torch_result: MAEEvaluation,
    parity: StatsActiveParity,
) -> dict[str, object]:
    """Bind complete T3 source artifacts and apply every immutable outcome gate."""

    if (
        training_run.get("artifact_type") != "smolvla-mlx-lora-run"
        or training_run.get("status") != "trained_and_exported"
    ):
        raise ValueError("fine-tune outcome evaluation requires a complete training run")
    base_report_sha256 = _require_sha256("base report", base_report_sha256)
    if base_report_sha256 != FROZEN_BASE_REPORT_SHA256:
        raise ValueError("base report differs from the pre-training frozen base report")
    split = training_run.get("split")
    lora = training_run.get("lora")
    if not isinstance(split, Mapping) or not isinstance(lora, Mapping):
        raise ValueError("complete training run is missing split or LoRA metadata")
    train_episodes = list(split.get("train_episodes", []))
    heldout_episodes = list(split.get("holdout_episodes", []))
    if not train_episodes or not heldout_episodes or set(train_episodes) & set(
        heldout_episodes
    ):
        raise ValueError("training run episode split is invalid")

    if (
        evaluation_manifest.get("artifact_type")
        != "smolvla-lora-heldout-evaluation"
        or evaluation_manifest.get("sample_count") != EVALUATION_SAMPLE_COUNT
    ):
        raise ValueError("held-out evaluation manifest identity is invalid")
    evaluation_manifest_sha256 = _require_sha256(
        "evaluation manifest",
        evaluation_manifest.get("manifest_sha256"),
    )
    if evaluation_manifest_sha256 != FROZEN_EVALUATION_MANIFEST_SHA256:
        raise ValueError("evaluation tensor manifest differs from the frozen population")
    evaluation_metadata_sha256 = _require_sha256(
        "evaluation metadata",
        evaluation_metadata_sha256,
    )
    if evaluation_metadata_sha256 != FROZEN_EVALUATION_METADATA_SHA256:
        raise ValueError("evaluation metadata differs from the frozen population")
    train_statistics_sha256 = _require_sha256(
        "training statistics",
        training_run.get("train_statistics_sha256"),
    )
    if train_statistics_sha256 != FROZEN_TRAIN_STATISTICS_SHA256:
        raise ValueError("training statistics differ from the pre-training frozen statistics")
    if (
        evaluation_manifest.get("train_statistics_sha256")
        != train_statistics_sha256
        or evaluation_manifest.get("train_episodes") != train_episodes
        or evaluation_manifest.get("heldout_episodes") != heldout_episodes
    ):
        raise ValueError("held-out evaluation population differs from the training run")

    if (
        base_report.get("artifact_type")
        != "smolvla-lora-base-heldout-evaluation"
        or base_report.get("evaluation_manifest_sha256")
        != evaluation_manifest_sha256
        or base_report.get("train_statistics_sha256")
        != train_statistics_sha256
        or base_report.get("sample_count") != EVALUATION_SAMPLE_COUNT
        or base_report.get("element_count") != EVALUATION_SAMPLE_COUNT * 6
    ):
        raise ValueError("base evaluation is not bound to the frozen population")
    _validate_base_evaluation_evidence(base_report, evaluation_manifest)

    export_metadata = export_manifest.get("metadata")
    if (
        export_manifest.get("artifact_type")
        != "smolvla-mlx-merged-training-checkpoint"
        or export_manifest.get("tensor_count") != 500
        or export_manifest.get("parameter_count") != 450_046_176
        or not isinstance(export_metadata, Mapping)
    ):
        raise ValueError("merged export manifest identity is invalid")
    adapter_sha256 = _require_sha256("adapter", training_run.get("adapter_sha256"))
    required_training_artifacts = {
        "adapter",
        "adapter_metadata",
        "final_checkpoint_metadata",
        "final_checkpoint_model",
        "final_checkpoint_optimizer",
    }
    if set(training_artifact_sha256) != required_training_artifacts:
        raise ValueError("completed training artifact digests are incomplete")
    validated_training_artifacts = {
        name: _require_sha256(name.replace("_", " "), digest)
        for name, digest in training_artifact_sha256.items()
    }
    if (
        validated_training_artifacts["adapter"] != adapter_sha256
        or validated_training_artifacts["final_checkpoint_model"] != adapter_sha256
    ):
        raise ValueError("adapter and final checkpoint model digests differ")
    expected_export_metadata = _expected_export_metadata(training_run)
    if dict(export_metadata) != expected_export_metadata:
        raise ValueError("merged export audit metadata is incomplete or changed")
    required_native_conversion = {
        "native_conversion_model",
        "native_conversion_name_map",
    }
    if set(native_conversion_sha256) != required_native_conversion:
        raise ValueError("native conversion artifact digests are incomplete")
    validated_native_conversion = {
        name: _require_sha256(name.replace("_", " "), digest)
        for name, digest in native_conversion_sha256.items()
    }

    expected_elements = EVALUATION_SAMPLE_COUNT * 6
    if (
        fine_mlx.framework != "mlx"
        or torch_result.framework != "torch"
        or fine_mlx.sample_count != EVALUATION_SAMPLE_COUNT
        or torch_result.sample_count != EVALUATION_SAMPLE_COUNT
        or fine_mlx.element_count != expected_elements
        or torch_result.element_count != expected_elements
    ):
        raise ValueError("fine-tuned MAE results differ from the frozen population")
    _validate_mae_evaluation_evidence(
        fine_mlx,
        evaluation_manifest,
        label="fine-tuned",
    )
    _validate_mae_evaluation_evidence(
        torch_result,
        evaluation_manifest,
        label="Torch round-trip",
    )
    parity_samples = parity.samples
    evaluation_cases = evaluation_manifest.get("cases")
    if (
        parity.sample_count != EVALUATION_SAMPLE_COUNT
        or len(parity_samples) != EVALUATION_SAMPLE_COUNT
        or not isinstance(evaluation_cases, list)
        or len(evaluation_cases) != EVALUATION_SAMPLE_COUNT
    ):
        raise ValueError("stats-active parity does not cover every frozen held-out case")
    parity_identity_fields = ("ordinal", "episode", "frame_index", "absolute_index")
    for ordinal, (sample, case) in enumerate(
        zip(parity_samples, evaluation_cases, strict=True)
    ):
        if (
            not isinstance(sample, Mapping)
            or not isinstance(case, Mapping)
            or sample.get("ordinal") != ordinal
            or any(sample.get(field) != case.get(field) for field in parity_identity_fields)
        ):
            raise ValueError("stats-active parity differs from the frozen case identities")
    parity_fields = (
        "image_preprocessing_max_abs",
        "state_preprocessing_max_abs",
        "preprocessing_max_abs",
        "normalized_action_max_abs",
        "physical_action_max_abs",
        "physical_action_standardized_max_abs",
    )
    sample_maxima = {field: 0.0 for field in parity_fields}
    for sample in parity_samples:
        for field in parity_fields:
            try:
                value = float(sample[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("stats-active parity sample evidence is invalid") from error
            if not math.isfinite(value) or value < 0:
                raise ValueError("stats-active parity sample evidence is invalid")
            sample_maxima[field] = max(sample_maxima[field], value)
    summary_maxima = {
        "image_preprocessing_max_abs": parity.image_preprocessing_max_abs,
        "state_preprocessing_max_abs": parity.state_preprocessing_max_abs,
        "preprocessing_max_abs": parity.preprocessing_max_abs,
        "normalized_action_max_abs": parity.normalized_action_max_abs,
        "physical_action_max_abs": parity.physical_action_max_abs,
        "physical_action_standardized_max_abs": (
            parity.physical_action_standardized_max_abs
        ),
    }
    if sample_maxima != summary_maxima:
        raise ValueError("stats-active parity summary differs from its sample evidence")
    if parity.preprocessing_max_abs != max(
        parity.image_preprocessing_max_abs,
        parity.state_preprocessing_max_abs,
    ):
        raise ValueError("stats-active preprocessing aggregate is inconsistent")
    expected_parity_max = max(summary_maxima.values())
    if parity.gate_max_abs != expected_parity_max:
        raise ValueError("stats-active parity gate maximum excludes a parity boundary")
    expected_parity_passed = _stats_active_parity_passed(
        image_preprocessing_max_abs=parity.image_preprocessing_max_abs,
        state_preprocessing_max_abs=parity.state_preprocessing_max_abs,
        normalized_action_max_abs=parity.normalized_action_max_abs,
        physical_action_max_abs=parity.physical_action_max_abs,
        physical_action_standardized_max_abs=(
            parity.physical_action_standardized_max_abs
        ),
    )
    if parity.passed != expected_parity_passed:
        raise ValueError("stats-active parity decision differs from its immutable threshold")

    gates = evaluate_outcome_gates(
        base_mlx_mae=float(base_report["mlx_mae"]),
        fine_mlx_mae=fine_mlx.mae,
        torch_mae=torch_result.mae,
        parity_max_abs=parity.gate_max_abs,
        image_preprocessing_max_abs=parity.image_preprocessing_max_abs,
        state_preprocessing_max_abs=parity.state_preprocessing_max_abs,
    )
    source_sha256 = {
        "training_run": _require_sha256("training run", training_run_sha256),
        "metrics_csv": _require_sha256("metrics CSV", metrics_sha256),
        **validated_training_artifacts,
        **validated_native_conversion,
        "dataset_revision_tree": FROZEN_DATASET_REVISION_TREE_SHA256,
        "evaluation_manifest": evaluation_manifest_sha256,
        "evaluation_metadata": evaluation_metadata_sha256,
        "base_report": base_report_sha256,
        "export_manifest": _require_sha256("export manifest", export_manifest_sha256),
        "train_statistics": train_statistics_sha256,
    }
    parity_document = asdict(parity)
    parity_document["samples"] = [dict(sample) for sample in parity.samples]
    return {
        "format_version": 1,
        "artifact_type": "smolvla-lora-finetune-outcome",
        "source_sha256": source_sha256,
        "thresholds": {
            "fine_to_base_mae_ratio_maximum": MAE_IMPROVEMENT_RATIO_MAXIMUM,
            "torch_to_mlx_mae_ratio_minimum": TORCH_MLX_MAE_RATIO_MINIMUM,
            "torch_to_mlx_mae_ratio_maximum": TORCH_MLX_MAE_RATIO_MAXIMUM,
            "image_preprocessing_max_abs": (
                IMAGE_PREPROCESSING_MAX_ABSOLUTE_TOLERANCE
            ),
            "state_preprocessing_max_abs": (
                STATE_PREPROCESSING_MAX_ABSOLUTE_TOLERANCE
            ),
            "stats_active_parity_max_abs": INFERENCE_MAX_ABSOLUTE_TOLERANCE,
        },
        "population": {
            "sample_count": EVALUATION_SAMPLE_COUNT,
            "element_count": expected_elements,
            "train_episodes": train_episodes,
            "heldout_episodes": heldout_episodes,
        },
        "training": {
            "selected_steps": int(training_run["selected_steps"]),
            "effective_batch_size": int(training_run["effective_batch_size"]),
            "actual_training_seconds": float(training_run["actual_training_seconds"]),
            "peak_memory_bytes": int(training_run["peak_memory_bytes"]),
            "final_loss": float(training_run["final_loss"]),
            "final_smoothed_loss": float(training_run["final_smoothed_loss"]),
            "lora": dict(lora),
        },
        "base_mlx_evaluation": dict(base_report),
        "fine_mlx_evaluation": fine_mlx.as_dict(),
        "torch_evaluation": torch_result.as_dict(),
        "stats_active_parity": parity_document,
        "gates": asdict(gates),
    }


def _read_json_document(path: Path, *, label: str) -> tuple[dict[str, object], str]:
    path = _require_real_file(path, label=label)
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _validate_completed_training_artifacts(
    run_dir: Path,
    training_run: Mapping[str, object],
) -> tuple[dict[str, str], CheckpointState]:
    run_dir = _require_real_directory(run_dir, label="fine-tune run directory")
    run_config_sha256 = _validated_training_run_config_sha256(training_run)
    adapter_path = _require_real_file(
        run_dir / "adapter.safetensors",
        label="final LoRA adapter",
    )
    adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    expected_adapter_sha256 = _require_sha256(
        "completed-run adapter",
        training_run.get("adapter_sha256"),
    )
    if adapter_sha256 != expected_adapter_sha256:
        raise ValueError("final LoRA adapter digest differs from the completed run")
    adapter_metadata, adapter_metadata_sha256 = _read_json_document(
        run_dir / "adapter.json",
        label="final LoRA adapter metadata",
    )
    lora = training_run.get("lora")
    if not isinstance(lora, Mapping):
        raise ValueError("completed run is missing LoRA metadata")
    expected_adapter_metadata = {
        "format_version": 1,
        "rank": int(lora["rank"]),
        "alpha": float(lora["alpha"]),
        "dropout": float(lora["dropout"]),
        "adapter_count": int(lora["adapter_count"]),
        "tensor_count": int(lora["trainable_tensor_count"]),
        "scalar_count": int(lora["trainable_scalar_count"]),
        "sha256": expected_adapter_sha256,
    }
    if "scope" in lora:
        expected_adapter_metadata["scope"] = str(lora["scope"])
    if adapter_metadata != expected_adapter_metadata:
        raise ValueError("final LoRA adapter metadata differs from the completed run")

    last_checkpoint = training_run.get("last_checkpoint")
    if not isinstance(last_checkpoint, Mapping):
        raise ValueError("completed run is missing its final checkpoint identity")
    selected_steps = int(training_run["selected_steps"])
    checkpoint_path = _require_real_directory(
        run_dir / "checkpoints" / f"step-{selected_steps:06d}",
        label="final training checkpoint",
    )
    try:
        recorded_path = Path(str(last_checkpoint["path"]))
        recorded_step = int(last_checkpoint["step"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("completed run has invalid final checkpoint identity") from error
    if (
        recorded_step != selected_steps
        or recorded_path.resolve() != checkpoint_path
    ):
        raise ValueError("completed run points at a different final checkpoint")
    checkpoint, model_tensors, optimizer_tensors = _read_checkpoint_directory(
        checkpoint_path,
        expected_run_config_sha256=run_config_sha256,
    )
    del model_tensors
    recorded_checkpoint_hashes = {
        "metadata_sha256": checkpoint.metadata_sha256,
        "model_sha256": checkpoint.model_sha256,
        "optimizer_sha256": checkpoint.optimizer_sha256,
    }
    if any(
        last_checkpoint.get(name) != digest
        for name, digest in recorded_checkpoint_hashes.items()
    ):
        raise ValueError("final checkpoint digests differ from the completed run")
    if (
        checkpoint.state.completed_step != selected_steps
        or checkpoint.state.selected_steps != selected_steps
        or checkpoint.state.last_update.loss != float(training_run["final_loss"])
        or checkpoint.state.smoothed_loss
        != float(training_run["final_smoothed_loss"])
        or checkpoint.state.peak_memory_bytes != int(training_run["peak_memory_bytes"])
        or checkpoint.state.elapsed_training_seconds
        > float(training_run["actual_training_seconds"])
        or checkpoint.model_sha256 != adapter_sha256
        or int(np.asarray(optimizer_tensors["step"]).item()) != selected_steps
        or float(np.asarray(optimizer_tensors["learning_rate"]).item())
        != float(np.float32(checkpoint.state.last_update.learning_rate))
    ):
        raise ValueError("final checkpoint state differs from the completed run")
    return (
        {
            "adapter": adapter_sha256,
            "adapter_metadata": adapter_metadata_sha256,
            "final_checkpoint_metadata": checkpoint.metadata_sha256,
            "final_checkpoint_model": checkpoint.model_sha256,
            "final_checkpoint_optimizer": checkpoint.optimizer_sha256,
        },
        checkpoint.state,
    )


def _validate_export_statistics(
    *,
    export_dir: Path,
    cache_dir: Path,
    train_episodes: tuple[int, ...],
    expected_sha256: object,
) -> str:
    expected_sha256 = _require_sha256("training statistics", expected_sha256)
    if expected_sha256 != FROZEN_TRAIN_STATISTICS_SHA256:
        raise ValueError("training statistics differ from the frozen train-only rows")
    dataset_root = _require_pinned_dataset_root(cache_dir)
    export_dir = _require_real_directory(export_dir, label="merged export directory")
    statistics = compute_train_statistics(
        dataset_root,
        train_episodes,
    )
    if statistics.sha256 != expected_sha256:
        raise ValueError("recomputed train-only statistics digest differs from the run")
    pre_path = _require_real_file(
        export_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors",
        label="exported preprocessor statistics",
    )
    post_path = _require_real_file(
        export_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        label="exported postprocessor statistics",
    )
    preprocessor = mx.load(str(pre_path))
    postprocessor = mx.load(str(post_path))
    for feature in ("observation.state", "action"):
        for statistic_name, expected_values in statistics.stats[feature].items():
            name = f"{feature}.{statistic_name}"
            if name not in preprocessor:
                raise ValueError(f"exported preprocessor is missing train statistic {name}")
            actual = np.asarray(preprocessor[name].astype(mx.float32))
            expected = np.asarray(expected_values, dtype=np.float32)
            if not np.array_equal(actual, expected):
                raise ValueError(f"exported preprocessor train statistic changed: {name}")
            if feature == "action":
                if name not in postprocessor:
                    raise ValueError(
                        f"exported postprocessor is missing train statistic {name}"
                    )
                post_actual = np.asarray(postprocessor[name].astype(mx.float32))
                if not np.array_equal(post_actual, expected):
                    raise ValueError(
                        f"exported postprocessor train statistic changed: {name}"
                    )
    return statistics.sha256


def _validate_native_conversion_for_export(
    export_dir: Path,
    converted_weights_path: Path,
    *,
    expected_source_sha256: object,
) -> dict[str, str]:
    """Bind the native cached tensors used for scoring to the validated export."""

    export_dir = _require_real_directory(export_dir, label="merged export directory")
    converted_weights_path = _require_real_file(
        _require_repository_cache_path(
            converted_weights_path,
            label="native converted checkpoint",
        ),
        label="native converted checkpoint",
    )
    name_map_path = _require_real_file(
        converted_weights_path.with_name("name_map.json"),
        label="native conversion name map",
    )
    validation = validate_converted_checkpoint(
        export_dir,
        converted_weights_path,
        name_map_path,
        dtype="float32",
        expected_tensor_count=500,
    )
    if (
        validation.parameter_count != 450_046_176
        or validation.source_model_sha256
        != _require_sha256("exported model", expected_source_sha256)
    ):
        raise ValueError("native conversion differs from the validated merged export")
    return {
        "native_conversion_model": validation.converted_model_sha256,
        "native_conversion_name_map": validation.name_map_sha256,
    }


def _validated_metrics_sha256(
    path: Path,
    *,
    expected_steps: int,
    training_run: Mapping[str, object],
    checkpoint_state: CheckpointState | None = None,
) -> str:
    path = _require_real_file(path, label="training metrics")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("training metrics are not valid UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != METRICS_FIELDS:
        raise ValueError("training metrics fields differ from the frozen schema")
    optimizer_config = _optimizer_config_from_training_run(training_run)
    if optimizer_config.training_horizon != expected_steps:
        raise ValueError("training metrics optimizer horizon differs from the run")
    count = 0
    previous_elapsed = 0.0
    previous_peak = -1
    previous_smoothed: float | None = None
    final_values: dict[str, float | int] | None = None
    for count, row in enumerate(reader, start=1):
        if set(row) != set(METRICS_FIELDS) or any(
            row[name] is None for name in METRICS_FIELDS
        ):
            raise ValueError(f"training metrics row {count} is incomplete")
        try:
            values: dict[str, float | int] = {
                "step": int(row["step"]),
                "loss": float(row["loss"]),
                "smoothed_loss": float(row["smoothed_loss"]),
                "learning_rate": float(row["learning_rate"]),
                "gradient_norm": float(row["gradient_norm"]),
                "clip_coefficient": float(row["clip_coefficient"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "updates_per_second": float(row["updates_per_second"]),
                "peak_memory_bytes": int(row["peak_memory_bytes"]),
            }
        except (TypeError, ValueError) as error:
            raise ValueError(f"training metrics row {count} is invalid") from error
        step = int(values["step"])
        loss = float(values["loss"])
        smoothed = float(values["smoothed_loss"])
        learning_rate = float(values["learning_rate"])
        gradient_norm = float(values["gradient_norm"])
        clip_coefficient = float(values["clip_coefficient"])
        elapsed = float(values["elapsed_seconds"])
        throughput = float(values["updates_per_second"])
        peak = int(values["peak_memory_bytes"])
        scalars = (
            loss,
            smoothed,
            learning_rate,
            gradient_norm,
            clip_coefficient,
            elapsed,
            throughput,
        )
        expected_smoothed = (
            loss
            if previous_smoothed is None
            else 0.98 * previous_smoothed + 0.02 * loss
        )
        expected_learning_rate = cosine_decay_with_warmup_lr(
            count - 1,
            optimizer_config,
        )
        if (
            step != count
            or not all(math.isfinite(value) for value in scalars)
            or loss < 0
            or smoothed < 0
            or learning_rate < 0
            or learning_rate != expected_learning_rate
            or gradient_norm < 0
            or not 0 < clip_coefficient <= 1
            or elapsed <= previous_elapsed
            or throughput <= 0
            or throughput != step / elapsed
            or peak < 0
            or peak < previous_peak
            or smoothed != expected_smoothed
        ):
            raise ValueError(f"training metrics row {count} is invalid")
        previous_elapsed = elapsed
        previous_peak = peak
        previous_smoothed = smoothed
        final_values = values
    if count != expected_steps:
        raise ValueError(
            f"training metrics contain {count} updates, expected {expected_steps}"
        )
    if final_values is None:
        raise ValueError("training metrics contain no optimizer updates")
    try:
        actual_training_seconds = float(training_run["actual_training_seconds"])
        run_peak = int(training_run["peak_memory_bytes"])
        run_loss = float(training_run["final_loss"])
        run_smoothed = float(training_run["final_smoothed_loss"])
        run_steps = int(training_run["selected_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("completed run has invalid final training metrics") from error
    if (
        not math.isfinite(actual_training_seconds)
        or actual_training_seconds < float(final_values["elapsed_seconds"])
        or run_steps != expected_steps
        or run_peak != int(final_values["peak_memory_bytes"])
        or run_loss != float(final_values["loss"])
        or run_smoothed != float(final_values["smoothed_loss"])
    ):
        raise ValueError("final training metrics differ from the completed run")
    if checkpoint_state is not None:
        try:
            effective_batch_size = int(training_run["effective_batch_size"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("completed run has invalid sample-count metadata") from error
        final_update = checkpoint_state.last_update
        if (
            checkpoint_state.completed_step != expected_steps
            or checkpoint_state.selected_steps != expected_steps
            or checkpoint_state.smoothed_loss != float(final_values["smoothed_loss"])
            or checkpoint_state.elapsed_training_seconds
            != float(final_values["elapsed_seconds"])
            or checkpoint_state.peak_memory_bytes
            != int(final_values["peak_memory_bytes"])
            or checkpoint_state.samples_consumed
            != expected_steps * effective_batch_size
            or checkpoint_state.flow_draw_count
            != expected_steps * effective_batch_size
            or final_update.loss != float(final_values["loss"])
            or final_update.learning_rate != float(final_values["learning_rate"])
            or final_update.gradient_norm != float(final_values["gradient_norm"])
            or final_update.clip_coefficient
            != float(final_values["clip_coefficient"])
            or not 0 < final_update.seconds <= checkpoint_state.elapsed_training_seconds
        ):
            raise ValueError("final training metrics differ from the final checkpoint")
    return hashlib.sha256(payload).hexdigest()


def _expected_export_metadata(training_run: Mapping[str, object]) -> dict[str, object]:
    split = training_run.get("split")
    lora = training_run.get("lora")
    if not isinstance(split, Mapping) or not isinstance(lora, Mapping):
        raise ValueError("training run is missing split or LoRA export metadata")
    metadata = {
        "seed": int(training_run["seed"]),
        "sampler_seed": int(training_run["sampler_seed"]),
        "selected_steps": int(training_run["selected_steps"]),
        "effective_batch_size": int(training_run["effective_batch_size"]),
        "rank": int(lora["rank"]),
        "alpha": float(lora["alpha"]),
        "dropout": float(lora["dropout"]),
        "adapter_sha256": str(training_run["adapter_sha256"]),
        "train_statistics_sha256": str(training_run["train_statistics_sha256"]),
        "train_episodes": list(split["train_episodes"]),
        "holdout_episodes": list(split["holdout_episodes"]),
        "merge_adapter_count": int(lora["adapter_count"]),
        **frozen_export_audit_metadata(
            _validated_training_run_config_sha256(training_run)
        ),
    }
    if "scope" in lora:
        metadata["lora_scope"] = str(lora["scope"])
        export = training_run.get("export")
        file_sha256 = export.get("file_sha256") if isinstance(export, Mapping) else None
        if not isinstance(file_sha256, Mapping) or set(file_sha256) != (
            _EXPORT_SUPPORT_FILE_NAMES | {"model.safetensors"}
        ):
            raise ValueError("training run export file inventory is invalid")
        metadata["support_file_sha256"] = {
            name: _require_sha256(f"export support file {name}", file_sha256[name])
            for name in sorted(_EXPORT_SUPPORT_FILE_NAMES)
        }
    return metadata


def run_finetune_outcome_evaluation(
    *,
    cache_dir: str | Path,
    native_cache: str | Path,
    run_dir: str | Path,
    evaluation_dir: str | Path,
    base_report_path: str | Path,
    output_path: str | Path,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[dict[str, object], str]:
    """Evaluate the completed export in MLX/Torch and persist all T3 gates."""

    run_dir = _require_real_directory(
        _require_repository_cache_path(
            Path(run_dir),
            label="fine-tune run directory",
        ),
        label="fine-tune run directory",
    )
    cache_dir = _require_real_directory(
        _require_repository_cache_path(
            Path(cache_dir),
            label="Hugging Face cache directory",
        ),
        label="Hugging Face cache directory",
    )
    native_cache = _require_real_directory(
        _require_repository_cache_path(
            Path(native_cache),
            label="native model cache directory",
        ),
        label="native model cache directory",
    )
    evaluation_dir = _require_real_directory(
        _require_repository_cache_path(
            Path(evaluation_dir),
            label="held-out evaluation directory",
        ),
        label="held-out evaluation directory",
    )
    base_report_path = _require_repository_cache_path(
        Path(base_report_path),
        label="base held-out evaluation",
    )
    output_path = _require_repository_cache_path(
        Path(output_path),
        label="fine-tune outcome report",
    )
    if output_path.exists() or output_path.is_symlink():
        _require_real_file(output_path, label="fine-tune outcome report")
    else:
        _require_real_directory(
            output_path.parent,
            label="fine-tune outcome report parent directory",
        )

    _require_minimum_free_bytes(shutil.disk_usage(_REPOSITORY_CACHE).free)
    dataset_root = _require_pinned_dataset_root(cache_dir)

    training_run, training_run_sha256 = _read_json_document(
        run_dir / "run.json",
        label="fine-tune run state",
    )
    if (
        training_run.get("artifact_type") != "smolvla-mlx-lora-run"
        or training_run.get("status") != "trained_and_exported"
    ):
        raise ValueError("fine-tune outcome evaluation requires a complete training run")
    selected_steps = int(training_run["selected_steps"])
    training_artifact_sha256, checkpoint_state = (
        _validate_completed_training_artifacts(
            run_dir,
            training_run,
        )
    )
    metrics_sha256 = _validated_metrics_sha256(
        run_dir / "metrics.csv",
        expected_steps=selected_steps,
        training_run=training_run,
        checkpoint_state=checkpoint_state,
    )
    cases = load_evaluation_cases(
        evaluation_dir,
        dataset_root=dataset_root,
    )
    evaluation_artifact, evaluation_metadata_sha256 = _frozen_evaluation_artifact(
        evaluation_dir,
        dataset_root=dataset_root,
    )
    evaluation_manifest = evaluation_artifact.metadata
    base_report, base_report_sha256 = _read_json_document(
        base_report_path,
        label="base held-out evaluation",
    )
    export_dir = run_dir / "export"
    expected_export_metadata = _expected_export_metadata(training_run)
    export_report = validate_merged_checkpoint_export(
        export_dir,
        expected_metadata=expected_export_metadata,
    )
    export_manifest, export_manifest_sha256 = _read_json_document(
        export_dir / "training_manifest.json",
        label="merged export manifest",
    )
    _validate_export_statistics(
        export_dir=export_dir,
        cache_dir=cache_dir,
        train_episodes=tuple(training_run["split"]["train_episodes"]),
        expected_sha256=training_run.get("train_statistics_sha256"),
    )
    recorded_export = training_run.get("export")
    if (
        not isinstance(recorded_export, Mapping)
        or Path(str(recorded_export.get("path"))).resolve() != export_dir.resolve()
        or recorded_export.get("tensor_count") != export_report.tensor_count
        or recorded_export.get("parameter_count") != export_report.parameter_count
        or recorded_export.get("file_sha256") != dict(export_report.file_sha256)
    ):
        raise ValueError("completed run state differs from its validated merged export")

    with mx.stream(mx.cpu):
        mlx_policy = load_stats_aware_policy(
            export_dir,
            cache_dir=native_cache,
            dtype=mx.float32,
        )
        native_conversion_sha256 = _validate_native_conversion_for_export(
            export_dir,
            mlx_policy.converted_weights_path,
            expected_source_sha256=export_report.file_sha256["model.safetensors"],
        )
        fine_mlx = evaluate_mlx_policy(
            mlx_policy,
            cases,
            progress=(
                None
                if progress is None
                else lambda completed, total: progress("mlx", completed, total)
            ),
        )
    del mlx_policy
    gc.collect()
    mx.clear_cache()

    from training.reference_export import TorchExportPolicy

    torch_policy = TorchExportPolicy.load(export_dir, cache_dir=cache_dir)
    torch_result = evaluate_torch_export(
        torch_policy,
        cases,
        progress=(
            None
            if progress is None
            else lambda completed, total: progress("torch", completed, total)
        ),
    )
    with mx.stream(mx.cpu):
        mlx_policy = load_stats_aware_policy(
            export_dir,
            cache_dir=native_cache,
            dtype=mx.float32,
        )
        parity_conversion_sha256 = _validate_native_conversion_for_export(
            export_dir,
            mlx_policy.converted_weights_path,
            expected_source_sha256=export_report.file_sha256["model.safetensors"],
        )
        if parity_conversion_sha256 != native_conversion_sha256:
            raise ValueError("native conversion changed between outcome evaluations")
        parity = run_stats_active_parity(mlx_policy, torch_policy, cases)
    del mlx_policy, torch_policy
    gc.collect()
    mx.clear_cache()

    report = assemble_finetune_outcome_report(
        training_run=training_run,
        training_run_sha256=training_run_sha256,
        metrics_sha256=metrics_sha256,
        training_artifact_sha256=training_artifact_sha256,
        native_conversion_sha256=native_conversion_sha256,
        evaluation_manifest=evaluation_manifest,
        evaluation_metadata_sha256=evaluation_metadata_sha256,
        base_report=base_report,
        base_report_sha256=base_report_sha256,
        export_manifest=export_manifest,
        export_manifest_sha256=export_manifest_sha256,
        fine_mlx=fine_mlx,
        torch_result=torch_result,
        parity=parity,
    )
    report_sha256 = write_run_state(output_path, report)
    return report, report_sha256


@dataclass(frozen=True)
class TrainedComparisonStart:
    """Validated floor and one-shot marker state captured before model work."""

    floor: dict[str, object]
    floor_sha256: str
    floor_file_mtime_ns: int
    floor_bundle_sha256: str
    start_marker: dict[str, object]
    start_marker_sha256: str
    start_marker_file_mtime_ns: int
    snapshots: tuple[object, ...]


def validate_trained_comparison_start_files(
    *,
    floor_path: str | Path,
    variant_root: str | Path,
    start_marker_path: str | Path,
    comparison_path: str | Path,
) -> TrainedComparisonStart:
    """Bind the prospective floor and marker before any comparison inference."""

    from smolvla_mlx.training import trained_parity as parity

    floor_path = Path(floor_path)
    start_marker_path = Path(start_marker_path)
    comparison_path = Path(comparison_path)
    if len(
        {
            str(floor_path.resolve()),
            str(start_marker_path.resolve()),
            str(comparison_path.resolve()),
        }
    ) != 3:
        raise ValueError("floor, start marker, and comparison paths must be distinct")
    if comparison_path.exists() or comparison_path.is_symlink():
        raise FileExistsError(f"comparison output already exists: {comparison_path}")

    floor, floor_snapshot = parity._snapshot_json(
        floor_path,
        label="prospective self-consistency floor",
    )
    bundle = parity._load_floor_bundle(floor, variant_root=variant_root)
    marker, marker_snapshot = parity._snapshot_json(
        start_marker_path,
        label="comparison start marker",
    )
    if not isinstance(floor, Mapping) or floor.get("purpose") != "prospective_gate":
        raise ValueError("trained comparison requires a prospective floor")
    if not isinstance(marker, Mapping):
        raise ValueError("comparison start marker must be an object")
    marker = parity.validate_comparison_start_marker(marker)
    if Path(marker["comparison_path"]).resolve() != comparison_path.resolve():
        raise ValueError("comparison start marker was issued for a different comparison path")
    expected_marker_binding = {
        "floor_sha256": floor_snapshot.sha256,
        "floor_procedure_id": bundle.report["procedure_id"],
        "floor_created_at_ns": bundle.report["created_at_ns"],
        "floor_file_mtime_ns": floor_snapshot.mtime_ns,
        "floor_bundle_sha256": bundle.bundle_sha256,
        "checkpoint_path": bundle.report["checkpoint_path"],
        "input_combined_sha256": bundle.report["input_sha256"]["combined_sha256"],
    }
    for field, expected in expected_marker_binding.items():
        if marker[field] != expected:
            raise ValueError(
                f"comparison marker {field.replace('_', ' ')} differs from the floor file"
            )
    if not (
        bundle.report["created_at_ns"]
        <= floor_snapshot.mtime_ns
        < marker["created_at_ns"]
        <= marker_snapshot.mtime_ns
    ):
        raise ValueError("comparison marker chronology is invalid")
    snapshots = (floor_snapshot, *bundle.snapshots, marker_snapshot)
    parity._revalidate_snapshots(snapshots)
    if comparison_path.exists() or comparison_path.is_symlink():
        raise FileExistsError(f"comparison output already exists: {comparison_path}")
    return TrainedComparisonStart(
        floor=dict(bundle.report),
        floor_sha256=floor_snapshot.sha256,
        floor_file_mtime_ns=floor_snapshot.mtime_ns,
        floor_bundle_sha256=bundle.bundle_sha256,
        start_marker=dict(marker),
        start_marker_sha256=marker_snapshot.sha256,
        start_marker_file_mtime_ns=marker_snapshot.mtime_ns,
        snapshots=snapshots,
    )


def _comparison_utc_from_ns(value_ns: int) -> str:
    if isinstance(value_ns, bool) or not isinstance(value_ns, int) or value_ns <= 0:
        raise ValueError("comparison timestamp must be a positive integer")
    seconds, nanoseconds = divmod(value_ns, 1_000_000_000)
    value = datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(
        microseconds=nanoseconds // 1_000
    )
    return value.isoformat(timespec="microseconds")


def assemble_trained_comparison_report(
    *,
    floor: Mapping[str, object],
    floor_sha256: str,
    floor_file_mtime_ns: int,
    floor_bundle_sha256: str,
    start_marker: Mapping[str, object],
    start_marker_sha256: str,
    start_marker_file_mtime_ns: int,
    floor_input_evidence: Mapping[str, object],
    evidence_files: Mapping[str, Mapping[str, str]],
    conversion_validation: Mapping[str, object],
    base_mlx_evaluation: Mapping[str, object],
    fine_mlx_evaluation: Mapping[str, object],
    torch_evaluation: Mapping[str, object],
    stats_active_parity: Mapping[str, object],
    created_at_ns: int,
) -> dict[str, object]:
    """Assemble and schema-check the immutable trained comparison document."""

    from smolvla_mlx.training import trained_parity as parity

    parity_evidence = {
        name: deepcopy(value)
        for name, value in stats_active_parity.items()
        if name != "passed"
    }
    metrics = {
        "base_mlx_mae": float(base_mlx_evaluation["mlx_mae"]),
        "fine_mlx_mae": float(fine_mlx_evaluation["mae"]),
        "torch_mae": float(torch_evaluation["mae"]),
        "image_preprocessing_max_abs": float(
            parity_evidence["image_preprocessing_max_abs"]
        ),
        "state_preprocessing_max_abs": float(
            parity_evidence["state_preprocessing_max_abs"]
        ),
        "normalized_action_max_abs": float(
            parity_evidence["normalized_action_max_abs"]
        ),
    }
    comparison = {
        "format_version": 1,
        "artifact_type": parity.COMPARISON_ARTIFACT_TYPE,
        "procedure_id": parity.PROCEDURE_ID,
        "created_at_utc": _comparison_utc_from_ns(created_at_ns),
        "created_at_ns": created_at_ns,
        "checkpoint_path": floor["checkpoint_path"],
        "sample_count": floor["sample_count"],
        "normalized_action_chunk_shape": deepcopy(
            floor["normalized_action_chunk_shape"]
        ),
        "floor_binding": {
            "floor_sha256": floor_sha256,
            "floor_procedure_id": floor["procedure_id"],
            "floor_created_at_ns": floor["created_at_ns"],
            "floor_file_mtime_ns": floor_file_mtime_ns,
            "input_combined_sha256": floor["input_sha256"]["combined_sha256"],
            "floor_bundle_sha256": floor_bundle_sha256,
        },
        "start_marker_binding": {
            "marker_sha256": start_marker_sha256,
            "marker_created_at_ns": start_marker["created_at_ns"],
            "marker_file_mtime_ns": start_marker_file_mtime_ns,
            "floor_bundle_sha256": floor_bundle_sha256,
        },
        "source_identity": deepcopy(floor["source_identity"]),
        "input_sha256": deepcopy(floor["input_sha256"]),
        "floor_input_evidence": deepcopy(dict(floor_input_evidence)),
        "case_identities": deepcopy(floor["case_identities"]),
        "evidence_files": deepcopy(dict(evidence_files)),
        "conversion_validation": deepcopy(dict(conversion_validation)),
        "base_mlx_evaluation": deepcopy(dict(base_mlx_evaluation)),
        "fine_mlx_evaluation": deepcopy(dict(fine_mlx_evaluation)),
        "torch_evaluation": deepcopy(dict(torch_evaluation)),
        "stats_active_parity": parity_evidence,
        "metrics": metrics,
    }
    validated, _, _ = parity._validate_comparison(comparison)
    return validated


def _recorded_comparison_path(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(_REPOSITORY_ROOT.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise ValueError(f"comparison evidence is outside the repository: {path}") from error


def _expected_native_conversion_path(export_dir: Path, native_cache: Path) -> Path:
    identity = hashlib.sha256(str(export_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    return (
        native_cache.resolve()
        / "converted"
        / identity
        / "float32"
        / "model.float32.safetensors"
    )


def run_trained_comparison_evaluation(
    *,
    floor_path: str | Path,
    variant_root: str | Path,
    start_marker_path: str | Path,
    comparison_path: str | Path,
    outcome_path: str | Path,
    cache_dir: str | Path,
    native_cache: str | Path,
    run_dir: str | Path,
    evaluation_dir: str | Path,
    base_report_path: str | Path,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[dict[str, object], str, dict[str, object], str]:
    """Validate the floor first, then evaluate and install its bound comparison."""

    from smolvla_mlx.training import trained_parity as parity
    from training.self_consistency import (
        collect_floor_input_evidence,
        collect_floor_input_hashes,
    )

    start = validate_trained_comparison_start_files(
        floor_path=floor_path,
        variant_root=variant_root,
        start_marker_path=start_marker_path,
        comparison_path=comparison_path,
    )
    comparison_path = _require_repository_cache_path(
        Path(comparison_path),
        label="trained comparison output",
    )
    outcome_path = _require_repository_cache_path(
        Path(outcome_path),
        label="fine-tune outcome report",
    )
    if outcome_path.exists() or outcome_path.is_symlink():
        raise FileExistsError(f"fine-tune outcome already exists: {outcome_path}")
    _require_real_directory(
        outcome_path.parent,
        label="fine-tune outcome report parent directory",
    )
    if outcome_path.resolve() == comparison_path.resolve():
        raise ValueError("fine-tune outcome and trained comparison paths must be distinct")

    floor_inputs, _ = collect_floor_input_hashes(
        checkpoint_dir=Path(run_dir) / "export",
        evaluation_dir=evaluation_dir,
        cache_dir=cache_dir,
    )
    if floor_inputs != start.floor["input_sha256"]:
        raise ValueError("current comparison inputs differ from the prospective floor")
    floor_input_evidence = collect_floor_input_evidence(
        checkpoint_dir=Path(run_dir) / "export",
        evaluation_dir=evaluation_dir,
        cache_dir=cache_dir,
    )
    parity._validated_floor_input_evidence(
        floor_input_evidence,
        expected_inputs=start.floor["input_sha256"],
    )
    expected_checkpoint = _recorded_comparison_path(
        _require_real_directory(Path(run_dir) / "export", label="merged export directory")
    )
    if start.floor["checkpoint_path"] != expected_checkpoint:
        raise ValueError("comparison run export differs from the floor checkpoint")
    parity._revalidate_snapshots(start.snapshots)

    outcome, outcome_sha256 = run_finetune_outcome_evaluation(
        cache_dir=cache_dir,
        native_cache=native_cache,
        run_dir=run_dir,
        evaluation_dir=evaluation_dir,
        base_report_path=base_report_path,
        output_path=outcome_path,
        progress=progress,
    )
    export_dir = _require_real_directory(
        Path(run_dir) / "export",
        label="merged export directory",
    )
    converted_path = _require_real_file(
        _expected_native_conversion_path(export_dir, Path(native_cache)),
        label="native converted checkpoint",
    )
    name_map_path = _require_real_file(
        converted_path.with_name("name_map.json"),
        label="native conversion name map",
    )
    conversion = validate_converted_checkpoint(
        export_dir,
        converted_path,
        name_map_path,
        dtype="float32",
        expected_tensor_count=500,
    )
    if conversion.parameter_count != 450_046_176:
        raise ValueError("native conversion parameter count differs from SmolVLA")
    source_sha256 = outcome.get("source_sha256")
    if not isinstance(source_sha256, Mapping) or (
        source_sha256.get("native_conversion_model")
        != conversion.converted_model_sha256
        or source_sha256.get("native_conversion_name_map")
        != conversion.name_map_sha256
    ):
        raise ValueError("outcome conversion evidence differs from the evaluated files")

    base_report_path = _require_real_file(
        Path(base_report_path),
        label="base held-out evaluation",
    )
    implementation_path = _require_real_file(
        Path(__file__),
        label="comparison implementation",
    )
    evidence_files = {
        "base_report": {
            "path": _recorded_comparison_path(base_report_path),
            "sha256": _require_sha256(
                "base report",
                source_sha256.get("base_report"),
            ),
        },
        "native_conversion_model": {
            "path": _recorded_comparison_path(converted_path),
            "sha256": conversion.converted_model_sha256,
        },
        "native_conversion_name_map": {
            "path": _recorded_comparison_path(name_map_path),
            "sha256": conversion.name_map_sha256,
        },
        "comparison_implementation": {
            "path": _recorded_comparison_path(implementation_path),
            "sha256": hashlib.sha256(implementation_path.read_bytes()).hexdigest(),
        },
    }
    conversion_validation = {
        "source_model_sha256": conversion.source_model_sha256,
        "converted_model_sha256": conversion.converted_model_sha256,
        "name_map_sha256": conversion.name_map_sha256,
        "dtype": conversion.dtype,
        "tensor_count": conversion.tensor_count,
        "parameter_count": conversion.parameter_count,
    }
    comparison = assemble_trained_comparison_report(
        floor=start.floor,
        floor_sha256=start.floor_sha256,
        floor_file_mtime_ns=start.floor_file_mtime_ns,
        floor_bundle_sha256=start.floor_bundle_sha256,
        start_marker=start.start_marker,
        start_marker_sha256=start.start_marker_sha256,
        start_marker_file_mtime_ns=start.start_marker_file_mtime_ns,
        floor_input_evidence=floor_input_evidence,
        evidence_files=evidence_files,
        conversion_validation=conversion_validation,
        base_mlx_evaluation=outcome["base_mlx_evaluation"],
        fine_mlx_evaluation=outcome["fine_mlx_evaluation"],
        torch_evaluation=outcome["torch_evaluation"],
        stats_active_parity=outcome["stats_active_parity"],
        created_at_ns=time.time_ns(),
    )
    parity._revalidate_snapshots(start.snapshots)
    comparison_sha256 = parity._atomic_json_no_clobber(comparison_path, comparison)
    return comparison, comparison_sha256, outcome, outcome_sha256


def capture_and_evaluate_base(
    *,
    cache_dir: str | Path,
    native_cache: str | Path,
    evaluation_dir: str | Path,
    output_path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[MAEEvaluation, str]:
    """Freeze held-out inputs and baseline MAE before the training run exists."""

    metadata = capture_evaluation_artifact(
        cache_dir=cache_dir,
        output_dir=evaluation_dir,
    )
    cases = load_evaluation_cases(evaluation_dir)
    policy = load_base_stats_policy(cache_dir=cache_dir, native_cache=native_cache)
    evaluation = evaluate_mlx_policy(policy, cases, progress=progress)
    report = {
        "format_version": 1,
        "artifact_type": "smolvla-lora-base-heldout-evaluation",
        "evaluation_manifest_sha256": metadata["manifest_sha256"],
        "train_statistics_sha256": metadata["train_statistics_sha256"],
        "sample_count": evaluation.sample_count,
        "element_count": evaluation.element_count,
        "mlx_mae": evaluation.mae,
        "absolute_error_sum": evaluation.absolute_error_sum,
        "device": evaluation.device,
        "dtype": evaluation.dtype,
        "samples": [dict(sample) for sample in evaluation.samples],
    }
    digest = write_run_state(output_path, report)
    return evaluation, digest
