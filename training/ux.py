"""Public Stage T4 training UX and exact-resume evidence.

This module is deliberately separate from :mod:`training.finetune`.  The
completed T3/T3B LoRA runner is an immutable experiment record; T4 reuses its
audited numerical and checkpoint primitives without changing that runner's
configuration or launch schemas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Callable, ClassVar, Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from huggingface_hub import snapshot_download
import numpy as np
from safetensors import safe_open

from reference.discovery import DATASET_ID
from reference.discovery import DATASET_REVISION
from smolvla_mlx.policy import SmolVLAMLX
from training.dataset import (
    TrainingDataBridge,
    compute_train_statistics,
    make_episode_split,
)
from training.export import export_merged_checkpoint, resolve_base_checkpoint
from training.finetune import (
    CheckpointState,
    UpdateResult,
    _save_adapter_checkpoint,
    accumulate_gradients,
    advance_flow_random_state,
    load_latest_training_checkpoint,
    save_training_checkpoint,
    training_base_artifact_identity,
    training_batch_from_bridge,
    write_run_state,
)
from training.gradients import configure_reference_trainable
from training.lora import (
    EXPERT_ONLY_SCOPE,
    LoRAConfig,
    LoRAInstallationReport,
    install_lora,
    merge_lora,
)
from training.model import SmolVLATrainingModel, training_loss
from training.optimizer import (
    SmolVLAAdamW,
    SmolVLAOptimizerConfig,
    clip_gradients_by_global_norm,
)


PARAMETER_MAX_ABS_GATE = 1e-6
LOSS_MAX_ABS_GATE = 1e-7
_MINIMUM_FREE_BYTES = 40 * 1024**3
_DRAW_CHAIN_INITIAL = hashlib.sha256(b"smolvla-mlx-t4-draw-chain-v1").hexdigest()
_METRIC_FIELDS = (
    "step",
    "loss",
    "smoothed_loss",
    "learning_rate",
    "gradient_norm",
    "clip_coefficient",
    "elapsed_seconds",
    "peak_memory_bytes",
    "draw_sha256",
)


def _stable_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stable_json_sha256(value: object) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """Controls shared by the two explicit public training modes."""

    mode: ClassVar[str]
    dataset: str | Path = DATASET_ID
    steps: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    output_dir: Path = Path(".cache/training/t4")
    cache_dir: Path = Path(".cache/hf")
    native_cache: Path = Path(".cache/smolvla_mlx/policy-float32")
    seed: int = 20_260_901
    sampler_seed: int = 20_260_901
    checkpoint_interval: int = 25
    resume: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, (str, Path)) or not str(self.dataset).strip():
            raise ValueError("training dataset must be a nonempty repo id or local path")
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("training steps must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("training batch size must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("training learning rate must be finite and positive")
        if type(self.checkpoint_interval) is not int or self.checkpoint_interval <= 0:
            raise ValueError("checkpoint interval must be a positive integer")
        if type(self.seed) is not int or type(self.sampler_seed) is not int:
            raise ValueError("training seeds must be integers")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "native_cache", Path(self.native_cache))


@dataclass(frozen=True, kw_only=True)
class FullTrainingConfig(TrainingConfig):
    """Train the complete parameter set enabled by LeRobot's SmolVLA policy.

    LeRobot freezes the vision encoder and language backbone by default while
    training the action expert and state projection.  ``full`` means every
    parameter in that reference trainable policy, with no low-rank adapters.
    """

    mode: ClassVar[str] = "full"


@dataclass(frozen=True, kw_only=True)
class LoRATrainingConfig(TrainingConfig):
    """Train only explicit native MLX LoRA adapter parameters."""

    mode: ClassVar[str] = "lora"
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    lora_scope: str = EXPERT_ONLY_SCOPE

    def __post_init__(self) -> None:
        super().__post_init__()
        LoRAConfig(
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            scope=self.lora_scope,
        )


@dataclass(frozen=True)
class TrainingTopologyReport:
    """Exact trainable parameter identity for one public training mode."""

    mode: str
    trainable_names: tuple[str, ...]
    trainable_tensor_count: int
    trainable_scalar_count: int
    trainable_dtype_counts: Mapping[str, int]
    adapter_count: int = 0


def _topology_report(
    model: nn.Module,
    *,
    mode: str,
    adapter_count: int = 0,
) -> TrainingTopologyReport:
    flat = tuple(tree_flatten(model.trainable_parameters()))
    if not flat:
        raise RuntimeError(f"{mode} training exposed no trainable parameters")
    dtype_counts: dict[str, int] = {}
    for _, value in flat:
        dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + 1
    return TrainingTopologyReport(
        mode=mode,
        trainable_names=tuple(name for name, _ in flat),
        trainable_tensor_count=len(flat),
        trainable_scalar_count=sum(int(value.size) for _, value in flat),
        trainable_dtype_counts=dict(sorted(dtype_counts.items())),
        adapter_count=adapter_count,
    )


def configure_full_training(model: nn.Module) -> TrainingTopologyReport:
    """Apply and prove the reference full-training freeze policy."""

    expected_names = configure_reference_trainable(model)
    # MLX AdamW promotes bf16 parameters when applying fp32 accumulated
    # gradients.  Make that master-parameter policy explicit before optimizer
    # initialization so fresh and reconstructed resume schemas are identical.
    model.update(
        tree_map(
            lambda value: value.astype(mx.float32),
            model.trainable_parameters(),
        )
    )
    mx.eval(model.trainable_parameters())
    report = _topology_report(model, mode="full")
    if report.trainable_names != expected_names:
        raise RuntimeError("full training topology changed after reference configuration")
    if not all(
        name.startswith(("state_proj.", "expert."))
        for name in report.trainable_names
    ):
        raise RuntimeError("full training exposed parameters outside the reference policy")
    if not all(
        value.dtype == mx.float32
        for _, value in tree_flatten(model.trainable_parameters())
    ):
        raise RuntimeError("full training master parameters must all remain fp32")
    return report


def configure_lora_training(
    model: nn.Module,
    config: LoRATrainingConfig,
) -> tuple[TrainingTopologyReport, LoRAInstallationReport]:
    """Install the requested LoRA topology and return its exact identity."""

    installed = install_lora(
        model,
        LoRAConfig(
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            scope=config.lora_scope,
        ),
    )
    report = _topology_report(
        model,
        mode="lora",
        adapter_count=installed.adapter_count,
    )
    if report.trainable_names != installed.trainable_names:
        raise RuntimeError("LoRA installation and public topology reports differ")
    return report, installed


def optimizer_coverage_evidence(
    model: nn.Module,
    optimizer: SmolVLAAdamW,
    topology: TrainingTopologyReport,
) -> dict[str, object]:
    """Initialize and prove that AdamW owns two moments for every trainable."""

    parameters = model.trainable_parameters()
    parameter_names = tuple(name for name, _ in tree_flatten(parameters))
    if parameter_names != topology.trainable_names:
        raise ValueError("optimizer coverage model differs from the topology report")
    optimizer.initialize(parameters)
    optimizer.validate_state_for(parameters)
    state_names = tuple(name for name, _ in tree_flatten(optimizer.state))
    moment_names = tuple(
        name for name in state_names if name not in {"step", "learning_rate"}
    )
    expected_moments = {
        f"{name}.{suffix}" for name in parameter_names for suffix in ("m", "v")
    }
    covered = set(moment_names) == expected_moments
    if not covered:
        raise RuntimeError("optimizer moments do not exactly cover the trainable set")
    return {
        "covered": True,
        "parameter_tensor_count": len(parameter_names),
        "moment_tensor_count": len(moment_names),
        "optimizer_state_tensor_count": len(state_names),
        "parameter_names_sha256": _stable_json_sha256(parameter_names),
        "moment_names_sha256": _stable_json_sha256(tuple(sorted(moment_names))),
    }


_TRAJECTORY_FIELDS = (
    "completed_step",
    "selected_steps",
    "smoothed_loss",
    "samples_consumed",
    "flow_draw_count",
    "last_loss",
    "learning_rate",
    "gradient_norm",
    "clip_coefficient",
)


def trajectory_state_sha256(state: Mapping[str, object]) -> str:
    """Hash numerical continuation state while excluding resource measurements."""

    missing = set(_TRAJECTORY_FIELDS).difference(state)
    if missing:
        raise ValueError(f"trajectory state is missing fields: {sorted(missing)}")
    return _stable_json_sha256({name: state[name] for name in _TRAJECTORY_FIELDS})


def _read_json_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required training artifact is absent or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"training artifact must contain an object: {path}")
    return value


def _checkpoint_tensors(root: Path, kind: str, step: int) -> dict[str, np.ndarray]:
    path = root / "checkpoints" / f"step-{step:06d}" / f"{kind}.safetensors"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"resume evidence tensor file is absent or unsafe: {path}")
    with safe_open(path, framework="np") as source:
        return {name: np.array(source.get_tensor(name), copy=True) for name in source.keys()}


def _compare_tensor_maps(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> tuple[float, bool]:
    if set(left) != set(right):
        raise ValueError("resume evidence tensor name sets differ")
    maximum = 0.0
    exact = True
    for name in sorted(left):
        a = left[name]
        b = right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"resume evidence tensor schema differs for {name}")
        exact = exact and np.array_equal(a, b)
        if a.size:
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(a.astype(np.float64) - b.astype(np.float64))
                    )
                ),
            )
    return maximum, exact


def _read_metrics(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"training metrics are absent or unsafe: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"training metrics contain no rows: {path}")
    return rows


def evaluate_resume_exactness(
    uninterrupted_dir: str | Path,
    resumed_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Compare a completed uninterrupted run with a 50+resume trajectory.

    Parameter and loss tolerances are the immutable T4 gates.  Optimizer,
    draws, sampler continuation state, and canonical step state must match
    exactly.
    """

    uninterrupted = Path(uninterrupted_dir).resolve(strict=True)
    resumed = Path(resumed_dir).resolve(strict=True)
    left_run = _read_json_object(uninterrupted / "run.json")
    right_run = _read_json_object(resumed / "run.json")
    for field in ("mode", "selected_steps", "run_config_sha256"):
        if left_run.get(field) != right_run.get(field):
            raise ValueError(f"resume run {field} values differ")
    steps = left_run.get("selected_steps")
    if type(steps) is not int or steps <= 0:
        raise ValueError("resume run selected_steps is invalid")

    left_parameters = _checkpoint_tensors(uninterrupted, "model", steps)
    right_parameters = _checkpoint_tensors(resumed, "model", steps)
    parameter_max_abs, parameters_exact = _compare_tensor_maps(
        left_parameters, right_parameters
    )
    left_optimizer = _checkpoint_tensors(uninterrupted, "optimizer", steps)
    right_optimizer = _checkpoint_tensors(resumed, "optimizer", steps)
    optimizer_max_abs, optimizer_exact = _compare_tensor_maps(
        left_optimizer, right_optimizer
    )

    left_metrics = _read_metrics(uninterrupted / "metrics.csv")
    right_metrics = _read_metrics(resumed / "metrics.csv")
    if len(left_metrics) != steps or len(right_metrics) != steps:
        raise ValueError("resume metrics do not contain the selected number of steps")
    if [row.get("step") for row in left_metrics] != [row.get("step") for row in right_metrics]:
        raise ValueError("resume metric step identities differ")
    loss_max_abs = max(
        abs(float(left["loss"]) - float(right["loss"]))
        for left, right in zip(left_metrics, right_metrics, strict=True)
    )
    numerical_fields = (
        "loss",
        "smoothed_loss",
        "learning_rate",
        "gradient_norm",
        "clip_coefficient",
    )
    metric_max_abs = max(
        abs(float(left[field]) - float(right[field]))
        for left, right in zip(left_metrics, right_metrics, strict=True)
        for field in numerical_fields
    )
    draw_rows_exact = all(
        left.get("draw_sha256") == right.get("draw_sha256")
        for left, right in zip(left_metrics, right_metrics, strict=True)
    )
    left_evidence = left_run.get("final_evidence")
    right_evidence = right_run.get("final_evidence")
    if not isinstance(left_evidence, Mapping) or not isinstance(right_evidence, Mapping):
        raise ValueError("resume runs lack final continuation evidence")
    exact_fields = {
        "draw_chain_exact": "draw_chain_sha256",
        "sampler_state_exact": "sampler_state_sha256",
        "trajectory_state_exact": "trajectory_state_sha256",
    }
    exact_results = {
        result_name: left_evidence.get(field) == right_evidence.get(field)
        for result_name, field in exact_fields.items()
    }
    draw_chain_exact = exact_results["draw_chain_exact"] and draw_rows_exact
    passed = (
        parameter_max_abs <= PARAMETER_MAX_ABS_GATE
        and loss_max_abs <= LOSS_MAX_ABS_GATE
        and metric_max_abs <= LOSS_MAX_ABS_GATE
        and optimizer_exact
        and draw_chain_exact
        and exact_results["sampler_state_exact"]
        and exact_results["trajectory_state_exact"]
    )
    report: dict[str, object] = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-resume-exactness",
        "mode": left_run["mode"],
        "selected_steps": steps,
        "run_config_sha256": left_run["run_config_sha256"],
        "gates": {
            "parameter_max_abs": PARAMETER_MAX_ABS_GATE,
            "loss_max_abs": LOSS_MAX_ABS_GATE,
        },
        "parameter_max_abs": parameter_max_abs,
        "parameters_exact": parameters_exact,
        "loss_max_abs": loss_max_abs,
        "metric_max_abs": metric_max_abs,
        "optimizer_max_abs": optimizer_max_abs,
        "optimizer_exact": optimizer_exact,
        "draw_chain_exact": draw_chain_exact,
        "sampler_state_exact": exact_results["sampler_state_exact"],
        "trajectory_state_exact": exact_results["trajectory_state_exact"],
        "passed": passed,
    }
    report["report_sha256"] = _stable_json_sha256(report)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


@dataclass
class _TrainingComponents:
    """Materialized objects and immutable identities for one run."""

    dataset: Mapping[str, object]
    train_episodes: tuple[int, ...]
    holdout_episodes: tuple[int, ...]
    train_statistics_sha256: str
    processor_stats: Mapping[str, Mapping[str, object]]
    model: nn.Module
    bridge: object
    optimizer: SmolVLAAdamW
    topology: TrainingTopologyReport
    optimizer_coverage: Mapping[str, object]
    base_artifact: Mapping[str, str]
    lora_report: LoRAInstallationReport | None


@dataclass(frozen=True)
class TrainingRunResult:
    """Final public result for one T4 training invocation."""

    mode: str
    selected_steps: int
    final_loss: float
    final_smoothed_loss: float
    loss_decreased: bool
    peak_memory_bytes: int
    training_seconds: float
    final_checkpoint: Path
    export_dir: Path
    run_config_sha256: str
    run_state_sha256: str
    draw_chain_sha256: str

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["final_checkpoint"] = str(self.final_checkpoint)
        value["export_dir"] = str(self.export_dir)
        return value


def _dataset_source(config: TrainingConfig) -> dict[str, object]:
    requested = str(config.dataset)
    candidate = Path(requested).expanduser()
    if candidate.exists() or candidate.is_symlink():
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(f"local training dataset is unsafe: {candidate}")
        root = candidate.resolve(strict=True)
        info_path = root / "meta" / "info.json"
        if not info_path.is_file() or info_path.is_symlink():
            raise FileNotFoundError(f"local LeRobot dataset lacks meta/info.json: {root}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if not isinstance(info, Mapping):
            raise ValueError("local LeRobot dataset info must be an object")
        total_episodes = info.get("total_episodes")
        if type(total_episodes) is not int or total_episodes < 2:
            raise ValueError("local LeRobot dataset needs at least two episodes")
        recorded_repo_id = info.get("repo_id")
        repo_id = (
            recorded_repo_id
            if isinstance(recorded_repo_id, str) and "/" in recorded_repo_id
            else (DATASET_ID if root.name == "svla_so101_pickplace" else f"local/{root.name}")
        )
        revision = None
    else:
        if candidate.is_absolute() or requested.startswith((".", "~")):
            raise FileNotFoundError(f"local training dataset does not exist: {candidate}")
        if requested.count("/") != 1:
            raise ValueError("dataset repo id must have the form owner/name")
        repo_id = requested
        root_name = "svla_so101_pickplace" if repo_id == DATASET_ID else repo_id.replace("/", "__")
        root = (config.cache_dir / "datasets" / root_name).resolve()
        revision = DATASET_REVISION if repo_id == DATASET_ID else None
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        metadata = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
        if not tuple((root / "data").glob("chunk-*/file-*.parquet")):
            snapshot_download(
                repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=root,
            )
            metadata = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
        total_episodes = int(metadata.total_episodes)
        if total_episodes < 2:
            raise ValueError("LeRobot training dataset needs at least two episodes")
        revision = metadata.revision
    return {
        "requested": requested,
        "repo_id": repo_id,
        "root": str(root),
        "revision": revision,
        "total_episodes": total_episodes,
    }


def _prepare_training(
    config: FullTrainingConfig | LoRATrainingConfig,
) -> _TrainingComponents:
    dataset = _dataset_source(config)
    split = make_episode_split(
        num_episodes=int(dataset["total_episodes"]),
        seed=config.seed,
    )
    stats = compute_train_statistics(dataset["root"], split.train_episodes)
    mx.random.seed(config.seed)
    model = SmolVLATrainingModel.from_pretrained(
        cache_dir=config.native_cache,
        dtype=mx.bfloat16,
    )
    model.train()
    if isinstance(config, FullTrainingConfig):
        topology = configure_full_training(model)
        lora_report = None
    elif isinstance(config, LoRATrainingConfig):
        topology, lora_report = configure_lora_training(model, config)
    else:
        raise TypeError(f"unsupported training configuration: {type(config).__name__}")
    bridge = TrainingDataBridge(
        cache_dir=config.cache_dir,
        episodes=split.train_episodes,
        sampler_seed=config.sampler_seed,
        stats=stats.processor_stats,
        dataset_id=str(dataset["repo_id"]),
        dataset_root=str(dataset["root"]),
        dataset_revision=(
            None if dataset["revision"] is None else str(dataset["revision"])
        ),
    )
    optimizer_config = replace(
        SmolVLAOptimizerConfig(),
        lr=config.learning_rate,
        decay_lr=min(2.5e-6, config.learning_rate),
        training_horizon=config.steps,
    )
    optimizer = SmolVLAAdamW(optimizer_config)
    coverage = optimizer_coverage_evidence(model, optimizer, topology)
    return _TrainingComponents(
        dataset=dataset,
        train_episodes=split.train_episodes,
        holdout_episodes=split.holdout_episodes,
        train_statistics_sha256=stats.sha256,
        processor_stats=stats.processor_stats,
        model=model,
        bridge=bridge,
        optimizer=optimizer,
        topology=topology,
        optimizer_coverage=coverage,
        base_artifact=training_base_artifact_identity(model),
        lora_report=lora_report,
    )


def _config_payload(
    config: FullTrainingConfig | LoRATrainingConfig,
    components: _TrainingComponents,
) -> dict[str, object]:
    mode_details: dict[str, object] = {}
    if isinstance(config, LoRATrainingConfig):
        mode_details = {
            "rank": config.rank,
            "alpha": config.alpha,
            "dropout": config.dropout,
            "scope": config.lora_scope,
        }
    return {
        "format_version": 1,
        "mode": config.mode,
        "dataset": dict(components.dataset),
        "steps": config.steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "sampler_seed": config.sampler_seed,
        "checkpoint_interval": config.checkpoint_interval,
        "native_cache": str(config.native_cache.resolve()),
        "train_episodes": list(components.train_episodes),
        "holdout_episodes": list(components.holdout_episodes),
        "train_statistics_sha256": components.train_statistics_sha256,
        "base_artifact": dict(components.base_artifact),
        "topology": asdict(components.topology),
        "optimizer": asdict(components.optimizer.config),
        "mode_details": mode_details,
        "base_dtype": "bfloat16",
    }


def _array_evidence(value: mx.array) -> dict[str, object]:
    array = np.array(value, copy=True)
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _perform_update(
    *,
    model: nn.Module,
    bridge: object,
    optimizer: SmolVLAAdamW,
    batch_size: int,
    draw_chain_sha256: str,
) -> tuple[UpdateResult, str, Mapping[str, object] | None]:
    start = time.perf_counter()
    batches = []
    last_observation: Mapping[str, object] | None = None
    chain = draw_chain_sha256
    for microbatch_index in range(batch_size):
        bridge_batch = bridge.next_batch()
        batch = training_batch_from_bridge(bridge_batch)
        mx.eval(batch.noise, batch.timesteps)
        draw_record = {
            "microbatch_index": microbatch_index,
            "episode": bridge_batch.episode,
            "frame_index": bridge_batch.frame_index,
            "absolute_index": bridge_batch.absolute_index,
            "noise": _array_evidence(batch.noise),
            "timesteps": _array_evidence(batch.timesteps),
        }
        chain = hashlib.sha256(
            bytes.fromhex(chain) + _stable_json_bytes(draw_record)
        ).hexdigest()
        batches.append(batch)
        last_observation = bridge_batch.observation
    accumulated = accumulate_gradients(model, batches, training_loss)
    clipped = clip_gradients_by_global_norm(
        accumulated.gradients,
        optimizer.config.grad_clip_norm,
    )
    learning_rate = optimizer.update(model, clipped.gradients)
    mx.eval(model.trainable_parameters(), optimizer.state)
    return (
        UpdateResult(
            loss=float(accumulated.mean_loss),
            learning_rate=learning_rate,
            gradient_norm=float(clipped.total_norm),
            clip_coefficient=float(clipped.coefficient),
            seconds=time.perf_counter() - start,
        ),
        chain,
        last_observation,
    )


def _prepare_metrics(path: Path, *, resume_step: int | None) -> list[dict[str, str]]:
    if path.is_symlink():
        raise ValueError(f"training metrics path is unsafe: {path}")
    if resume_step is None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite training metrics: {path}")
        with path.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=_METRIC_FIELDS).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        return []
    rows = _read_metrics(path)
    if resume_step <= 0 or len(rows) < resume_step:
        raise ValueError("training metrics end before the resume checkpoint")
    for index, row in enumerate(rows, start=1):
        if set(row) != set(_METRIC_FIELDS) or int(row["step"]) != index:
            raise ValueError("training metrics have an invalid schema or step sequence")
    rows = rows[:resume_step]
    temporary = path.with_name(f".{path.name}.resume-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"training metrics staging path exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return rows


def _append_metric(path: Path, row: Mapping[str, object]) -> None:
    if set(row) != set(_METRIC_FIELDS):
        raise ValueError("training metric row differs from the public schema")
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=_METRIC_FIELDS).writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_binding(checkpoint: object) -> dict[str, object]:
    return {
        "step": checkpoint.state.completed_step,
        "path": str(checkpoint.path),
        "metadata_sha256": checkpoint.metadata_sha256,
        "model_sha256": checkpoint.model_sha256,
        "optimizer_sha256": checkpoint.optimizer_sha256,
    }


def _finalize_export(
    *,
    config: FullTrainingConfig | LoRATrainingConfig,
    components: _TrainingComponents,
    output_dir: Path,
    run_config_sha256: str,
    last_observation: Mapping[str, object] | None,
    validate_action: bool,
) -> dict[str, object]:
    adapter_sha256: str | None = None
    merge_adapter_count = 0
    if isinstance(config, LoRATrainingConfig):
        if components.lora_report is None:
            raise RuntimeError("LoRA training lacks its installation report")
        adapter_sha256 = _save_adapter_checkpoint(
            components.model,
            output_dir / "adapter.safetensors",
            lora_report=components.lora_report,
        )
        merge_report = merge_lora(components.model, dtype=mx.float32)
        merge_adapter_count = merge_report.adapter_count
    metadata = {
        "training_mode": config.mode,
        "run_config_sha256": run_config_sha256,
        "selected_steps": config.steps,
        "effective_batch_size": config.batch_size,
        "train_statistics_sha256": components.train_statistics_sha256,
        "train_episodes": list(components.train_episodes),
        "holdout_episodes": list(components.holdout_episodes),
        "topology": asdict(components.topology),
        "adapter_sha256": adapter_sha256,
        "merge_adapter_count": merge_adapter_count,
    }
    # The exporter validates its JSON round trip byte-for-structure.  Canonicalize
    # dataclass tuples now so the in-memory expectation exactly matches the
    # manifest's decoded list representation.
    export_metadata = json.loads(_stable_json_bytes(metadata))
    export_report = export_merged_checkpoint(
        model=components.model,
        source_checkpoint_dir=resolve_base_checkpoint(config.cache_dir),
        output_dir=output_dir / "export",
        processor_stats=components.processor_stats,
        metadata=export_metadata,
    )
    action_validation: dict[str, object]
    if validate_action:
        if last_observation is None:
            raise RuntimeError("training export validation has no real observation")
        mx.random.seed(config.seed + config.steps)
        policy = SmolVLAMLX.from_pretrained(
            export_report.output_dir,
            cache_dir=config.native_cache.parent,
            dtype=mx.bfloat16,
            execution_mode="production",
        )
        action = np.asarray(policy.select_action(last_observation), dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise RuntimeError(f"training export emitted an invalid action: {action}")
        action_validation = {
            "finite": True,
            "shape": list(action.shape),
            "dtype": str(action.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest(),
        }
    else:
        action_validation = {"finite": None, "skipped": True}
    return {
        "path": str(export_report.output_dir),
        "tensor_count": export_report.tensor_count,
        "parameter_count": export_report.parameter_count,
        "dtype": export_report.dtype,
        "file_sha256": dict(export_report.file_sha256),
        "adapter_sha256": adapter_sha256,
        "action_validation": action_validation,
    }


def run_training(
    config: FullTrainingConfig | LoRATrainingConfig,
    *,
    progress: Callable[[int, int, UpdateResult], None] | None = None,
    validate_action: bool = True,
) -> TrainingRunResult:
    """Train, checkpoint, exactly resume, export, and validate one MLX run."""

    return _run_training_impl(
        config,
        progress=progress,
        validate_action=validate_action,
    )


def _run_training_impl(
    config: FullTrainingConfig | LoRATrainingConfig,
    *,
    progress: Callable[[int, int, UpdateResult], None] | None,
    validate_action: bool,
) -> TrainingRunResult:
    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"native training requires Metal GPU, got {mx.default_device()}")
    output_dir = Path(os.path.abspath(config.output_dir.expanduser()))
    if output_dir.is_symlink():
        raise ValueError(f"training output is unsafe: {output_dir}")
    if config.resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"training output is not resumable: {output_dir}")
    else:
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite training output: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output_dir.parent).free < _MINIMUM_FREE_BYTES:
            raise RuntimeError("native training requires at least 40 GiB free")
        output_dir.mkdir()
    if shutil.disk_usage(output_dir).free < _MINIMUM_FREE_BYTES:
        raise RuntimeError("native training requires at least 40 GiB free")

    components = _prepare_training(config)
    configuration = _config_payload(config, components)
    run_config_sha256 = _stable_json_sha256(configuration)
    run_path = output_dir / "run.json"
    metrics_path = output_dir / "metrics.csv"
    checkpoint_root = output_dir / "checkpoints"
    start_step = 0
    elapsed_before = 0.0
    previous_peak = 0
    smoothed_loss: float | None = None
    draw_chain = _DRAW_CHAIN_INITIAL
    last_update: UpdateResult | None = None
    last_observation: Mapping[str, object] | None = None
    latest_checkpoint = None

    if config.resume:
        run_document = _read_json_object(run_path)
        if run_document.get("status") not in {"running", "interrupted"}:
            raise ValueError(f"training run is not resumable: {run_document.get('status')!r}")
        if run_document.get("run_config_sha256") != run_config_sha256:
            raise ValueError("resume configuration differs from the original trajectory")
        latest_checkpoint = load_latest_training_checkpoint(
            model=components.model,
            optimizer=components.optimizer,
            checkpoint_root=checkpoint_root,
            trainable_names=components.topology.trainable_names,
            expected_run_config_sha256=run_config_sha256,
            expected_selected_steps=config.steps,
            expected_effective_batch_size=config.batch_size,
            expected_checkpoint_interval=config.checkpoint_interval,
            expected_last_checkpoint=run_document.get("last_checkpoint"),
        )
        checkpoint_state = latest_checkpoint.state
        start_step = checkpoint_state.completed_step
        elapsed_before = checkpoint_state.elapsed_training_seconds
        previous_peak = checkpoint_state.peak_memory_bytes
        smoothed_loss = checkpoint_state.smoothed_loss
        last_update = checkpoint_state.last_update
        bridge_state = components.bridge.state_dict()
        num_samples = int(bridge_state["num_samples"])
        epoch, start_index = divmod(checkpoint_state.samples_consumed, num_samples)
        bridge_state.update(
            {
                "samples_consumed": checkpoint_state.samples_consumed,
                "epoch": epoch,
                "start_index": start_index,
            }
        )
        components.bridge.load_state_dict(bridge_state)
        advance_flow_random_state(
            draw_count=checkpoint_state.flow_draw_count,
            shape=(1, 50, 32),
        )
        prior_rows = _prepare_metrics(metrics_path, resume_step=start_step)
        draw_chain = prior_rows[-1]["draw_sha256"]
        run_document = {
            **run_document,
            "status": "running",
            "resume_count": int(run_document.get("resume_count", 0)) + 1,
            "resumed_from_step": start_step,
        }
    else:
        _prepare_metrics(metrics_path, resume_step=None)
        run_document = {
            "format_version": 1,
            "artifact_type": "smolvla-mlx-public-training-run",
            "status": "running",
            "mode": config.mode,
            "selected_steps": config.steps,
            "run_config_sha256": run_config_sha256,
            "configuration": configuration,
            "topology": asdict(components.topology),
            "optimizer_coverage": dict(components.optimizer_coverage),
            "checkpoint_count": 0,
            "resume_count": 0,
            "last_completed_step": 0,
        }
    write_run_state(run_path, run_document)

    training_start = time.perf_counter()
    mx.reset_peak_memory()
    completed_step = start_step
    loss_values = [float(row["loss"]) for row in _read_metrics(metrics_path)] if start_step else []
    try:
        for step_index in range(start_step, config.steps):
            update, draw_chain, observation = _perform_update(
                model=components.model,
                bridge=components.bridge,
                optimizer=components.optimizer,
                batch_size=config.batch_size,
                draw_chain_sha256=draw_chain,
            )
            completed_step = step_index + 1
            last_update = update
            if observation is not None:
                last_observation = observation
            smoothed_loss = (
                update.loss
                if smoothed_loss is None
                else 0.98 * smoothed_loss + 0.02 * update.loss
            )
            elapsed = elapsed_before + time.perf_counter() - training_start
            peak_memory = max(previous_peak, int(mx.get_peak_memory()))
            loss_values.append(update.loss)
            _append_metric(
                metrics_path,
                {
                    "step": completed_step,
                    "loss": update.loss,
                    "smoothed_loss": smoothed_loss,
                    "learning_rate": update.learning_rate,
                    "gradient_norm": update.gradient_norm,
                    "clip_coefficient": update.clip_coefficient,
                    "elapsed_seconds": elapsed,
                    "peak_memory_bytes": peak_memory,
                    "draw_sha256": draw_chain,
                },
            )
            on_checkpoint = (
                completed_step == 1
                or completed_step % config.checkpoint_interval == 0
                or completed_step == config.steps
            )
            if on_checkpoint:
                bridge_state = components.bridge.state_dict()
                expected_draws = completed_step * config.batch_size
                if (
                    int(bridge_state["samples_consumed"]) != expected_draws
                    or components.optimizer.step_index != completed_step
                ):
                    raise RuntimeError("training sampler/optimizer counters diverged")
                latest_checkpoint = save_training_checkpoint(
                    model=components.model,
                    optimizer=components.optimizer,
                    checkpoint_root=checkpoint_root,
                    state=CheckpointState(
                        completed_step=completed_step,
                        selected_steps=config.steps,
                        smoothed_loss=float(smoothed_loss),
                        elapsed_training_seconds=float(elapsed),
                        peak_memory_bytes=peak_memory,
                        samples_consumed=expected_draws,
                        flow_draw_count=expected_draws,
                        last_update=update,
                        run_config_sha256=run_config_sha256,
                    ),
                    trainable_names=components.topology.trainable_names,
                    keep_last=3,
                )
                run_document = {
                    **run_document,
                    "last_completed_step": completed_step,
                    "last_checkpoint": _checkpoint_binding(latest_checkpoint),
                    "checkpoint_count": int(run_document.get("checkpoint_count", 0)) + 1,
                    "last_pruned_checkpoints": list(latest_checkpoint.pruned_checkpoints),
                    "draw_chain_sha256": draw_chain,
                }
                write_run_state(run_path, run_document)
            if progress is not None:
                progress(completed_step, config.steps, update)
    except BaseException as error:
        write_run_state(
            run_path,
            {
                **run_document,
                "status": "interrupted",
                "last_completed_step": completed_step,
                "interruption": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )
        raise

    if (
        latest_checkpoint is None
        or latest_checkpoint.state.completed_step != config.steps
        or last_update is None
        or smoothed_loss is None
    ):
        raise RuntimeError("training completed without a final checkpoint")
    window = min(10, max(1, len(loss_values) // 2))
    first_window_mean = float(np.mean(loss_values[:window]))
    last_window_mean = float(np.mean(loss_values[-window:]))
    loss_decreased = last_window_mean < first_window_mean
    bridge_state = components.bridge.state_dict()
    trajectory = {
        "completed_step": config.steps,
        "selected_steps": config.steps,
        "smoothed_loss": float(smoothed_loss),
        "samples_consumed": int(bridge_state["samples_consumed"]),
        "flow_draw_count": config.steps * config.batch_size,
        "last_loss": last_update.loss,
        "learning_rate": last_update.learning_rate,
        "gradient_norm": last_update.gradient_norm,
        "clip_coefficient": last_update.clip_coefficient,
        "elapsed_training_seconds": latest_checkpoint.state.elapsed_training_seconds,
        "peak_memory_bytes": latest_checkpoint.state.peak_memory_bytes,
    }
    final_evidence = {
        "draw_chain_sha256": draw_chain,
        "sampler_state_sha256": _stable_json_sha256(bridge_state),
        "trajectory_state_sha256": trajectory_state_sha256(trajectory),
        "checkpoint": _checkpoint_binding(latest_checkpoint),
    }
    exporting_state = {
        **run_document,
        "status": "exporting",
        "last_completed_step": config.steps,
        "first_window_mean_loss": first_window_mean,
        "last_window_mean_loss": last_window_mean,
        "loss_decreased": loss_decreased,
        "final_evidence": final_evidence,
    }
    write_run_state(run_path, exporting_state)
    export = _finalize_export(
        config=config,
        components=components,
        output_dir=output_dir,
        run_config_sha256=run_config_sha256,
        last_observation=last_observation,
        validate_action=validate_action,
    )
    training_seconds = elapsed_before + time.perf_counter() - training_start
    peak_memory = max(previous_peak, int(mx.get_peak_memory()))
    final_state = {
        **exporting_state,
        "status": "trained_and_exported",
        "final_loss": last_update.loss,
        "final_smoothed_loss": float(smoothed_loss),
        "resource_observation": {
            "classification": "functional-smoke-non-benchmark",
            "training_seconds": training_seconds,
            "peak_memory_bytes": peak_memory,
        },
        "export": export,
    }
    run_state_sha256 = write_run_state(run_path, final_state)
    gc.collect()
    return TrainingRunResult(
        mode=config.mode,
        selected_steps=config.steps,
        final_loss=last_update.loss,
        final_smoothed_loss=float(smoothed_loss),
        loss_decreased=loss_decreased,
        peak_memory_bytes=peak_memory,
        training_seconds=training_seconds,
        final_checkpoint=latest_checkpoint.path,
        export_dir=Path(str(export["path"])),
        run_config_sha256=run_config_sha256,
        run_state_sha256=run_state_sha256,
        draw_chain_sha256=draw_chain,
    )
