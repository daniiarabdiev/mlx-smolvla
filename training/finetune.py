"""Measured native MLX LoRA training loop for the Stage T3 outcome gate."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import Callable, Iterable, Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from reference.discovery import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    DATASET_ID,
    DATASET_REVISION,
)
from smolvla_mlx.types import ProcessedObservation
from training.dataset import (
    BridgeBatch,
    SAMPLER_SEED,
    SPLIT_SEED,
    TrainingDataBridge,
    compute_train_statistics,
    make_episode_split,
)
from training.export import (
    export_merged_checkpoint,
    resolve_base_checkpoint,
    validate_merged_checkpoint_export,
)
from training.lora import LoRAConfig, install_lora, merge_lora
from training.model import SmolVLATrainingModel, TrainingBatch, training_loss
from training.optimizer import (
    SmolVLAAdamW,
    SmolVLAOptimizerConfig,
    clip_gradients_by_global_norm,
)
from training.t3_contract import frozen_export_audit_metadata


METRICS_FIELDS = (
    "step",
    "loss",
    "smoothed_loss",
    "learning_rate",
    "gradient_norm",
    "clip_coefficient",
    "elapsed_seconds",
    "updates_per_second",
    "peak_memory_bytes",
)
_MINIMUM_FREE_BYTES = 40 * 1024**3


@dataclass(frozen=True)
class FlowDraws:
    """One native free-running flow-matching noise/timestep draw."""

    noise: mx.array
    timesteps: mx.array


def sample_flow_draws(shape: tuple[int, ...]) -> FlowDraws:
    """Sample SmolVLA's N(0,1) noise and scaled Beta(1.5,1) time."""

    if len(shape) != 3 or shape[0] <= 0:
        raise ValueError(f"flow action shape must be [batch, chunk, width], got {shape}")
    noise = mx.random.normal(shape).astype(mx.float32)
    uniform = mx.random.uniform(shape=(shape[0],)).astype(mx.float32)
    beta = mx.power(uniform, 1.0 / 1.5)
    timesteps = beta * 0.999 + 0.001
    return FlowDraws(noise=noise, timesteps=timesteps.astype(mx.float32))


def advance_flow_random_state(*, draw_count: int, shape: tuple[int, ...]) -> None:
    """Advance the global MLX PRNG by complete flow draws for exact resume."""

    if draw_count < 0:
        raise ValueError("flow draw count must be nonnegative")
    for _ in range(draw_count):
        draws = sample_flow_draws(shape)
        mx.eval(draws.noise, draws.timesteps)


def training_batch_from_bridge(batch: BridgeBatch) -> TrainingBatch:
    """Cross the NumPy boundary and attach one free-running native draw."""

    actions = mx.array(batch.actions).astype(mx.float32)
    draws = sample_flow_draws(tuple(actions.shape))
    return TrainingBatch(
        processed=ProcessedObservation(
            pixel_values=mx.array(batch.pixel_values).astype(mx.float32),
            pixel_attention_mask=mx.array(batch.pixel_attention_mask).astype(mx.bool_),
            input_ids=mx.array(batch.input_ids).astype(mx.int32),
            text_attention_mask=mx.array(batch.text_attention_mask).astype(mx.bool_),
            state=mx.array(batch.state).astype(mx.float32),
        ),
        actions=actions,
        action_is_pad=mx.array(batch.action_is_pad).astype(mx.bool_),
        noise=draws.noise,
        timesteps=draws.timesteps,
        action_dim=batch.physical_action_dim,
    )


@dataclass(frozen=True)
class AccumulationResult:
    """Mean loss and mean gradient tree over distinct microbatches."""

    mean_loss: mx.array
    gradients: dict
    microbatch_count: int


def accumulate_gradients(
    model: nn.Module,
    batches: Iterable[object],
    loss_fn: Callable[[nn.Module, object], mx.array],
) -> AccumulationResult:
    """Accumulate evaluated fp32 gradients and divide once by batch count."""

    batch_values = tuple(batches)
    if not batch_values:
        raise ValueError("gradient accumulation requires at least one microbatch")
    value_and_grad = nn.value_and_grad(model, lambda batch: loss_fn(model, batch))
    accumulated = None
    loss_sum = 0.0
    expected_names = tuple(name for name, _ in tree_flatten(model.trainable_parameters()))
    for batch in batch_values:
        loss, gradients = value_and_grad(batch)
        flat_gradients = tuple(tree_flatten(gradients))
        if tuple(name for name, _ in flat_gradients) != expected_names:
            raise RuntimeError("microbatch gradient tree differs from the trainable parameter tree")
        mx.eval(loss, gradients)
        if not bool(mx.isfinite(loss)):
            raise RuntimeError("microbatch loss is non-finite")
        if not all(bool(mx.all(mx.isfinite(value))) for _, value in flat_gradients):
            raise RuntimeError("microbatch gradient tree contains a non-finite value")
        loss_sum += float(loss)
        if accumulated is None:
            accumulated = tree_map(lambda value: value.astype(mx.float32), gradients)
        else:
            accumulated = tree_map(
                lambda total, value: total + value.astype(mx.float32),
                accumulated,
                gradients,
            )
        mx.eval(accumulated)
    assert accumulated is not None
    divisor = mx.array(float(len(batch_values)), dtype=mx.float32)
    averaged = tree_map(lambda value: value / divisor, accumulated)
    mean_loss = mx.array(loss_sum / len(batch_values), dtype=mx.float32)
    mx.eval(mean_loss, averaged)
    return AccumulationResult(
        mean_loss=mean_loss,
        gradients=averaged,
        microbatch_count=len(batch_values),
    )


def select_step_budget(
    median_update_seconds: float,
    *,
    nominal_steps: int = 3_000,
    training_seconds: float = 6_900.0,
) -> int:
    """Cap the nominal run using the frozen measured-time budget formula."""

    if not math.isfinite(median_update_seconds) or median_update_seconds <= 0:
        raise ValueError("median update time must be finite and positive")
    if nominal_steps <= 0 or not math.isfinite(training_seconds) or training_seconds <= 0:
        raise ValueError("nominal steps and training seconds must be positive")
    return max(1, min(nominal_steps, math.floor(training_seconds / median_update_seconds)))


class MetricsWriter:
    """Durable CSV writer for a fresh run or validated checkpoint resume."""

    def __init__(
        self,
        path: str | Path,
        *,
        resume_from_step: int | None = None,
        checkpoint_state: CheckpointState | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.recovery_path: Path | None = None
        if resume_from_step is None:
            if self.path.exists():
                raise FileExistsError(f"refusing to append to existing metrics file {self.path}")
            self._handle = self.path.open("x", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._handle, fieldnames=METRICS_FIELDS)
            self._writer.writeheader()
            self._next_step = 1
            self._sync()
            return
        if resume_from_step < 0:
            raise ValueError("metrics resume step must be nonnegative")
        if checkpoint_state is not None and checkpoint_state.completed_step != resume_from_step:
            raise ValueError("metrics resume step differs from checkpoint state")
        if self.path.is_symlink() or not self.path.is_file():
            raise FileNotFoundError(f"metrics file is missing or unsafe for resume: {self.path}")
        self.recovery_path = self._next_recovery_path()
        shutil.copy2(self.path, self.recovery_path)
        _sync_file(self.recovery_path)
        _sync_directory(self.path.parent)
        rows = self._read_checkpoint_prefix(self.recovery_path, resume_from_step)
        if checkpoint_state is not None and resume_from_step > 0:
            self._validate_checkpoint_boundary(rows[-1], checkpoint_state)
        self._replace_with_prefix(rows)
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=METRICS_FIELDS)
        self._next_step = resume_from_step + 1

    def _read_checkpoint_prefix(
        self,
        source: Path,
        resume_from_step: int,
    ) -> list[dict[str, str]]:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != METRICS_FIELDS:
                raise ValueError("metrics header differs from the frozen schema")
            rows: list[dict[str, str]] = []
            for expected_step in range(1, resume_from_step + 1):
                try:
                    row = next(reader)
                except StopIteration as error:
                    raise ValueError(
                        f"metrics end before checkpoint step {resume_from_step}"
                    ) from error
                if set(row) != set(METRICS_FIELDS) or any(
                    row[name] is None for name in METRICS_FIELDS
                ):
                    raise ValueError(f"metrics checkpoint prefix row {expected_step} is torn")
                try:
                    step = int(row["step"])
                    peak_memory = int(row["peak_memory_bytes"])
                    scalars = [
                        float(row[name])
                        for name in METRICS_FIELDS
                        if name not in {"step", "peak_memory_bytes"}
                    ]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"metrics checkpoint prefix row {expected_step} is invalid"
                    ) from error
                if step != expected_step or peak_memory < 0 or not all(
                    math.isfinite(value) for value in scalars
                ):
                    raise ValueError(
                        f"metrics checkpoint prefix row {expected_step} is invalid"
                    )
                rows.append(row)
        return rows

    @staticmethod
    def _validate_checkpoint_boundary(
        row: Mapping[str, str],
        checkpoint_state: CheckpointState,
    ) -> None:
        expected = {
            "step": checkpoint_state.completed_step,
            "loss": checkpoint_state.last_update.loss,
            "smoothed_loss": checkpoint_state.smoothed_loss,
            "learning_rate": checkpoint_state.last_update.learning_rate,
            "gradient_norm": checkpoint_state.last_update.gradient_norm,
            "clip_coefficient": checkpoint_state.last_update.clip_coefficient,
            "elapsed_seconds": checkpoint_state.elapsed_training_seconds,
            "updates_per_second": checkpoint_state.completed_step
            / checkpoint_state.elapsed_training_seconds,
            "peak_memory_bytes": checkpoint_state.peak_memory_bytes,
        }
        actual = {
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
        if actual != expected:
            raise ValueError(
                f"metrics/checkpoint boundary differs: actual={actual}, expected={expected}"
            )

    def _next_recovery_path(self) -> Path:
        for index in range(1, 1_000_000):
            candidate = self.path.with_name(
                f"{self.path.stem}.recovery-{index:06d}{self.path.suffix}"
            )
            if not candidate.exists():
                return candidate
        raise RuntimeError("metrics recovery namespace is exhausted")

    def _replace_with_prefix(self, rows: list[dict[str, str]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)
            _sync_directory(self.path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def write(self, **values: object) -> None:
        if tuple(values) != METRICS_FIELDS:
            missing = tuple(name for name in METRICS_FIELDS if name not in values)
            unexpected = tuple(name for name in values if name not in METRICS_FIELDS)
            raise ValueError(f"metrics row fields differ; missing={missing}, unexpected={unexpected}")
        if int(values["step"]) != self._next_step:
            raise ValueError(
                f"metrics step must be {self._next_step}, got {values['step']}"
            )
        self._writer.writerow(values)
        self._next_step += 1
        self._sync()

    def close(self) -> None:
        if not self._handle.closed:
            self._sync()
            self._handle.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def write_run_state(path: str | Path, value: object) -> str:
    """Atomically replace one complete JSON run-state document."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FineTuneConfig:
    """Frozen settings for one measured Stage T3 run."""

    cache_dir: Path = Path(".cache/hf")
    native_cache: Path = Path(".cache/smolvla_mlx/policy-float32")
    output_dir: Path = Path(".cache/training/t3")
    seed: int = SPLIT_SEED
    sampler_seed: int = SAMPLER_SEED
    nominal_steps: int = 3_000
    effective_batch_size: int = 8
    training_seconds: float = 6_900.0
    benchmark_warmup_updates: int = 3
    benchmark_measured_updates: int = 10
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    checkpoint_interval: int = 100
    resume: bool = False

    def __post_init__(self) -> None:
        if self.nominal_steps <= 0 or self.effective_batch_size <= 0:
            raise ValueError("fine-tune steps and effective batch size must be positive")
        if self.benchmark_warmup_updates < 0 or self.benchmark_measured_updates <= 0:
            raise ValueError("benchmark warmup/measured update counts are invalid")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint interval must be positive")
        LoRAConfig(rank=self.rank, alpha=self.alpha, dropout=self.dropout)


def training_run_config_sha256(
    config: FineTuneConfig,
    *,
    selected_steps: int,
    train_statistics_sha256: str,
    train_episodes: tuple[int, ...],
    holdout_episodes: tuple[int, ...],
    base_artifact: Mapping[str, str],
    optimizer_config: SmolVLAOptimizerConfig,
) -> str:
    """Hash every setting that can affect the resumed numerical trajectory."""

    payload = {
        "format_version": 1,
        "checkpoint": {"id": CHECKPOINT_ID, "revision": CHECKPOINT_REVISION},
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "seed": config.seed,
        "sampler_seed": config.sampler_seed,
        "nominal_steps": config.nominal_steps,
        "selected_steps": selected_steps,
        "effective_batch_size": config.effective_batch_size,
        "training_seconds": config.training_seconds,
        "benchmark_warmup_updates": config.benchmark_warmup_updates,
        "benchmark_measured_updates": config.benchmark_measured_updates,
        "lora": {
            "rank": config.rank,
            "alpha": config.alpha,
            "dropout": config.dropout,
        },
        "checkpoint_interval": config.checkpoint_interval,
        "base_artifact": dict(base_artifact),
        "optimizer": asdict(optimizer_config),
        "train_statistics_sha256": train_statistics_sha256,
        "train_episodes": list(train_episodes),
        "holdout_episodes": list(holdout_episodes),
        "base_dtype": "bfloat16",
        "adapter_dtype": "float32",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_base_artifact_identity(model: SmolVLATrainingModel) -> dict[str, str]:
    """Hash the exact converted base bytes rebuilt by a future resume."""

    model_path = model.converted_weights_path
    if model_path is None or not model_path.is_file():
        raise FileNotFoundError("training model has no converted base-weight artifact")
    name_map_path = model_path.parent / "name_map.json"
    if not name_map_path.is_file():
        raise FileNotFoundError(f"converted base name map is missing: {name_map_path}")
    return {
        "model_file": model_path.name,
        "model_sha256": _file_sha256(model_path),
        "name_map_file": name_map_path.name,
        "name_map_sha256": _file_sha256(name_map_path),
    }


@dataclass(frozen=True)
class UpdateResult:
    """One effective-batch optimizer update."""

    loss: float
    learning_rate: float
    gradient_norm: float
    clip_coefficient: float
    seconds: float


@dataclass(frozen=True)
class CheckpointState:
    """Non-tensor state required for sample- and optimizer-exact resume."""

    completed_step: int
    selected_steps: int
    smoothed_loss: float
    elapsed_training_seconds: float
    peak_memory_bytes: int
    samples_consumed: int
    flow_draw_count: int
    last_update: UpdateResult
    run_config_sha256: str


@dataclass(frozen=True)
class TrainingCheckpoint:
    """One fully published atomic training checkpoint."""

    path: Path
    state: CheckpointState
    metadata_sha256: str
    model_sha256: str
    optimizer_sha256: str
    pruned_checkpoints: tuple[str, ...] = ()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tensor_specs(tensors: Mapping[str, mx.array]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in tensors.items()
    }


def _checkpoint_directory_step(path: Path) -> int | None:
    if path.is_symlink() or not path.is_dir() or not path.name.startswith("step-"):
        return None
    suffix = path.name.removeprefix("step-")
    if len(suffix) != 6 or not suffix.isdigit():
        return None
    return int(suffix)


def _checkpoint_recovery_path(target: Path) -> Path:
    for index in range(1, 1_000_000):
        candidate = target.with_name(f".recovery-{target.name}-{index:06d}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError("checkpoint recovery namespace is exhausted")


def prune_training_checkpoints(
    checkpoint_root: str | Path,
    *,
    keep_last: int = 3,
    expected_run_config_sha256: str,
    trainable_names: tuple[str, ...],
    expected_model_tensors: Mapping[str, mx.array],
    expected_optimizer_tensors: Mapping[str, mx.array],
) -> tuple[str, ...]:
    """Remove only older complete checkpoints belonging to the active run."""

    if keep_last <= 0:
        raise ValueError("checkpoint retention must be positive")
    checkpoint_root = Path(checkpoint_root)
    complete: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        step = _checkpoint_directory_step(path)
        if step is None:
            continue
        try:
            _read_checkpoint_directory(
                path,
                expected_run_config_sha256=expected_run_config_sha256,
                trainable_names=trainable_names,
                expected_model_tensors=expected_model_tensors,
                expected_optimizer_tensors=expected_optimizer_tensors,
            )
        except Exception:
            continue
        complete.append((step, path))
    complete.sort()
    removed: list[str] = []
    for _, path in complete[:-keep_last]:
        shutil.rmtree(path)
        removed.append(path.name)
    if removed:
        _sync_directory(checkpoint_root)
    return tuple(removed)


def _checkpoint_state_dict(state: CheckpointState) -> dict[str, object]:
    return {
        **asdict(state),
        "last_update": asdict(state.last_update),
    }


def _checkpoint_state_from_dict(value: Mapping[str, object]) -> CheckpointState:
    required = {
        "completed_step",
        "selected_steps",
        "smoothed_loss",
        "elapsed_training_seconds",
        "peak_memory_bytes",
        "samples_consumed",
        "flow_draw_count",
        "last_update",
        "run_config_sha256",
    }
    if set(value) != required:
        raise ValueError("checkpoint state fields differ from the frozen schema")
    last_update = value["last_update"]
    if not isinstance(last_update, Mapping):
        raise ValueError("checkpoint last update must be an object")
    state = CheckpointState(
        completed_step=int(value["completed_step"]),
        selected_steps=int(value["selected_steps"]),
        smoothed_loss=float(value["smoothed_loss"]),
        elapsed_training_seconds=float(value["elapsed_training_seconds"]),
        peak_memory_bytes=int(value["peak_memory_bytes"]),
        samples_consumed=int(value["samples_consumed"]),
        flow_draw_count=int(value["flow_draw_count"]),
        last_update=UpdateResult(
            loss=float(last_update["loss"]),
            learning_rate=float(last_update["learning_rate"]),
            gradient_norm=float(last_update["gradient_norm"]),
            clip_coefficient=float(last_update["clip_coefficient"]),
            seconds=float(last_update["seconds"]),
        ),
        run_config_sha256=str(value["run_config_sha256"]),
    )
    if not 0 < state.completed_step <= state.selected_steps:
        raise ValueError("checkpoint completed step is outside its training horizon")
    if state.elapsed_training_seconds <= 0:
        raise ValueError("checkpoint elapsed training time must be positive")
    if state.samples_consumed < 0 or state.flow_draw_count < 0:
        raise ValueError("checkpoint draw/sample counts must be nonnegative")
    if len(state.run_config_sha256) != 64:
        raise ValueError("checkpoint run-config digest is invalid")
    numeric = (
        state.smoothed_loss,
        state.elapsed_training_seconds,
        state.last_update.loss,
        state.last_update.learning_rate,
        state.last_update.gradient_norm,
        state.last_update.clip_coefficient,
        state.last_update.seconds,
    )
    if not all(math.isfinite(item) for item in numeric):
        raise ValueError("checkpoint scalar state contains a non-finite value")
    return state


def _read_checkpoint_directory(
    path: Path,
    *,
    expected_run_config_sha256: str | None = None,
    trainable_names: tuple[str, ...] | None = None,
    expected_model_tensors: Mapping[str, mx.array] | None = None,
    expected_optimizer_tensors: Mapping[str, mx.array] | None = None,
) -> tuple[TrainingCheckpoint, dict[str, mx.array], dict[str, mx.array]]:
    """Validate one complete checkpoint directory without mutating live state."""

    path = Path(path)
    directory_step = _checkpoint_directory_step(path)
    if directory_step is None:
        raise ValueError(f"not a complete checkpoint directory name: {path}")
    metadata_path = path / "metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError(f"checkpoint metadata is missing or unsafe: {metadata_path}")
    metadata_sha256 = _file_sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if set(metadata) != {"format_version", "artifact_type", "state", "model", "optimizer"}:
        raise ValueError("checkpoint metadata fields differ from the frozen schema")
    if metadata["format_version"] != 1 or metadata["artifact_type"] != (
        "smolvla-mlx-training-checkpoint"
    ):
        raise ValueError("checkpoint metadata identity is invalid")
    state = _checkpoint_state_from_dict(metadata["state"])
    if state.completed_step != directory_step:
        raise ValueError("checkpoint directory name differs from its completed step")
    if (
        expected_run_config_sha256 is not None
        and state.run_config_sha256 != expected_run_config_sha256
    ):
        raise ValueError("checkpoint was produced by a different training configuration")

    def load_tensor_file(
        kind: str,
        expected_file: str,
        expected_tensors: Mapping[str, mx.array] | None,
        expected_names: tuple[str, ...] | None,
    ) -> tuple[dict[str, mx.array], str, tuple[str, ...]]:
        section = metadata[kind]
        if not isinstance(section, Mapping) or set(section) != {
            "file",
            "sha256",
            "tensor_names",
            "tensor_specs",
        }:
            raise ValueError(f"checkpoint {kind} metadata fields differ from the schema")
        if section["file"] != expected_file:
            raise ValueError(f"checkpoint {kind} filename is invalid")
        tensor_path = path / expected_file
        if tensor_path.is_symlink() or not tensor_path.is_file():
            raise ValueError(f"checkpoint {kind} tensor file is missing or unsafe")
        digest = _file_sha256(tensor_path)
        if digest != section["sha256"]:
            raise ValueError(f"checkpoint {kind} tensor digest is invalid")
        names = tuple(str(name) for name in section["tensor_names"])
        specs = section["tensor_specs"]
        if not isinstance(specs, Mapping) or set(specs) != set(names):
            raise ValueError(f"checkpoint {kind} tensor specs differ from its names")
        loaded = mx.load(str(tensor_path))
        if set(loaded) != set(names):
            raise ValueError(f"checkpoint {kind} tensor set differs from its metadata")
        loaded = {name: loaded[name] for name in names}
        if expected_names is not None and names != expected_names:
            raise ValueError(f"checkpoint {kind} tensor names differ from the current schema")
        if expected_tensors is not None and set(loaded) != set(expected_tensors):
            raise ValueError(f"checkpoint {kind} tensor set differs from the current schema")
        for name in names:
            spec = specs[name]
            if not isinstance(spec, Mapping) or set(spec) != {"shape", "dtype"}:
                raise ValueError(f"checkpoint {kind} tensor spec is invalid for {name}")
            if list(loaded[name].shape) != spec["shape"] or str(loaded[name].dtype) != spec[
                "dtype"
            ]:
                raise ValueError(f"checkpoint {kind} tensor differs from metadata for {name}")
            if expected_tensors is not None:
                expected = expected_tensors[name]
                if loaded[name].shape != expected.shape or loaded[name].dtype != expected.dtype:
                    raise ValueError(
                        f"checkpoint {kind} tensor shape/dtype changed for {name}"
                    )
        return loaded, digest, names

    loaded_model, model_sha256, model_names = load_tensor_file(
        "model",
        "model.safetensors",
        expected_model_tensors,
        trainable_names,
    )
    loaded_optimizer, optimizer_sha256, _ = load_tensor_file(
        "optimizer",
        "optimizer.safetensors",
        expected_optimizer_tensors,
        None,
    )
    mx.eval(loaded_model, loaded_optimizer)
    if "step" not in loaded_optimizer or int(loaded_optimizer["step"]) != state.completed_step:
        raise ValueError("checkpoint optimizer internal step differs from completed step")
    if "learning_rate" not in loaded_optimizer:
        raise ValueError("checkpoint optimizer is missing its learning rate")
    checkpoint = TrainingCheckpoint(
        path=path,
        state=state,
        metadata_sha256=metadata_sha256,
        model_sha256=model_sha256,
        optimizer_sha256=optimizer_sha256,
    )
    if model_names != tuple(metadata["model"]["tensor_names"]):
        raise AssertionError("validated checkpoint model-name order changed")
    return checkpoint, loaded_model, loaded_optimizer


def _write_latest_checkpoint_pointer(
    checkpoint_root: Path,
    checkpoint: TrainingCheckpoint,
) -> None:
    write_run_state(
        checkpoint_root / "latest.json",
        {
            "format_version": 1,
            "checkpoint": checkpoint.path.name,
            "completed_step": checkpoint.state.completed_step,
            "metadata_sha256": checkpoint.metadata_sha256,
        },
    )
    _sync_directory(checkpoint_root)


def save_training_checkpoint(
    *,
    model: nn.Module,
    optimizer: SmolVLAAdamW,
    checkpoint_root: str | Path,
    state: CheckpointState,
    trainable_names: tuple[str, ...],
    keep_last: int = 3,
) -> TrainingCheckpoint:
    """Atomically publish model, optimizer, and exact continuation state."""

    if optimizer.step_index != state.completed_step:
        raise ValueError(
            f"optimizer/checkpoint step mismatch: {optimizer.step_index} != {state.completed_step}"
        )
    checkpoint_root = Path(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    target = checkpoint_root / f"step-{state.completed_step:06d}"
    model_tensors = dict(tree_flatten(model.trainable_parameters()))
    if tuple(model_tensors) != trainable_names:
        raise ValueError("checkpoint model tensor names differ from the trainable contract")
    optimizer_tensors = dict(tree_flatten(optimizer.state))
    if not optimizer_tensors:
        raise ValueError("cannot checkpoint an uninitialized optimizer")
    optimizer.validate_state_for(model.trainable_parameters())
    mx.eval(model_tensors, optimizer_tensors)

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=checkpoint_root))
    try:
        model_path = temporary / "model.safetensors"
        optimizer_path = temporary / "optimizer.safetensors"
        mx.save_safetensors(str(model_path), model_tensors)
        mx.save_safetensors(str(optimizer_path), optimizer_tensors)
        _sync_file(model_path)
        _sync_file(optimizer_path)
        model_sha256 = _file_sha256(model_path)
        optimizer_sha256 = _file_sha256(optimizer_path)
        metadata = {
            "format_version": 1,
            "artifact_type": "smolvla-mlx-training-checkpoint",
            "state": _checkpoint_state_dict(state),
            "model": {
                "file": model_path.name,
                "sha256": model_sha256,
                "tensor_names": list(trainable_names),
                "tensor_specs": _tensor_specs(model_tensors),
            },
            "optimizer": {
                "file": optimizer_path.name,
                "sha256": optimizer_sha256,
                "tensor_names": list(optimizer_tensors),
                "tensor_specs": _tensor_specs(optimizer_tensors),
            },
        }
        metadata_sha256 = write_run_state(temporary / "metadata.json", metadata)
        _sync_directory(temporary)
        candidate = TrainingCheckpoint(
            path=target,
            state=state,
            metadata_sha256=metadata_sha256,
            model_sha256=model_sha256,
            optimizer_sha256=optimizer_sha256,
        )
        if target.exists() or target.is_symlink():
            try:
                existing, _, _ = _read_checkpoint_directory(
                    target,
                    expected_run_config_sha256=state.run_config_sha256,
                    trainable_names=trainable_names,
                    expected_model_tensors=model_tensors,
                    expected_optimizer_tensors=optimizer_tensors,
                )
            except Exception:
                existing = None
            if (
                existing is not None
                and existing.state == state
                and existing.model_sha256 == model_sha256
                and existing.optimizer_sha256 == optimizer_sha256
            ):
                shutil.rmtree(temporary)
                _write_latest_checkpoint_pointer(checkpoint_root, existing)
                pruned_checkpoints = prune_training_checkpoints(
                    checkpoint_root,
                    keep_last=keep_last,
                    expected_run_config_sha256=state.run_config_sha256,
                    trainable_names=trainable_names,
                    expected_model_tensors=model_tensors,
                    expected_optimizer_tensors=optimizer_tensors,
                )
                return TrainingCheckpoint(
                    path=existing.path,
                    state=existing.state,
                    metadata_sha256=existing.metadata_sha256,
                    model_sha256=existing.model_sha256,
                    optimizer_sha256=existing.optimizer_sha256,
                    pruned_checkpoints=pruned_checkpoints,
                )
            target.replace(_checkpoint_recovery_path(target))
            _sync_directory(checkpoint_root)
        temporary.replace(target)
        _sync_directory(checkpoint_root)
        _write_latest_checkpoint_pointer(checkpoint_root, candidate)
        pruned_checkpoints = prune_training_checkpoints(
            checkpoint_root,
            keep_last=keep_last,
            expected_run_config_sha256=state.run_config_sha256,
            trainable_names=trainable_names,
            expected_model_tensors=model_tensors,
            expected_optimizer_tensors=optimizer_tensors,
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return TrainingCheckpoint(
        path=target,
        state=state,
        metadata_sha256=metadata_sha256,
        model_sha256=model_sha256,
        optimizer_sha256=optimizer_sha256,
        pruned_checkpoints=pruned_checkpoints,
    )


def load_latest_training_checkpoint(
    *,
    model: nn.Module,
    optimizer: SmolVLAAdamW,
    checkpoint_root: str | Path,
    trainable_names: tuple[str, ...],
    expected_run_config_sha256: str,
) -> TrainingCheckpoint:
    """Discover, repair the pointer to, and restore the newest valid checkpoint."""

    checkpoint_root = Path(checkpoint_root)
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise FileNotFoundError(f"training checkpoint directory is missing: {checkpoint_root}")
    current_tensors = dict(tree_flatten(model.trainable_parameters()))
    if tuple(current_tensors) != trainable_names:
        raise ValueError("current model trainable names differ from the resume contract")
    optimizer.initialize(model.trainable_parameters())
    expected_optimizer_tensors = dict(tree_flatten(optimizer.state))

    valid: list[
        tuple[TrainingCheckpoint, dict[str, mx.array], dict[str, mx.array]]
    ] = []
    invalid: list[str] = []
    for path in checkpoint_root.iterdir():
        if _checkpoint_directory_step(path) is None:
            continue
        try:
            valid.append(
                _read_checkpoint_directory(
                    path,
                    expected_run_config_sha256=expected_run_config_sha256,
                    trainable_names=trainable_names,
                    expected_model_tensors=current_tensors,
                    expected_optimizer_tensors=expected_optimizer_tensors,
                )
            )
        except Exception as error:
            invalid.append(f"{path.name}: {type(error).__name__}: {error}")
    if not valid:
        detail = "; ".join(sorted(invalid)) if invalid else "no step directories"
        raise FileNotFoundError(f"no valid training checkpoint found in {checkpoint_root}: {detail}")
    valid.sort(key=lambda item: item[0].state.completed_step)
    checkpoint, loaded_model, loaded_optimizer = valid[-1]
    _write_latest_checkpoint_pointer(checkpoint_root, checkpoint)
    pruned_checkpoints = prune_training_checkpoints(
        checkpoint_root,
        keep_last=3,
        expected_run_config_sha256=expected_run_config_sha256,
        trainable_names=trainable_names,
        expected_model_tensors=current_tensors,
        expected_optimizer_tensors=expected_optimizer_tensors,
    )

    model.update(tree_unflatten([(name, loaded_model[name]) for name in trainable_names]))
    optimizer_names = tuple(loaded_optimizer)
    optimizer.load_state(
        tree_unflatten([(name, loaded_optimizer[name]) for name in optimizer_names]),
        step_index=checkpoint.state.completed_step,
    )
    mx.eval(model.trainable_parameters(), optimizer.state)
    return TrainingCheckpoint(
        path=checkpoint.path,
        state=checkpoint.state,
        metadata_sha256=checkpoint.metadata_sha256,
        model_sha256=checkpoint.model_sha256,
        optimizer_sha256=checkpoint.optimizer_sha256,
        pruned_checkpoints=pruned_checkpoints,
    )


def _optimizer_update(
    *,
    model: SmolVLATrainingModel,
    bridge: TrainingDataBridge,
    optimizer: SmolVLAAdamW,
    effective_batch_size: int,
) -> UpdateResult:
    start = time.perf_counter()
    batches = tuple(
        training_batch_from_bridge(bridge.next_batch())
        for _ in range(effective_batch_size)
    )
    accumulated = accumulate_gradients(model, batches, training_loss)
    clipped = clip_gradients_by_global_norm(
        accumulated.gradients,
        optimizer.config.grad_clip_norm,
    )
    learning_rate = optimizer.update(model, clipped.gradients)
    mx.eval(model.trainable_parameters(), optimizer.state)
    return UpdateResult(
        loss=float(accumulated.mean_loss),
        learning_rate=learning_rate,
        gradient_norm=float(clipped.total_norm),
        clip_coefficient=float(clipped.coefficient),
        seconds=time.perf_counter() - start,
    )


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured real effective-batch Metal update timings."""

    warmup_updates: int
    measured_updates: int
    effective_batch_size: int
    update_seconds: tuple[float, ...]
    median_update_seconds: float
    selected_steps: int
    nominal_steps: int
    estimated_training_seconds: float
    peak_memory_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BenchmarkResult":
        required = {
            "warmup_updates",
            "measured_updates",
            "effective_batch_size",
            "update_seconds",
            "median_update_seconds",
            "selected_steps",
            "nominal_steps",
            "estimated_training_seconds",
            "peak_memory_bytes",
        }
        if set(value) != required:
            raise ValueError("benchmark fields differ from the frozen schema")
        result = cls(
            warmup_updates=int(value["warmup_updates"]),
            measured_updates=int(value["measured_updates"]),
            effective_batch_size=int(value["effective_batch_size"]),
            update_seconds=tuple(float(item) for item in value["update_seconds"]),
            median_update_seconds=float(value["median_update_seconds"]),
            selected_steps=int(value["selected_steps"]),
            nominal_steps=int(value["nominal_steps"]),
            estimated_training_seconds=float(value["estimated_training_seconds"]),
            peak_memory_bytes=int(value["peak_memory_bytes"]),
        )
        if len(result.update_seconds) != result.measured_updates:
            raise ValueError("benchmark timing count differs from measured updates")
        if result.selected_steps <= 0 or result.selected_steps > result.nominal_steps:
            raise ValueError("benchmark selected step count is invalid")
        if not all(
            math.isfinite(item) and item > 0
            for item in (*result.update_seconds, result.median_update_seconds)
        ):
            raise ValueError("benchmark contains an invalid update timing")
        return result


def _build_training_components(config: FineTuneConfig, *, training_horizon: int):
    split = make_episode_split(num_episodes=50, seed=config.seed)
    stats = compute_train_statistics(
        config.cache_dir / "datasets" / "svla_so101_pickplace",
        split.train_episodes,
    )
    mx.random.seed(config.seed)
    model = SmolVLATrainingModel.from_pretrained(
        cache_dir=config.native_cache,
        dtype=mx.bfloat16,
    )
    model.train()
    lora_report = install_lora(
        model,
        LoRAConfig(rank=config.rank, alpha=config.alpha, dropout=config.dropout),
    )
    bridge = TrainingDataBridge(
        cache_dir=config.cache_dir,
        episodes=split.train_episodes,
        sampler_seed=config.sampler_seed,
        stats=stats.processor_stats,
    )
    optimizer_config = replace(
        SmolVLAOptimizerConfig(),
        training_horizon=training_horizon,
    )
    optimizer = SmolVLAAdamW(optimizer_config)
    return split, stats, model, lora_report, bridge, optimizer


def benchmark_lora_updates(config: FineTuneConfig) -> BenchmarkResult:
    """Measure 3+10 real effective-batch updates and select the frozen budget."""

    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"LoRA benchmark requires Metal GPU, got {mx.default_device()}")
    disk_free = shutil.disk_usage(Path.cwd()).free
    if disk_free < _MINIMUM_FREE_BYTES:
        raise RuntimeError(f"LoRA benchmark requires {_MINIMUM_FREE_BYTES} free bytes, got {disk_free}")
    _, _, model, _, bridge, optimizer = _build_training_components(
        config,
        training_horizon=config.nominal_steps,
    )
    mx.reset_peak_memory()
    measured: list[float] = []
    total = config.benchmark_warmup_updates + config.benchmark_measured_updates
    for update_index in range(total):
        result = _optimizer_update(
            model=model,
            bridge=bridge,
            optimizer=optimizer,
            effective_batch_size=config.effective_batch_size,
        )
        if update_index >= config.benchmark_warmup_updates:
            measured.append(result.seconds)
    median = statistics.median(measured)
    selected_steps = select_step_budget(
        median,
        nominal_steps=config.nominal_steps,
        training_seconds=config.training_seconds,
    )
    result = BenchmarkResult(
        warmup_updates=config.benchmark_warmup_updates,
        measured_updates=config.benchmark_measured_updates,
        effective_batch_size=config.effective_batch_size,
        update_seconds=tuple(measured),
        median_update_seconds=median,
        selected_steps=selected_steps,
        nominal_steps=config.nominal_steps,
        estimated_training_seconds=selected_steps * median,
        peak_memory_bytes=int(mx.get_peak_memory()),
    )
    del model, bridge, optimizer
    gc.collect()
    mx.clear_cache()
    return result


def _save_adapter_checkpoint(
    model: SmolVLATrainingModel,
    path: Path,
    *,
    lora_report,
) -> str:
    tensors = {
        name: value.astype(mx.float32)
        for name, value in tree_flatten(model.trainable_parameters())
    }
    if tuple(tensors) != lora_report.trainable_names:
        raise RuntimeError("final adapter tensor names changed during training")
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        mx.save_safetensors(str(temporary), tensors)
        _sync_file(temporary)
        temporary.replace(path)
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = _file_sha256(path)
    write_run_state(
        path.with_suffix(".json"),
        {
            "format_version": 1,
            "rank": lora_report.rank,
            "alpha": lora_report.alpha,
            "dropout": lora_report.dropout,
            "adapter_count": lora_report.adapter_count,
            "tensor_count": len(tensors),
            "scalar_count": lora_report.trainable_scalar_count,
            "sha256": digest,
        },
    )
    return digest


@dataclass(frozen=True)
class FineTuneResult:
    """Local artifacts and measurements from one completed training/export run."""

    selected_steps: int
    benchmark: BenchmarkResult
    training_seconds: float
    final_loss: float
    final_smoothed_loss: float
    peak_memory_bytes: int
    adapter_sha256: str
    export_dir: Path
    run_state_sha256: str


def run_lora_finetune(
    config: FineTuneConfig,
    *,
    progress: Callable[[int, int, UpdateResult], None] | None = None,
) -> FineTuneResult:
    """Benchmark, train or exactly resume, save adapters, merge, and export."""

    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"LoRA fine-tuning requires Metal GPU, got {mx.default_device()}")
    output_dir = config.output_dir.resolve()
    is_resume = output_dir.exists()
    if is_resume and not config.resume:
        raise FileExistsError(f"refusing to overwrite existing fine-tune run {output_dir}")
    disk_free_before = shutil.disk_usage(output_dir.parent).free
    if disk_free_before < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"LoRA fine-tuning requires {_MINIMUM_FREE_BYTES} free bytes, got {disk_free_before}"
        )

    run_path = output_dir / "run.json"
    if is_resume:
        if not run_path.is_file() or not (output_dir / "benchmark.json").is_file():
            raise FileNotFoundError("fine-tune output exists without resumable run metadata")
        run_document = json.loads(run_path.read_text(encoding="utf-8"))
        if run_document.get("status") == "trained_and_exported":
            raise FileExistsError(f"fine-tune run is already complete: {output_dir}")
        if run_document.get("status") not in {"running", "interrupted", "exporting"}:
            raise ValueError(f"fine-tune run status is not resumable: {run_document.get('status')!r}")
        benchmark = BenchmarkResult.from_dict(
            json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))
        )
    else:
        benchmark = benchmark_lora_updates(config)
    selected_steps = benchmark.selected_steps
    if (
        benchmark.warmup_updates != config.benchmark_warmup_updates
        or benchmark.measured_updates != config.benchmark_measured_updates
        or benchmark.effective_batch_size != config.effective_batch_size
        or benchmark.nominal_steps != config.nominal_steps
        or selected_steps
        != select_step_budget(
            benchmark.median_update_seconds,
            nominal_steps=config.nominal_steps,
            training_seconds=config.training_seconds,
        )
    ):
        raise ValueError("benchmark artifact differs from the requested training configuration")
    if not is_resume:
        output_dir.mkdir(parents=True)
        write_run_state(output_dir / "benchmark.json", benchmark.as_dict())
    split, stats, model, lora_report, bridge, optimizer = _build_training_components(
        config,
        training_horizon=selected_steps,
    )
    base_artifact = training_base_artifact_identity(model)
    run_config_sha256 = training_run_config_sha256(
        config,
        selected_steps=selected_steps,
        train_statistics_sha256=stats.sha256,
        train_episodes=split.train_episodes,
        holdout_episodes=split.holdout_episodes,
        base_artifact=base_artifact,
        optimizer_config=optimizer.config,
    )
    trainable_names = lora_report.trainable_names
    checkpoint_root = output_dir / "checkpoints"
    start_step = 0
    elapsed_before = 0.0
    previous_peak_memory = 0
    smoothed_loss: float | None = None
    final_update: UpdateResult | None = None
    resume_checkpoint_state: CheckpointState | None = None
    if is_resume:
        if run_document.get("run_config_sha256") != run_config_sha256:
            raise ValueError("existing run metadata differs from the requested configuration")
        checkpoint = load_latest_training_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=checkpoint_root,
            trainable_names=trainable_names,
            expected_run_config_sha256=run_config_sha256,
        )
        checkpoint_state = checkpoint.state
        resume_checkpoint_state = checkpoint_state
        expected_draws = checkpoint_state.completed_step * config.effective_batch_size
        if (
            checkpoint_state.selected_steps != selected_steps
            or checkpoint_state.samples_consumed != expected_draws
            or checkpoint_state.flow_draw_count != expected_draws
        ):
            raise ValueError("checkpoint counters differ from the requested training trajectory")
        bridge_state = bridge.state_dict()
        num_samples = int(bridge_state["num_samples"])
        epoch, start_index = divmod(checkpoint_state.samples_consumed, num_samples)
        bridge_state.update(
            {
                "samples_consumed": checkpoint_state.samples_consumed,
                "epoch": epoch,
                "start_index": start_index,
            }
        )
        bridge.load_state_dict(bridge_state)
        advance_flow_random_state(
            draw_count=checkpoint_state.flow_draw_count,
            shape=(1, 50, 32),
        )
        start_step = checkpoint_state.completed_step
        elapsed_before = checkpoint_state.elapsed_training_seconds
        previous_peak_memory = checkpoint_state.peak_memory_bytes
        smoothed_loss = checkpoint_state.smoothed_loss
        final_update = checkpoint_state.last_update
        run_document = {
            **run_document,
            "status": "running",
            "resume_count": int(run_document.get("resume_count", 0)) + 1,
            "resumed_from_step": start_step,
            "last_interruption": run_document.get("interruption"),
        }
    else:
        run_document = {
            "format_version": 1,
            "artifact_type": "smolvla-mlx-lora-run",
            "status": "running",
            "seed": config.seed,
            "sampler_seed": config.sampler_seed,
            "nominal_steps": config.nominal_steps,
            "selected_steps": selected_steps,
            "effective_batch_size": config.effective_batch_size,
            "training_seconds_budget": config.training_seconds,
            "benchmark": benchmark.as_dict(),
            "lora": {
                "rank": config.rank,
                "alpha": config.alpha,
                "dropout": config.dropout,
                "adapter_count": lora_report.adapter_count,
                "trainable_tensor_count": lora_report.trainable_tensor_count,
                "trainable_scalar_count": lora_report.trainable_scalar_count,
            },
            "split": {
                "train_episodes": list(split.train_episodes),
                "holdout_episodes": list(split.holdout_episodes),
                "holdout_fraction": split.holdout_fraction,
            },
            "train_statistics_sha256": stats.sha256,
            "run_config_sha256": run_config_sha256,
            "base_artifact": base_artifact,
            "optimizer": asdict(optimizer.config),
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "device": str(mx.default_device()),
            "base_dtype": "bfloat16",
            "adapter_dtype": "float32",
            "checkpoint_interval": config.checkpoint_interval,
            "checkpoint_count": 0,
            "resume_count": 0,
            "metrics_recoveries": [],
            "disk_free_before_bytes": disk_free_before,
        }
    write_run_state(run_path, run_document)

    train_start = time.perf_counter()
    mx.reset_peak_memory()
    completed_step = start_step
    try:
        with MetricsWriter(
            output_dir / "metrics.csv",
            resume_from_step=start_step if is_resume else None,
            checkpoint_state=resume_checkpoint_state,
        ) as metrics:
            if metrics.recovery_path is not None:
                run_document = {
                    **run_document,
                    "metrics_recoveries": [
                        *run_document.get("metrics_recoveries", []),
                        metrics.recovery_path.name,
                    ],
                }
                write_run_state(run_path, run_document)
            for step_index in range(start_step, selected_steps):
                update = _optimizer_update(
                    model=model,
                    bridge=bridge,
                    optimizer=optimizer,
                    effective_batch_size=config.effective_batch_size,
                )
                completed_step = step_index + 1
                final_update = update
                smoothed_loss = (
                    update.loss
                    if smoothed_loss is None
                    else 0.98 * smoothed_loss + 0.02 * update.loss
                )
                elapsed = elapsed_before + time.perf_counter() - train_start
                peak_memory = max(previous_peak_memory, int(mx.get_peak_memory()))
                metrics.write(
                    step=completed_step,
                    loss=update.loss,
                    smoothed_loss=smoothed_loss,
                    learning_rate=update.learning_rate,
                    gradient_norm=update.gradient_norm,
                    clip_coefficient=update.clip_coefficient,
                    elapsed_seconds=elapsed,
                    updates_per_second=completed_step / elapsed,
                    peak_memory_bytes=peak_memory,
                )
                if (
                    completed_step == 1
                    or completed_step % config.checkpoint_interval == 0
                    or completed_step == selected_steps
                ):
                    bridge_checkpoint_state = bridge.state_dict()
                    checkpoint = save_training_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        checkpoint_root=checkpoint_root,
                        state=CheckpointState(
                            completed_step=completed_step,
                            selected_steps=selected_steps,
                            smoothed_loss=smoothed_loss,
                            elapsed_training_seconds=elapsed,
                            peak_memory_bytes=peak_memory,
                            samples_consumed=int(
                                bridge_checkpoint_state["samples_consumed"]
                            ),
                            flow_draw_count=completed_step
                            * config.effective_batch_size,
                            last_update=update,
                            run_config_sha256=run_config_sha256,
                        ),
                        trainable_names=trainable_names,
                    )
                    run_document = {
                        **run_document,
                        "last_completed_step": completed_step,
                        "last_checkpoint": {
                            "step": completed_step,
                            "path": str(checkpoint.path),
                            "metadata_sha256": checkpoint.metadata_sha256,
                            "model_sha256": checkpoint.model_sha256,
                            "optimizer_sha256": checkpoint.optimizer_sha256,
                        },
                        "last_pruned_checkpoints": list(
                            checkpoint.pruned_checkpoints
                        ),
                        "checkpoint_count": int(run_document.get("checkpoint_count", 0))
                        + 1,
                    }
                    write_run_state(run_path, run_document)
                if progress is not None:
                    progress(completed_step, selected_steps, update)
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
    if final_update is None or smoothed_loss is None:
        raise RuntimeError("fine-tune loop completed without an optimizer update")
    actual_training_seconds = elapsed_before + time.perf_counter() - train_start
    peak_memory = max(previous_peak_memory, int(mx.get_peak_memory()))
    run_document = {
        **run_document,
        "status": "exporting",
        "last_completed_step": completed_step,
        "actual_training_seconds": actual_training_seconds,
        "peak_memory_bytes": peak_memory,
    }
    write_run_state(run_path, run_document)
    adapter_sha256 = _save_adapter_checkpoint(
        model,
        output_dir / "adapter.safetensors",
        lora_report=lora_report,
    )
    merge_report = merge_lora(model, dtype=mx.float32)
    source_checkpoint = resolve_base_checkpoint(config.cache_dir)
    export_metadata = {
        "seed": config.seed,
        "sampler_seed": config.sampler_seed,
        "selected_steps": selected_steps,
        "effective_batch_size": config.effective_batch_size,
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "adapter_sha256": adapter_sha256,
        "train_statistics_sha256": stats.sha256,
        "train_episodes": list(split.train_episodes),
        "holdout_episodes": list(split.holdout_episodes),
        "merge_adapter_count": merge_report.adapter_count,
        **frozen_export_audit_metadata(run_config_sha256),
    }
    export_dir = output_dir / "export"
    if export_dir.exists() or export_dir.is_symlink():
        export_report = validate_merged_checkpoint_export(
            export_dir,
            expected_metadata=export_metadata,
        )
    else:
        export_report = export_merged_checkpoint(
            model=model,
            source_checkpoint_dir=source_checkpoint,
            output_dir=export_dir,
            processor_stats=stats.processor_stats,
            metadata=export_metadata,
        )
    disk_free_after = shutil.disk_usage(output_dir.parent).free
    final_state = {
        **run_document,
        "status": "trained_and_exported",
        "actual_training_seconds": actual_training_seconds,
        "final_loss": final_update.loss,
        "final_smoothed_loss": smoothed_loss,
        "peak_memory_bytes": peak_memory,
        "adapter_sha256": adapter_sha256,
        "export": {
            "path": str(export_report.output_dir),
            "tensor_count": export_report.tensor_count,
            "parameter_count": export_report.parameter_count,
            "file_sha256": dict(export_report.file_sha256),
        },
        "disk_free_after_bytes": disk_free_after,
    }
    run_state_sha256 = write_run_state(output_dir / "run.json", final_state)
    return FineTuneResult(
        selected_steps=selected_steps,
        benchmark=benchmark,
        training_seconds=actual_training_seconds,
        final_loss=final_update.loss,
        final_smoothed_loss=smoothed_loss,
        peak_memory_bytes=peak_memory,
        adapter_sha256=adapter_sha256,
        export_dir=export_report.output_dir,
        run_state_sha256=run_state_sha256,
    )
