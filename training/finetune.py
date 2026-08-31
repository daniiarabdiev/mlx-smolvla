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
from typing import Callable, Iterable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from reference.discovery import DATASET_ID, DATASET_REVISION
from smolvla_mlx.types import ProcessedObservation
from training.dataset import (
    BridgeBatch,
    SAMPLER_SEED,
    SPLIT_SEED,
    TrainingDataBridge,
    compute_train_statistics,
    make_episode_split,
)
from training.export import export_merged_checkpoint, resolve_base_checkpoint
from training.lora import LoRAConfig, install_lora, merge_lora
from training.model import SmolVLATrainingModel, TrainingBatch, training_loss
from training.optimizer import (
    SmolVLAAdamW,
    SmolVLAOptimizerConfig,
    clip_gradients_by_global_norm,
)


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
    """Durable append-only CSV writer for one fresh fine-tuning run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"refusing to append to existing metrics file {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=METRICS_FIELDS)
        self._writer.writeheader()
        self._sync()

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def write(self, **values: object) -> None:
        if tuple(values) != METRICS_FIELDS:
            missing = tuple(name for name in METRICS_FIELDS if name not in values)
            unexpected = tuple(name for name in values if name not in METRICS_FIELDS)
            raise ValueError(f"metrics row fields differ; missing={missing}, unexpected={unexpected}")
        self._writer.writerow(values)
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

    def __post_init__(self) -> None:
        if self.nominal_steps <= 0 or self.effective_batch_size <= 0:
            raise ValueError("fine-tune steps and effective batch size must be positive")
        if self.benchmark_warmup_updates < 0 or self.benchmark_measured_updates <= 0:
            raise ValueError("benchmark warmup/measured update counts are invalid")
        LoRAConfig(rank=self.rank, alpha=self.alpha, dropout=self.dropout)


@dataclass(frozen=True)
class UpdateResult:
    """One effective-batch optimizer update."""

    loss: float
    learning_rate: float
    gradient_norm: float
    clip_coefficient: float
    seconds: float


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
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    mx.save_safetensors(str(temporary), tensors)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
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
    """Benchmark, train from a fresh base, save adapters, merge, and export."""

    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"LoRA fine-tuning requires Metal GPU, got {mx.default_device()}")
    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing fine-tune run {output_dir}")
    disk_free_before = shutil.disk_usage(output_dir.parent).free
    if disk_free_before < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"LoRA fine-tuning requires {_MINIMUM_FREE_BYTES} free bytes, got {disk_free_before}"
        )

    benchmark = benchmark_lora_updates(config)
    selected_steps = benchmark.selected_steps
    output_dir.mkdir(parents=True)
    write_run_state(output_dir / "benchmark.json", benchmark.as_dict())
    split, stats, model, lora_report, bridge, optimizer = _build_training_components(
        config,
        training_horizon=selected_steps,
    )
    initial_state = {
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
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "device": str(mx.default_device()),
        "base_dtype": "bfloat16",
        "adapter_dtype": "float32",
        "disk_free_before_bytes": disk_free_before,
    }
    write_run_state(output_dir / "run.json", initial_state)

    smoothed_loss = None
    final_update = None
    train_start = time.perf_counter()
    mx.reset_peak_memory()
    with MetricsWriter(output_dir / "metrics.csv") as metrics:
        for step_index in range(selected_steps):
            update = _optimizer_update(
                model=model,
                bridge=bridge,
                optimizer=optimizer,
                effective_batch_size=config.effective_batch_size,
            )
            final_update = update
            smoothed_loss = (
                update.loss
                if smoothed_loss is None
                else 0.98 * smoothed_loss + 0.02 * update.loss
            )
            elapsed = time.perf_counter() - train_start
            metrics.write(
                step=step_index + 1,
                loss=update.loss,
                smoothed_loss=smoothed_loss,
                learning_rate=update.learning_rate,
                gradient_norm=update.gradient_norm,
                clip_coefficient=update.clip_coefficient,
                elapsed_seconds=elapsed,
                updates_per_second=(step_index + 1) / elapsed,
                peak_memory_bytes=int(mx.get_peak_memory()),
            )
            if progress is not None:
                progress(step_index + 1, selected_steps, update)
    if final_update is None or smoothed_loss is None:
        raise RuntimeError("fine-tune loop completed without an optimizer update")
    actual_training_seconds = time.perf_counter() - train_start
    peak_memory = int(mx.get_peak_memory())
    adapter_sha256 = _save_adapter_checkpoint(
        model,
        output_dir / "adapter.safetensors",
        lora_report=lora_report,
    )
    merge_report = merge_lora(model, dtype=mx.float32)
    source_checkpoint = resolve_base_checkpoint(config.cache_dir)
    export_report = export_merged_checkpoint(
        model=model,
        source_checkpoint_dir=source_checkpoint,
        output_dir=output_dir / "export",
        processor_stats=stats.processor_stats,
        metadata={
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
        },
    )
    disk_free_after = shutil.disk_usage(output_dir.parent).free
    final_state = {
        **initial_state,
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
