"""Measured native MLX LoRA training loop for the Stage T3 outcome gate."""

from __future__ import annotations

import base64
import csv
import ctypes
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import errno
import fcntl
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import statistics
import sys
import tempfile
import time
import traceback
from typing import BinaryIO, Callable, Collection, Iterable, Iterator, Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np

from reference.discovery import (
    BASE_VLM_ID,
    BASE_VLM_REVISION,
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    DATASET_ID,
    DATASET_REVISION,
)
from mlx_smolvla.types import ProcessedObservation
from training.dataset import (
    BridgeBatch,
    SAMPLER_SEED,
    SPLIT_SEED,
    TrainingDataBridge,
    compute_train_statistics,
    make_episode_split,
)
from training.export import (
    expected_merged_checkpoint_support_file_sha256,
    export_merged_checkpoint,
    resolve_base_checkpoint,
    validate_merged_checkpoint_export,
    validate_bound_merged_checkpoint_export,
)
from training.gradients import canonical_parameter_name
from training.lora import (
    EXPERT_ONLY_SCOPE,
    LEGACY_FULL_SCOPE,
    LoRAConfig,
    install_lora,
    merge_lora,
)
from training.model import SmolVLATrainingModel, TrainingBatch, training_loss
from training.optimizer import (
    SmolVLAAdamW,
    SmolVLAOptimizerConfig,
    clip_gradients_by_global_norm,
)
from training.t3_contract import (
    FROZEN_BASE_REPORT_SHA256,
    FROZEN_CHECKPOINT_REVISION_TREE_SHA256,
    FROZEN_DATASET_REVISION_TREE_SHA256,
    FROZEN_EVALUATION_MANIFEST_SHA256,
    FROZEN_EVALUATION_METADATA_SHA256,
    FROZEN_TRAIN_STATISTICS_SHA256,
    FROZEN_TOKENIZER_REVISION_TREE_SHA256,
    frozen_export_audit_metadata,
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
ADAPTIVE_BUDGET_MODE = "adaptive_benchmark"
FIXED_BUDGET_MODE = "fixed_steps"
_VALID_BUDGET_MODES = {ADAPTIVE_BUDGET_MODE, FIXED_BUDGET_MODE}
_T3B_PRESTART_FILES = frozenset(
    {"launch.json", "training.lock", "training.log", "training.pid"}
)
_T3B_FROZEN_SETTINGS = {
    "seed": SPLIT_SEED,
    "sampler_seed": SAMPLER_SEED,
    "nominal_steps": 3_000,
    "effective_batch_size": 8,
    "rank": 8,
    "alpha": 16.0,
    "dropout": 0.0,
    "lora_scope": EXPERT_ONLY_SCOPE,
    "budget_mode": FIXED_BUDGET_MODE,
    "checkpoint_interval": 100,
}
_T3B_CHECKPOINT_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
)
_T3B_TOKENIZER_FILES = frozenset(
    {
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
)
_T3B_NATIVE_TOKENIZER_FILES = frozenset(
    {"tokenizer.json", "tokenizer_config.json"}
)
_T3B_EVALUATION_FILES = frozenset(
    {"manifest.json", "metadata.json"}
    | {
        f"cases/{ordinal:03d}/{name}.npy"
        for ordinal in range(56)
        for name in ("camera1", "camera2", "noise", "state", "target_action")
    }
)
_T3B_DATASET_FILES = frozenset(
    {
        "data/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "videos/observation.images.side/chunk-000/file-000.mp4",
        "videos/observation.images.up/chunk-000/file-000.mp4",
        f"revision/{DATASET_REVISION}.json",
    }
)


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
        reinitialize_zero_step: bool = False,
        parent_descriptor: int | None = None,
        expected_parent_snapshot: _DirectorySnapshot | None = None,
        expected_source_snapshot: _StableFileSnapshot | None = None,
    ) -> None:
        self.path = Path(os.path.abspath(Path(path).expanduser()))
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        if parent_descriptor is None:
            self._parent_snapshot = _ensure_safe_directory(
                self.path.parent,
                label="metrics directory",
            )
            self._parent_descriptor = os.open(self.path.parent, directory_flags)
        else:
            if expected_parent_snapshot is None:
                raise ValueError("bound metrics require a parent snapshot")
            if self.path.parent != expected_parent_snapshot.path:
                raise ValueError("metrics path differs from its bound parent")
            self._parent_snapshot = expected_parent_snapshot
            self._parent_descriptor = os.dup(parent_descriptor)
        parent_identity = os.fstat(self._parent_descriptor)
        _, parent_device, parent_inode = self._parent_snapshot.components[-1]
        if (parent_identity.st_dev, parent_identity.st_ino) != (
            parent_device,
            parent_inode,
        ):
            os.close(self._parent_descriptor)
            raise RuntimeError("metrics parent descriptor changed")
        self.recovery_path: Path | None = None
        self._parent_closed = False
        try:
            if resume_from_step is None:
                if reinitialize_zero_step:
                    raise ValueError("zero-step metrics reinitialization requires resume step 0")
                try:
                    existing = os.stat(
                        self.path.name,
                        dir_fd=self._parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    existing = None
                if existing is not None:
                    raise FileExistsError(
                        f"refusing to append to existing metrics file {self.path}"
                    )
                descriptor = os.open(
                    self.path.name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=self._parent_descriptor,
                )
                metrics_identity = os.fstat(descriptor)
                self._handle = os.fdopen(
                    descriptor,
                    "w",
                    newline="",
                    encoding="utf-8",
                )
                self._writer = csv.DictWriter(self._handle, fieldnames=METRICS_FIELDS)
                self._writer.writeheader()
                self._next_step = 1
                self._metrics_device = metrics_identity.st_dev
                self._metrics_inode = metrics_identity.st_ino
                self._sync()
                os.fsync(self._parent_descriptor)
                self._revalidate_metrics_handle()
                return
            if resume_from_step < 0:
                raise ValueError("metrics resume step must be nonnegative")
            if reinitialize_zero_step and resume_from_step != 0:
                raise ValueError("metrics reinitialization is only valid at step 0")
            if (
                checkpoint_state is not None
                and checkpoint_state.completed_step != resume_from_step
            ):
                raise ValueError("metrics resume step differs from checkpoint state")
            source = os.stat(
                self.path.name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(source.st_mode):
                raise FileNotFoundError(
                    f"metrics file is missing or unsafe for resume: {self.path}"
                )
            if expected_source_snapshot is not None:
                current_source = _snapshot_regular_file_at(
                    self._parent_descriptor,
                    self.path.name,
                    label="fine-tune metrics",
                    capture_payload=True,
                )
                if not _same_bound_file_snapshot(
                    current_source,
                    expected_source_snapshot,
                ):
                    raise RuntimeError(
                        "fine-tune metrics changed after checkpoint selection"
                    )
            self.recovery_path, recovery_identity = self._copy_to_next_recovery(source)
            if expected_source_snapshot is not None:
                copied_source = _snapshot_regular_file_at(
                    self._parent_descriptor,
                    self.path.name,
                    label="fine-tune metrics",
                    capture_payload=True,
                )
                if not _same_bound_file_snapshot(
                    copied_source,
                    expected_source_snapshot,
                ):
                    raise RuntimeError(
                        "fine-tune metrics changed during checkpoint recovery"
                    )
            rows = (
                []
                if reinitialize_zero_step
                else self._read_checkpoint_prefix(
                    self.recovery_path,
                    resume_from_step,
                    expected_identity=(
                        recovery_identity.st_dev,
                        recovery_identity.st_ino,
                    ),
                )
            )
            if checkpoint_state is not None and resume_from_step > 0:
                self._validate_checkpoint_boundary(rows[-1], checkpoint_state)
            published_device, published_inode = self._replace_with_prefix(
                rows,
                expected_destination=source,
            )
            descriptor = os.open(
                self.path.name,
                os.O_WRONLY
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._parent_descriptor,
            )
            opened_metrics = os.fstat(descriptor)
            if (opened_metrics.st_dev, opened_metrics.st_ino) != (
                published_device,
                published_inode,
            ):
                os.close(descriptor)
                raise RuntimeError("metrics changed before append reopen")
            self._handle = os.fdopen(
                descriptor,
                "a",
                newline="",
                encoding="utf-8",
            )
            self._writer = csv.DictWriter(self._handle, fieldnames=METRICS_FIELDS)
            self._next_step = resume_from_step + 1
            self._metrics_device = published_device
            self._metrics_inode = published_inode
            self._revalidate_metrics_handle()
        except BaseException:
            os.close(self._parent_descriptor)
            self._parent_closed = True
            raise

    def _read_checkpoint_prefix(
        self,
        source: Path,
        resume_from_step: int,
        *,
        expected_identity: tuple[int, int],
    ) -> list[dict[str, str]]:
        descriptor = os.open(
            source.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=self._parent_descriptor,
        )
        opened = os.fstat(descriptor)
        named_before = os.stat(
            source.name,
            dir_fd=self._parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != expected_identity
            or (named_before.st_dev, named_before.st_ino) != expected_identity
        ):
            os.close(descriptor)
            raise RuntimeError("metrics recovery changed before prefix validation")
        with os.fdopen(descriptor, newline="", encoding="utf-8") as handle:
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
                    raise ValueError(
                        f"metrics checkpoint prefix row {expected_step} is torn"
                    )
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
            opened_after = os.fstat(handle.fileno())
            named_after = os.stat(
                source.name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            if (
                (opened_after.st_dev, opened_after.st_ino) != expected_identity
                or (named_after.st_dev, named_after.st_ino) != expected_identity
            ):
                raise RuntimeError("metrics recovery changed during prefix validation")
        return rows

    def _copy_to_next_recovery(
        self,
        source: os.stat_result,
    ) -> tuple[Path, os.stat_result]:
        recovery = self._next_recovery_path()
        source_descriptor = os.open(
            self.path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=self._parent_descriptor,
        )
        recovery_descriptor = os.open(
            recovery.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=self._parent_descriptor,
        )
        try:
            opened = os.fstat(source_descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (source.st_dev, source.st_ino)
            ):
                raise RuntimeError("fine-tune metrics changed before recovery")
            while chunk := os.read(source_descriptor, 1024 * 1024):
                view = memoryview(chunk)
                while view:
                    written = os.write(recovery_descriptor, view)
                    view = view[written:]
            os.fsync(recovery_descriptor)
            recovery_identity = os.fstat(recovery_descriptor)
            after = os.fstat(source_descriptor)
            named = os.stat(
                self.path.name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            if (
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                != (source.st_dev, source.st_ino, source.st_size, source.st_mtime_ns)
                or (named.st_dev, named.st_ino) != (source.st_dev, source.st_ino)
            ):
                raise RuntimeError("fine-tune metrics changed during recovery")
            os.fsync(self._parent_descriptor)
        finally:
            os.close(recovery_descriptor)
            os.close(source_descriptor)
        return recovery, recovery_identity

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
            try:
                os.stat(
                    candidate.name,
                    dir_fd=self._parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return candidate
        raise RuntimeError("metrics recovery namespace is exhausted")

    def _replace_with_prefix(
        self,
        rows: list[dict[str, str]],
        *,
        expected_destination: os.stat_result,
    ) -> tuple[int, int]:
        descriptor, temporary_name = _create_staged_file_at(
            self._parent_descriptor,
            prefix=f".{self.path.name}.",
        )
        staged_identity = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            _publish_staged_file_at(
                parent_descriptor=self._parent_descriptor,
                staged_name=temporary_name,
                destination_name=self.path.name,
                staged_device=staged_identity.st_dev,
                staged_inode=staged_identity.st_ino,
                expected_destination=expected_destination,
            )
            published = os.stat(
                self.path.name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            if (published.st_dev, published.st_ino) != (
                staged_identity.st_dev,
                staged_identity.st_ino,
            ):
                raise RuntimeError("metrics prefix publication changed")
            return published.st_dev, published.st_ino
        finally:
            try:
                remaining = os.stat(
                    temporary_name,
                    dir_fd=self._parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                remaining = None
            if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
                staged_identity.st_dev,
                staged_identity.st_ino,
            ):
                os.unlink(temporary_name, dir_fd=self._parent_descriptor)
                os.fsync(self._parent_descriptor)

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def _revalidate_metrics_handle(self) -> None:
        opened = os.fstat(self._handle.fileno())
        named = os.stat(
            self.path.name,
            dir_fd=self._parent_descriptor,
            follow_symlinks=False,
        )
        expected = (self._metrics_device, self._metrics_inode)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != expected
            or (named.st_dev, named.st_ino) != expected
        ):
            raise RuntimeError("metrics file changed while open")

    def write(self, **values: object) -> None:
        self._revalidate_metrics_handle()
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
        self._revalidate_metrics_handle()

    def close(self) -> None:
        if not self._handle.closed:
            self._revalidate_metrics_handle()
            self._sync()
            self._revalidate_metrics_handle()
            self._handle.close()
        if not self._parent_closed:
            os.close(self._parent_descriptor)
            self._parent_closed = True

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def _create_staged_file_at(
    parent_descriptor: int,
    *,
    prefix: str,
    suffix: str = "",
) -> tuple[int, str]:
    """Create one unpredictable, exclusive regular file under a bound directory."""

    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _ in range(256):
        name = f"{prefix}{os.urandom(12).hex()}{suffix}"
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise RuntimeError(f"staged file is not regular: {name}")
        return descriptor, name
    raise RuntimeError("staged-file namespace is exhausted")


def _create_exclusive_child_file_at(
    parent_descriptor: int,
    name: str,
    *,
    mode: int = 0o600,
) -> tuple[int, os.stat_result]:
    """Create and bind one exact regular child without following a path."""

    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("bound child filename must be direct")
    descriptor = os.open(
        name,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
        dir_fd=parent_descriptor,
    )
    opened = os.fstat(descriptor)
    try:
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except BaseException:
        os.close(descriptor)
        raise
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        os.close(descriptor)
        raise RuntimeError(f"bound child changed during creation: {name}")
    return descriptor, opened


def _save_safetensors_child_at(
    parent_descriptor: int,
    name: str,
    tensors: Mapping[str, mx.array],
) -> _StableFileSnapshot:
    """Serialize safetensors through an already-bound child descriptor."""

    descriptor, identity = _create_exclusive_child_file_at(
        parent_descriptor,
        name,
    )
    with os.fdopen(descriptor, "w+b") as handle:
        mx.save_safetensors(handle, tensors)
        handle.flush()
        os.fsync(handle.fileno())
        after = os.fstat(handle.fileno())
        if (after.st_dev, after.st_ino) != (identity.st_dev, identity.st_ino):
            raise RuntimeError(f"bound safetensors child changed: {name}")
    snapshot = _snapshot_regular_file_at(
        parent_descriptor,
        name,
        label="bound safetensors child",
    )
    if (snapshot.device, snapshot.inode) != (identity.st_dev, identity.st_ino):
        raise RuntimeError(f"bound safetensors child name changed: {name}")
    os.fsync(parent_descriptor)
    return snapshot


def _descriptor_path(descriptor: int) -> Path:
    """Return the current absolute Darwin path naming one open descriptor."""

    if sys.platform != "darwin" or not hasattr(fcntl, "F_GETPATH"):
        raise RuntimeError("descriptor path resolution requires macOS F_GETPATH")
    payload = fcntl.fcntl(descriptor, fcntl.F_GETPATH, b"\0" * 1024)
    path = Path(os.fsdecode(payload.rstrip(b"\0")))
    named = os.stat(path, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
        raise RuntimeError("descriptor path does not name the opened inode")
    return path


def _require_exact_directory_inventory_at(
    descriptor: int,
    expected: Collection[str],
    *,
    label: str,
) -> None:
    """Reject any added, removed, or renamed direct child of a bound directory."""

    expected_names = frozenset(expected)
    if any(Path(name).name != name or name in {"", ".", ".."} for name in expected_names):
        raise ValueError(f"{label} expected inventory contains an unsafe name")
    actual_names = frozenset(os.listdir(descriptor))
    if actual_names != expected_names:
        raise RuntimeError(
            f"{label} namespace changed: actual={sorted(actual_names)}, "
            f"expected={sorted(expected_names)}"
        )


def _create_staged_directory_at(
    parent_descriptor: int,
    *,
    prefix: str,
    expected_parent_inventory: Collection[str] | None = None,
) -> tuple[int, str, Path]:
    """Create and open one unpredictable directory below a bound parent."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    expected_inventory = (
        None
        if expected_parent_inventory is None
        else frozenset(expected_parent_inventory)
    )
    for _ in range(256):
        if expected_inventory is not None:
            _require_exact_directory_inventory_at(
                parent_descriptor,
                expected_inventory,
                label="staged-directory parent",
            )
        name = f"{prefix}{os.urandom(12).hex()}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            os.close(descriptor)
            raise RuntimeError(f"staged directory changed during creation: {name}")
        if expected_inventory is not None:
            _require_exact_directory_inventory_at(
                parent_descriptor,
                {*expected_inventory, name},
                label="staged-directory parent",
            )
        return descriptor, name, _descriptor_path(descriptor)
    raise RuntimeError("staged-directory namespace is exhausted")


def _renameatx_np(
    *,
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
    flags: int,
) -> None:
    """Invoke Darwin's descriptor-relative atomic rename primitive."""

    if sys.platform != "darwin":
        raise RuntimeError("atomic descriptor-relative publication requires macOS")
    libc = ctypes.CDLL(None, use_errno=True)
    renameatx = libc.renameatx_np
    renameatx.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameatx.restype = ctypes.c_int
    result = renameatx(
        source_descriptor,
        os.fsencode(source_name),
        destination_descriptor,
        os.fsencode(destination_name),
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {
        errno.EEXIST,
        errno.ENOTEMPTY,
        errno.EISDIR,
        errno.ELOOP,
    }:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    raise OSError(error_number, os.strerror(error_number), source_name)


def _publish_staged_file_at(
    *,
    parent_descriptor: int,
    staged_name: str,
    destination_name: str,
    staged_device: int,
    staged_inode: int,
    expected_destination: os.stat_result | None,
) -> None:
    """Publish a bound staged file without ever displacing an unbound target."""

    staged_snapshot = _snapshot_regular_file_at(
        parent_descriptor,
        staged_name,
        label="staged state file",
    )
    if (
        (staged_snapshot.device, staged_snapshot.inode)
        != (staged_device, staged_inode)
    ):
        raise RuntimeError(f"staged state file changed before publication: {staged_name}")

    def quarantine_current_destination(current: _StableFileSnapshot) -> str:
        for _ in range(1_000_000):
            failed_name = (
                f".{destination_name}.publication-failed-{os.urandom(12).hex()}"
            )
            try:
                _renameatx_np(
                    source_descriptor=parent_descriptor,
                    source_name=destination_name,
                    destination_descriptor=parent_descriptor,
                    destination_name=failed_name,
                    flags=0x00000004 | 0x00000010,
                )
            except FileExistsError:
                continue
            quarantined = _snapshot_regular_file_at(
                parent_descriptor,
                failed_name,
                label="failed state publication",
            )
            if not _same_bound_file_snapshot(current, quarantined):
                raise RuntimeError(
                    f"published state file changed during quarantine: {destination_name}"
                )
            os.fsync(parent_descriptor)
            return failed_name
        raise RuntimeError("state publication failure namespace is exhausted")
    try:
        destination = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        destination = None
    if expected_destination is None:
        if destination is not None:
            raise FileExistsError(
                f"state destination appeared before publication: {destination_name}"
            )
        _rename_entry_no_clobber_at(
            source_descriptor=parent_descriptor,
            source_name=staged_name,
            destination_descriptor=parent_descriptor,
            destination_name=destination_name,
            expected_device=staged_device,
            expected_inode=staged_inode,
            expected_directory=False,
        )
        published_snapshot = _snapshot_regular_file_at(
            parent_descriptor,
            destination_name,
            label="published state file",
        )
        if not _same_bound_file_snapshot(staged_snapshot, published_snapshot):
            quarantine_current_destination(published_snapshot)
            raise RuntimeError(f"published state file changed: {destination_name}")
        os.fsync(parent_descriptor)
        return

    if (
        destination is None
        or not stat.S_ISREG(destination.st_mode)
        or (destination.st_dev, destination.st_ino)
        != (expected_destination.st_dev, expected_destination.st_ino)
    ):
        raise RuntimeError(
            f"state destination changed before publication: {destination_name}"
        )
    previous_name = (
        f".{destination_name}.previous-{os.urandom(12).hex()}"
    )
    _renameatx_np(
        source_descriptor=parent_descriptor,
        source_name=destination_name,
        destination_descriptor=parent_descriptor,
        destination_name=previous_name,
        flags=0x00000004 | 0x00000010,
    )
    previous = os.stat(
        previous_name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(previous.st_mode)
        or (previous.st_dev, previous.st_ino)
        != (expected_destination.st_dev, expected_destination.st_ino)
    ):
        try:
            _renameatx_np(
                source_descriptor=parent_descriptor,
                source_name=previous_name,
                destination_descriptor=parent_descriptor,
                destination_name=destination_name,
                flags=0x00000004 | 0x00000010,
            )
        finally:
            os.fsync(parent_descriptor)
        raise RuntimeError(
            f"state destination changed during publication: {destination_name}"
        )
    os.fsync(parent_descriptor)
    try:
        _rename_entry_no_clobber_at(
            source_descriptor=parent_descriptor,
            source_name=staged_name,
            destination_descriptor=parent_descriptor,
            destination_name=destination_name,
            expected_device=staged_device,
            expected_inode=staged_inode,
            expected_directory=False,
        )
    except BaseException:
        try:
            os.stat(
                destination_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            _renameatx_np(
                source_descriptor=parent_descriptor,
                source_name=previous_name,
                destination_descriptor=parent_descriptor,
                destination_name=destination_name,
                flags=0x00000004 | 0x00000010,
            )
        os.fsync(parent_descriptor)
        raise
    try:
        published_snapshot = _snapshot_regular_file_at(
            parent_descriptor,
            destination_name,
            label="published state file",
        )
    except FileNotFoundError:
        _renameatx_np(
            source_descriptor=parent_descriptor,
            source_name=previous_name,
            destination_descriptor=parent_descriptor,
            destination_name=destination_name,
            flags=0x00000004 | 0x00000010,
        )
        os.fsync(parent_descriptor)
        raise RuntimeError(f"published state file changed: {destination_name}")
    if not _same_bound_file_snapshot(staged_snapshot, published_snapshot):
        quarantine_current_destination(published_snapshot)
        _renameatx_np(
            source_descriptor=parent_descriptor,
            source_name=previous_name,
            destination_descriptor=parent_descriptor,
            destination_name=destination_name,
            flags=0x00000004 | 0x00000010,
        )
        os.fsync(parent_descriptor)
        raise RuntimeError(f"published state file changed: {destination_name}")
    preserved = os.stat(
        previous_name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(preserved.st_mode)
        or (preserved.st_dev, preserved.st_ino)
        != (expected_destination.st_dev, expected_destination.st_ino)
    ):
        raise RuntimeError(f"preserved state file changed: {destination_name}")
    os.unlink(previous_name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


_UNSPECIFIED_STATE_DESTINATION = object()


def _snapshot_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
    capture_payload: bool = False,
) -> _StableFileSnapshot:
    """Hash one stable no-follow child of an already-bound directory."""

    snapshot, descriptor = _open_bound_regular_file_at(
        parent_descriptor,
        name,
        label=label,
        capture_payload=capture_payload,
    )
    os.close(descriptor)
    return snapshot


def _open_bound_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
    capture_payload: bool = False,
) -> tuple[_StableFileSnapshot, int]:
    """Hash and retain one no-follow child of an already-bound directory."""

    if Path(name).name != name:
        raise ValueError(f"{label} name must be a direct child")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FileNotFoundError(f"{label} is not a regular file: {name}")
        digest = hashlib.sha256()
        payload = bytearray() if capture_payload else None
        byte_count = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
            if payload is not None:
                payload.extend(chunk)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            or (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or byte_count != after.st_size
        ):
            raise RuntimeError(f"{label} changed while it was read: {name}")
        snapshot = _StableFileSnapshot(
            path=_descriptor_path(parent_descriptor) / name,
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
            sha256=digest.hexdigest(),
            payload=None if payload is None else bytes(payload),
        )
        return snapshot, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_bound_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    expected: _StableFileSnapshot,
    descriptor: int,
    label: str,
    verify_bytes: bool,
) -> None:
    """Require a retained file descriptor to remain the same named bytes."""

    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"{label} changed while bound: {name}") from error
    expected_identity = (
        expected.device,
        expected.inode,
        expected.size,
        expected.mtime_ns,
    )
    opened_identity = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    )
    named_identity = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
    )
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or opened_identity != expected_identity
        or named_identity != expected_identity
    ):
        raise RuntimeError(f"{label} changed while bound: {name}")
    if not verify_bytes:
        return
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
        byte_count += len(chunk)
    after = os.fstat(descriptor)
    try:
        named_after = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise RuntimeError(f"{label} changed while bound: {name}") from error
    if (
        (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        != expected_identity
        or (
            named_after.st_dev,
            named_after.st_ino,
            named_after.st_size,
            named_after.st_mtime_ns,
        )
        != expected_identity
        or not stat.S_ISREG(named_after.st_mode)
        or byte_count != expected.size
        or digest.hexdigest() != expected.sha256
    ):
        raise RuntimeError(f"{label} changed while bound: {name}")


def _same_bound_file_snapshot(
    left: _StableFileSnapshot,
    right: _StableFileSnapshot,
) -> bool:
    """Compare file identity and bytes while ignoring a renamed parent path."""

    return (
        left.device,
        left.inode,
        left.size,
        left.mtime_ns,
        left.sha256,
    ) == (
        right.device,
        right.inode,
        right.size,
        right.mtime_ns,
        right.sha256,
    )


def _write_run_state_with_binding(
    path: str | Path,
    value: object,
    *,
    parent_descriptor: int | None = None,
    expected_parent_snapshot: _DirectorySnapshot | None = None,
    expected_destination_snapshot: _StableFileSnapshot | None | object = (
        _UNSPECIFIED_STATE_DESTINATION
    ),
) -> tuple[str, _StableFileSnapshot]:
    """Publish JSON with an optional exact prior-inode-and-bytes CAS binding."""

    path = Path(os.path.abspath(Path(path).expanduser()))
    owns_parent_descriptor = parent_descriptor is None
    if parent_descriptor is None:
        parent_snapshot = _ensure_safe_directory(
            path.parent,
            label="run-state directory",
        )
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        bound_parent = os.open(path.parent, directory_flags)
    else:
        if expected_parent_snapshot is None:
            raise ValueError("bound run-state writes require a parent snapshot")
        if path.parent != expected_parent_snapshot.path:
            raise ValueError("run-state path differs from its bound parent")
        parent_snapshot = expected_parent_snapshot
        bound_parent = os.dup(parent_descriptor)
    parent_identity = os.fstat(bound_parent)
    _, parent_device, parent_inode = parent_snapshot.components[-1]
    if (parent_identity.st_dev, parent_identity.st_ino) != (
        parent_device,
        parent_inode,
    ):
        os.close(bound_parent)
        raise RuntimeError("run-state parent descriptor changed")
    try:
        try:
            current_destination = _snapshot_regular_file_at(
                bound_parent,
                path.name,
                label="run-state destination",
            )
        except FileNotFoundError:
            current_destination = None
        if expected_destination_snapshot is _UNSPECIFIED_STATE_DESTINATION:
            destination_binding = current_destination
        elif expected_destination_snapshot is None:
            if current_destination is not None:
                raise FileExistsError(
                    f"run-state destination appeared before publication: {path.name}"
                )
            destination_binding = None
        else:
            assert isinstance(expected_destination_snapshot, _StableFileSnapshot)
            if current_destination is None or not _same_bound_file_snapshot(
                current_destination,
                expected_destination_snapshot,
            ):
                raise RuntimeError(
                    f"run-state destination changed before publication: {path.name}"
                )
            destination_binding = current_destination
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        descriptor, temporary_name = _create_staged_file_at(
            bound_parent,
            prefix=f".{path.name}.",
        )
        staged_identity = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if destination_binding is not None:
                before_publish = _snapshot_regular_file_at(
                    bound_parent,
                    path.name,
                    label="run-state destination",
                )
                if not _same_bound_file_snapshot(
                    before_publish,
                    destination_binding,
                ):
                    raise RuntimeError(
                        f"run-state destination changed before publication: {path.name}"
                    )
                expected_destination = os.stat(
                    path.name,
                    dir_fd=bound_parent,
                    follow_symlinks=False,
                )
            else:
                expected_destination = None
            _publish_staged_file_at(
                parent_descriptor=bound_parent,
                staged_name=temporary_name,
                destination_name=path.name,
                staged_device=staged_identity.st_dev,
                staged_inode=staged_identity.st_ino,
                expected_destination=expected_destination,
            )
            verification_descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=bound_parent,
            )
            try:
                published_payload = bytearray()
                while chunk := os.read(verification_descriptor, 1024 * 1024):
                    published_payload.extend(chunk)
                published_identity = os.fstat(verification_descriptor)
            finally:
                os.close(verification_descriptor)
            if (
                bytes(published_payload) != payload
                or (published_identity.st_dev, published_identity.st_ino)
                != (staged_identity.st_dev, staged_identity.st_ino)
            ):
                raise RuntimeError(f"published run-state bytes changed: {path.name}")
            published_snapshot = _StableFileSnapshot(
                path=_descriptor_path(bound_parent) / path.name,
                device=published_identity.st_dev,
                inode=published_identity.st_ino,
                size=published_identity.st_size,
                mtime_ns=published_identity.st_mtime_ns,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        finally:
            try:
                remaining = os.stat(
                    temporary_name,
                    dir_fd=bound_parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                remaining = None
            if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
                staged_identity.st_dev,
                staged_identity.st_ino,
            ):
                os.unlink(temporary_name, dir_fd=bound_parent)
                os.fsync(bound_parent)
        if owns_parent_descriptor:
            _revalidate_directory_snapshot(
                parent_snapshot,
                label="run-state directory",
            )
        return published_snapshot.sha256, published_snapshot
    finally:
        os.close(bound_parent)


def write_run_state(
    path: str | Path,
    value: object,
    *,
    parent_descriptor: int | None = None,
    expected_parent_snapshot: _DirectorySnapshot | None = None,
) -> str:
    """Atomically publish one complete JSON document under a bound directory."""

    digest, _ = _write_run_state_with_binding(
        path,
        value,
        parent_descriptor=parent_descriptor,
        expected_parent_snapshot=expected_parent_snapshot,
    )
    return digest


@dataclass(frozen=True)
class FineTuneConfig:
    """Frozen settings for one measured Stage T3 run."""

    cache_dir: Path = Path(".cache/hf")
    native_cache: Path = Path(".cache/mlx_smolvla/policy-float32")
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
    lora_scope: str = LEGACY_FULL_SCOPE
    budget_mode: str = ADAPTIVE_BUDGET_MODE
    checkpoint_interval: int = 100
    resume: bool = False

    def __post_init__(self) -> None:
        if self.nominal_steps <= 0 or self.effective_batch_size <= 0:
            raise ValueError("fine-tune steps and effective batch size must be positive")
        if self.benchmark_warmup_updates < 0 or self.benchmark_measured_updates <= 0:
            raise ValueError("benchmark warmup/measured update counts are invalid")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint interval must be positive")
        if self.budget_mode not in _VALID_BUDGET_MODES:
            raise ValueError(
                "fine-tune budget mode must be one of "
                f"{sorted(_VALID_BUDGET_MODES)}, got {self.budget_mode!r}"
            )
        LoRAConfig(
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            scope=self.lora_scope,
        )


def fixed_step_budget(config: FineTuneConfig) -> dict[str, object]:
    """Return the deterministic, timing-free fixed-step commitment."""

    if config.budget_mode != FIXED_BUDGET_MODE:
        raise ValueError("fixed step budget requires fixed_steps mode")
    return {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-fixed-step-budget",
        "mode": FIXED_BUDGET_MODE,
        "timing_measurements": False,
        "selected_steps": config.nominal_steps,
        "nominal_steps": config.nominal_steps,
        "effective_batch_size": config.effective_batch_size,
    }


def validate_fixed_step_budget(
    value: object,
    *,
    config: FineTuneConfig,
) -> dict[str, object]:
    """Validate that a persisted fixed budget exactly matches its config."""

    expected = fixed_step_budget(config)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("fixed step budget differs from the requested configuration")
    return dict(value)


def _launch_utc_from_ns(value_ns: int) -> str:
    if isinstance(value_ns, bool) or not isinstance(value_ns, int) or value_ns <= 0:
        raise ValueError("launch timestamp must be a positive integer")
    seconds, nanoseconds = divmod(value_ns, 1_000_000_000)
    value = datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(
        microseconds=nanoseconds // 1_000
    )
    return value.isoformat(timespec="microseconds")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _launch_training_config(config: FineTuneConfig) -> dict[str, object]:
    return {
        "cache_dir": str(config.cache_dir.resolve()),
        "native_cache": str(config.native_cache.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "seed": config.seed,
        "sampler_seed": config.sampler_seed,
        "nominal_steps": config.nominal_steps,
        "effective_batch_size": config.effective_batch_size,
        "training_seconds": config.training_seconds,
        "benchmark_warmup_updates": config.benchmark_warmup_updates,
        "benchmark_measured_updates": config.benchmark_measured_updates,
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "lora_scope": config.lora_scope,
        "budget_mode": config.budget_mode,
        "checkpoint_interval": config.checkpoint_interval,
    }


def _requires_t3b_launch_config(config: FineTuneConfig) -> bool:
    return (
        config.lora_scope == EXPERT_ONLY_SCOPE
        and config.budget_mode == FIXED_BUDGET_MODE
    )


def _validate_t3b_frozen_config(config: FineTuneConfig) -> None:
    actual = {name: getattr(config, name) for name in _T3B_FROZEN_SETTINGS}
    if actual != _T3B_FROZEN_SETTINGS:
        raise ValueError(
            f"T3B launch configuration differs from the frozen plan: {actual}"
        )


def _validate_t3b_train_statistics_sha256(value: object) -> str:
    if value != FROZEN_TRAIN_STATISTICS_SHA256:
        raise ValueError(
            "T3B training statistics differ from the frozen training statistics"
        )
    return FROZEN_TRAIN_STATISTICS_SHA256


def _resolve_t3b_launch_path(
    config: FineTuneConfig,
    launch_config_path: str | Path | None,
) -> Path:
    output_dir = _safe_t3b_output_path(
        config.output_dir,
        must_exist=True,
        label="T3B fine-tune output path",
    )
    expected = output_dir / "launch.json"
    candidate = (
        expected
        if launch_config_path is None
        else Path(os.path.abspath(Path(launch_config_path).expanduser()))
    )
    if candidate != expected:
        raise ValueError("T3B launch configuration must be output_dir/launch.json")
    return expected


def _validate_prepared_t3b_output(output_dir: str | Path) -> None:
    """Require a pristine prepared run directory before run.json exists."""

    output_dir = Path(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise FileExistsError(f"prepared T3B output directory is unsafe: {output_dir}")
    entries = tuple(output_dir.iterdir())
    unexpected = sorted(entry.name for entry in entries if entry.name not in _T3B_PRESTART_FILES)
    if unexpected:
        raise FileExistsError(
            f"prepared T3B output has unexpected entries: {unexpected}"
        )
    launch_path = output_dir / "launch.json"
    if launch_path not in entries:
        raise FileNotFoundError(f"prepared T3B output has no launch.json: {output_dir}")
    unsafe = sorted(
        entry.name for entry in entries if entry.is_symlink() or not entry.is_file()
    )
    if unsafe:
        raise FileExistsError(f"prepared T3B output has unsafe entries: {unsafe}")


@dataclass
class _T3BTrainingLease:
    """Live lock-file ownership bound to the original output-directory inode."""

    lock_descriptor: int
    output_descriptor: int
    output_snapshot: _DirectorySnapshot
    lock_device: int
    lock_inode: int
    checkpoint_root_descriptor: int | None = None
    checkpoint_root_snapshot: _DirectorySnapshot | None = None
    export_descriptor: int | None = None
    export_snapshot: _DirectorySnapshot | None = None
    export_file_bindings: dict[
        str, tuple[_StableFileSnapshot, int]
    ] | None = None
    adapter_file_bindings: dict[
        str, tuple[_StableFileSnapshot, int]
    ] | None = None
    metrics_file_binding: tuple[_StableFileSnapshot, int] | None = None
    final_checkpoint_binding_stack: ExitStack | None = None
    final_checkpoint_bindings: dict[str, _BoundCheckpointCandidate] | None = None
    final_checkpoint_pointer_binding: tuple[_StableFileSnapshot, int] | None = None
    final_checkpoint_inventory: frozenset[str] | None = None


def _acquire_t3b_training_lock(output_dir: str | Path) -> _T3BTrainingLease:
    """Take one nonblocking OS lock before any mutable training artifact."""

    output_dir = Path(output_dir)
    directory_snapshot = _snapshot_directory(
        output_dir,
        label="T3B lock directory",
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    output_descriptor = os.open(output_dir, directory_flags)
    output_identity = os.fstat(output_descriptor)
    _, output_device, output_inode = directory_snapshot.components[-1]
    if (output_identity.st_dev, output_identity.st_ino) != (
        output_device,
        output_inode,
    ):
        os.close(output_descriptor)
        raise RuntimeError(f"T3B output directory changed before lock: {output_dir}")
    path = output_dir / "training.lock"
    try:
        existing_lock = os.stat(
            path.name,
            dir_fd=output_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        existing_lock = None
    if existing_lock is not None and not stat.S_ISREG(existing_lock.st_mode):
        os.close(output_descriptor)
        raise FileExistsError(f"T3B training lock is unsafe: {path}")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(
            path.name,
            flags,
            0o600,
            dir_fd=output_descriptor,
        )
    except BaseException:
        os.close(output_descriptor)
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FileExistsError(f"T3B training lock is not a regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BlockingIOError(
                error.errno,
                f"T3B training run is already owned: {output_dir}",
            ) from error
        locked = os.fstat(descriptor)
        try:
            named = os.stat(
                path.name,
                dir_fd=output_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise RuntimeError(
                f"T3B training lock changed while it was acquired: {path}"
            ) from error
        if (
            not stat.S_ISREG(named.st_mode)
            or (locked.st_dev, locked.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError(
                f"T3B training lock changed while it was acquired: {path}"
            )
        lease = _T3BTrainingLease(
            lock_descriptor=descriptor,
            output_descriptor=output_descriptor,
            output_snapshot=directory_snapshot,
            lock_device=locked.st_dev,
            lock_inode=locked.st_ino,
        )
        _revalidate_t3b_training_lock(lease)
        return lease
    except BaseException:
        os.close(descriptor)
        os.close(output_descriptor)
        raise


def _revalidate_t3b_training_lock(lease: _T3BTrainingLease) -> None:
    """Prove that the held lock still names the originally acquired output tree."""

    try:
        output_identity = os.fstat(lease.output_descriptor)
    except OSError as error:
        raise RuntimeError("T3B output directory lease is no longer open") from error
    _, output_device, output_inode = lease.output_snapshot.components[-1]
    if (output_identity.st_dev, output_identity.st_ino) != (
        output_device,
        output_inode,
    ):
        raise RuntimeError("T3B output directory descriptor changed")
    try:
        _revalidate_directory_snapshot(
            lease.output_snapshot,
            label="T3B output directory",
        )
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeError(
            f"T3B output directory changed while locked: {lease.output_snapshot.path}"
        ) from error
    lock_path = lease.output_snapshot.path / "training.lock"
    try:
        named = os.stat(
            lock_path.name,
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
        locked = os.fstat(lease.lock_descriptor)
    except OSError as error:
        raise RuntimeError(f"T3B training lock changed while held: {lock_path}") from error
    if (
        not stat.S_ISREG(named.st_mode)
        or (locked.st_dev, locked.st_ino)
        != (lease.lock_device, lease.lock_inode)
        or (named.st_dev, named.st_ino)
        != (lease.lock_device, lease.lock_inode)
    ):
        raise RuntimeError(f"T3B training lock changed while held: {lock_path}")


def _bind_t3b_checkpoint_root(
    lease: _T3BTrainingLease,
    *,
    allow_existing: bool,
) -> None:
    """Create/open `checkpoints` once and retain its inode for the whole run."""

    if lease.checkpoint_root_descriptor is not None:
        raise RuntimeError("T3B checkpoint root is already bound")
    _revalidate_t3b_training_lock(lease)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        named = os.stat(
            "checkpoints",
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        named = None
    if named is None:
        try:
            os.mkdir("checkpoints", mode=0o700, dir_fd=lease.output_descriptor)
        except FileExistsError as error:
            raise FileExistsError(
                "T3B checkpoint root appeared before exclusive creation"
            ) from error
        named = os.stat(
            "checkpoints",
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
        os.fsync(lease.output_descriptor)
    elif not allow_existing:
        raise FileExistsError("T3B checkpoint root appeared before fresh-run binding")
    if not stat.S_ISDIR(named.st_mode):
        raise FileExistsError("T3B checkpoint root is unsafe")
    descriptor = os.open(
        "checkpoints",
        directory_flags,
        dir_fd=lease.output_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("T3B checkpoint root changed during binding")
        snapshot = _snapshot_directory(
            _descriptor_path(descriptor),
            label="T3B checkpoint root",
        )
        lease.checkpoint_root_descriptor = descriptor
        lease.checkpoint_root_snapshot = snapshot
        descriptor = -1
        _revalidate_t3b_checkpoint_root(lease)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _revalidate_t3b_checkpoint_root(lease: _T3BTrainingLease) -> None:
    """Require the retained checkpoint inode to remain the named output child."""

    descriptor = lease.checkpoint_root_descriptor
    snapshot = lease.checkpoint_root_snapshot
    if descriptor is None or snapshot is None:
        raise RuntimeError("T3B checkpoint root is not bound")
    try:
        opened = os.fstat(descriptor)
        named = os.stat(
            "checkpoints",
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise RuntimeError("T3B checkpoint root changed while bound") from error
    _, expected_device, expected_inode = snapshot.components[-1]
    if (
        not stat.S_ISDIR(named.st_mode)
        or (opened.st_dev, opened.st_ino) != (expected_device, expected_inode)
        or (named.st_dev, named.st_ino) != (expected_device, expected_inode)
    ):
        raise RuntimeError("T3B checkpoint root changed while bound")
    try:
        _revalidate_directory_snapshot(snapshot, label="T3B checkpoint root")
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeError("T3B checkpoint root changed while bound") from error


def _bind_t3b_export_root(lease: _T3BTrainingLease) -> None:
    """Open the published export once and retain it through run completion."""

    if lease.export_descriptor is not None:
        raise RuntimeError("T3B export directory is already bound")
    _revalidate_t3b_training_lock(lease)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    named = os.stat(
        "export",
        dir_fd=lease.output_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(named.st_mode):
        raise ValueError("T3B export is not a directory")
    descriptor = os.open(
        "export",
        directory_flags,
        dir_fd=lease.output_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("T3B export changed during binding")
        snapshot = _snapshot_directory(
            _descriptor_path(descriptor),
            label="T3B export directory",
        )
        lease.export_descriptor = descriptor
        lease.export_snapshot = snapshot
        descriptor = -1
        _revalidate_t3b_export_root(lease)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _bind_t3b_adapter_files(
    lease: _T3BTrainingLease,
    *,
    expected_sha256: str,
    lora_report,
) -> None:
    """Retain the exact final adapter pair until run-state publication ends."""

    if lease.adapter_file_bindings is not None:
        raise RuntimeError("T3B adapter files are already bound")
    _revalidate_t3b_training_lock(lease)
    bindings: dict[str, tuple[_StableFileSnapshot, int]] = {}
    try:
        for name in ("adapter.safetensors", "adapter.json"):
            bindings[name] = _open_bound_regular_file_at(
                lease.output_descriptor,
                name,
                label="T3B adapter file",
                capture_payload=name == "adapter.json",
            )
        adapter_snapshot, _ = bindings["adapter.safetensors"]
        metadata_snapshot, _ = bindings["adapter.json"]
        if adapter_snapshot.sha256 != expected_sha256:
            raise RuntimeError("T3B adapter digest changed before binding")
        assert metadata_snapshot.payload is not None
        try:
            metadata = json.loads(metadata_snapshot.payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("T3B adapter metadata is not valid JSON") from error
        expected_metadata: dict[str, object] = {
            "format_version": 1,
            "rank": lora_report.rank,
            "alpha": lora_report.alpha,
            "dropout": lora_report.dropout,
            "adapter_count": lora_report.adapter_count,
            "tensor_count": len(lora_report.trainable_names),
            "scalar_count": lora_report.trainable_scalar_count,
            "sha256": expected_sha256,
        }
        scope = getattr(lora_report, "scope", LEGACY_FULL_SCOPE)
        if scope != LEGACY_FULL_SCOPE:
            expected_metadata["scope"] = scope
        if metadata != expected_metadata:
            raise ValueError("T3B adapter metadata differs from the trained adapter")
        lease.adapter_file_bindings = bindings
        _revalidate_t3b_adapter_files(lease, verify_bytes=True)
    except BaseException:
        if lease.adapter_file_bindings is bindings:
            lease.adapter_file_bindings = None
        for _, descriptor in bindings.values():
            os.close(descriptor)
        raise


def _final_metrics_evidence(
    snapshot: _StableFileSnapshot,
    *,
    checkpoint_state: CheckpointState,
) -> dict[str, object]:
    """Validate one complete metrics payload and return its immutable evidence."""

    _validate_metrics_checkpoint_snapshot(snapshot, checkpoint_state)
    assert snapshot.payload is not None
    try:
        reader = csv.DictReader(
            io.StringIO(snapshot.payload.decode("utf-8"), newline="")
        )
        rows = list(reader)
    except UnicodeDecodeError as error:
        raise ValueError("final metrics is not valid UTF-8") from error
    if (
        tuple(reader.fieldnames or ()) != METRICS_FIELDS
        or len(rows) != checkpoint_state.completed_step
    ):
        raise ValueError("final metrics row count differs from training")
    return {
        "file": "metrics.csv",
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size,
        "row_count": len(rows),
    }


def _validate_t3b_resume_metrics_evidence(
    run_document: Mapping[str, object],
    *,
    snapshot: _StableFileSnapshot,
    checkpoint_state: CheckpointState,
) -> None:
    """Bind a completed metrics file to the evidence committed before a crash."""

    recorded = run_document.get("metrics")
    if recorded is None:
        return
    current = _final_metrics_evidence(
        snapshot,
        checkpoint_state=checkpoint_state,
    )
    if not _same_canonical_json_value(recorded, current):
        raise ValueError("final metrics differ from the committed evidence")


def _bind_t3b_metrics_file(
    lease: _T3BTrainingLease,
    *,
    checkpoint_state: CheckpointState,
) -> dict[str, object]:
    """Retain and validate the complete final metrics file through publication."""

    if lease.metrics_file_binding is not None:
        raise RuntimeError("T3B metrics file is already bound")
    snapshot, descriptor = _open_bound_regular_file_at(
        lease.output_descriptor,
        "metrics.csv",
        label="T3B final metrics",
        capture_payload=True,
    )
    try:
        evidence = _final_metrics_evidence(
            snapshot,
            checkpoint_state=checkpoint_state,
        )
        lease.metrics_file_binding = (snapshot, descriptor)
        _revalidate_t3b_metrics_file(lease, verify_bytes=True)
        return evidence
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_t3b_metrics_file(
    lease: _T3BTrainingLease,
    *,
    verify_bytes: bool = False,
) -> None:
    """Require the final metrics name, inode, and optionally bytes to stay fixed."""

    binding = lease.metrics_file_binding
    if binding is None:
        raise RuntimeError("T3B final metrics file is not bound")
    snapshot, descriptor = binding
    _revalidate_bound_regular_file_at(
        lease.output_descriptor,
        "metrics.csv",
        expected=snapshot,
        descriptor=descriptor,
        label="T3B final metrics",
        verify_bytes=verify_bytes,
    )


def _revalidate_t3b_adapter_files(
    lease: _T3BTrainingLease,
    *,
    verify_bytes: bool = False,
) -> None:
    """Require both retained adapter children to remain the same named bytes."""

    bindings = lease.adapter_file_bindings
    if bindings is None or set(bindings) != {"adapter.safetensors", "adapter.json"}:
        raise RuntimeError("T3B adapter files are not bound")
    for name, (snapshot, descriptor) in bindings.items():
        _revalidate_bound_regular_file_at(
            lease.output_descriptor,
            name,
            expected=snapshot,
            descriptor=descriptor,
            label="T3B adapter file",
            verify_bytes=verify_bytes,
        )


def _bind_t3b_export_files(
    lease: _T3BTrainingLease,
    *,
    expected_report,
    expected_metadata: Mapping[str, object],
) -> None:
    """Retain every validated export child through final-state publication."""

    if lease.export_descriptor is None:
        raise RuntimeError("T3B export directory is not bound")
    if lease.export_file_bindings is not None:
        raise RuntimeError("T3B export files are already bound")
    expected_inventory = set(_T3B_CHECKPOINT_FILES) | {"training_manifest.json"}
    if set(os.listdir(lease.export_descriptor)) != expected_inventory:
        raise ValueError("T3B export inventory changed before file binding")
    bindings: dict[str, tuple[_StableFileSnapshot, int]] = {}
    try:
        for name in sorted(expected_inventory):
            bindings[name] = _open_bound_regular_file_at(
                lease.export_descriptor,
                name,
                label="T3B export file",
                capture_payload=name == "training_manifest.json",
            )
        actual_hashes = {
            name: bindings[name][0].sha256
            for name in _T3B_CHECKPOINT_FILES
        }
        if actual_hashes != dict(expected_report.file_sha256):
            raise RuntimeError("T3B export bytes changed after validation")
        manifest_snapshot, _ = bindings["training_manifest.json"]
        assert manifest_snapshot.payload is not None
        try:
            manifest = json.loads(manifest_snapshot.payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("T3B export manifest is not valid JSON") from error
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("metadata") != dict(expected_metadata)
            or manifest.get("file_sha256") != actual_hashes
        ):
            raise ValueError("T3B export manifest changed after validation")
        lease.export_file_bindings = bindings
        _revalidate_t3b_export_root(lease, verify_bytes=True)
    except BaseException:
        if lease.export_file_bindings is bindings:
            lease.export_file_bindings = None
        for _, descriptor in bindings.values():
            os.close(descriptor)
        raise


def _revalidate_t3b_export_root(
    lease: _T3BTrainingLease,
    *,
    verify_bytes: bool = False,
) -> None:
    """Require the retained export tree to remain the same named bytes."""

    descriptor = lease.export_descriptor
    snapshot = lease.export_snapshot
    if descriptor is None or snapshot is None:
        raise RuntimeError("T3B export directory is not bound")
    try:
        opened = os.fstat(descriptor)
        named = os.stat(
            "export",
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise RuntimeError("T3B export changed while bound") from error
    _, expected_device, expected_inode = snapshot.components[-1]
    if (
        not stat.S_ISDIR(named.st_mode)
        or (opened.st_dev, opened.st_ino) != (expected_device, expected_inode)
        or (named.st_dev, named.st_ino) != (expected_device, expected_inode)
    ):
        raise RuntimeError("T3B export changed while bound")
    try:
        _revalidate_directory_snapshot(snapshot, label="T3B export directory")
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeError("T3B export changed while bound") from error
    bindings = lease.export_file_bindings
    if bindings is not None:
        if set(os.listdir(descriptor)) != set(bindings):
            raise RuntimeError("T3B export file inventory changed while bound")
        for name, (file_snapshot, file_descriptor) in bindings.items():
            _revalidate_bound_regular_file_at(
                descriptor,
                name,
                expected=file_snapshot,
                descriptor=file_descriptor,
                label="T3B export file",
                verify_bytes=verify_bytes,
            )


def _release_t3b_training_lock(lease: _T3BTrainingLease) -> None:
    try:
        fcntl.flock(lease.lock_descriptor, fcntl.LOCK_UN)
    finally:
        if lease.final_checkpoint_binding_stack is not None:
            lease.final_checkpoint_binding_stack.close()
            lease.final_checkpoint_binding_stack = None
            lease.final_checkpoint_bindings = None
            lease.final_checkpoint_pointer_binding = None
            lease.final_checkpoint_inventory = None
        if lease.checkpoint_root_descriptor is not None:
            os.close(lease.checkpoint_root_descriptor)
            lease.checkpoint_root_descriptor = None
            lease.checkpoint_root_snapshot = None
        if lease.export_descriptor is not None:
            os.close(lease.export_descriptor)
            lease.export_descriptor = None
            lease.export_snapshot = None
        if lease.export_file_bindings is not None:
            for _, descriptor in lease.export_file_bindings.values():
                os.close(descriptor)
            lease.export_file_bindings = None
        if lease.adapter_file_bindings is not None:
            for _, descriptor in lease.adapter_file_bindings.values():
                os.close(descriptor)
            lease.adapter_file_bindings = None
        if lease.metrics_file_binding is not None:
            _, descriptor = lease.metrics_file_binding
            os.close(descriptor)
            lease.metrics_file_binding = None
        os.close(lease.lock_descriptor)
        os.close(lease.output_descriptor)


@dataclass(frozen=True)
class _T3BTrainingLogLease:
    descriptor: int
    device: int
    inode: int
    header: bytes
    identity: Mapping[str, object]


def _training_log_identity(
    descriptor: int,
    *,
    header: bytes,
) -> dict[str, object]:
    """Validate the immutable first-line header and return its inode binding."""

    try:
        document = json.loads(header)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("T3B training log header is invalid") from error
    opened = os.fstat(descriptor)
    expected = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-training-log",
        "file": "training.log",
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "created_at_ns": document.get("created_at_ns") if isinstance(document, dict) else None,
    }
    if (
        not isinstance(document, dict)
        or set(document) != set(expected)
        or document != expected
        or isinstance(expected["created_at_ns"], bool)
        or not isinstance(expected["created_at_ns"], int)
        or expected["created_at_ns"] <= 0
    ):
        raise ValueError("T3B training log header differs from its inode")
    return {**expected, "header_sha256": hashlib.sha256(header).hexdigest()}


def _read_training_log_header(descriptor: int) -> bytes:
    payload = os.pread(descriptor, 4096, 0)
    header, separator, _ = payload.partition(b"\n")
    if not separator or not header:
        raise ValueError("T3B training log has no complete identity header")
    return header + b"\n"


def _revalidate_t3b_training_log(
    log_lease: _T3BTrainingLogLease,
    output_lease: _T3BTrainingLease,
) -> None:
    """Prove the named log still refers to the retained writable descriptor."""

    _revalidate_t3b_training_lock(output_lease)
    try:
        opened = os.fstat(log_lease.descriptor)
        named = os.stat(
            "training.log",
            dir_fd=output_lease.output_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise RuntimeError("T3B training log changed while held") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or (opened.st_dev, opened.st_ino) != (log_lease.device, log_lease.inode)
        or (named.st_dev, named.st_ino) != (log_lease.device, log_lease.inode)
        or os.pread(log_lease.descriptor, len(log_lease.header), 0)
        != log_lease.header
    ):
        raise RuntimeError("T3B training log changed while held")


def _quarantine_prestart_training_log(
    output_lease: _T3BTrainingLease,
    *,
    expected_device: int,
    expected_inode: int,
) -> str:
    """Preserve a complete pre-run log before creating a fresh launch log."""

    try:
        os.mkdir("startup-recoveries", mode=0o700, dir_fd=output_lease.output_descriptor)
    except FileExistsError:
        pass
    recovery_identity = os.stat(
        "startup-recoveries",
        dir_fd=output_lease.output_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(recovery_identity.st_mode):
        raise FileExistsError("T3B startup recovery root is unsafe")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    recovery_descriptor = os.open(
        "startup-recoveries",
        flags,
        dir_fd=output_lease.output_descriptor,
    )
    try:
        opened_recovery = os.fstat(recovery_descriptor)
        if (opened_recovery.st_dev, opened_recovery.st_ino) != (
            recovery_identity.st_dev,
            recovery_identity.st_ino,
        ):
            raise RuntimeError("T3B startup recovery root changed during open")
        recovered_name = _move_entry_to_unique_recovery_at(
            source_descriptor=output_lease.output_descriptor,
            source_name="training.log",
            destination_descriptor=recovery_descriptor,
            destination_prefix="training-log-prestart-",
            expected_device=expected_device,
            expected_inode=expected_inode,
            expected_directory=False,
        )
        os.fsync(recovery_descriptor)
        os.fsync(output_lease.output_descriptor)
        return recovered_name
    finally:
        os.close(recovery_descriptor)


def _open_t3b_training_log(
    lease: _T3BTrainingLease,
    path: str | Path,
    *,
    resume: bool,
    expected_identity: Mapping[str, object] | None = None,
) -> _T3BTrainingLogLease:
    """Open the in-run log below the held output dirfd without following it."""

    path = Path(os.path.abspath(Path(path).expanduser()))
    expected = lease.output_snapshot.path / "training.log"
    if path != expected:
        raise ValueError("T3B training log must be OUTPUT/training.log")
    _revalidate_t3b_training_lock(lease)
    try:
        named_before = os.stat(
            path.name,
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        named_before = None
    if named_before is not None and not stat.S_ISREG(named_before.st_mode):
        raise FileExistsError(f"T3B training log already exists or is unsafe: {path}")
    if named_before is None and resume:
        raise FileNotFoundError(f"resumable T3B training log is missing: {path}")

    flags = (
        os.O_RDWR
        | os.O_APPEND
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if named_before is not None and not resume:
        existing_descriptor = os.open(
            path.name,
            flags,
            dir_fd=lease.output_descriptor,
        )
        try:
            existing = os.fstat(existing_descriptor)
            try:
                existing_header = _read_training_log_header(existing_descriptor)
                _training_log_identity(existing_descriptor, header=existing_header)
            except ValueError as error:
                raise FileExistsError(
                    f"T3B training log already exists without a valid identity: {path}"
                ) from error
            named = os.stat(
                path.name,
                dir_fd=lease.output_descriptor,
                follow_symlinks=False,
            )
            if (named.st_dev, named.st_ino) != (existing.st_dev, existing.st_ino):
                raise RuntimeError("T3B prestart training log changed during open")
            _quarantine_prestart_training_log(
                lease,
                expected_device=existing.st_dev,
                expected_inode=existing.st_ino,
            )
        finally:
            os.close(existing_descriptor)
        named_before = None
    staging_descriptor: int | None = None
    staging_name: str | None = None
    staging_identity: os.stat_result | None = None
    if named_before is None:
        staging_descriptor, staging_name = _create_staged_file_at(
            lease.output_descriptor,
            prefix=".training.log.",
        )
        staging_identity = os.fstat(staging_descriptor)
        header_document = {
            "format_version": 1,
            "artifact_type": "smolvla-mlx-training-log",
            "file": path.name,
            "device": staging_identity.st_dev,
            "inode": staging_identity.st_ino,
            "created_at_ns": time.time_ns(),
        }
        header = json.dumps(
            header_document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        os.write(staging_descriptor, header + b"\n")
        os.fsync(staging_descriptor)
        try:
            _rename_entry_no_clobber_at(
                source_descriptor=lease.output_descriptor,
                source_name=staging_name,
                destination_descriptor=lease.output_descriptor,
                destination_name=path.name,
                expected_device=staging_identity.st_dev,
                expected_inode=staging_identity.st_ino,
                expected_directory=False,
            )
            os.fsync(lease.output_descriptor)
            descriptor = staging_descriptor
            staging_descriptor = None
            staging_name = None
            staging_identity = None
            current_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            fcntl.fcntl(descriptor, fcntl.F_SETFL, current_flags | os.O_APPEND)
        finally:
            if staging_descriptor is not None:
                os.close(staging_descriptor)
            if staging_name is not None and staging_identity is not None:
                try:
                    remaining = os.stat(
                        staging_name,
                        dir_fd=lease.output_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    remaining = None
                if remaining is not None and (
                    remaining.st_dev,
                    remaining.st_ino,
                ) == (staging_identity.st_dev, staging_identity.st_ino):
                    os.unlink(staging_name, dir_fd=lease.output_descriptor)
                    os.fsync(lease.output_descriptor)
    else:
        descriptor = os.open(
            path.name,
            flags,
            dir_fd=lease.output_descriptor,
        )
    try:
        opened = os.fstat(descriptor)
        header = _read_training_log_header(descriptor)
        identity = _training_log_identity(descriptor, header=header)
        if resume and (
            expected_identity is None or dict(expected_identity) != identity
        ):
            raise ValueError("resumable T3B training log differs from run metadata")
        named = os.stat(
            path.name,
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError("T3B training log changed during open")
        _revalidate_t3b_training_lock(lease)
        log_lease = _T3BTrainingLogLease(
            descriptor=descriptor,
            device=opened.st_dev,
            inode=opened.st_ino,
            header=header,
            identity=identity,
        )
        _revalidate_t3b_training_log(log_lease, lease)
        return log_lease
    except ValueError as error:
        os.close(descriptor)
        if not resume:
            raise FileExistsError(
                f"T3B training log already exists without a valid identity: {path}"
            ) from error
        raise
    except BaseException:
        os.close(descriptor)
        raise


def _read_bound_t3b_run_document(lease: _T3BTrainingLease) -> dict[str, object]:
    """Read run.json through the locked output dirfd for resume-log binding."""

    descriptor = os.open(
        "run.json",
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=lease.output_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("T3B run metadata is not a regular file")
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        named = os.stat(
            "run.json",
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError("T3B run metadata changed during log binding")
        value = json.loads(bytes(payload))
        if not isinstance(value, dict):
            raise ValueError("T3B run metadata must be an object")
        return value
    finally:
        os.close(descriptor)


@contextmanager
def _redirect_standard_streams_to_log(descriptor: int) -> Iterator[None]:
    """Redirect Python and native stdout/stderr to one already-bound log fd."""

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    log_stream = os.fdopen(
        os.dup(descriptor),
        "a",
        encoding="utf-8",
        buffering=1,
    )
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
        with redirect_stdout(log_stream), redirect_stderr(log_stream):
            yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            log_stream.flush()
            os.fsync(descriptor)
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            log_stream.close()


def _startup_recovery_inventory(
    output_dir: Path,
    *,
    output_descriptor: int | None = None,
    expected_output_snapshot: _DirectorySnapshot | None = None,
) -> tuple[str, ...] | None:
    """Return the exact safe pre-run recovery inventory through a retained dirfd."""

    output_dir = Path(os.path.abspath(output_dir))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if output_descriptor is None:
        output_snapshot = _snapshot_directory(
            output_dir,
            label="T3B startup recovery output",
        )
        bound_output = os.open(output_dir, directory_flags)
    else:
        if expected_output_snapshot is None:
            raise ValueError("bound startup recovery inventory requires an output snapshot")
        if output_dir != expected_output_snapshot.path:
            raise ValueError("startup recovery output differs from its bound directory")
        output_snapshot = expected_output_snapshot
        bound_output = os.dup(output_descriptor)
    recovery_descriptor: int | None = None
    try:
        output_identity = os.fstat(bound_output)
        _, output_device, output_inode = output_snapshot.components[-1]
        if (output_identity.st_dev, output_identity.st_ino) != (
            output_device,
            output_inode,
        ):
            raise RuntimeError("startup recovery output descriptor changed")
        try:
            recovery_named = os.stat(
                "startup-recoveries",
                dir_fd=bound_output,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return ()
        if not stat.S_ISDIR(recovery_named.st_mode):
            return None
        recovery_descriptor = os.open(
            "startup-recoveries",
            directory_flags,
            dir_fd=bound_output,
        )
        recovery_opened = os.fstat(recovery_descriptor)
        recovery_identity = (recovery_named.st_dev, recovery_named.st_ino)
        if (recovery_opened.st_dev, recovery_opened.st_ino) != recovery_identity:
            raise RuntimeError("T3B startup recovery root changed during open")
        names = tuple(sorted(os.listdir(recovery_descriptor)))
        allowed_prefixes = (
            "budget-json-partial-",
            "run-json-partial-",
            "training-pid-partial-",
            "training-log-partial-",
            "training-log-prestart-",
            "training-pid-prestart-",
        )
        recovered: list[tuple[str, _StableFileSnapshot]] = []
        for name in names:
            matching = next(
                (prefix for prefix in allowed_prefixes if name.startswith(prefix)),
                None,
            )
            suffix = "" if matching is None else name.removeprefix(matching)
            if matching is None or len(suffix) != 6 or not suffix.isdigit():
                return None
            try:
                child = os.stat(
                    name,
                    dir_fd=recovery_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return None
            if not stat.S_ISREG(child.st_mode):
                return None
            try:
                snapshot = _snapshot_regular_file_at(
                    recovery_descriptor,
                    name,
                    label="T3B startup recovery artifact",
                )
            except (FileNotFoundError, OSError):
                return None
            recovered.append((name, snapshot))
        if tuple(sorted(os.listdir(recovery_descriptor))) != names:
            return None
        for name, snapshot in recovered:
            try:
                current = _snapshot_regular_file_at(
                    recovery_descriptor,
                    name,
                    label="T3B startup recovery artifact",
                )
            except (FileNotFoundError, OSError):
                return None
            if not _same_bound_file_snapshot(snapshot, current):
                return None
        recovery_after = os.fstat(recovery_descriptor)
        named_after = os.stat(
            "startup-recoveries",
            dir_fd=bound_output,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named_after.st_mode)
            or (recovery_after.st_dev, recovery_after.st_ino) != recovery_identity
            or (named_after.st_dev, named_after.st_ino) != recovery_identity
        ):
            raise RuntimeError("T3B startup recovery root changed while bound")
        _revalidate_directory_snapshot(
            output_snapshot,
            label="T3B startup recovery output",
        )
        return tuple(f"startup-recoveries/{name}" for name, _ in recovered)
    finally:
        if recovery_descriptor is not None:
            os.close(recovery_descriptor)
        os.close(bound_output)


def _reconcile_t3b_prestart_output(
    output_dir: Path,
    *,
    config: FineTuneConfig,
    lease: _T3BTrainingLease,
) -> tuple[str, ...]:
    """Recover a crash before run.json without accepting any mutable run state."""

    _revalidate_t3b_training_lock(lease)
    output_dir = Path(os.path.abspath(output_dir))
    if output_dir != lease.output_snapshot.path:
        raise ValueError("T3B prestart output differs from the locked directory")
    expected_budget = fixed_step_budget(config)
    expected_budget_payload = (
        json.dumps(expected_budget, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    allowed_regular = _T3B_PRESTART_FILES | {"budget.json"}
    staging_prefixes = {
        ".budget.json.": "budget-json-partial-",
        ".run.json.": "run-json-partial-",
        ".training.pid.": "training-pid-partial-",
        ".training.log.": "training-log-partial-",
    }
    staging: list[tuple[Path, _StableFileSnapshot, str]] = []
    unexpected: list[str] = []
    for entry in sorted(output_dir.iterdir(), key=lambda path: path.name):
        if entry.name == "training.pid":
            if entry.is_symlink() or not entry.is_file():
                raise FileExistsError(f"prepared T3B output has unsafe entry: {entry}")
            staging.append(
                (
                    entry,
                    _snapshot_regular_file(
                        entry,
                        label="T3B stale prestart process identity",
                    ),
                    "training-pid-prestart-",
                )
            )
            continue
        if entry.name in allowed_regular:
            if entry.is_symlink() or not entry.is_file():
                raise FileExistsError(f"prepared T3B output has unsafe entry: {entry}")
            continue
        if entry.name == "startup-recoveries":
            continue
        matching = next(
            (
                (prefix, recovery_prefix)
                for prefix, recovery_prefix in staging_prefixes.items()
                if entry.name.startswith(prefix) and entry.name != prefix
            ),
            None,
        )
        if matching is None or entry.is_symlink() or not entry.is_file():
            unexpected.append(entry.name)
            continue
        staging.append(
            (
                entry,
                _snapshot_regular_file(entry, label="T3B startup staging artifact"),
                matching[1],
            )
        )
    if unexpected:
        raise FileExistsError(
            f"prepared T3B output has unexpected entries: {sorted(unexpected)}"
        )
    launch_path = output_dir / "launch.json"
    if not launch_path.is_file() or launch_path.is_symlink():
        raise FileNotFoundError(f"prepared T3B output has no safe launch.json: {output_dir}")
    budget_path = output_dir / "budget.json"
    if budget_path.exists() or budget_path.is_symlink():
        budget_snapshot = _snapshot_regular_file(
            budget_path,
            label="T3B prestart fixed budget",
            capture_payload=True,
        )
        if (
            budget_snapshot.payload != expected_budget_payload
            or budget_snapshot.sha256
            != hashlib.sha256(expected_budget_payload).hexdigest()
        ):
            raise ValueError("T3B prestart fixed budget differs from the frozen budget")
    existing = _startup_recovery_inventory(
        output_dir,
        output_descriptor=lease.output_descriptor,
        expected_output_snapshot=lease.output_snapshot,
    )
    if existing is None:
        raise FileExistsError("T3B startup recovery inventory is unsafe")
    if staging:
        recovery_root = output_dir / "startup-recoveries"
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            os.mkdir(recovery_root.name, mode=0o700, dir_fd=lease.output_descriptor)
        except FileExistsError:
            pass
        recovery_named = os.stat(
            recovery_root.name,
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(recovery_named.st_mode):
            raise FileExistsError("T3B startup recovery root is unsafe")
        recovery_descriptor = os.open(
            recovery_root.name,
            directory_flags,
            dir_fd=lease.output_descriptor,
        )
        try:
            recovery_identity = os.fstat(recovery_descriptor)
            if (recovery_identity.st_dev, recovery_identity.st_ino) != (
                recovery_named.st_dev,
                recovery_named.st_ino,
            ):
                raise RuntimeError("T3B startup recovery root changed")
            for entry, snapshot, recovery_prefix in staging:
                _move_entry_to_unique_recovery_at(
                    source_descriptor=lease.output_descriptor,
                    source_name=entry.name,
                    destination_descriptor=recovery_descriptor,
                    destination_prefix=recovery_prefix,
                    expected_device=snapshot.device,
                    expected_inode=snapshot.inode,
                    expected_directory=False,
                )
                os.fsync(recovery_descriptor)
                os.fsync(lease.output_descriptor)
        finally:
            os.close(recovery_descriptor)
    recovered = _startup_recovery_inventory(
        output_dir,
        output_descriptor=lease.output_descriptor,
        expected_output_snapshot=lease.output_snapshot,
    )
    if recovered is None:
        raise RuntimeError("T3B startup recovery inventory changed")
    _revalidate_t3b_training_lock(lease)
    return recovered


@dataclass(frozen=True)
class _DirectorySnapshot:
    """Identity of every directory component in one symlink-free path."""

    path: Path
    components: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class _StableFileSnapshot:
    """Bytes and identity read from one stable no-follow file descriptor."""

    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str
    payload: bytes | None = None


@dataclass(frozen=True)
class _T3BStartupBindings:
    """Immutable output children captured through the locked output dirfd."""

    run: _StableFileSnapshot | None
    launch: _StableFileSnapshot
    budget: _StableFileSnapshot | None
    process_identity: _StableFileSnapshot | None


def _snapshot_directory(path: str | Path, *, label: str) -> _DirectorySnapshot:
    """Reject symlinked ancestry and bind every directory inode in ``path``."""

    absolute = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(absolute.anchor)
    components: list[tuple[str, int, int]] = []
    paths = [current]
    for part in absolute.parts[1:]:
        current /= part
        paths.append(current)
    for candidate in paths:
        try:
            value = os.lstat(candidate)
        except OSError as error:
            raise FileNotFoundError(f"{label} is missing or unsafe: {absolute}") from error
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise FileNotFoundError(f"{label} has unsafe ancestry: {candidate}")
        components.append((str(candidate), value.st_dev, value.st_ino))
    if Path(os.path.realpath(absolute)) != absolute:
        raise FileNotFoundError(f"{label} has symlinked ancestry: {absolute}")
    return _DirectorySnapshot(path=absolute, components=tuple(components))


def _revalidate_directory_snapshot(
    expected: _DirectorySnapshot,
    *,
    label: str,
) -> None:
    current = _snapshot_directory(expected.path, label=label)
    if current != expected:
        raise RuntimeError(f"{label} changed while it was in use: {expected.path}")


def _ensure_safe_directory(path: str | Path, *, label: str) -> _DirectorySnapshot:
    """Create one final directory without following a pre-existing symlink."""

    path = Path(os.path.abspath(Path(path).expanduser()))
    parent = _snapshot_directory(path.parent, label=f"{label} parent")
    try:
        os.mkdir(path, mode=0o700)
    except FileExistsError:
        pass
    snapshot = _snapshot_directory(path, label=label)
    _revalidate_directory_snapshot(parent, label=f"{label} parent")
    return snapshot


def _create_safe_directory_no_clobber(
    path: str | Path,
    *,
    label: str,
) -> tuple[_DirectorySnapshot, int]:
    """Create one directory and fail if any competing entry already won."""

    path = Path(os.path.abspath(Path(path).expanduser()))
    parent = _snapshot_directory(path.parent, label=f"{label} parent")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(path.parent, directory_flags)
    child_descriptor: int | None = None
    try:
        opened_parent = os.fstat(parent_descriptor)
        _, parent_device, parent_inode = parent.components[-1]
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_device,
            parent_inode,
        ):
            raise RuntimeError(f"{label} parent changed before creation")
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError as error:
            raise FileExistsError(f"{label} already exists: {path}") from error
        named_child = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(named_child.st_mode):
            raise RuntimeError(f"{label} is not a directory after creation")
        child_descriptor = os.open(
            path.name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        opened_child = os.fstat(child_descriptor)
        if (opened_child.st_dev, opened_child.st_ino) != (
            named_child.st_dev,
            named_child.st_ino,
        ):
            raise RuntimeError(f"{label} changed during open")
        bound_path = _descriptor_path(child_descriptor)
        snapshot = _snapshot_directory(bound_path, label=label)
        _revalidate_directory_snapshot(parent, label=f"{label} parent")
        if bound_path != path:
            raise RuntimeError(f"{label} path changed during creation")
        result_descriptor = child_descriptor
        child_descriptor = None
        return snapshot, result_descriptor
    finally:
        if child_descriptor is not None:
            os.close(child_descriptor)
        os.close(parent_descriptor)


def _snapshot_regular_file(
    path: str | Path,
    *,
    label: str,
    capture_payload: bool = False,
) -> _StableFileSnapshot:
    """Hash one regular file through a stable descriptor without following it."""

    path = Path(os.path.abspath(Path(path).expanduser()))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}") from error
    chunks: list[bytes] | None = [] if capture_payload else None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FileNotFoundError(f"{label} is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    try:
        named = os.lstat(path)
    except OSError as error:
        raise RuntimeError(f"{label} path changed while it was read: {path}") from error
    named_identity = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
    )
    if (
        identity_before != identity_after
        or identity_after != named_identity
        or not stat.S_ISREG(named.st_mode)
        or byte_count != after.st_size
    ):
        raise RuntimeError(f"{label} changed while it was read: {path}")
    return _StableFileSnapshot(
        path=path,
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest.hexdigest(),
        payload=b"".join(chunks) if chunks is not None else None,
    )


def _same_file_snapshot(
    left: _StableFileSnapshot,
    right: _StableFileSnapshot,
) -> bool:
    return (
        left.path == right.path
        and left.device == right.device
        and left.inode == right.inode
        and left.size == right.size
        and left.mtime_ns == right.mtime_ns
        and left.sha256 == right.sha256
    )


def _optional_t3b_startup_file(
    lease: _T3BTrainingLease,
    name: str,
    *,
    label: str,
    capture_payload: bool,
) -> _StableFileSnapshot | None:
    try:
        return _snapshot_regular_file_at(
            lease.output_descriptor,
            name,
            label=label,
            capture_payload=capture_payload,
        )
    except FileNotFoundError:
        return None


def _capture_t3b_startup_bindings(
    lease: _T3BTrainingLease,
) -> _T3BStartupBindings:
    """Capture all startup state through the already-retained output inode."""

    _revalidate_t3b_training_lock(lease)
    try:
        launch = _snapshot_regular_file_at(
            lease.output_descriptor,
            "launch.json",
            label="T3B launch configuration",
            capture_payload=True,
        )
    except FileNotFoundError as error:
        raise FileNotFoundError("prepared T3B output has no safe launch.json") from error
    result = _T3BStartupBindings(
        run=_optional_t3b_startup_file(
            lease,
            "run.json",
            label="T3B run metadata",
            capture_payload=True,
        ),
        launch=launch,
        budget=_optional_t3b_startup_file(
            lease,
            "budget.json",
            label="T3B fixed budget",
            capture_payload=True,
        ),
        process_identity=_optional_t3b_startup_file(
            lease,
            "training.pid",
            label="T3B process identity",
            capture_payload=True,
        ),
    )
    _revalidate_t3b_training_lock(lease)
    return result


def _is_lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_from_stable_snapshot(
    snapshot: _StableFileSnapshot,
    *,
    label: str,
) -> object:
    if snapshot.payload is None:
        raise RuntimeError(f"{label} was not captured with its payload")
    try:
        return json.loads(snapshot.payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _validate_resume_process_identity(
    run_document: Mapping[str, object],
    snapshot: _StableFileSnapshot | None,
) -> None:
    """Require the prior run's PID document to match its immutable binding."""

    process = run_document.get("process")
    if not isinstance(process, Mapping):
        raise ValueError("resumable run has no process identity binding")
    required = {
        "format_version",
        "artifact_type",
        "pid",
        "parent_pid",
        "started_at_ns",
        "working_directory",
        "executable",
        "launch_config",
        "launch_file_sha256",
        "configuration_sha256",
        "run_config_sha256",
        "identity_file",
        "identity_sha256",
        "training_log",
    }
    if set(process) != required:
        raise ValueError("resumable run process identity fields differ")
    if (
        process["format_version"] != 1
        or process["artifact_type"] != "smolvla-mlx-training-process"
        or process["identity_file"] != "training.pid"
        or type(process["pid"]) is not int
        or type(process["parent_pid"]) is not int
        or type(process["started_at_ns"]) is not int
        or process["pid"] <= 0
        or process["parent_pid"] < 0
        or process["started_at_ns"] <= 0
        or not all(
            isinstance(process[name], str) and bool(process[name])
            for name in (
                "working_directory",
                "executable",
                "launch_config",
            )
        )
        or not all(
            _is_lowercase_sha256(process[name])
            for name in (
                "launch_file_sha256",
                "configuration_sha256",
                "run_config_sha256",
                "identity_sha256",
            )
        )
    ):
        raise ValueError("resumable run process identity is invalid")
    training_log = process["training_log"]
    if (
        not isinstance(training_log, Mapping)
        or set(training_log)
        != {
            "format_version",
            "artifact_type",
            "file",
            "device",
            "inode",
            "created_at_ns",
            "header_sha256",
        }
        or training_log["format_version"] != 1
        or training_log["artifact_type"] != "smolvla-mlx-training-log"
        or training_log["file"] != "training.log"
        or any(
            type(training_log[name]) is not int or training_log[name] <= 0
            for name in ("device", "inode", "created_at_ns")
        )
        or not _is_lowercase_sha256(training_log["header_sha256"])
    ):
        raise ValueError("resumable run training-log identity is invalid")
    if snapshot is None:
        raise FileNotFoundError("resumable run process identity file is missing")
    if snapshot.sha256 != process["identity_sha256"]:
        raise ValueError("resumable run process identity digest differs")
    persisted = _json_from_stable_snapshot(
        snapshot,
        label="fine-tune process identity",
    )
    expected = {
        name: value
        for name, value in process.items()
        if name not in {"identity_file", "identity_sha256"}
    }
    if persisted != expected:
        raise ValueError("resumable run process identity content differs")


def _same_canonical_json_value(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""

    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_t3b_resume_run_document(
    document: Mapping[str, object],
    *,
    expected_immutable: Mapping[str, object],
    selected_steps: int,
    checkpoint_interval: int,
) -> None:
    """Validate the full resumable run schema before recovery or mutation."""

    required_mutable = {
        "status",
        "checkpoint_count",
        "resume_count",
        "metrics_recoveries",
        "checkpoint_recoveries",
        "startup_recoveries",
        "disk_free_before_bytes",
        "process",
    }
    optional_mutable = {
        "last_completed_step",
        "last_checkpoint",
        "last_pruned_checkpoints",
        "interruption",
        "resumed_from_step",
        "last_interruption",
        "actual_training_seconds",
        "peak_memory_bytes",
        "metrics",
    }
    expected_keys = set(expected_immutable) | required_mutable
    if not expected_keys <= set(document) or not set(document) <= (
        expected_keys | optional_mutable
    ):
        raise ValueError("resumable run metadata fields differ from the schema")
    for name, expected in expected_immutable.items():
        if not _same_canonical_json_value(document.get(name), expected):
            raise ValueError(f"resumable run immutable field differs: {name}")
    if document["status"] not in {"running", "interrupted", "exporting"}:
        raise ValueError("resumable run status is invalid")
    metrics = document.get("metrics")
    if document["status"] == "exporting" and metrics is None:
        raise ValueError("resumable exporting run has no metrics evidence")
    if metrics is not None and (
        not isinstance(metrics, Mapping)
        or set(metrics) != {"file", "sha256", "size_bytes", "row_count"}
        or metrics["file"] != "metrics.csv"
        or not _is_lowercase_sha256(metrics["sha256"])
        or type(metrics["size_bytes"]) is not int
        or metrics["size_bytes"] <= 0
        or type(metrics["row_count"]) is not int
        or metrics["row_count"] != selected_steps
    ):
        raise ValueError("resumable run metrics evidence is invalid")
    for name in ("checkpoint_count", "resume_count", "disk_free_before_bytes"):
        if type(document[name]) is not int or document[name] < 0:
            raise ValueError(f"resumable run {name} must be a nonnegative integer")
    for name in (
        "metrics_recoveries",
        "checkpoint_recoveries",
        "startup_recoveries",
    ):
        value = document[name]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(f"resumable run {name} must contain path strings")
    for name in ("last_completed_step", "resumed_from_step"):
        if name in document and (
            type(document[name]) is not int
            or not 0 <= document[name] <= selected_steps
        ):
            raise ValueError(f"resumable run {name} is invalid")
    if "last_pruned_checkpoints" in document and (
        not isinstance(document["last_pruned_checkpoints"], list)
        or any(
            not isinstance(item, str) or not item.startswith("step-")
            for item in document["last_pruned_checkpoints"]
        )
    ):
        raise ValueError("resumable run pruned checkpoint list is invalid")
    for name in ("interruption", "last_interruption"):
        if name not in document or document[name] is None:
            continue
        value = document[name]
        if (
            not isinstance(value, Mapping)
            or set(value) != {"type", "message"}
            or not all(isinstance(item, str) for item in value.values())
        ):
            raise ValueError(f"resumable run {name} is invalid")
    if "actual_training_seconds" in document and (
        type(document["actual_training_seconds"]) is not float
        or not math.isfinite(document["actual_training_seconds"])
        or document["actual_training_seconds"] < 0
    ):
        raise ValueError("resumable run actual training time is invalid")
    if "peak_memory_bytes" in document and (
        type(document["peak_memory_bytes"]) is not int
        or document["peak_memory_bytes"] < 0
    ):
        raise ValueError("resumable run peak memory is invalid")
    last_checkpoint = document.get("last_checkpoint")
    if last_checkpoint is None:
        if document["checkpoint_count"] != 0:
            raise ValueError("resumable run checkpoint count has no checkpoint")
        if metrics is not None:
            raise ValueError("resumable run metrics evidence has no checkpoint")
        return
    if not isinstance(last_checkpoint, Mapping) or set(last_checkpoint) != {
        "step",
        "path",
        "metadata_sha256",
        "model_sha256",
        "optimizer_sha256",
    }:
        raise ValueError("resumable run last checkpoint is invalid")
    step = last_checkpoint["step"]
    if type(step) is not int or not 0 < step <= selected_steps:
        raise ValueError("resumable run last checkpoint step is invalid")
    if (
        not isinstance(last_checkpoint["path"], str)
        or Path(last_checkpoint["path"]).name != f"step-{step:06d}"
        or not all(
            _is_lowercase_sha256(last_checkpoint[name])
            for name in (
                "metadata_sha256",
                "model_sha256",
                "optimizer_sha256",
            )
        )
    ):
        raise ValueError("resumable run last checkpoint identity is invalid")
    cadence = {1}
    cadence.update(range(checkpoint_interval, step + 1, checkpoint_interval))
    if step == selected_steps:
        cadence.add(step)
    if step not in cadence or document["checkpoint_count"] != len(cadence):
        raise ValueError("resumable run checkpoint trajectory is invalid")
    if metrics is not None and (
        step != selected_steps
        or document.get("last_completed_step") != selected_steps
    ):
        raise ValueError("resumable run metrics evidence is not at the final checkpoint")


def _validate_metrics_checkpoint_snapshot(
    snapshot: _StableFileSnapshot,
    checkpoint_state: CheckpointState,
) -> None:
    """Validate a checkpoint boundary against one immutable metrics payload."""

    if snapshot.payload is None:
        raise RuntimeError("metrics snapshot was not captured with its payload")
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("metrics file is not valid UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != METRICS_FIELDS:
        raise ValueError("metrics header differs from the frozen schema")
    boundary: dict[str, str] | None = None
    for expected_step in range(1, checkpoint_state.completed_step + 1):
        try:
            row = next(reader)
        except StopIteration as error:
            raise ValueError(
                "metrics end before checkpoint step "
                f"{checkpoint_state.completed_step}"
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
        boundary = row
    assert boundary is not None
    MetricsWriter._validate_checkpoint_boundary(boundary, checkpoint_state)


def _snapshot_open_regular_file(
    descriptor: int,
    *,
    path: Path,
    label: str,
) -> _StableFileSnapshot:
    """Hash one retained regular-file descriptor without changing its offset."""

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, byte_count)
        if not chunk:
            break
        digest.update(chunk)
        byte_count += len(chunk)
    after = os.fstat(descriptor)
    if (
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        or byte_count != after.st_size
    ):
        raise RuntimeError(f"{label} changed while its descriptor was read")
    return _StableFileSnapshot(
        path=path,
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest.hexdigest(),
    )


def _read_stable_json(path: str | Path, *, label: str) -> tuple[object, str]:
    snapshot = _snapshot_regular_file(path, label=label, capture_payload=True)
    assert snapshot.payload is not None
    try:
        value = json.loads(snapshot.payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return value, snapshot.sha256


def _copy_file_no_clobber(source: Path, destination: Path) -> None:
    """Copy a trusted private file to a new regular path without following links."""

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor,
            "wb",
            closefd=False,
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, 1024 * 1024)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    finally:
        os.close(descriptor)


def _copy_stable_file_no_clobber(
    source: Path,
    destination: Path,
    *,
    label: str,
    destination_parent_descriptor: int | None = None,
    expected_destination_parent_snapshot: _DirectorySnapshot | None = None,
) -> _StableFileSnapshot:
    """Stream one stable regular inode into a new owner-only file."""

    source = Path(os.path.abspath(source))
    destination = Path(os.path.abspath(destination))
    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    write_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if destination_parent_descriptor is None:
        parent_snapshot = _snapshot_directory(
            destination.parent,
            label=f"private {label} parent",
        )
        bound_parent = os.open(destination.parent, directory_flags)
        revalidate_parent_path = True
    else:
        if expected_destination_parent_snapshot is None:
            raise ValueError("bound stable copy requires a destination-parent snapshot")
        parent_snapshot = expected_destination_parent_snapshot
        bound_parent = os.dup(destination_parent_descriptor)
        revalidate_parent_path = False
    parent_identity = os.fstat(bound_parent)
    _, expected_parent_device, expected_parent_inode = parent_snapshot.components[-1]
    if (parent_identity.st_dev, parent_identity.st_ino) != (
        expected_parent_device,
        expected_parent_inode,
    ):
        os.close(bound_parent)
        raise RuntimeError(f"private {label} parent descriptor changed")
    try:
        source_descriptor = os.open(source, read_flags)
    except OSError as error:
        os.close(bound_parent)
        raise FileNotFoundError(f"{label} is missing or unsafe: {source}") from error
    destination_descriptor: int | None = None
    destination_identity: os.stat_result | None = None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FileNotFoundError(f"{label} is not a regular file: {source}")
        destination_descriptor = os.open(
            destination.name,
            write_flags,
            0o600,
            dir_fd=bound_parent,
        )
        destination_identity = os.fstat(destination_descriptor)
        with os.fdopen(
            destination_descriptor,
            "wb",
            closefd=False,
        ) as destination_handle:
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                destination_handle.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        after = os.fstat(source_descriptor)
        named = os.lstat(source)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        named_identity = (
            named.st_dev,
            named.st_ino,
            named.st_size,
            named.st_mtime_ns,
        )
        if (
            identity_before != identity_after
            or identity_after != named_identity
            or not stat.S_ISREG(named.st_mode)
            or byte_count != after.st_size
        ):
            raise RuntimeError(f"{label} changed while it was copied")
        named_destination = os.stat(
            destination.name,
            dir_fd=bound_parent,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(named_destination.st_mode)
            or (named_destination.st_dev, named_destination.st_ino)
            != (destination_identity.st_dev, destination_identity.st_ino)
            or named_destination.st_size != byte_count
        ):
            raise RuntimeError(f"private {label} changed while it was copied")
        copied_path = _descriptor_path(bound_parent) / destination.name
        copied = _snapshot_regular_file(copied_path, label=f"private {label}")
        if (
            (copied.device, copied.inode)
            != (destination_identity.st_dev, destination_identity.st_ino)
            or copied.sha256 != digest.hexdigest()
            or copied.size != byte_count
        ):
            raise RuntimeError(f"private {label} differs from its captured bytes")
        if revalidate_parent_path:
            _revalidate_directory_snapshot(
                parent_snapshot,
                label=f"private {label} parent",
            )
    except BaseException:
        if destination_identity is not None:
            try:
                remaining = os.stat(
                    destination.name,
                    dir_fd=bound_parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                remaining = None
            if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
                destination_identity.st_dev,
                destination_identity.st_ino,
            ):
                os.unlink(destination.name, dir_fd=bound_parent)
        raise
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)
        os.close(bound_parent)
    return copied


@dataclass(frozen=True)
class _BoundPrivateFile:
    """A private regular file whose inode remains open for materialized reads."""

    descriptor: int

    @property
    def path(self) -> Path:
        return _descriptor_path(self.descriptor)

    @contextmanager
    def open_reader(self) -> Iterator[BinaryIO]:
        duplicate = os.dup(self.descriptor)
        try:
            os.lseek(duplicate, 0, os.SEEK_SET)
            with os.fdopen(duplicate, "rb") as handle:
                duplicate = -1
                yield handle
        finally:
            if duplicate >= 0:
                os.close(duplicate)


@contextmanager
def _bound_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    expected: _StableFileSnapshot,
    label: str,
) -> Iterator[_BoundPrivateFile]:
    """Retain and repeatedly validate one exact child of a bound directory."""

    if Path(name).name != name:
        raise ValueError(f"{label} name must be a direct child")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    try:
        opened = _snapshot_open_regular_file(
            descriptor,
            path=_descriptor_path(descriptor),
            label=label,
        )
        named = _snapshot_regular_file_at(
            parent_descriptor,
            name,
            label=label,
        )
        if not (
            _same_bound_file_snapshot(opened, expected)
            and _same_bound_file_snapshot(named, expected)
        ):
            raise RuntimeError(f"{label} differs from its captured bytes")
        yield _BoundPrivateFile(descriptor)
        opened_after = _snapshot_open_regular_file(
            descriptor,
            path=_descriptor_path(descriptor),
            label=label,
        )
        try:
            named_after = _snapshot_regular_file_at(
                parent_descriptor,
                name,
                label=label,
            )
        except (FileNotFoundError, OSError) as error:
            raise RuntimeError(f"{label} changed while it was in use") from error
        if not (
            _same_bound_file_snapshot(opened_after, expected)
            and _same_bound_file_snapshot(named_after, expected)
        ):
            raise RuntimeError(f"{label} changed while it was in use")
    finally:
        os.close(descriptor)


@contextmanager
def _private_stable_file(
    path: str | Path,
    *,
    label: str,
    source_parent_descriptor: int | None = None,
) -> Iterator[tuple[_BoundPrivateFile, str]]:
    """Copy stable bytes and retain the exact private inode across its load."""

    path = Path(path)
    if source_parent_descriptor is None:
        source = _snapshot_regular_file(path, label=label, capture_payload=True)
    else:
        if path.name != str(path):
            raise ValueError(f"{label} must be a direct bound-directory child")
        source = _snapshot_regular_file_at(
            source_parent_descriptor,
            path.name,
            label=label,
            capture_payload=True,
        )
    assert source.payload is not None
    temporary_parent = Path(os.path.realpath(tempfile.gettempdir()))
    parent_snapshot = _snapshot_directory(
        temporary_parent,
        label="private stable-file temporary parent",
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(temporary_parent, directory_flags)
    parent_identity = os.fstat(parent_descriptor)
    _, parent_device, parent_inode = parent_snapshot.components[-1]
    if (parent_identity.st_dev, parent_identity.st_ino) != (
        parent_device,
        parent_inode,
    ):
        os.close(parent_descriptor)
        raise RuntimeError("private stable-file temporary parent changed")
    descriptor, temporary_name = _create_staged_file_at(
        parent_descriptor,
        prefix=".smolvla-private-",
        suffix=source.path.suffix,
    )
    temporary_identity = os.fstat(descriptor)
    temporary = _descriptor_path(parent_descriptor) / temporary_name
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            handle.write(source.payload)
            handle.flush()
            os.fsync(handle.fileno())
        expected_private = _snapshot_open_regular_file(
            descriptor,
            path=temporary,
            label=f"private {label}",
        )
        named_private = _snapshot_regular_file_at(
            parent_descriptor,
            temporary_name,
            label=f"private {label}",
        )
        if (
            expected_private.sha256 != source.sha256
            or expected_private.size != source.size
            or not _same_bound_file_snapshot(expected_private, named_private)
        ):
            raise RuntimeError(f"private {label} differs from its captured bytes")
        yield _BoundPrivateFile(descriptor), source.sha256
        opened_after = _snapshot_open_regular_file(
            descriptor,
            path=_descriptor_path(descriptor),
            label=f"private {label}",
        )
        try:
            named_after = _snapshot_regular_file_at(
                parent_descriptor,
                temporary_name,
                label=f"private {label}",
            )
        except (FileNotFoundError, OSError) as error:
            raise RuntimeError(
                f"private {label} changed while it was in use"
            ) from error
        if not (
            _same_bound_file_snapshot(opened_after, expected_private)
            and _same_bound_file_snapshot(named_after, expected_private)
        ):
            raise RuntimeError(f"private {label} changed while it was in use")
        current = (
            _snapshot_regular_file(source.path, label=label)
            if source_parent_descriptor is None
            else _snapshot_regular_file_at(
                source_parent_descriptor,
                path.name,
                label=label,
            )
        )
        if not (
            _same_file_snapshot(source, current)
            if source_parent_descriptor is None
            else _same_bound_file_snapshot(source, current)
        ):
            raise RuntimeError(f"{label} changed while its private snapshot was used")
    finally:
        try:
            remaining = os.stat(
                temporary_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            remaining = None
        if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
            temporary_identity.st_dev,
            temporary_identity.st_ino,
        ):
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        os.close(descriptor)
        os.close(parent_descriptor)


@contextmanager
def _working_directory_at(descriptor: int) -> Iterator[Path]:
    """Temporarily make a retained directory fd the process-relative path root."""

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    previous = os.open(".", directory_flags)
    expected = os.fstat(descriptor)
    try:
        os.fchdir(descriptor)
        current = os.stat(".", follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("bound working directory changed during entry")
        yield Path(".")
        current = os.stat(".", follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("bound working directory changed while in use")
    finally:
        os.fchdir(previous)
        os.close(previous)


@dataclass(frozen=True)
class _BoundDirectoryLease:
    descriptor: int

    @property
    def path(self) -> Path:
        return _descriptor_path(self.descriptor)

    @property
    def name(self) -> str:
        return self.path.name

    def with_name(self, name: str) -> Path:
        return self.path.with_name(name)

    def __truediv__(self, child: str | Path) -> Path:
        return self.path / child

    def __fspath__(self) -> str:
        return os.fspath(self.path)


@contextmanager
def _bound_temporary_directory(
    parent: str | Path,
    *,
    prefix: str,
    label: str,
) -> Iterator[_BoundDirectoryLease]:
    """Create and clean one temporary directory without deleting a replacement."""

    parent_snapshot = _snapshot_directory(parent, label=f"{label} parent")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(parent_snapshot.path, directory_flags)
    staged_descriptor: int | None = None
    staged_name: str | None = None
    staged_device: int | None = None
    staged_inode: int | None = None
    try:
        opened_parent = os.fstat(parent_descriptor)
        _, parent_device, parent_inode = parent_snapshot.components[-1]
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_device,
            parent_inode,
        ):
            raise RuntimeError(f"{label} parent changed before creation")
        staged_descriptor, staged_name, staged_path = _create_staged_directory_at(
            parent_descriptor,
            prefix=prefix,
        )
        staged_identity = os.fstat(staged_descriptor)
        staged_device = staged_identity.st_dev
        staged_inode = staged_identity.st_ino
        yield _BoundDirectoryLease(staged_descriptor)
        named = os.stat(
            staged_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named.st_mode)
            or (named.st_dev, named.st_ino) != (staged_device, staged_inode)
        ):
            raise RuntimeError(f"{label} changed while it was in use")
        _revalidate_directory_snapshot(parent_snapshot, label=f"{label} parent")
    finally:
        if staged_descriptor is not None:
            os.close(staged_descriptor)
        if staged_name is not None and staged_device is not None and staged_inode is not None:
            try:
                remaining = os.stat(
                    staged_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                remaining = None
            if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
                staged_device,
                staged_inode,
            ):
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise RuntimeError("safe temporary cleanup requires fd-based rmtree")
                shutil.rmtree(staged_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
        os.close(parent_descriptor)


def _snapshot_tree_evidence(
    *,
    root: str | Path,
    paths: Mapping[str, Path],
    label: str,
    allowed_symlink_root: str | Path | None = None,
) -> dict[str, object]:
    """Hash an explicit logical file tree with stable files and bound symlinks."""

    root_snapshot = _snapshot_directory(root, label=f"{label} root")
    root_path = root_snapshot.path
    allowed_snapshot = (
        None
        if allowed_symlink_root is None
        else _snapshot_directory(
            allowed_symlink_root,
            label=f"{label} allowed symlink root",
        )
    )
    parent_snapshots: dict[Path, _DirectorySnapshot] = {}

    def bind_parent(parent: Path) -> None:
        absolute = Path(os.path.abspath(parent))
        if absolute not in parent_snapshots:
            parent_snapshots[absolute] = _snapshot_directory(
                absolute,
                label=f"{label} file parent",
            )

    files: dict[str, str] = {}
    links: dict[str, dict[str, str]] = {}
    for logical_name, raw_path in sorted(paths.items()):
        logical = Path(logical_name)
        if logical.is_absolute() or ".." in logical.parts or logical.as_posix() != logical_name:
            raise ValueError(f"{label} has an unsafe logical path: {logical_name}")
        path = Path(os.path.abspath(raw_path))
        if not path.is_relative_to(root_path):
            raise ValueError(f"{label} input escapes its root: {logical_name}")
        bind_parent(path.parent)
        if path.is_symlink():
            if allowed_snapshot is None:
                raise ValueError(f"{label} contains a symlink: {logical_name}")
            before = os.lstat(path)
            target = os.readlink(path)
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise FileNotFoundError(
                    f"{label} contains a broken symlink: {logical_name}"
                ) from error
            if not resolved.is_relative_to(allowed_snapshot.path):
                raise ValueError(f"{label} symlink escapes its allowed root: {logical_name}")
            bind_parent(resolved.parent)
            snapshot = _snapshot_regular_file(
                resolved,
                label=f"{label} file {logical_name}",
            )
            after = os.lstat(path)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or not stat.S_ISLNK(after.st_mode)
                or os.readlink(path) != target
                or path.resolve(strict=True) != resolved
            ):
                raise RuntimeError(f"{label} symlink changed while it was read")
            links[logical_name] = {
                "target": target,
                "resolved_path": str(resolved),
            }
        else:
            snapshot = _snapshot_regular_file(
                path,
                label=f"{label} file {logical_name}",
            )
        files[logical_name] = snapshot.sha256
    for snapshot in parent_snapshots.values():
        _revalidate_directory_snapshot(snapshot, label=f"{label} file parent")
    _revalidate_directory_snapshot(root_snapshot, label=f"{label} root")
    if allowed_snapshot is not None:
        _revalidate_directory_snapshot(
            allowed_snapshot,
            label=f"{label} allowed symlink root",
        )
    canonical = {
        "files": dict(sorted(files.items())),
        "links": {name: dict(record) for name, record in sorted(links.items())},
    }
    tree_sha256 = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"tree_sha256": tree_sha256, **canonical}


def _hf_revision_tree_path(snapshot_root: Path, revision: str) -> Path:
    snapshot_root = Path(os.path.abspath(snapshot_root))
    if snapshot_root.name != revision or snapshot_root.parent.name != "snapshots":
        raise ValueError(f"Hugging Face snapshot path is not pinned to {revision}")
    return snapshot_root.parent.parent / "trees" / f"{revision}.json"


def _validate_tree_files_against_revision(
    *,
    evidence: Mapping[str, object],
    paths: Mapping[str, Path],
    revision_tree_path: Path,
    expected_revision_sha256: str,
    label: str,
) -> str:
    """Prove selected local bytes match one independently frozen Hub tree."""

    revision, revision_sha256 = _read_stable_json(
        revision_tree_path,
        label=f"{label} revision tree",
    )
    records = revision.get("files") if isinstance(revision, Mapping) else None
    evidence_files = evidence.get("files") if isinstance(evidence, Mapping) else None
    if (
        revision_sha256 != expected_revision_sha256
        or not isinstance(revision, Mapping)
        or revision.get("format_version") != 1
        or not isinstance(records, Mapping)
        or not isinstance(evidence_files, Mapping)
        or set(paths) != set(evidence_files) - {
            f"revision/{DATASET_REVISION}.json"
        }
    ):
        raise ValueError(f"{label} revision evidence is invalid")
    for logical_name, raw_path in sorted(paths.items()):
        record = records.get(logical_name)
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} revision omits {logical_name}")
        expected_size = record.get("size")
        lfs_sha256 = record.get("lfs_sha256")
        blob_id = record.get("blob_id")
        capture_payload = not isinstance(lfs_sha256, str)
        path = raw_path.resolve(strict=True) if raw_path.is_symlink() else raw_path
        snapshot = _snapshot_regular_file(
            path,
            label=f"{label} revision file {logical_name}",
            capture_payload=capture_payload,
        )
        if snapshot.size != expected_size or snapshot.sha256 != evidence_files[logical_name]:
            raise ValueError(f"{label} file differs from its captured tree: {logical_name}")
        if isinstance(lfs_sha256, str):
            valid_revision_digest = snapshot.sha256 == lfs_sha256
        elif isinstance(blob_id, str) and snapshot.payload is not None:
            header = f"blob {snapshot.size}\0".encode("ascii")
            valid_revision_digest = (
                hashlib.sha1(header + snapshot.payload).hexdigest() == blob_id
            )
        else:
            valid_revision_digest = False
        if not valid_revision_digest:
            raise ValueError(f"{label} file differs from revision: {logical_name}")
    return revision_sha256


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _reject_t3b_output_input_overlap(
    output_dir: str | Path,
    inputs: Mapping[str, str | Path],
) -> None:
    output = Path(os.path.realpath(Path(output_dir).expanduser()))
    for label, raw_path in inputs.items():
        path = Path(os.path.realpath(Path(raw_path).expanduser()))
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(f"T3B output overlaps {label}: {path}")


def _safe_t3b_output_path(
    path: str | Path,
    *,
    must_exist: bool,
    label: str,
) -> Path:
    """Return a lexical absolute output only after rejecting symlinked ancestry."""

    absolute = Path(os.path.abspath(Path(path).expanduser()))
    if must_exist:
        return _snapshot_directory(absolute, label=label).path
    _snapshot_directory(absolute.parent, label=f"{label} parent")
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"{label} already exists: {absolute}")
    return absolute


def _validate_t3b_tree_evidence(
    value: object,
    *,
    label: str,
    expected_files: frozenset[str],
) -> dict[str, object]:
    required = {"tree_sha256", "files", "links"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} evidence fields differ from the frozen schema")
    files = value["files"]
    links = value["links"]
    if (
        not isinstance(files, Mapping)
        or set(files) != expected_files
        or any(not _is_sha256(digest) for digest in files.values())
        or not isinstance(links, Mapping)
        or not set(links).issubset(expected_files)
        or any(
            not isinstance(record, Mapping)
            or set(record) != {"target", "resolved_path"}
            or not isinstance(record["target"], str)
            or not record["target"]
            or not isinstance(record["resolved_path"], str)
            or not record["resolved_path"]
            for record in links.values()
        )
    ):
        raise ValueError(f"{label} file evidence is invalid")
    canonical = {
        "files": dict(sorted(files.items())),
        "links": {
            name: dict(record) for name, record in sorted(links.items())
        },
    }
    computed = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if value["tree_sha256"] != computed:
        raise ValueError(f"{label} tree digest differs from its files")
    return {"tree_sha256": computed, **canonical}


def validate_t3b_frozen_input_evidence(value: object) -> dict[str, object]:
    """Validate the complete physical input commitment for T3B training."""

    required = {
        "format_version",
        "revision_trees",
        "train_statistics_sha256",
        "processor_statistics_sha256",
        "source_checkpoint",
        "native_checkpoint",
        "native_conversion",
        "pinned_dataset",
        "tokenizer_snapshot",
        "native_tokenizer_snapshot",
        "evaluation_artifact",
        "base_report",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("T3B frozen input fields differ from the frozen schema")
    train_statistics_sha256 = _validate_t3b_train_statistics_sha256(
        value["train_statistics_sha256"]
    )
    if value["format_version"] != 1 or not _is_sha256(
        value["processor_statistics_sha256"]
    ):
        raise ValueError("T3B statistics evidence is invalid")
    revision_trees = value["revision_trees"]
    if not isinstance(revision_trees, Mapping) or dict(revision_trees) != {
        "checkpoint_sha256": FROZEN_CHECKPOINT_REVISION_TREE_SHA256,
        "dataset_sha256": FROZEN_DATASET_REVISION_TREE_SHA256,
        "tokenizer_sha256": FROZEN_TOKENIZER_REVISION_TREE_SHA256,
    }:
        raise ValueError("T3B revision-tree evidence is invalid")
    source_checkpoint = _validate_t3b_tree_evidence(
        value["source_checkpoint"],
        label="source checkpoint",
        expected_files=_T3B_CHECKPOINT_FILES,
    )
    native_checkpoint = _validate_t3b_tree_evidence(
        value["native_checkpoint"],
        label="native checkpoint",
        expected_files=_T3B_CHECKPOINT_FILES,
    )
    pinned_dataset = _validate_t3b_tree_evidence(
        value["pinned_dataset"],
        label="pinned dataset",
        expected_files=_T3B_DATASET_FILES,
    )
    tokenizer_snapshot = _validate_t3b_tree_evidence(
        value["tokenizer_snapshot"],
        label="tokenizer snapshot",
        expected_files=_T3B_TOKENIZER_FILES,
    )
    native_tokenizer_snapshot = _validate_t3b_tree_evidence(
        value["native_tokenizer_snapshot"],
        label="native tokenizer snapshot",
        expected_files=_T3B_NATIVE_TOKENIZER_FILES,
    )
    evaluation_artifact = _validate_t3b_tree_evidence(
        value["evaluation_artifact"],
        label="evaluation artifact",
        expected_files=_T3B_EVALUATION_FILES,
    )
    revision_name = f"revision/{DATASET_REVISION}.json"
    if (
        source_checkpoint["files"] != native_checkpoint["files"]
        or any(
            tokenizer_snapshot["files"][name]
            != native_tokenizer_snapshot["files"][name]
            for name in _T3B_NATIVE_TOKENIZER_FILES
        )
        or pinned_dataset["files"][revision_name]
        != FROZEN_DATASET_REVISION_TREE_SHA256
        or evaluation_artifact["files"]["manifest.json"]
        != FROZEN_EVALUATION_MANIFEST_SHA256
        or evaluation_artifact["files"]["metadata.json"]
        != FROZEN_EVALUATION_METADATA_SHA256
    ):
        raise ValueError("T3B dataset/evaluation inputs differ from the frozen evidence")
    base_report = value["base_report"]
    if (
        not isinstance(base_report, Mapping)
        or set(base_report) != {"file", "sha256"}
        or base_report["file"] != "t3-base-evaluation.json"
        or base_report["sha256"] != FROZEN_BASE_REPORT_SHA256
    ):
        raise ValueError("T3B base evaluation report differs from the frozen evidence")
    conversion = value["native_conversion"]
    conversion_fields = {
        "model_file",
        "model_sha256",
        "name_map_file",
        "name_map_sha256",
        "source_model_sha256",
        "tensor_count",
        "parameter_count",
        "dtype",
    }
    if (
        not isinstance(conversion, Mapping)
        or set(conversion) != conversion_fields
        or conversion["model_file"] != "model.bfloat16.safetensors"
        or conversion["name_map_file"] != "name_map.json"
        or not _is_sha256(conversion["model_sha256"])
        or not _is_sha256(conversion["name_map_sha256"])
        or conversion["source_model_sha256"]
        != source_checkpoint["files"]["model.safetensors"]
        or conversion["source_model_sha256"]
        != native_checkpoint["files"]["model.safetensors"]
        or conversion["tensor_count"] != 500
        or conversion["parameter_count"] != 450_046_176
        or conversion["dtype"] != "bfloat16"
    ):
        raise ValueError("T3B native conversion evidence is invalid")
    return {
        "format_version": 1,
        "revision_trees": dict(revision_trees),
        "train_statistics_sha256": train_statistics_sha256,
        "processor_statistics_sha256": value["processor_statistics_sha256"],
        "source_checkpoint": source_checkpoint,
        "native_checkpoint": native_checkpoint,
        "native_conversion": dict(conversion),
        "pinned_dataset": pinned_dataset,
        "tokenizer_snapshot": tokenizer_snapshot,
        "native_tokenizer_snapshot": native_tokenizer_snapshot,
        "evaluation_artifact": evaluation_artifact,
        "base_report": dict(base_report),
    }


def _checkpoint_input_paths(root: Path) -> dict[str, Path]:
    return {name: root / name for name in sorted(_T3B_CHECKPOINT_FILES)}


def _tokenizer_input_paths(
    root: Path,
    *,
    names: frozenset[str] = _T3B_TOKENIZER_FILES,
) -> dict[str, Path]:
    return {name: root / name for name in sorted(names)}


def _dataset_input_paths(root: Path) -> dict[str, Path]:
    result = {
        name: root / name
        for name in sorted(_T3B_DATASET_FILES)
        if not name.startswith("revision/")
    }
    revision_name = f"revision/{DATASET_REVISION}.json"
    result[revision_name] = (
        root
        / ".cache"
        / "huggingface"
        / "trees"
        / f"{DATASET_REVISION}.json"
    )
    return result


def _validate_t3b_dataset_inventory(root: str | Path) -> tuple[str, ...]:
    """Require the complete data/meta/video population frozen by the revision."""

    root_snapshot = _snapshot_directory(root, label="T3B pinned dataset root")
    expected = set(_T3B_DATASET_FILES) - {f"revision/{DATASET_REVISION}.json"}
    actual: set[str] = set()
    for top_name in ("data", "meta", "videos"):
        top = root_snapshot.path / top_name
        top_snapshot = _snapshot_directory(
            top,
            label=f"T3B pinned dataset {top_name} inventory",
        )
        for directory, directory_names, file_names in os.walk(
            top,
            followlinks=False,
        ):
            directory_path = Path(directory)
            for name in directory_names:
                candidate = directory_path / name
                value = os.lstat(candidate)
                if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
                    relative = candidate.relative_to(root_snapshot.path).as_posix()
                    raise ValueError(
                        f"T3B dataset inventory has an unsafe directory: {relative}"
                    )
            for name in file_names:
                candidate = directory_path / name
                relative = candidate.relative_to(root_snapshot.path).as_posix()
                value = os.lstat(candidate)
                if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
                    raise ValueError(
                        f"T3B dataset inventory has an unsafe file: {relative}"
                    )
                actual.add(relative)
        _revalidate_directory_snapshot(
            top_snapshot,
            label=f"T3B pinned dataset {top_name} inventory",
        )
    _revalidate_directory_snapshot(
        root_snapshot,
        label="T3B pinned dataset root",
    )
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"T3B dataset inventory differs; missing={missing}, extra={extra}"
        )
    return tuple(sorted(actual))


def _evaluation_input_paths(root: Path) -> dict[str, Path]:
    return {name: root / name for name in sorted(_T3B_EVALUATION_FILES)}


def _validate_t3b_conversion_from_stable_hardlinks(
    *,
    source_model_path: Path,
    converted_model_path: Path,
    name_map_path: Path,
) -> dict[str, object]:
    """Validate BF16 conversion semantics on hard links to captured inodes."""

    from mlx_smolvla.convert import validate_converted_checkpoint

    source_path = source_model_path.resolve(strict=True)
    source_before = _snapshot_regular_file(
        source_path,
        label="T3B native source model",
    )
    converted_before = _snapshot_regular_file(
        converted_model_path,
        label="T3B converted model",
    )
    name_map_before = _snapshot_regular_file(
        name_map_path,
        label="T3B conversion name map",
    )
    with _bound_temporary_directory(
        converted_model_path.parent,
        prefix=".t3b-conversion-validation-",
        label="T3B conversion validation directory",
    ) as temporary_root:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        os.mkdir("source", mode=0o700, dir_fd=temporary_root.descriptor)
        os.mkdir("converted", mode=0o700, dir_fd=temporary_root.descriptor)
        source_descriptor = os.open(
            "source",
            directory_flags,
            dir_fd=temporary_root.descriptor,
        )
        converted_descriptor = os.open(
            "converted",
            directory_flags,
            dir_fd=temporary_root.descriptor,
        )
        try:
            source_dir = _descriptor_path(source_descriptor)
            converted_dir = _descriptor_path(converted_descriptor)
            source_snapshot = _snapshot_directory(
                source_dir,
                label="private conversion source directory",
            )
            converted_snapshot = _snapshot_directory(
                converted_dir,
                label="private conversion output directory",
            )
            private_source = source_dir / "model.safetensors"
            private_converted = converted_dir / "model.bfloat16.safetensors"
            private_name_map = converted_dir / "name_map.json"
            _copy_stable_file_no_clobber(
                source_path,
                private_source,
                label="T3B native source model",
                destination_parent_descriptor=source_descriptor,
                expected_destination_parent_snapshot=source_snapshot,
            )
            _copy_stable_file_no_clobber(
                converted_model_path,
                private_converted,
                label="T3B converted model",
                destination_parent_descriptor=converted_descriptor,
                expected_destination_parent_snapshot=converted_snapshot,
            )
            _copy_stable_file_no_clobber(
                name_map_path,
                private_name_map,
                label="T3B conversion name map",
                destination_parent_descriptor=converted_descriptor,
                expected_destination_parent_snapshot=converted_snapshot,
            )
            report = validate_converted_checkpoint(
                _descriptor_path(source_descriptor),
                _descriptor_path(converted_descriptor)
                / "model.bfloat16.safetensors",
                _descriptor_path(converted_descriptor) / "name_map.json",
                dtype="bfloat16",
                expected_tensor_count=500,
            )
        finally:
            os.close(converted_descriptor)
            os.close(source_descriptor)
    for expected, path, label in (
        (source_before, source_path, "T3B native source model"),
        (converted_before, converted_model_path, "T3B converted model"),
        (name_map_before, name_map_path, "T3B conversion name map"),
    ):
        current = _snapshot_regular_file(path, label=label)
        if not _same_file_snapshot(expected, current):
            raise RuntimeError(f"{label} changed during semantic validation")
    if (
        report.tensor_count != 500
        or report.parameter_count != 450_046_176
        or report.dtype != "bfloat16"
        or report.source_model_sha256 != source_before.sha256
        or report.converted_model_sha256 != converted_before.sha256
        or report.name_map_sha256 != name_map_before.sha256
    ):
        raise RuntimeError("T3B conversion validation report changed")
    return {
        "model_file": converted_model_path.name,
        "model_sha256": converted_before.sha256,
        "name_map_file": name_map_path.name,
        "name_map_sha256": name_map_before.sha256,
        "source_model_sha256": source_before.sha256,
        "tensor_count": report.tensor_count,
        "parameter_count": report.parameter_count,
        "dtype": report.dtype,
    }


def _validate_runtime_model_matches_converted_checkpoint(
    model: nn.Module,
    converted_model_path: Path,
) -> None:
    """Prove the live pre-update base tree equals the committed BF16 bytes."""

    live: dict[str, mx.array] = {}
    for name, value in tree_flatten(model.parameters()):
        if name.endswith((".lora_a", ".lora_b")):
            continue
        canonical = canonical_parameter_name(name.replace(".base.", "."))
        if canonical in live:
            raise RuntimeError(f"duplicate live base parameter: {canonical}")
        live[canonical] = value
    before = _snapshot_regular_file(
        converted_model_path,
        label="T3B converted model",
    )
    with _bound_temporary_directory(
        converted_model_path.parent,
        prefix=".t3b-runtime-model-validation-",
        label="T3B runtime model validation directory",
    ) as temporary_root:
        private_model = temporary_root / converted_model_path.name
        temporary_snapshot = _snapshot_directory(
            temporary_root,
            label="T3B private runtime-model validation directory",
        )
        private_snapshot = _copy_stable_file_no_clobber(
            converted_model_path,
            private_model,
            label="T3B converted model",
            destination_parent_descriptor=temporary_root.descriptor,
            expected_destination_parent_snapshot=temporary_snapshot,
        )
        with _bound_regular_file_at(
            temporary_root.descriptor,
            converted_model_path.name,
            expected=private_snapshot,
            label="private T3B converted model",
        ) as private_file:
            with private_file.open_reader() as handle:
                converted = mx.load(handle, format="safetensors")
                mx.eval(converted)
            if set(live) != set(converted):
                raise RuntimeError(
                    "live model tensor inventory differs from the converted checkpoint"
                )
            for name in sorted(live):
                if (
                    live[name].shape != converted[name].shape
                    or live[name].dtype != converted[name].dtype
                    or not bool(mx.array_equal(live[name], converted[name]))
                ):
                    raise RuntimeError(
                        f"live model tensor differs from converted checkpoint: {name}"
                    )
    after = _snapshot_regular_file(
        converted_model_path,
        label="T3B converted model",
    )
    if not _same_file_snapshot(before, after):
        raise RuntimeError("T3B converted model changed during runtime validation")


def _validate_t3b_revision_trees(
    *,
    source_checkpoint_root: Path,
    source_checkpoint: Mapping[str, object],
    native_checkpoint_root: Path,
    native_checkpoint: Mapping[str, object],
    dataset_root: Path,
    pinned_dataset: Mapping[str, object],
    tokenizer_root: Path,
    tokenizer_snapshot: Mapping[str, object],
    native_tokenizer_root: Path,
    native_tokenizer_snapshot: Mapping[str, object],
) -> dict[str, str]:
    checkpoint_revision_path = _hf_revision_tree_path(
        source_checkpoint_root,
        CHECKPOINT_REVISION,
    )
    native_checkpoint_revision_path = _hf_revision_tree_path(
        native_checkpoint_root,
        CHECKPOINT_REVISION,
    )
    tokenizer_revision_path = _hf_revision_tree_path(
        tokenizer_root,
        BASE_VLM_REVISION,
    )
    native_tokenizer_revision_path = _hf_revision_tree_path(
        native_tokenizer_root,
        BASE_VLM_REVISION,
    )
    dataset_paths = _dataset_input_paths(dataset_root)
    dataset_revision_name = f"revision/{DATASET_REVISION}.json"
    dataset_revision_path = dataset_paths.pop(dataset_revision_name)
    checkpoint_sha256 = _validate_tree_files_against_revision(
        evidence=source_checkpoint,
        paths=_checkpoint_input_paths(source_checkpoint_root),
        revision_tree_path=checkpoint_revision_path,
        expected_revision_sha256=FROZEN_CHECKPOINT_REVISION_TREE_SHA256,
        label="T3B source checkpoint",
    )
    native_checkpoint_sha256 = _validate_tree_files_against_revision(
        evidence=native_checkpoint,
        paths=_checkpoint_input_paths(native_checkpoint_root),
        revision_tree_path=native_checkpoint_revision_path,
        expected_revision_sha256=FROZEN_CHECKPOINT_REVISION_TREE_SHA256,
        label="T3B native checkpoint",
    )
    dataset_sha256 = _validate_tree_files_against_revision(
        evidence=pinned_dataset,
        paths=dataset_paths,
        revision_tree_path=dataset_revision_path,
        expected_revision_sha256=FROZEN_DATASET_REVISION_TREE_SHA256,
        label="T3B pinned dataset",
    )
    tokenizer_sha256 = _validate_tree_files_against_revision(
        evidence=tokenizer_snapshot,
        paths=_tokenizer_input_paths(tokenizer_root),
        revision_tree_path=tokenizer_revision_path,
        expected_revision_sha256=FROZEN_TOKENIZER_REVISION_TREE_SHA256,
        label="T3B source tokenizer",
    )
    native_tokenizer_sha256 = _validate_tree_files_against_revision(
        evidence=native_tokenizer_snapshot,
        paths=_tokenizer_input_paths(
            native_tokenizer_root,
            names=_T3B_NATIVE_TOKENIZER_FILES,
        ),
        revision_tree_path=native_tokenizer_revision_path,
        expected_revision_sha256=FROZEN_TOKENIZER_REVISION_TREE_SHA256,
        label="T3B native tokenizer",
    )
    if (
        native_checkpoint_sha256 != checkpoint_sha256
        or native_tokenizer_sha256 != tokenizer_sha256
    ):
        raise RuntimeError("T3B cache copies use different revision trees")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "tokenizer_sha256": tokenizer_sha256,
    }


@dataclass(frozen=True)
class _T3BInputPaths:
    source_checkpoint_root: Path
    native_checkpoint_root: Path
    dataset_root: Path
    tokenizer_root: Path
    native_tokenizer_root: Path
    evaluation_root: Path
    base_report_path: Path
    converted_model_path: Path
    name_map_path: Path


@dataclass(frozen=True)
class _T3BPhysicalInputCapture:
    revision_trees: dict[str, str]
    source_checkpoint: dict[str, object]
    native_checkpoint: dict[str, object]
    pinned_dataset: dict[str, object]
    dataset_inventory: tuple[str, ...]
    tokenizer_snapshot: dict[str, object]
    native_tokenizer_snapshot: dict[str, object]
    evaluation_artifact: dict[str, object]
    base_report: _StableFileSnapshot
    converted_model: _StableFileSnapshot
    name_map: _StableFileSnapshot


def _resolve_t3b_input_paths(
    config: FineTuneConfig,
    model: SmolVLATrainingModel | None,
) -> _T3BInputPaths:
    from huggingface_hub import snapshot_download
    from training.evaluation import _require_pinned_dataset_root
    from training.reference_export import resolve_tokenizer_snapshot

    cache_snapshot = _snapshot_directory(
        Path(os.path.abspath(config.cache_dir)),
        label="T3B configured Hugging Face cache",
    )
    native_cache_snapshot = _snapshot_directory(
        Path(os.path.abspath(config.native_cache)),
        label="T3B configured native cache",
    )
    cache_dir = cache_snapshot.path
    native_cache = native_cache_snapshot.path
    source_checkpoint_root = resolve_base_checkpoint(cache_dir)
    native_checkpoint_root = resolve_base_checkpoint(native_cache / "hf")
    tokenizer_root = resolve_tokenizer_snapshot(cache_dir)
    native_tokenizer_root = Path(
        snapshot_download(
            BASE_VLM_ID,
            revision=BASE_VLM_REVISION,
            cache_dir=native_cache / "hf",
            allow_patterns=sorted(_T3B_NATIVE_TOKENIZER_FILES),
            local_files_only=True,
        )
    )
    dataset_root = _require_pinned_dataset_root(cache_dir)
    output_parent = Path(os.path.abspath(config.output_dir)).parent
    evaluation_root = output_parent / "t3-evaluation"
    base_report_path = output_parent / "t3-base-evaluation.json"
    conversion_identity = hashlib.sha256(
        str(native_checkpoint_root).encode("utf-8")
    ).hexdigest()[:16]
    expected_converted_model_path = (
        native_cache
        / "converted"
        / conversion_identity
        / "bfloat16"
        / "model.bfloat16.safetensors"
    )
    if model is None:
        converted_model_path = expected_converted_model_path
    else:
        converted_model_path = model.converted_weights_path
        if converted_model_path is None:
            raise FileNotFoundError("T3B training model has no converted weight path")
        converted_model_path = Path(os.path.abspath(converted_model_path))
        if converted_model_path != expected_converted_model_path:
            raise RuntimeError(
                "T3B training model was not loaded from the frozen converted path"
            )
    name_map_path = converted_model_path.parent / "name_map.json"
    paths = _T3BInputPaths(
        source_checkpoint_root=source_checkpoint_root,
        native_checkpoint_root=native_checkpoint_root,
        dataset_root=dataset_root,
        tokenizer_root=tokenizer_root,
        native_tokenizer_root=native_tokenizer_root,
        evaluation_root=evaluation_root,
        base_report_path=base_report_path,
        converted_model_path=converted_model_path,
        name_map_path=name_map_path,
    )
    _reject_t3b_output_input_overlap(
        config.output_dir,
        {
            "source checkpoint": paths.source_checkpoint_root,
            "native checkpoint": paths.native_checkpoint_root,
            "source tokenizer": paths.tokenizer_root,
            "native tokenizer": paths.native_tokenizer_root,
            "pinned dataset": paths.dataset_root,
            "held-out evaluation": paths.evaluation_root,
            "base evaluation report": paths.base_report_path,
            "converted model": paths.converted_model_path,
            "conversion name map": paths.name_map_path,
        },
    )
    return paths


def _capture_t3b_physical_inputs(
    config: FineTuneConfig,
    paths: _T3BInputPaths,
) -> _T3BPhysicalInputCapture:
    cache_dir = Path(config.cache_dir).resolve()
    native_cache = Path(config.native_cache).resolve()
    conversion_parent = _snapshot_directory(
        paths.converted_model_path.parent,
        label="T3B conversion directory",
    )
    report_parent = _snapshot_directory(
        paths.base_report_path.parent,
        label="T3B base report directory",
    )
    source_checkpoint = _snapshot_tree_evidence(
        root=paths.source_checkpoint_root,
        paths=_checkpoint_input_paths(paths.source_checkpoint_root),
        label="T3B source checkpoint",
        allowed_symlink_root=cache_dir,
    )
    native_checkpoint = _snapshot_tree_evidence(
        root=paths.native_checkpoint_root,
        paths=_checkpoint_input_paths(paths.native_checkpoint_root),
        label="T3B native checkpoint",
        allowed_symlink_root=native_cache / "hf",
    )
    dataset_inventory = _validate_t3b_dataset_inventory(paths.dataset_root)
    pinned_dataset = _snapshot_tree_evidence(
        root=paths.dataset_root,
        paths=_dataset_input_paths(paths.dataset_root),
        label="T3B pinned dataset",
    )
    tokenizer_snapshot = _snapshot_tree_evidence(
        root=paths.tokenizer_root,
        paths=_tokenizer_input_paths(paths.tokenizer_root),
        label="T3B source tokenizer",
        allowed_symlink_root=cache_dir,
    )
    native_tokenizer_snapshot = _snapshot_tree_evidence(
        root=paths.native_tokenizer_root,
        paths=_tokenizer_input_paths(
            paths.native_tokenizer_root,
            names=_T3B_NATIVE_TOKENIZER_FILES,
        ),
        label="T3B native tokenizer",
        allowed_symlink_root=native_cache / "hf",
    )
    evaluation_artifact = _snapshot_tree_evidence(
        root=paths.evaluation_root,
        paths=_evaluation_input_paths(paths.evaluation_root),
        label="T3B held-out evaluation",
    )
    base_report = _snapshot_regular_file(
        paths.base_report_path,
        label="T3B base evaluation report",
    )
    converted_model = _snapshot_regular_file(
        paths.converted_model_path,
        label="T3B converted model",
    )
    name_map = _snapshot_regular_file(
        paths.name_map_path,
        label="T3B conversion name map",
    )
    revision_trees = _validate_t3b_revision_trees(
        source_checkpoint_root=paths.source_checkpoint_root,
        source_checkpoint=source_checkpoint,
        native_checkpoint_root=paths.native_checkpoint_root,
        native_checkpoint=native_checkpoint,
        dataset_root=paths.dataset_root,
        pinned_dataset=pinned_dataset,
        tokenizer_root=paths.tokenizer_root,
        tokenizer_snapshot=tokenizer_snapshot,
        native_tokenizer_root=paths.native_tokenizer_root,
        native_tokenizer_snapshot=native_tokenizer_snapshot,
    )
    _revalidate_directory_snapshot(
        conversion_parent,
        label="T3B conversion directory",
    )
    _revalidate_directory_snapshot(
        report_parent,
        label="T3B base report directory",
    )
    return _T3BPhysicalInputCapture(
        revision_trees=revision_trees,
        source_checkpoint=source_checkpoint,
        native_checkpoint=native_checkpoint,
        pinned_dataset=pinned_dataset,
        dataset_inventory=dataset_inventory,
        tokenizer_snapshot=tokenizer_snapshot,
        native_tokenizer_snapshot=native_tokenizer_snapshot,
        evaluation_artifact=evaluation_artifact,
        base_report=base_report,
        converted_model=converted_model,
        name_map=name_map,
    )


def _require_unchanged_t3b_inputs(
    before: object,
    after: object,
    *,
    context: str,
) -> None:
    if before != after:
        raise RuntimeError(f"T3B physical inputs changed during {context}")


def collect_t3b_frozen_input_evidence(
    config: FineTuneConfig,
    model: SmolVLATrainingModel | None,
    *,
    runtime_statistics=None,
    validate_runtime_model: bool = True,
) -> dict[str, object]:
    """Rebuild every physical input commitment used by a T3B process."""

    _validate_t3b_frozen_config(config)
    from training.evaluation import _frozen_evaluation_artifact

    if validate_runtime_model and model is None:
        raise ValueError("runtime model validation requires a loaded T3B model")
    paths = _resolve_t3b_input_paths(config, model)

    # Commit the first view before any statistics, evaluator, tokenizer, or
    # conversion semantic read. The final view must be exactly identical.
    initial = _capture_t3b_physical_inputs(config, paths)
    if initial.base_report.sha256 != FROZEN_BASE_REPORT_SHA256:
        raise ValueError("T3B base evaluation report differs from the frozen report")

    _frozen_evaluation_artifact(paths.evaluation_root, dataset_root=paths.dataset_root)
    manifest, manifest_sha256 = _read_stable_json(
        paths.evaluation_root / "manifest.json",
        label="T3B held-out evaluation manifest",
    )
    expected_case_files = _T3B_EVALUATION_FILES - {"manifest.json", "metadata.json"}
    if (
        manifest_sha256 != FROZEN_EVALUATION_MANIFEST_SHA256
        or manifest_sha256
        != initial.evaluation_artifact["files"]["manifest.json"]
        or not isinstance(manifest, Mapping)
        or {
            str(record.get("path"))
            for record in manifest.values()
            if isinstance(record, Mapping)
        }
        != expected_case_files
        or any(
            not isinstance(record, Mapping)
            or initial.evaluation_artifact["files"].get(str(record.get("path")))
            != record.get("sha256")
            for record in manifest.values()
        )
    ):
        raise ValueError("T3B evaluation manifest differs from its tensor files")

    split = make_episode_split(num_episodes=50, seed=config.seed)
    statistics = compute_train_statistics(paths.dataset_root, split.train_episodes)
    _validate_t3b_train_statistics_sha256(statistics.sha256)
    processor_statistics_sha256 = _canonical_json_sha256(
        statistics.processor_stats
    )
    if runtime_statistics is not None and (
        runtime_statistics.sha256 != statistics.sha256
        or _canonical_json_sha256(runtime_statistics.processor_stats)
        != processor_statistics_sha256
    ):
        raise RuntimeError("live training statistics differ from the frozen inputs")

    native_conversion = _validate_t3b_conversion_from_stable_hardlinks(
        source_model_path=paths.native_checkpoint_root / "model.safetensors",
        converted_model_path=paths.converted_model_path,
        name_map_path=paths.name_map_path,
    )
    if (
        native_conversion["model_sha256"] != initial.converted_model.sha256
        or native_conversion["name_map_sha256"] != initial.name_map.sha256
        or native_conversion["source_model_sha256"]
        != initial.native_checkpoint["files"]["model.safetensors"]
    ):
        raise RuntimeError("T3B conversion differs from the initial physical inputs")
    if validate_runtime_model:
        assert model is not None
        _validate_runtime_model_matches_converted_checkpoint(
            model,
            paths.converted_model_path,
        )

    final = _capture_t3b_physical_inputs(config, paths)
    _require_unchanged_t3b_inputs(
        initial,
        final,
        context="frozen-input capture",
    )
    evidence = {
        "format_version": 1,
        "revision_trees": initial.revision_trees,
        "train_statistics_sha256": statistics.sha256,
        "processor_statistics_sha256": processor_statistics_sha256,
        "source_checkpoint": initial.source_checkpoint,
        "native_checkpoint": initial.native_checkpoint,
        "native_conversion": native_conversion,
        "pinned_dataset": initial.pinned_dataset,
        "tokenizer_snapshot": initial.tokenizer_snapshot,
        "native_tokenizer_snapshot": initial.native_tokenizer_snapshot,
        "evaluation_artifact": initial.evaluation_artifact,
        "base_report": {
            "file": paths.base_report_path.name,
            "sha256": initial.base_report.sha256,
        },
    }
    return validate_t3b_frozen_input_evidence(evidence)


@contextmanager
def _private_t3b_source_checkpoint(
    *,
    config: FineTuneConfig,
    expected_evidence: Mapping[str, object],
    output_descriptor: int | None = None,
    expected_output_snapshot: _DirectorySnapshot | None = None,
) -> Iterator[_BoundDirectoryLease]:
    """Copy the six committed export inputs into owner-only private files."""

    expected = validate_t3b_frozen_input_evidence(expected_evidence)[
        "source_checkpoint"
    ]
    source_root = resolve_base_checkpoint(config.cache_dir)
    current = _snapshot_tree_evidence(
        root=source_root,
        paths=_checkpoint_input_paths(source_root),
        label="T3B export source checkpoint",
        allowed_symlink_root=Path(config.cache_dir).resolve(),
    )
    if current != expected:
        raise RuntimeError("T3B export source differs from the launch commitment")
    output_dir = Path(os.path.abspath(config.output_dir))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if output_descriptor is None:
        output_snapshot = _snapshot_directory(
            output_dir,
            label="T3B export run directory",
        )
        bound_output = os.open(output_dir, directory_flags)
    else:
        if expected_output_snapshot is None:
            raise ValueError("bound private source requires an output snapshot")
        if output_dir != expected_output_snapshot.path:
            raise ValueError("private source output differs from its bound directory")
        output_snapshot = expected_output_snapshot
        bound_output = os.dup(output_descriptor)
    output_identity = os.fstat(bound_output)
    _, output_device, output_inode = output_snapshot.components[-1]
    if (output_identity.st_dev, output_identity.st_ino) != (
        output_device,
        output_inode,
    ):
        os.close(bound_output)
        raise RuntimeError("private source output descriptor changed")
    private_descriptor: int | None = None
    private_name: str | None = None
    private_device: int | None = None
    private_inode: int | None = None
    try:
        private_descriptor, private_name, private_root = _create_staged_directory_at(
            bound_output,
            prefix=".source-checkpoint-",
        )
        private_identity = os.fstat(private_descriptor)
        private_device = private_identity.st_dev
        private_inode = private_identity.st_ino
        private_snapshot = _snapshot_directory(
            private_root,
            label="T3B private export source checkpoint",
        )
        for name, source_path in _checkpoint_input_paths(source_root).items():
            copied = _copy_stable_file_no_clobber(
                source_path.resolve(strict=True),
                private_root / name,
                label=f"T3B export source {name}",
                destination_parent_descriptor=private_descriptor,
                expected_destination_parent_snapshot=private_snapshot,
            )
            if copied.sha256 != expected["files"][name]:
                raise RuntimeError(
                    f"private T3B export source differs from launch input: {name}"
                )
        private_root = _descriptor_path(private_descriptor)
        private = _snapshot_tree_evidence(
            root=private_root,
            paths=_checkpoint_input_paths(private_root),
            label="T3B private export source checkpoint",
        )
        if private["files"] != expected["files"] or private["links"]:
            raise RuntimeError("private T3B export source differs from launch inputs")
        _revalidate_directory_snapshot(
            private_snapshot,
            label="T3B private export source checkpoint",
        )
        yield _BoundDirectoryLease(private_descriptor)
        private_root = _descriptor_path(private_descriptor)
        private_after = _snapshot_tree_evidence(
            root=private_root,
            paths=_checkpoint_input_paths(private_root),
            label="T3B private export source checkpoint",
        )
        source_after = _snapshot_tree_evidence(
            root=source_root,
            paths=_checkpoint_input_paths(source_root),
            label="T3B export source checkpoint",
            allowed_symlink_root=Path(config.cache_dir).resolve(),
        )
        if (
            private_after["files"] != expected["files"]
            or private_after["links"]
            or source_after != expected
        ):
            raise RuntimeError("T3B source checkpoint changed during export")
        _revalidate_directory_snapshot(
            private_snapshot,
            label="T3B private export source checkpoint",
        )
        _revalidate_directory_snapshot(
            output_snapshot,
            label="T3B export run directory",
        )
    finally:
        if private_descriptor is not None:
            os.close(private_descriptor)
        if private_name is not None and private_device is not None and private_inode is not None:
            try:
                remaining = os.stat(
                    private_name,
                    dir_fd=bound_output,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                remaining = None
            if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
                private_device,
                private_inode,
            ):
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise RuntimeError("safe private-source cleanup requires fd-based rmtree")
                shutil.rmtree(private_name, dir_fd=bound_output)
                os.fsync(bound_output)
        os.close(bound_output)


@contextmanager
def _private_t3b_tokenizer_snapshot(
    *,
    config: FineTuneConfig,
    expected_evidence: Mapping[str, object],
    output_descriptor: int | None = None,
    expected_output_snapshot: _DirectorySnapshot | None = None,
) -> Iterator[_BoundDirectoryLease]:
    """Copy the launch-bound tokenizer into owner-only descriptor-held files."""

    from training.reference_export import resolve_tokenizer_snapshot

    expected = validate_t3b_frozen_input_evidence(expected_evidence)[
        "tokenizer_snapshot"
    ]
    source_root = resolve_tokenizer_snapshot(Path(config.cache_dir).resolve())
    current = _snapshot_tree_evidence(
        root=source_root,
        paths=_tokenizer_input_paths(source_root),
        label="T3B export tokenizer",
        allowed_symlink_root=Path(config.cache_dir).resolve(),
    )
    if current != expected:
        raise RuntimeError("T3B export tokenizer differs from the launch commitment")
    output_dir = Path(os.path.abspath(config.output_dir))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if output_descriptor is None:
        output_snapshot = _snapshot_directory(
            output_dir,
            label="T3B tokenizer run directory",
        )
        bound_output = os.open(output_dir, directory_flags)
    else:
        if expected_output_snapshot is None:
            raise ValueError("bound private tokenizer requires an output snapshot")
        output_snapshot = expected_output_snapshot
        bound_output = os.dup(output_descriptor)
    opened_output = os.fstat(bound_output)
    _, output_device, output_inode = output_snapshot.components[-1]
    if (opened_output.st_dev, opened_output.st_ino) != (
        output_device,
        output_inode,
    ):
        os.close(bound_output)
        raise RuntimeError("private tokenizer output descriptor changed")
    private_descriptor: int | None = None
    private_name: str | None = None
    private_device: int | None = None
    private_inode: int | None = None
    try:
        private_descriptor, private_name, private_root = _create_staged_directory_at(
            bound_output,
            prefix=".tokenizer-snapshot-",
        )
        private_identity = os.fstat(private_descriptor)
        private_device = private_identity.st_dev
        private_inode = private_identity.st_ino
        private_snapshot = _snapshot_directory(
            private_root,
            label="T3B private export tokenizer",
        )
        for name, source_path in _tokenizer_input_paths(source_root).items():
            copied = _copy_stable_file_no_clobber(
                source_path.resolve(strict=True),
                private_root / name,
                label=f"T3B export tokenizer {name}",
                destination_parent_descriptor=private_descriptor,
                expected_destination_parent_snapshot=private_snapshot,
            )
            if copied.sha256 != expected["files"][name]:
                raise RuntimeError(
                    f"private T3B export tokenizer differs from launch input: {name}"
                )
        private_root = _descriptor_path(private_descriptor)
        private = _snapshot_tree_evidence(
            root=private_root,
            paths=_tokenizer_input_paths(private_root),
            label="T3B private export tokenizer",
        )
        if private["files"] != expected["files"] or private["links"]:
            raise RuntimeError("private T3B export tokenizer differs from launch inputs")
        _revalidate_directory_snapshot(
            private_snapshot,
            label="T3B private export tokenizer",
        )
        yield _BoundDirectoryLease(private_descriptor)
        private_root = _descriptor_path(private_descriptor)
        private_after = _snapshot_tree_evidence(
            root=private_root,
            paths=_tokenizer_input_paths(private_root),
            label="T3B private export tokenizer",
        )
        source_after = _snapshot_tree_evidence(
            root=source_root,
            paths=_tokenizer_input_paths(source_root),
            label="T3B export tokenizer",
            allowed_symlink_root=Path(config.cache_dir).resolve(),
        )
        if (
            private_after["files"] != expected["files"]
            or private_after["links"]
            or source_after != expected
        ):
            raise RuntimeError("T3B tokenizer changed during export")
        _revalidate_directory_snapshot(
            private_snapshot,
            label="T3B private export tokenizer",
        )
        _revalidate_directory_snapshot(
            output_snapshot,
            label="T3B tokenizer run directory",
        )
    finally:
        if private_descriptor is not None:
            os.close(private_descriptor)
        if private_name is not None and private_device is not None and private_inode is not None:
            try:
                remaining = os.stat(
                    private_name,
                    dir_fd=bound_output,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                remaining = None
            if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
                private_device,
                private_inode,
            ):
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise RuntimeError("safe private-tokenizer cleanup requires fd rmtree")
                shutil.rmtree(private_name, dir_fd=bound_output)
                os.fsync(bound_output)
        os.close(bound_output)


def _validate_training_bridge_evidence(value: object) -> dict[str, object]:
    """Validate the self-hashed materialized dataset/processor/loader commitment."""

    if not isinstance(value, Mapping) or set(value) != {
        "format_version",
        "sha256",
        "components",
    }:
        raise ValueError("training bridge evidence fields differ")
    components = value["components"]
    expected_components = {
        "bridge",
        "config",
        "metadata",
        "dataset",
        "sampler",
        "preprocessor",
        "loader",
    }
    if (
        value["format_version"] != 1
        or not _is_sha256(value["sha256"])
        or not isinstance(components, Mapping)
        or set(components) != expected_components
        or any(
            not isinstance(name, str) or not _is_sha256(digest)
            for name, digest in components.items()
        )
    ):
        raise ValueError("training bridge evidence is invalid")
    canonical_components = dict(sorted(components.items()))
    if value["sha256"] != _canonical_json_sha256(canonical_components):
        raise ValueError("training bridge evidence digest differs")
    return {
        "format_version": 1,
        "sha256": value["sha256"],
        "components": canonical_components,
    }


def assemble_finetune_launch_config(
    *,
    config: FineTuneConfig,
    budget: Mapping[str, object],
    train_statistics_sha256: str,
    train_episodes: tuple[int, ...],
    holdout_episodes: tuple[int, ...],
    base_artifact: Mapping[str, str],
    optimizer_config: SmolVLAOptimizerConfig,
    lora_report,
    reference_freeze_policy: Mapping[str, object],
    implementation_sha256: Mapping[str, str],
    frozen_inputs: Mapping[str, object],
    training_bridge: Mapping[str, object],
    created_at_ns: int,
) -> dict[str, object]:
    """Assemble the complete pre-update T3B launch commitment."""

    budget = validate_fixed_step_budget(budget, config=config)
    frozen_inputs = validate_t3b_frozen_input_evidence(frozen_inputs)
    training_bridge = _validate_training_bridge_evidence(training_bridge)
    if train_statistics_sha256 != frozen_inputs["train_statistics_sha256"]:
        raise ValueError("launch statistics differ from the frozen physical inputs")
    if (
        base_artifact.get("model_file")
        != frozen_inputs["native_conversion"]["model_file"]
        or base_artifact.get("model_sha256")
        != frozen_inputs["native_conversion"]["model_sha256"]
        or base_artifact.get("name_map_file")
        != frozen_inputs["native_conversion"]["name_map_file"]
        or base_artifact.get("name_map_sha256")
        != frozen_inputs["native_conversion"]["name_map_sha256"]
    ):
        raise ValueError("launch base artifact differs from the validated conversion")
    selected_steps = int(budget["selected_steps"])
    run_config_sha256 = training_run_config_sha256(
        config,
        selected_steps=selected_steps,
        train_statistics_sha256=train_statistics_sha256,
        train_episodes=train_episodes,
        holdout_episodes=holdout_episodes,
        base_artifact=base_artifact,
        optimizer_config=optimizer_config,
    )
    document: dict[str, object] = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-finetune-launch-config",
        "procedure_id": "smolvla-t3b-expert-only-lora-v1",
        "created_at_utc": _launch_utc_from_ns(created_at_ns),
        "created_at_ns": created_at_ns,
        "training": _launch_training_config(config),
        "budget": dict(budget),
        "source_identity": {
            "checkpoint": {"id": CHECKPOINT_ID, "revision": CHECKPOINT_REVISION},
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "base_vlm": {"id": BASE_VLM_ID, "revision": BASE_VLM_REVISION},
        },
        "split": {
            "train_episodes": list(train_episodes),
            "holdout_episodes": list(holdout_episodes),
        },
        "train_statistics_sha256": train_statistics_sha256,
        "base_artifact": dict(base_artifact),
        "optimizer": {
            **asdict(optimizer_config),
            "betas": list(optimizer_config.betas),
        },
        "lora_topology": {
            "scope": lora_report.scope,
            "rank": lora_report.rank,
            "alpha": lora_report.alpha,
            "dropout": lora_report.dropout,
            "adapter_count": lora_report.adapter_count,
            "target_names": list(lora_report.target_names),
            "trainable_names": list(lora_report.trainable_names),
            "trainable_tensor_count": lora_report.trainable_tensor_count,
            "trainable_scalar_count": lora_report.trainable_scalar_count,
        },
        "reference_freeze_policy": dict(reference_freeze_policy),
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "frozen_inputs": frozen_inputs,
        "training_bridge": training_bridge,
        "export_audit": frozen_export_audit_metadata(run_config_sha256),
        "run_config_sha256": run_config_sha256,
    }
    document["configuration_sha256"] = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return validate_finetune_launch_config(document)


def validate_finetune_launch_config(value: object) -> dict[str, object]:
    """Validate and recompute one self-contained pretraining commitment."""

    required = {
        "format_version",
        "artifact_type",
        "procedure_id",
        "created_at_utc",
        "created_at_ns",
        "training",
        "budget",
        "source_identity",
        "split",
        "train_statistics_sha256",
        "base_artifact",
        "optimizer",
        "lora_topology",
        "reference_freeze_policy",
        "implementation_sha256",
        "frozen_inputs",
        "training_bridge",
        "export_audit",
        "run_config_sha256",
        "configuration_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("launch configuration fields differ from the frozen schema")
    document = dict(value)
    recorded_configuration_sha256 = document.pop("configuration_sha256")
    computed_configuration_sha256 = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if recorded_configuration_sha256 != computed_configuration_sha256:
        raise ValueError("launch configuration digest differs from its contents")
    if (
        document["format_version"] != 1
        or document["artifact_type"] != "smolvla-mlx-finetune-launch-config"
        or document["procedure_id"] != "smolvla-t3b-expert-only-lora-v1"
        or not _is_sha256(recorded_configuration_sha256)
    ):
        raise ValueError("launch configuration identity is invalid")
    created_at_ns = document["created_at_ns"]
    if document["created_at_utc"] != _launch_utc_from_ns(created_at_ns):
        raise ValueError("launch configuration timestamps differ")
    training = document["training"]
    training_fields = set(_launch_training_config(FineTuneConfig()))
    if not isinstance(training, Mapping) or set(training) != training_fields:
        raise ValueError("launch training fields differ from the frozen schema")
    try:
        config = FineTuneConfig(
            cache_dir=Path(str(training["cache_dir"])),
            native_cache=Path(str(training["native_cache"])),
            output_dir=Path(str(training["output_dir"])),
            seed=int(training["seed"]),
            sampler_seed=int(training["sampler_seed"]),
            nominal_steps=int(training["nominal_steps"]),
            effective_batch_size=int(training["effective_batch_size"]),
            training_seconds=float(training["training_seconds"]),
            benchmark_warmup_updates=int(training["benchmark_warmup_updates"]),
            benchmark_measured_updates=int(training["benchmark_measured_updates"]),
            rank=int(training["rank"]),
            alpha=float(training["alpha"]),
            dropout=float(training["dropout"]),
            lora_scope=str(training["lora_scope"]),
            budget_mode=str(training["budget_mode"]),
            checkpoint_interval=int(training["checkpoint_interval"]),
        )
        _validate_t3b_frozen_config(config)
        validate_fixed_step_budget(document["budget"], config=config)
        optimizer = SmolVLAOptimizerConfig(**dict(document["optimizer"]))
    except (TypeError, ValueError) as error:
        raise ValueError("launch training configuration is invalid") from error
    if document["source_identity"] != {
        "checkpoint": {"id": CHECKPOINT_ID, "revision": CHECKPOINT_REVISION},
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "base_vlm": {"id": BASE_VLM_ID, "revision": BASE_VLM_REVISION},
    }:
        raise ValueError("launch source identity changed")
    split = document["split"]
    base_artifact = document["base_artifact"]
    if (
        not isinstance(split, Mapping)
        or set(split) != {"train_episodes", "holdout_episodes"}
        or not isinstance(split["train_episodes"], list)
        or not isinstance(split["holdout_episodes"], list)
        or not isinstance(base_artifact, Mapping)
        or set(base_artifact)
        != {"model_file", "model_sha256", "name_map_file", "name_map_sha256"}
        or not _is_sha256(document["train_statistics_sha256"])
        or not _is_sha256(base_artifact["model_sha256"])
        or not _is_sha256(base_artifact["name_map_sha256"])
    ):
        raise ValueError("launch data/base evidence is invalid")
    topology = document["lora_topology"]
    topology_fields = {
        "scope",
        "rank",
        "alpha",
        "dropout",
        "adapter_count",
        "target_names",
        "trainable_names",
        "trainable_tensor_count",
        "trainable_scalar_count",
    }
    target_names = topology.get("target_names") if isinstance(topology, Mapping) else None
    trainable_names = (
        topology.get("trainable_names") if isinstance(topology, Mapping) else None
    )
    if (
        not isinstance(topology, Mapping)
        or set(topology) != topology_fields
        or not isinstance(target_names, list)
        or not all(isinstance(name, str) and name for name in target_names)
        or len(set(target_names)) != len(target_names)
        or not isinstance(trainable_names, list)
        or not all(isinstance(name, str) and name for name in trainable_names)
        or len(set(trainable_names)) != len(trainable_names)
        or topology["scope"] != config.lora_scope
        or topology["rank"] != config.rank
        or topology["alpha"] != config.alpha
        or topology["dropout"] != config.dropout
        or topology["adapter_count"] != len(target_names)
        or topology["trainable_tensor_count"] != len(trainable_names)
        or topology["trainable_tensor_count"] != 2 * topology["adapter_count"]
        or not isinstance(topology["trainable_scalar_count"], int)
        or topology["trainable_scalar_count"] <= 0
    ):
        raise ValueError("launch LoRA topology is invalid")
    reference_policy = document["reference_freeze_policy"]
    if (
        not isinstance(reference_policy, Mapping)
        or set(reference_policy)
        != {
            "lerobot_version",
            "freeze_vision_encoder",
            "train_expert_only",
            "train_state_proj",
            "configuration_source_sha256",
            "implementation_source_sha256",
        }
        or not isinstance(reference_policy["lerobot_version"], str)
        or not reference_policy["lerobot_version"]
        or any(
            reference_policy[field] is not True
            for field in (
                "freeze_vision_encoder",
                "train_expert_only",
                "train_state_proj",
            )
        )
        or not _is_sha256(reference_policy["configuration_source_sha256"])
        or not _is_sha256(reference_policy["implementation_source_sha256"])
    ):
        raise ValueError("launch reference freeze policy is invalid")
    implementation = document["implementation_sha256"]
    if (
        not isinstance(implementation, Mapping)
        or not implementation
        or any(
            not isinstance(name, str) or not name or not _is_sha256(digest)
            for name, digest in implementation.items()
        )
    ):
        raise ValueError("launch implementation hashes are invalid")
    frozen_inputs = validate_t3b_frozen_input_evidence(document["frozen_inputs"])
    training_bridge = _validate_training_bridge_evidence(document["training_bridge"])
    if (
        frozen_inputs["train_statistics_sha256"]
        != document["train_statistics_sha256"]
        or frozen_inputs["native_conversion"]["model_file"]
        != base_artifact["model_file"]
        or frozen_inputs["native_conversion"]["model_sha256"]
        != base_artifact["model_sha256"]
        or frozen_inputs["native_conversion"]["name_map_file"]
        != base_artifact["name_map_file"]
        or frozen_inputs["native_conversion"]["name_map_sha256"]
        != base_artifact["name_map_sha256"]
    ):
        raise ValueError("launch frozen inputs differ from the training/base evidence")
    if training_bridge != document["training_bridge"]:
        raise ValueError("launch training bridge evidence is not canonical")
    recomputed_run_config = training_run_config_sha256(
        config,
        selected_steps=int(document["budget"]["selected_steps"]),
        train_statistics_sha256=str(document["train_statistics_sha256"]),
        train_episodes=tuple(int(item) for item in split["train_episodes"]),
        holdout_episodes=tuple(int(item) for item in split["holdout_episodes"]),
        base_artifact={str(key): str(item) for key, item in base_artifact.items()},
        optimizer_config=optimizer,
    )
    if document["run_config_sha256"] != recomputed_run_config:
        raise ValueError("launch run configuration digest is invalid")
    if document["export_audit"] != frozen_export_audit_metadata(
        recomputed_run_config
    ):
        raise ValueError("launch export/evaluation audit inputs changed")
    return {**document, "configuration_sha256": recorded_configuration_sha256}


def write_finetune_launch_config(
    path: str | Path,
    value: object,
    *,
    parent_descriptor: int | None = None,
    expected_parent_snapshot: _DirectorySnapshot | None = None,
) -> str:
    """Install one launch commitment atomically without overwriting a winner."""

    document = validate_finetune_launch_config(value)
    path = Path(os.path.abspath(Path(path).expanduser()))
    if parent_descriptor is None:
        parent_snapshot = _ensure_safe_directory(
            path.parent,
            label="launch configuration directory",
        )
    else:
        if expected_parent_snapshot is None:
            raise ValueError("bound launch writer requires a parent snapshot")
        if path.parent != expected_parent_snapshot.path:
            raise ValueError("launch configuration differs from its bound parent")
        parent_snapshot = expected_parent_snapshot
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    bound_parent = (
        os.open(path.parent, directory_flags)
        if parent_descriptor is None
        else os.dup(parent_descriptor)
    )
    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    temporary_identity: os.stat_result | None = None
    try:
        opened_parent = os.fstat(bound_parent)
        _, parent_device, parent_inode = parent_snapshot.components[-1]
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_device,
            parent_inode,
        ):
            raise RuntimeError("launch configuration directory changed before open")
        temporary_descriptor, temporary_name = _create_staged_file_at(
            bound_parent,
            prefix=f".{path.name}.",
        )
        view = memoryview(payload)
        while view:
            written = os.write(temporary_descriptor, view)
            if written <= 0:
                raise OSError("short write while staging launch configuration")
            view = view[written:]
        os.fsync(temporary_descriptor)
        temporary_identity = os.fstat(temporary_descriptor)
        os.lseek(temporary_descriptor, 0, os.SEEK_SET)
        staged_payload = bytearray()
        while chunk := os.read(temporary_descriptor, 1024 * 1024):
            staged_payload.extend(chunk)
        if bytes(staged_payload) != payload:
            raise RuntimeError("staged launch configuration bytes changed")
        _rename_entry_no_clobber_at(
            source_descriptor=bound_parent,
            source_name=temporary_name,
            destination_descriptor=bound_parent,
            destination_name=path.name,
            expected_device=temporary_identity.st_dev,
            expected_inode=temporary_identity.st_ino,
            expected_directory=False,
        )
        os.fsync(bound_parent)
        published = os.stat(
            path.name,
            dir_fd=bound_parent,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(published.st_mode)
            or (published.st_dev, published.st_ino)
            != (temporary_identity.st_dev, temporary_identity.st_ino)
        ):
            raise RuntimeError("published launch configuration inode changed")
        os.lseek(temporary_descriptor, 0, os.SEEK_SET)
        published_payload = bytearray()
        while chunk := os.read(temporary_descriptor, 1024 * 1024):
            published_payload.extend(chunk)
        if bytes(published_payload) != payload:
            raise RuntimeError("published launch configuration bytes changed")
        _revalidate_directory_snapshot(
            parent_snapshot,
            label="launch configuration directory",
        )
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_name is not None and temporary_identity is not None:
            try:
                remaining = os.stat(
                    temporary_name,
                    dir_fd=bound_parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                remaining = None
            if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
                temporary_identity.st_dev,
                temporary_identity.st_ino,
            ):
                os.unlink(temporary_name, dir_fd=bound_parent)
                os.fsync(bound_parent)
        os.close(bound_parent)
    return hashlib.sha256(payload).hexdigest()


def _read_finetune_launch_config(
    path: str | Path,
) -> tuple[dict[str, object], str]:
    """Read one launch config through a stable no-follow descriptor."""

    path = Path(path)
    parent_snapshot = _snapshot_directory(
        path.parent,
        label="launch configuration directory",
    )
    snapshot = _snapshot_regular_file(
        path,
        label="launch configuration",
        capture_payload=True,
    )
    document, digest = _validate_finetune_launch_snapshot(snapshot)
    _revalidate_directory_snapshot(
        parent_snapshot,
        label="launch configuration directory",
    )
    return document, digest


def _validate_finetune_launch_snapshot(
    snapshot: _StableFileSnapshot,
) -> tuple[dict[str, object], str]:
    """Parse one already-bound launch snapshot without reopening its path."""

    document = _json_from_stable_snapshot(
        snapshot,
        label="launch configuration",
    )
    return validate_finetune_launch_config(document), snapshot.sha256


def validate_finetune_launch_runtime_binding(
    launch_document: object,
    *,
    config: FineTuneConfig,
    budget: Mapping[str, object],
    train_statistics_sha256: str,
    train_episodes: tuple[int, ...],
    holdout_episodes: tuple[int, ...],
    base_artifact: Mapping[str, str],
    optimizer_config: SmolVLAOptimizerConfig,
    lora_report,
    reference_freeze_policy: Mapping[str, object],
    implementation_sha256: Mapping[str, str],
    frozen_inputs: Mapping[str, object],
    training_bridge: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild a launch commitment from live inputs before the first update."""

    persisted = validate_finetune_launch_config(launch_document)
    expected = assemble_finetune_launch_config(
        config=config,
        budget=budget,
        train_statistics_sha256=train_statistics_sha256,
        train_episodes=train_episodes,
        holdout_episodes=holdout_episodes,
        base_artifact=base_artifact,
        optimizer_config=optimizer_config,
        lora_report=lora_report,
        reference_freeze_policy=reference_freeze_policy,
        implementation_sha256=implementation_sha256,
        frozen_inputs=frozen_inputs,
        training_bridge=training_bridge,
        created_at_ns=int(persisted["created_at_ns"]),
    )
    if persisted != expected:
        raise ValueError("launch configuration differs from the current runtime inputs")
    return persisted


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

    lora_config: dict[str, object] = {
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
    }
    if config.lora_scope != LEGACY_FULL_SCOPE:
        lora_config["scope"] = config.lora_scope
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
        "lora": lora_config,
        "checkpoint_interval": config.checkpoint_interval,
        "base_artifact": dict(base_artifact),
        "optimizer": asdict(optimizer_config),
        "train_statistics_sha256": train_statistics_sha256,
        "train_episodes": list(train_episodes),
        "holdout_episodes": list(holdout_episodes),
        "base_dtype": "bfloat16",
        "adapter_dtype": "float32",
    }
    if config.budget_mode != ADAPTIVE_BUDGET_MODE:
        payload["budget_mode"] = config.budget_mode
    if _requires_t3b_launch_config(config):
        payload["base_vlm"] = {"id": BASE_VLM_ID, "revision": BASE_VLM_REVISION}
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


def reference_freeze_policy_evidence() -> dict[str, object]:
    """Read and hash the installed LeRobot default SmolVLA freeze policy."""

    from importlib.metadata import version
    import lerobot.policies.smolvla.configuration_smolvla as configuration_module
    import lerobot.policies.smolvla.smolvlm_with_expert as implementation_module

    config_class = configuration_module.SmolVLAConfig
    fields = config_class.__dataclass_fields__
    policy = {
        "freeze_vision_encoder": fields["freeze_vision_encoder"].default,
        "train_expert_only": fields["train_expert_only"].default,
        "train_state_proj": fields["train_state_proj"].default,
    }
    if policy != {
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
    }:
        raise RuntimeError(f"installed reference freeze policy changed: {policy}")
    configuration_path = Path(configuration_module.__file__).resolve(strict=True)
    implementation_path = Path(implementation_module.__file__).resolve(strict=True)
    return {
        "lerobot_version": version("lerobot"),
        **policy,
        "configuration_source_sha256": _file_sha256(configuration_path),
        "implementation_source_sha256": _file_sha256(implementation_path),
    }


_T3B_RUNTIME_DISTRIBUTIONS = (
    "lerobot",
    "datasets",
    "pyarrow",
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "av",
    "huggingface-hub",
    "safetensors",
    "numpy",
    "mlx",
    "mlx-metal",
    "pillow",
    "pandas",
    "packaging",
    "einops",
    "fsspec",
)


def _resolve_installed_path_without_symlinks(
    path: Path,
    *,
    allowed_root: Path,
    label: str,
) -> Path:
    """Resolve an installed path only if every component below the root is direct."""

    allowed_root = allowed_root.resolve(strict=True)
    absolute = Path(os.path.abspath(path))
    if not absolute.is_relative_to(allowed_root):
        raise ValueError(f"{label} escapes its environment: {absolute}")
    current = allowed_root
    for component in absolute.relative_to(allowed_root).parts:
        current /= component
        identity = os.lstat(current)
        if stat.S_ISLNK(identity.st_mode):
            raise FileNotFoundError(f"{label} contains a symlink: {absolute}")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        raise FileNotFoundError(f"{label} contains a symlink: {absolute}")
    return resolved


def _installed_distribution_record_path(distribution_name: str) -> Path:
    """Resolve the installed wheel RECORD that identifies one runtime package."""

    from importlib.metadata import distribution

    installed = distribution(distribution_name)
    record = next(
        (
            item
            for item in installed.files or ()
            if item.name == "RECORD" and item.parent.name.endswith(".dist-info")
        ),
        None,
    )
    if record is None:
        raise FileNotFoundError(
            f"installed distribution has no RECORD: {distribution_name}"
        )
    return _resolve_installed_path_without_symlinks(
        Path(installed.locate_file(record)),
        allowed_root=Path(sys.prefix),
        label=f"installed distribution RECORD for {distribution_name}",
    )


def _hash_distribution_recorded_files(
    record_path: Path,
    *,
    install_root: Path,
    allowed_root: Path,
) -> str:
    """Verify one wheel RECORD against installed bytes and hash the full payload."""

    record_snapshot = _snapshot_regular_file(
        record_path,
        label="installed distribution RECORD",
        capture_payload=True,
    )
    assert record_snapshot.payload is not None
    try:
        rows = tuple(
            csv.reader(io.StringIO(record_snapshot.payload.decode("utf-8")))
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"installed distribution RECORD is not UTF-8: {record_path}") from error
    if not rows:
        raise ValueError(f"installed distribution RECORD is empty: {record_path}")
    install_root = install_root.resolve(strict=True)
    allowed_root = allowed_root.resolve(strict=True)
    evidence: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise ValueError(f"installed distribution RECORD row is invalid: {record_path}")
        recorded_name, recorded_digest, recorded_size = row
        if (
            not recorded_name
            or recorded_name in seen
            or Path(recorded_name).is_absolute()
            or "\x00" in recorded_name
        ):
            raise ValueError(f"installed distribution RECORD path is invalid: {recorded_name!r}")
        seen.add(recorded_name)
        candidate = _resolve_installed_path_without_symlinks(
            install_root / recorded_name,
            allowed_root=allowed_root,
            label=f"installed distribution file {recorded_name}",
        )
        if not candidate.is_relative_to(allowed_root):
            raise ValueError(
                f"installed distribution file escapes its environment: {recorded_name}"
            )
        if not candidate.is_file():
            raise FileNotFoundError(
                f"installed distribution file is unsafe: {recorded_name}"
            )
        snapshot = _snapshot_regular_file(
            candidate,
            label=f"installed distribution file {recorded_name}",
        )
        if recorded_size:
            try:
                expected_size = int(recorded_size)
            except ValueError as error:
                raise ValueError(
                    f"installed distribution RECORD size is invalid: {recorded_name}"
                ) from error
            if expected_size < 0 or snapshot.size != expected_size:
                raise RuntimeError(
                    f"installed distribution file size differs from RECORD: {recorded_name}"
                )
        if recorded_digest:
            algorithm, separator, encoded = recorded_digest.partition("=")
            if algorithm != "sha256" or not separator or not encoded:
                raise ValueError(
                    f"installed distribution RECORD digest is unsupported: {recorded_name}"
                )
            try:
                expected_digest = base64.urlsafe_b64decode(
                    encoded + "=" * (-len(encoded) % 4)
                ).hex()
            except Exception as error:
                raise ValueError(
                    f"installed distribution RECORD digest is invalid: {recorded_name}"
                ) from error
            if snapshot.sha256 != expected_digest:
                raise RuntimeError(
                    f"installed distribution file digest differs from RECORD: {recorded_name}"
                )
        evidence.append(
            {
                "path": recorded_name,
                "size": snapshot.size,
                "sha256": snapshot.sha256,
            }
        )
    final_record = _snapshot_regular_file(
        record_path,
        label="installed distribution RECORD",
        capture_payload=True,
    )
    if not _same_file_snapshot(record_snapshot, final_record):
        raise RuntimeError(f"installed distribution RECORD changed: {record_path}")
    return _canonical_json_sha256(evidence)


def _installed_distribution_payload_sha256(distribution_name: str) -> str:
    """Hash actual bytes of every file installed by one selected distribution."""

    record_path = _installed_distribution_record_path(distribution_name)
    return _hash_distribution_recorded_files(
        record_path,
        install_root=record_path.parent.parent,
        allowed_root=Path(sys.prefix),
    )


def _hash_distribution_package_inventory(
    record_path: Path,
    *,
    install_root: Path,
    allowed_root: Path,
    known_recorded_paths: Collection[Path] = (),
) -> str:
    """Reject unrecorded package files while binding canonical generated bytecode."""

    import importlib.util

    record_snapshot = _snapshot_regular_file(
        record_path,
        label="installed distribution RECORD inventory",
        capture_payload=True,
    )
    assert record_snapshot.payload is not None
    try:
        rows = tuple(csv.reader(io.StringIO(record_snapshot.payload.decode("utf-8"))))
    except UnicodeDecodeError as error:
        raise ValueError(f"installed distribution RECORD is not UTF-8: {record_path}") from error
    install_root = install_root.resolve(strict=True)
    allowed_root = allowed_root.resolve(strict=True)
    recorded_paths: set[Path] = set()
    top_level_roots: set[Path] = set()
    for row in rows:
        if len(row) != 3 or not row[0] or Path(row[0]).is_absolute():
            raise ValueError(f"installed distribution RECORD row is invalid: {record_path}")
        resolved = _resolve_installed_path_without_symlinks(
            install_root / row[0],
            allowed_root=allowed_root,
            label=f"installed distribution inventory file {row[0]}",
        )
        if not resolved.is_relative_to(allowed_root):
            raise ValueError(f"installed distribution inventory escapes environment: {row[0]}")
        recorded_paths.add(resolved)
        relative = Path(row[0])
        if (
            relative.parts
            and relative.parts[0] != ".."
            and not relative.parts[0].endswith(".dist-info")
        ):
            root = _resolve_installed_path_without_symlinks(
                install_root / relative.parts[0],
                allowed_root=allowed_root,
                label=f"installed package root {relative.parts[0]}",
            )
            if root.is_relative_to(allowed_root):
                top_level_roots.add(root)

    for known_path in known_recorded_paths:
        resolved = _resolve_installed_path_without_symlinks(
            Path(known_path),
            allowed_root=allowed_root,
            label="known installed distribution file",
        )
        if not resolved.is_relative_to(allowed_root):
            raise ValueError(
                f"installed distribution inventory escapes environment: {resolved}"
            )
        recorded_paths.add(resolved)

    allowed_bytecode: set[Path] = set()
    for source in recorded_paths:
        if source.suffix != ".py":
            continue
        for optimization in (None, "1", "2"):
            cached = Path(
                importlib.util.cache_from_source(
                    str(source),
                    optimization=optimization,
                )
            )
            allowed_bytecode.add(cached)

    evidence: list[dict[str, object]] = []
    for root in sorted(top_level_roots):
        if root.is_symlink():
            raise FileNotFoundError(f"installed package root is a symlink: {root}")
        if root.is_file():
            candidates = (root,)
        elif root.is_dir():
            candidates = tuple(sorted(root.rglob("*")))
        else:
            raise FileNotFoundError(f"installed package root is unsafe: {root}")
        for candidate in candidates:
            relative_name = candidate.relative_to(install_root).as_posix()
            if "__pycache__" in candidate.relative_to(install_root).parts:
                continue
            if candidate.is_symlink():
                raise FileNotFoundError(
                    f"installed package inventory contains a symlink: {relative_name}"
                )
            if candidate.is_dir():
                evidence.append({"path": f"{relative_name}/", "type": "directory"})
                continue
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"installed package inventory contains an unsafe entry: {relative_name}"
                )
            generated_bytecode = candidate in allowed_bytecode
            if (
                not generated_bytecode
                and candidate.suffix == ".pyc"
                and candidate.parent.name == "__pycache__"
                and "." in candidate.stem
            ):
                source_stem = candidate.stem.split(".", 1)[0]
                generated_bytecode = (
                    candidate.parent.parent / f"{source_stem}.py"
                ).resolve() in recorded_paths
            if candidate not in recorded_paths and not generated_bytecode:
                raise RuntimeError(
                    f"installed package contains an unrecorded file: {relative_name}"
                )
            if generated_bytecode:
                continue
            snapshot = _snapshot_regular_file(
                candidate,
                label=f"installed package inventory file {relative_name}",
            )
            evidence.append(
                {
                    "path": relative_name,
                    "type": "generated-bytecode" if generated_bytecode else "recorded",
                    "size": snapshot.size,
                    "sha256": snapshot.sha256,
                }
            )
    final_record = _snapshot_regular_file(
        record_path,
        label="installed distribution RECORD inventory",
        capture_payload=True,
    )
    if not _same_file_snapshot(record_snapshot, final_record):
        raise RuntimeError(f"installed distribution RECORD changed: {record_path}")
    return _canonical_json_sha256(evidence)


def _installed_runtime_recorded_paths() -> frozenset[Path]:
    """Return the union of wheel ownership for shared runtime package roots."""

    from importlib.metadata import distribution

    allowed_root = Path(sys.prefix).resolve(strict=True)
    recorded_paths: set[Path] = set()
    for distribution_name in _T3B_RUNTIME_DISTRIBUTIONS:
        installed = distribution(distribution_name)
        files = installed.files
        if files is None:
            raise FileNotFoundError(
                f"installed distribution has no file inventory: {distribution_name}"
            )
        for item in files:
            resolved = _resolve_installed_path_without_symlinks(
                Path(installed.locate_file(item)),
                allowed_root=allowed_root,
                label=f"installed distribution file {distribution_name}: {item}",
            )
            if not resolved.is_relative_to(allowed_root):
                raise ValueError(
                    "installed distribution file escapes its environment: "
                    f"{distribution_name}: {item}"
                )
            recorded_paths.add(resolved)
    return frozenset(recorded_paths)


def _installed_distribution_inventory_sha256(
    distribution_name: str,
    *,
    known_recorded_paths: Collection[Path] = (),
) -> str:
    record_path = _installed_distribution_record_path(distribution_name)
    return _hash_distribution_package_inventory(
        record_path,
        install_root=record_path.parent.parent,
        allowed_root=Path(sys.prefix),
        known_recorded_paths=known_recorded_paths,
    )


def _t3b_installed_runtime_inputs() -> dict[str, Path]:
    """Return installed sources and wheel records executed by the T3B bridge."""

    import lerobot

    lerobot_root = Path(lerobot.__file__).resolve(strict=True).parent
    relative_sources: set[Path] = {Path("__init__.py")}
    for pattern in (
        "datasets/*.py",
        "policies/*.py",
        "policies/common/*.py",
        "policies/smolvla/*.py",
        "processor/*.py",
    ):
        relative_sources.update(
            path.relative_to(lerobot_root) for path in lerobot_root.glob(pattern)
        )
    relative_sources.add(Path("utils/collate.py"))
    result = {
        f"installed/lerobot/{relative.as_posix()}": lerobot_root / relative
        for relative in sorted(relative_sources)
    }
    for distribution_name in _T3B_RUNTIME_DISTRIBUTIONS:
        result[f"distribution/{distribution_name}/RECORD"] = (
            _installed_distribution_record_path(distribution_name)
        )
    return result


_T3B_NATIVE_DEPENDENCY_SCOPE = (
    "direct-extension-origin-bound; transitive-dyld-images-inventory-hashed-only"
)
_T3B_REQUIRED_PROVENANCE_MODULES = frozenset(
    {
        "__main__",
        "training",
        "training.runtime_provenance",
        "training.finetune",
    }
)


def _require_t3b_runtime_provenance(
    *,
    allow_unfrozen: bool = False,
    freeze: bool = False,
) -> Mapping[str, object]:
    """Require the isolated bootstrap and a nonempty guarded module manifest."""

    if not (sys.flags.isolated and sys.flags.no_site):
        raise RuntimeError(
            "T3B runtime provenance requires the isolated -I -S launcher"
        )
    from training.runtime_provenance import (
        freeze_runtime_provenance,
        runtime_provenance_evidence,
    )

    evidence = (
        freeze_runtime_provenance() if freeze else runtime_provenance_evidence()
    )
    modules = evidence.get("modules")
    if (
        evidence.get("format_version") != 1
        or evidence.get("native_dependency_scope")
        != _T3B_NATIVE_DEPENDENCY_SCOPE
        or not isinstance(modules, Mapping)
        or not modules
    ):
        raise RuntimeError("T3B runtime provenance is not installed or is invalid")
    if not allow_unfrozen and evidence.get("frozen") is not True:
        raise RuntimeError("T3B runtime provenance is not frozen")
    missing = _T3B_REQUIRED_PROVENANCE_MODULES.difference(modules)
    if missing:
        raise RuntimeError(
            f"T3B runtime provenance lacks required modules: {sorted(missing)}"
        )
    for module_name, generations in modules.items():
        if not isinstance(module_name, str) or not isinstance(generations, list) or not generations:
            raise RuntimeError("T3B runtime provenance module evidence is invalid")
        for generation in generations:
            if (
                not isinstance(generation, Mapping)
                or set(generation)
                != {"origin", "kind", "file_sha256", "code_sha256"}
                or generation["kind"] not in {"source", "extension"}
                or not isinstance(generation["origin"], str)
                or not generation["origin"]
                or not _is_lowercase_sha256(generation["file_sha256"])
                or not _is_lowercase_sha256(generation["code_sha256"])
            ):
                raise RuntimeError("T3B runtime provenance module evidence is invalid")
    return evidence


def finetune_implementation_hashes() -> dict[str, str]:
    """Hash every direct implementation file that can affect T3B training."""

    _require_t3b_runtime_provenance(allow_unfrozen=True)
    repository_root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "mlx_smolvla/__init__.py",
        "mlx_smolvla/cache.py",
        "mlx_smolvla/config.py",
        "mlx_smolvla/connector.py",
        "mlx_smolvla/convert.py",
        "mlx_smolvla/expert.py",
        "mlx_smolvla/flow.py",
        "mlx_smolvla/language.py",
        "mlx_smolvla/policy.py",
        "mlx_smolvla/preprocessing.py",
        "mlx_smolvla/rmsnorm.py",
        "mlx_smolvla/types.py",
        "mlx_smolvla/vision.py",
        "training/finetune.py",
        "training/lora.py",
        "training/model.py",
        "training/objective.py",
        "training/optimizer.py",
        "training/data.py",
        "training/dataset.py",
        "training/export.py",
        "training/evaluation.py",
        "training/gradients.py",
        "training/reference_export.py",
        "training/t3_contract.py",
        "training/runtime_provenance.py",
        "scripts/finetune_lora",
        "scripts/finetune_lora.py",
        "reference/discovery.py",
        "uv.lock",
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = repository_root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"fine-tune implementation input is unsafe: {relative}")
        result[relative] = _file_sha256(path)
    native_extensions = tuple(
        sorted((repository_root / "mlx_smolvla").glob("_rmsnorm_native*.so"))
    )
    if len(native_extensions) != 1:
        raise FileNotFoundError(
            "fine-tune implementation requires exactly one native RMSNorm extension"
        )
    native_extension = native_extensions[0]
    if native_extension.is_symlink() or not native_extension.is_file():
        raise FileNotFoundError("fine-tune native RMSNorm extension is unsafe")
    result[str(native_extension.relative_to(repository_root))] = _file_sha256(
        native_extension
    )
    for name, path in sorted(_t3b_installed_runtime_inputs().items()):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"fine-tune installed runtime input is unsafe: {name}")
        result[name] = _file_sha256(path)
    from importlib.metadata import version

    # Materialize the complete bridge/evaluation/export lazy-import closure before
    # freezing.  The run repeats this closure and compares the resulting manifest
    # with launch.json before the first optimizer update.
    import huggingface_hub  # noqa: F401
    import lerobot.policies.common.vla_utils  # noqa: F401
    import lerobot.policies.factory  # noqa: F401
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401
    import safetensors.torch  # noqa: F401
    import mlx_smolvla.convert  # noqa: F401
    import torch  # noqa: F401
    import torch.utils.data._utils.collate  # noqa: F401
    import training.evaluation  # noqa: F401
    import training.reference_export  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401

    installed_runtime_recorded_paths = _installed_runtime_recorded_paths()
    for distribution_name in _T3B_RUNTIME_DISTRIBUTIONS:
        identity = f"{distribution_name}=={version(distribution_name)}".encode("utf-8")
        result[f"distribution/{distribution_name}/version"] = hashlib.sha256(
            identity
        ).hexdigest()
        result[f"distribution/{distribution_name}/installed-files"] = (
            _installed_distribution_payload_sha256(distribution_name)
        )
        result[f"distribution/{distribution_name}/package-inventory"] = (
            _installed_distribution_inventory_sha256(
                distribution_name,
                known_recorded_paths=installed_runtime_recorded_paths,
            )
        )
    runtime_provenance = _require_t3b_runtime_provenance(freeze=True)
    result["runtime-provenance/manifest"] = _canonical_json_sha256(
        runtime_provenance
    )
    modules = runtime_provenance.get("modules", {})
    if not isinstance(modules, Mapping):
        raise ValueError("runtime provenance module evidence is invalid")
    for module_name, evidence in sorted(modules.items()):
        if not isinstance(module_name, str) or not isinstance(evidence, list):
            raise ValueError("runtime provenance module evidence is invalid")
        result[f"runtime-provenance/module/{module_name}"] = (
            _canonical_json_sha256(evidence)
        )
    return result


def _validate_t3b_training_bridge_semantics(
    *,
    config: FineTuneConfig,
    bridge: TrainingDataBridge,
    stats: TrainStatistics,
    expected_frozen_inputs: Mapping[str, object],
) -> dict[str, object]:
    """Compare the live bridge with an independently reconstructed clean bridge."""

    live_evidence = bridge.semantic_evidence()
    live_state = bridge.state_dict()
    if live_state.get("samples_consumed") != 0:
        raise RuntimeError("training bridge was consumed before semantic validation")
    audit_bridge = TrainingDataBridge(
        cache_dir=config.cache_dir,
        episodes=bridge.episodes,
        sampler_seed=config.sampler_seed,
        stats=stats.processor_stats,
    )
    try:
        audit_evidence = audit_bridge.semantic_evidence()
        if live_evidence != audit_evidence:
            raise RuntimeError(
                "materialized training bridge differs from its clean reconstruction"
            )

        # Exercise the real video-decode and preprocessing path on the disposable
        # reconstruction before runtime provenance is frozen.  PyAV and related
        # libraries load behavior-relevant Python modules lazily on the first
        # decoded frame; the live bridge must remain fresh for update 1.
        audit_batch = audit_bridge.next_batch()
        if audit_batch is None:
            raise RuntimeError("training bridge audit returned an empty batch")
        del audit_batch
        after_audit_inputs = collect_t3b_frozen_input_evidence(
            config,
            None,
            validate_runtime_model=False,
        )
        _require_unchanged_t3b_inputs(
            expected_frozen_inputs,
            after_audit_inputs,
            context="training-bridge semantic reconstruction",
        )
        bridge.load_state_dict(live_state)
        reset_evidence = bridge.semantic_evidence()
        if reset_evidence != live_evidence:
            raise RuntimeError(
                "training bridge changed while its iterator was reconstructed"
            )
        return reset_evidence
    finally:
        del audit_bridge
        gc.collect()


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
class _CheckpointNamespaceEvidence:
    """Exact post-validation identities for all canonical retained checkpoints."""

    inventory: frozenset[str]
    directory_identities: Mapping[str, tuple[int, int]]
    file_snapshots: Mapping[str, Mapping[str, _StableFileSnapshot]]
    pointer_snapshot: _StableFileSnapshot


@dataclass(frozen=True)
class TrainingCheckpoint:
    """One fully published atomic training checkpoint."""

    path: Path
    state: CheckpointState
    metadata_sha256: str
    model_sha256: str
    optimizer_sha256: str
    pruned_checkpoints: tuple[str, ...] = ()
    namespace_evidence: _CheckpointNamespaceEvidence | None = None


@dataclass(frozen=True)
class _BoundCheckpointCandidate:
    """Retained authority for the checkpoint selected for live restoration."""

    name: str
    directory_descriptor: int
    directory_device: int
    directory_inode: int
    file_bindings: Mapping[str, tuple[_StableFileSnapshot, int]]


def _file_sha256(path: Path) -> str:
    return _snapshot_regular_file(path, label="hashed file").sha256


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


def _zero_step_checkpoint_recovery_inventory(
    output_dir: Path,
    *,
    output_descriptor: int | None = None,
    expected_output_snapshot: _DirectorySnapshot | None = None,
) -> tuple[str, ...] | None:
    """Return the exact safe recovery inventory, or ``None`` for unknown state."""

    output_dir = Path(os.path.abspath(output_dir))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if output_descriptor is None:
        output_snapshot = _snapshot_directory(
            output_dir,
            label="checkpoint recovery output",
        )
        bound_output = os.open(output_dir, directory_flags)
    else:
        if expected_output_snapshot is None:
            raise ValueError("bound recovery inventory requires an output snapshot")
        if output_dir != expected_output_snapshot.path:
            raise ValueError("recovery inventory output differs from its bound parent")
        output_snapshot = expected_output_snapshot
        bound_output = os.dup(output_descriptor)
    recovery_descriptor: int | None = None
    try:
        output_identity = os.fstat(bound_output)
        _, output_device, output_inode = output_snapshot.components[-1]
        if (output_identity.st_dev, output_identity.st_ino) != (
            output_device,
            output_inode,
        ):
            raise RuntimeError("checkpoint recovery output descriptor changed")
        try:
            recovery_identity = os.stat(
                "checkpoint-recoveries",
                dir_fd=bound_output,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return ()
        if not stat.S_ISDIR(recovery_identity.st_mode):
            return None
        recovery_descriptor = os.open(
            "checkpoint-recoveries",
            directory_flags,
            dir_fd=bound_output,
        )
        opened_recovery = os.fstat(recovery_descriptor)
        if (opened_recovery.st_dev, opened_recovery.st_ino) != (
            recovery_identity.st_dev,
            recovery_identity.st_ino,
        ):
            raise RuntimeError("checkpoint recovery root changed during open")
        recovery_root = _descriptor_path(recovery_descriptor)
        root_snapshot = _snapshot_directory(
            recovery_root,
            label="checkpoint recovery root",
        )
        directory_snapshots: list[_DirectorySnapshot] = []
        file_snapshots: list[tuple[str, _StableFileSnapshot]] = []
        recovered: list[str] = []
        for name in sorted(os.listdir(recovery_descriptor)):
            try:
                child = os.stat(
                    name,
                    dir_fd=recovery_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return None
            step_match = re.fullmatch(
                r"step-(\d{6})-"
                r"(partial|discarded|replaced|pruned|published|publication-failed)-"
                r"(\d{6})",
                name,
            )
            pointer_match = re.fullmatch(
                r"latest-pointer-"
                r"(partial|previous|published|publication-failed)-(\d{6})",
                name,
            )
            if step_match is not None:
                kind = step_match.group(2)
                expected_directory = kind != "replaced"
                if expected_directory and not stat.S_ISDIR(child.st_mode):
                    return None
                if not expected_directory and not (
                    stat.S_ISDIR(child.st_mode) or stat.S_ISREG(child.st_mode)
                ):
                    return None
            elif pointer_match is not None:
                if not stat.S_ISREG(child.st_mode):
                    return None
            else:
                return None
            if stat.S_ISDIR(child.st_mode):
                directory_snapshots.append(
                    _snapshot_directory(
                        recovery_root / name,
                        label="recovered checkpoint transaction",
                    )
                )
            else:
                file_snapshots.append(
                    (
                        name,
                        _snapshot_regular_file_at(
                            recovery_descriptor,
                            name,
                            label="recovered checkpoint transaction",
                        ),
                    )
                )
            recovered.append(f"checkpoint-recoveries/{name}")
        for snapshot in directory_snapshots:
            _revalidate_directory_snapshot(
                snapshot,
                label="recovered checkpoint transaction",
            )
        for name, snapshot in file_snapshots:
            current = _snapshot_regular_file_at(
                recovery_descriptor,
                name,
                label="recovered checkpoint transaction",
            )
            if not _same_bound_file_snapshot(snapshot, current):
                return None
        _revalidate_directory_snapshot(
            root_snapshot,
            label="checkpoint recovery root",
        )
        return tuple(recovered)
    finally:
        if recovery_descriptor is not None:
            os.close(recovery_descriptor)
        os.close(bound_output)


def _rename_entry_no_clobber_at(
    *,
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
    expected_device: int | None = None,
    expected_inode: int | None = None,
    expected_directory: bool | None = None,
) -> None:
    """Atomically move one direct child without replacement or symlink following."""

    if (expected_device is None) != (expected_inode is None):
        raise ValueError("source device and inode must be provided together")
    source = os.stat(
        source_name,
        dir_fd=source_descriptor,
        follow_symlinks=False,
    )
    if expected_device is not None and (source.st_dev, source.st_ino) != (
        expected_device,
        expected_inode,
    ):
        raise RuntimeError(f"staged entry changed before publication: {source_name}")
    if expected_directory is not None and stat.S_ISDIR(source.st_mode) != (
        expected_directory
    ):
        raise RuntimeError(f"staged entry type changed before publication: {source_name}")
    rename_excl = 0x00000004
    rename_nofollow_any = 0x00000010
    _renameatx_np(
        source_descriptor=source_descriptor,
        source_name=source_name,
        destination_descriptor=destination_descriptor,
        destination_name=destination_name,
        flags=rename_excl | rename_nofollow_any,
    )
    published = os.stat(
        destination_name,
        dir_fd=destination_descriptor,
        follow_symlinks=False,
    )
    identity_changed = (published.st_dev, published.st_ino) != (
        source.st_dev,
        source.st_ino,
    )
    type_changed = expected_directory is not None and stat.S_ISDIR(
        published.st_mode
    ) != expected_directory
    if identity_changed or type_changed:
        for _ in range(1_000_000):
            failed_name = (
                f".{destination_name}.publication-failed-{os.urandom(12).hex()}"
            )
            try:
                _renameatx_np(
                    source_descriptor=destination_descriptor,
                    source_name=destination_name,
                    destination_descriptor=destination_descriptor,
                    destination_name=failed_name,
                    flags=rename_excl | rename_nofollow_any,
                )
            except FileExistsError:
                continue
            quarantined = os.stat(
                failed_name,
                dir_fd=destination_descriptor,
                follow_symlinks=False,
            )
            if (quarantined.st_dev, quarantined.st_ino) != (
                published.st_dev,
                published.st_ino,
            ):
                raise RuntimeError(
                    f"published entry changed during quarantine: {destination_name}"
                )
            os.fsync(destination_descriptor)
            break
        else:
            raise RuntimeError("publication failure namespace is exhausted")
        if identity_changed:
            raise RuntimeError(
                f"staged entry changed during publication: {source_name}"
            )
        raise RuntimeError(f"published entry type changed: {destination_name}")


def _move_entry_to_unique_recovery_at(
    *,
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_prefix: str,
    expected_device: int,
    expected_inode: int,
    expected_directory: bool,
) -> str:
    """Move one bound entry into a numbered exclusive recovery name."""

    for index in range(1, 1_000_000):
        candidate = f"{destination_prefix}{index:06d}"
        try:
            _rename_entry_no_clobber_at(
                source_descriptor=source_descriptor,
                source_name=source_name,
                destination_descriptor=destination_descriptor,
                destination_name=candidate,
                expected_device=expected_device,
                expected_inode=expected_inode,
                expected_directory=expected_directory,
            )
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("recovery namespace is exhausted")


_T3B_RESUME_DIRECTORY_STAGES = {
    ".adapter-stage-": "adapter-stage-",
    ".source-checkpoint-": "source-checkpoint-",
    ".tokenizer-snapshot-": "tokenizer-snapshot-",
    ".export.": "export-stage-",
    ".export.publication-failed-": "export-publication-failed-",
}
_T3B_RESUME_FILE_STAGES = {
    ".run.json.": "run-json-partial-",
    ".training.pid.": "training-pid-partial-",
    ".metrics.csv.": "metrics-csv-partial-",
    ".training.log.": "training-log-partial-",
    ".budget.json.": "budget-json-partial-",
    ".launch.json.": "launch-json-partial-",
    ".run.json.previous-": "run-json-previous-",
    ".run.json.publication-failed-": "run-json-publication-failed-",
    ".metrics.csv.previous-": "metrics-csv-previous-",
    ".metrics.csv.publication-failed-": "metrics-csv-publication-failed-",
    ".training.pid.previous-": "training-pid-previous-",
    ".training.pid.publication-failed-": "training-pid-publication-failed-",
    ".training.log.previous-": "training-log-previous-",
    ".training.log.publication-failed-": "training-log-publication-failed-",
    ".budget.json.previous-": "budget-json-previous-",
    ".budget.json.publication-failed-": "budget-json-publication-failed-",
    ".launch.json.previous-": "launch-json-previous-",
    ".launch.json.publication-failed-": "launch-json-publication-failed-",
    ".adapter.safetensors.publication-failed-": (
        "adapter-safetensors-publication-failed-"
    ),
    ".adapter.json.publication-failed-": "adapter-json-publication-failed-",
}

_T3B_RESTART_STATE_DESTINATIONS = (
    "run.json",
    "training.pid",
    "metrics.csv",
    "training.log",
    "budget.json",
    "launch.json",
)


def _restore_missing_t3b_previous_generations(
    lease: _T3BTrainingLease,
) -> tuple[str, ...]:
    """Restore each unique durable prior generation whose canonical name is absent."""

    _revalidate_t3b_training_lock(lease)
    names = tuple(os.listdir(lease.output_descriptor))
    planned: list[tuple[str, str, _StableFileSnapshot]] = []
    for destination_name in _T3B_RESTART_STATE_DESTINATIONS:
        pattern = re.compile(
            rf"\.{re.escape(destination_name)}\.previous-[0-9a-f]{{24}}"
        )
        candidates = tuple(sorted(name for name in names if pattern.fullmatch(name)))
        try:
            current = os.stat(
                destination_name,
                dir_fd=lease.output_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            current = None
        if current is not None:
            continue
        if not candidates:
            continue
        if len(candidates) != 1:
            raise FileExistsError(
                "missing T3B state has ambiguous previous generations: "
                f"{destination_name}: {list(candidates)}"
            )
        previous_name = candidates[0]
        previous = _snapshot_regular_file_at(
            lease.output_descriptor,
            previous_name,
            label="T3B previous state generation",
            capture_payload=True,
        )
        planned.append((destination_name, previous_name, previous))

    restored: list[str] = []
    for destination_name, previous_name, previous in planned:
        current_previous = _snapshot_regular_file_at(
            lease.output_descriptor,
            previous_name,
            label="T3B previous state generation",
            capture_payload=True,
        )
        if not _same_bound_file_snapshot(previous, current_previous):
            raise RuntimeError(
                f"T3B previous generation changed before restoration: {previous_name}"
            )
        _rename_entry_no_clobber_at(
            source_descriptor=lease.output_descriptor,
            source_name=previous_name,
            destination_descriptor=lease.output_descriptor,
            destination_name=destination_name,
            expected_device=previous.device,
            expected_inode=previous.inode,
            expected_directory=False,
        )
        os.fsync(lease.output_descriptor)
        published = _snapshot_regular_file_at(
            lease.output_descriptor,
            destination_name,
            label="restored T3B state generation",
            capture_payload=True,
        )
        if not _same_bound_file_snapshot(previous, published):
            raise RuntimeError(
                f"restored T3B state generation changed: {destination_name}"
            )
        restored.append(destination_name)
    _revalidate_t3b_training_lock(lease)
    return tuple(restored)


def _metrics_recovery_inventory(
    lease: _T3BTrainingLease,
) -> tuple[str, ...]:
    """Bind the exact regular metrics-recovery namespace before resume mutation."""

    _revalidate_t3b_training_lock(lease)
    recovered: list[tuple[str, _StableFileSnapshot]] = []
    for name in sorted(os.listdir(lease.output_descriptor)):
        if not name.startswith("metrics.recovery-"):
            continue
        if re.fullmatch(r"metrics\.recovery-[0-9]{6}\.csv", name) is None:
            raise FileExistsError(f"T3B metrics recovery name is unsafe: {name}")
        recovered.append(
            (
                name,
                _snapshot_regular_file_at(
                    lease.output_descriptor,
                    name,
                    label="T3B metrics recovery",
                ),
            )
        )
    for name, snapshot in recovered:
        current = _snapshot_regular_file_at(
            lease.output_descriptor,
            name,
            label="T3B metrics recovery",
        )
        if not _same_bound_file_snapshot(snapshot, current):
            raise RuntimeError(f"T3B metrics recovery changed: {name}")
    _revalidate_t3b_training_lock(lease)
    return tuple(name for name, _ in recovered)


def _resume_output_recovery_inventory(
    lease: _T3BTrainingLease,
) -> tuple[str, ...] | None:
    """Return the exact safe inventory of preserved output-level transactions."""

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        named = os.stat(
            "resume-recoveries",
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return ()
    if not stat.S_ISDIR(named.st_mode):
        return None
    descriptor = os.open(
        "resume-recoveries",
        directory_flags,
        dir_fd=lease.output_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("T3B resume recovery root changed during open")
        root = _descriptor_path(descriptor)
        root_snapshot = _snapshot_directory(root, label="T3B resume recovery root")
        recovered: list[str] = []
        for name in sorted(os.listdir(descriptor)):
            directory_kind = next(
                (
                    prefix
                    for prefix in _T3B_RESUME_DIRECTORY_STAGES.values()
                    if re.fullmatch(re.escape(prefix) + r"\d{6}", name)
                ),
                None,
            )
            file_kind = next(
                (
                    prefix
                    for prefix in _T3B_RESUME_FILE_STAGES.values()
                    if re.fullmatch(re.escape(prefix) + r"\d{6}", name)
                ),
                None,
            )
            try:
                child = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return None
            if directory_kind is not None:
                if not stat.S_ISDIR(child.st_mode):
                    return None
                snapshot = _snapshot_directory(
                    root / name,
                    label="T3B recovered output transaction",
                )
                _revalidate_directory_snapshot(
                    snapshot,
                    label="T3B recovered output transaction",
                )
            elif file_kind is not None:
                if not stat.S_ISREG(child.st_mode):
                    return None
                _snapshot_regular_file_at(
                    descriptor,
                    name,
                    label="T3B recovered output transaction",
                )
            else:
                return None
            recovered.append(f"resume-recoveries/{name}")
        _revalidate_directory_snapshot(root_snapshot, label="T3B resume recovery root")
        return tuple(recovered)
    finally:
        os.close(descriptor)


def _reconcile_t3b_resume_output_staging(
    lease: _T3BTrainingLease,
) -> tuple[str, ...]:
    """Quarantine exact SIGKILL leftovers from output-level transactions."""

    _revalidate_t3b_training_lock(lease)
    existing = _resume_output_recovery_inventory(lease)
    if existing is None:
        raise FileExistsError("T3B resume recovery inventory is unsafe")
    output_root = _descriptor_path(lease.output_descriptor)
    staged: list[
        tuple[str, str, int, int, bool, _DirectorySnapshot | None, str | None]
    ] = []
    unexpected_hidden: list[str] = []
    for name in sorted(os.listdir(lease.output_descriptor)):
        if not name.startswith("."):
            continue
        matched: tuple[str, str, bool] | None = None
        for source_prefix, recovery_prefix in _T3B_RESUME_DIRECTORY_STAGES.items():
            suffix = name.removeprefix(source_prefix)
            if suffix != name and re.fullmatch(r"[0-9a-f]{24}", suffix):
                matched = (source_prefix, recovery_prefix, True)
                break
        if matched is None:
            for source_prefix, recovery_prefix in _T3B_RESUME_FILE_STAGES.items():
                suffix = name.removeprefix(source_prefix)
                if suffix != name and re.fullmatch(r"[0-9a-f]{24}", suffix):
                    matched = (source_prefix, recovery_prefix, False)
                    break
        if matched is None:
            unexpected_hidden.append(name)
            continue
        _, recovery_prefix, expected_directory = matched
        identity = os.stat(
            name,
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(identity.st_mode) != expected_directory:
            raise FileExistsError(f"T3B resume staging entry has unsafe type: {name}")
        if expected_directory:
            snapshot = _snapshot_directory(
                output_root / name,
                label="T3B interrupted output transaction",
            )
            device = snapshot.components[-1][1]
            inode = snapshot.components[-1][2]
            digest = None
        else:
            file_snapshot = _snapshot_regular_file_at(
                lease.output_descriptor,
                name,
                label="T3B interrupted output transaction",
            )
            snapshot = None
            device = file_snapshot.device
            inode = file_snapshot.inode
            digest = file_snapshot.sha256
        staged.append(
            (name, recovery_prefix, device, inode, expected_directory, snapshot, digest)
        )
    if unexpected_hidden:
        raise FileExistsError(
            f"T3B resume output has unexpected hidden entries: {unexpected_hidden}"
        )
    if not staged:
        return existing
    try:
        os.mkdir("resume-recoveries", mode=0o700, dir_fd=lease.output_descriptor)
    except FileExistsError:
        pass
    recovery_named = os.stat(
        "resume-recoveries",
        dir_fd=lease.output_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(recovery_named.st_mode):
        raise FileExistsError("T3B resume recovery root is unsafe")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    recovery_descriptor = os.open(
        "resume-recoveries",
        directory_flags,
        dir_fd=lease.output_descriptor,
    )
    try:
        recovery_opened = os.fstat(recovery_descriptor)
        if (recovery_opened.st_dev, recovery_opened.st_ino) != (
            recovery_named.st_dev,
            recovery_named.st_ino,
        ):
            raise RuntimeError("T3B resume recovery root changed during open")
        for name, prefix, device, inode, is_directory, snapshot, digest in staged:
            if snapshot is not None:
                _revalidate_directory_snapshot(
                    snapshot,
                    label="T3B interrupted output transaction",
                )
            recovered_name = _move_entry_to_unique_recovery_at(
                source_descriptor=lease.output_descriptor,
                source_name=name,
                destination_descriptor=recovery_descriptor,
                destination_prefix=prefix,
                expected_device=device,
                expected_inode=inode,
                expected_directory=is_directory,
            )
            moved = os.stat(
                recovered_name,
                dir_fd=recovery_descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISDIR(moved.st_mode) != is_directory
                or (moved.st_dev, moved.st_ino) != (device, inode)
            ):
                raise RuntimeError("T3B output transaction changed during recovery")
            if digest is not None:
                moved_snapshot = _snapshot_regular_file_at(
                    recovery_descriptor,
                    recovered_name,
                    label="T3B recovered output transaction",
                )
                if moved_snapshot.sha256 != digest:
                    raise RuntimeError("T3B recovered output bytes changed")
            os.fsync(recovery_descriptor)
            os.fsync(lease.output_descriptor)
    finally:
        os.close(recovery_descriptor)
    recovered = _resume_output_recovery_inventory(lease)
    if recovered is None:
        raise RuntimeError("T3B resume recovery inventory changed")
    _revalidate_t3b_training_lock(lease)
    return recovered


def _open_process_identity_recovery_root(
    lease: _T3BTrainingLease,
    *,
    create: bool,
) -> int | None:
    """Open the private directory that makes PID/run rotation restart-safe."""

    name = "process-identity-recoveries"
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        named = os.stat(
            name,
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        if not create:
            return None
        try:
            os.mkdir(name, mode=0o700, dir_fd=lease.output_descriptor)
        except FileExistsError:
            pass
        named = os.stat(
            name,
            dir_fd=lease.output_descriptor,
            follow_symlinks=False,
        )
        os.fsync(lease.output_descriptor)
    if not stat.S_ISDIR(named.st_mode):
        raise FileExistsError("process identity recovery root is unsafe")
    descriptor = os.open(
        name,
        directory_flags,
        dir_fd=lease.output_descriptor,
    )
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        os.close(descriptor)
        raise RuntimeError("process identity recovery root changed during open")
    return descriptor


def _process_identity_recovery_inventory(
    lease: _T3BTrainingLease,
) -> tuple[str, ...] | None:
    """Return the exact safe inventory of inert prior PID generations."""

    _revalidate_t3b_training_lock(lease)
    try:
        descriptor = _open_process_identity_recovery_root(lease, create=False)
    except FileExistsError:
        return None
    if descriptor is None:
        return ()
    try:
        recovered: list[tuple[str, _StableFileSnapshot]] = []
        for name in sorted(os.listdir(descriptor)):
            if (
                re.fullmatch(
                    r"training-pid-(?:previous|uncommitted)-[0-9]{6}",
                    name,
                )
                is None
            ):
                return None
            try:
                snapshot = _snapshot_regular_file_at(
                    descriptor,
                    name,
                    label="process identity recovery",
                )
            except (FileNotFoundError, OSError):
                return None
            recovered.append((name, snapshot))
        for name, snapshot in recovered:
            try:
                current = _snapshot_regular_file_at(
                    descriptor,
                    name,
                    label="process identity recovery",
                )
            except (FileNotFoundError, OSError):
                return None
            if not _same_bound_file_snapshot(snapshot, current):
                return None
        _revalidate_t3b_training_lock(lease)
        return tuple(
            f"process-identity-recoveries/{name}" for name, _ in recovered
        )
    finally:
        os.close(descriptor)


def _validate_recorded_t3b_recovery_inventories(
    run_document: Mapping[str, object],
    *,
    output_dir: Path,
    lease: _T3BTrainingLease,
) -> None:
    """Require every recorded inert recovery to remain a safe live artifact."""

    _revalidate_t3b_training_lock(lease)
    checkpoint_inventory = _zero_step_checkpoint_recovery_inventory(
        output_dir,
        output_descriptor=lease.output_descriptor,
        expected_output_snapshot=lease.output_snapshot,
    )
    startup_inventory = _startup_recovery_inventory(
        output_dir,
        output_descriptor=lease.output_descriptor,
        expected_output_snapshot=lease.output_snapshot,
    )
    resume_inventory = _resume_output_recovery_inventory(lease)
    process_inventory = _process_identity_recovery_inventory(lease)
    if any(
        inventory is None
        for inventory in (
            checkpoint_inventory,
            startup_inventory,
            resume_inventory,
            process_inventory,
        )
    ):
        raise FileExistsError("T3B recorded recovery inventory is unsafe")
    assert checkpoint_inventory is not None
    assert startup_inventory is not None
    assert resume_inventory is not None
    assert process_inventory is not None

    recorded_checkpoint = run_document.get("checkpoint_recoveries", [])
    recorded_startup = run_document.get("startup_recoveries", [])
    if not isinstance(recorded_checkpoint, list) or not isinstance(
        recorded_startup, list
    ):
        raise ValueError("recorded recovery inventories must be path lists")
    checkpoint_pattern = re.compile(
        r"checkpoint-recoveries/(?:"
        r"step-[0-9]{6}-(?:partial|discarded|replaced|pruned|published|publication-failed)-[0-9]{6}"
        r"|latest-pointer-(?:partial|previous|published|publication-failed)-[0-9]{6}"
        r")"
    )
    startup_pattern = re.compile(
        r"(?:"
        r"startup-recoveries/(?:budget-json-partial|run-json-partial|training-pid-partial|training-log-partial|training-log-prestart|training-pid-prestart)-[0-9]{6}"
        r"|resume-recoveries/[a-z0-9-]+-[0-9]{6}"
        r"|process-identity-recoveries/training-pid-(?:previous|uncommitted)-[0-9]{6}"
        r")"
    )
    if (
        any(
            not isinstance(path, str)
            or checkpoint_pattern.fullmatch(path) is None
            for path in recorded_checkpoint
        )
        or len(set(recorded_checkpoint)) != len(recorded_checkpoint)
    ):
        raise ValueError("recorded checkpoint recovery path is invalid")
    if (
        any(
            not isinstance(path, str) or startup_pattern.fullmatch(path) is None
            for path in recorded_startup
        )
        or len(set(recorded_startup)) != len(recorded_startup)
    ):
        raise ValueError("recorded startup recovery path is invalid")

    missing_checkpoint = set(recorded_checkpoint).difference(checkpoint_inventory)
    if missing_checkpoint:
        raise FileNotFoundError(
            "recorded checkpoint recoveries are missing: "
            f"{sorted(missing_checkpoint)}"
        )
    live_startup = set(startup_inventory) | set(resume_inventory) | set(
        process_inventory
    )
    missing_startup = set(recorded_startup).difference(live_startup)
    if missing_startup:
        raise FileNotFoundError(
            "recorded startup recoveries are missing: " f"{sorted(missing_startup)}"
        )
    _revalidate_t3b_training_lock(lease)


def _backup_resume_process_identity(
    lease: _T3BTrainingLease,
    expected: _StableFileSnapshot,
) -> str:
    """Atomically preserve the prior PID generation before publishing a new one."""

    _revalidate_t3b_training_lock(lease)
    current = _snapshot_regular_file_at(
        lease.output_descriptor,
        "training.pid",
        label="fine-tune process identity",
        capture_payload=True,
    )
    if not _same_bound_file_snapshot(current, expected):
        raise RuntimeError("fine-tune process identity changed before rotation")
    recovery_descriptor = _open_process_identity_recovery_root(lease, create=True)
    assert recovery_descriptor is not None
    try:
        name = _move_entry_to_unique_recovery_at(
            source_descriptor=lease.output_descriptor,
            source_name="training.pid",
            destination_descriptor=recovery_descriptor,
            destination_prefix="training-pid-previous-",
            expected_device=expected.device,
            expected_inode=expected.inode,
            expected_directory=False,
        )
        os.fsync(recovery_descriptor)
        os.fsync(lease.output_descriptor)
        preserved = _snapshot_regular_file_at(
            recovery_descriptor,
            name,
            label="preserved prior process identity",
            capture_payload=True,
        )
        if (
            preserved.device != expected.device
            or preserved.inode != expected.inode
            or preserved.size != expected.size
            or preserved.mtime_ns != expected.mtime_ns
            or preserved.sha256 != expected.sha256
        ):
            raise RuntimeError("preserved prior process identity changed during rotation")
        return name
    finally:
        os.close(recovery_descriptor)


def _reconcile_resume_process_identity(
    lease: _T3BTrainingLease,
    run_snapshot: _StableFileSnapshot,
) -> tuple[str, ...]:
    """Restore the PID generation still referenced by an unchanged run.json."""

    run_document = _json_from_stable_snapshot(
        run_snapshot,
        label="T3B run metadata",
    )
    if not isinstance(run_document, Mapping):
        raise ValueError("T3B run metadata must be an object")
    process = run_document.get("process")
    if (
        not isinstance(process, Mapping)
        or process.get("identity_file") != "training.pid"
        or not _is_lowercase_sha256(process.get("identity_sha256"))
    ):
        raise ValueError("resumable run has no valid process identity pointer")
    expected_digest = process["identity_sha256"]
    try:
        current = _snapshot_regular_file_at(
            lease.output_descriptor,
            "training.pid",
            label="fine-tune process identity",
            capture_payload=True,
        )
    except FileNotFoundError:
        current = None
    if current is not None and current.sha256 == expected_digest:
        return ()

    recovery_descriptor = _open_process_identity_recovery_root(lease, create=False)
    if recovery_descriptor is None:
        raise ValueError("resumable run process identity differs without recovery")
    moved: list[str] = []
    try:
        matching_name: str | None = None
        matching_snapshot: _StableFileSnapshot | None = None
        for name in sorted(os.listdir(recovery_descriptor)):
            if Path(name).name != name:
                raise ValueError("process identity recovery name is unsafe")
            try:
                candidate = _snapshot_regular_file_at(
                    recovery_descriptor,
                    name,
                    label="recovered process identity",
                    capture_payload=True,
                )
            except (FileNotFoundError, OSError) as error:
                raise ValueError("process identity recovery entry is unsafe") from error
            if candidate.sha256 == expected_digest:
                matching_name = name
                matching_snapshot = candidate
                break
        if matching_name is None or matching_snapshot is None:
            raise ValueError("prior bound process identity cannot be recovered")
        if current is not None:
            moved_name = _move_entry_to_unique_recovery_at(
                source_descriptor=lease.output_descriptor,
                source_name="training.pid",
                destination_descriptor=recovery_descriptor,
                destination_prefix="training-pid-uncommitted-",
                expected_device=current.device,
                expected_inode=current.inode,
                expected_directory=False,
            )
            preserved = _snapshot_regular_file_at(
                recovery_descriptor,
                moved_name,
                label="preserved uncommitted process identity",
                capture_payload=True,
            )
            if (
                preserved.device != current.device
                or preserved.inode != current.inode
                or preserved.size != current.size
                or preserved.mtime_ns != current.mtime_ns
                or preserved.sha256 != current.sha256
            ):
                raise RuntimeError(
                    "uncommitted process identity changed during recovery"
                )
            moved.append(f"process-identity-recoveries/{moved_name}")
        _rename_entry_no_clobber_at(
            source_descriptor=recovery_descriptor,
            source_name=matching_name,
            destination_descriptor=lease.output_descriptor,
            destination_name="training.pid",
            expected_device=matching_snapshot.device,
            expected_inode=matching_snapshot.inode,
            expected_directory=False,
        )
        os.fsync(recovery_descriptor)
        os.fsync(lease.output_descriptor)
        restored = _snapshot_regular_file_at(
            lease.output_descriptor,
            "training.pid",
            label="restored process identity",
            capture_payload=True,
        )
        if restored.sha256 != expected_digest:
            raise RuntimeError("restored process identity differs from run.json")
        _revalidate_t3b_training_lock(lease)
        return tuple(moved)
    finally:
        os.close(recovery_descriptor)


def _prepare_zero_step_checkpoint_replay(
    checkpoint_root: Path,
    *,
    output_dir: Path,
    output_descriptor: int | None = None,
    expected_output_snapshot: _DirectorySnapshot | None = None,
    checkpoint_root_descriptor: int | None = None,
    expected_checkpoint_root_snapshot: _DirectorySnapshot | None = None,
    expected_staging_step: int = 1,
    allow_published_entries: bool = False,
    allowed_published_steps: Collection[int] = (),
) -> tuple[str, ...] | None:
    """Reconcile recovery evidence and quarantine one expected partial save step.

    ``None`` means either namespace contains state that cannot be proven safe for a
    replay. Missing and empty roots are eligible. At step zero, the only accepted
    checkpoint-root entries are unpublished step-1 staging directories. For a later
    resume, already-published non-staging entries may be retained while exactly the
    next expected staging step is quarantined. Every partial is moved intact with a
    no-clobber/no-follow rename, and the complete inventory is returned on retries.
    """

    if expected_staging_step <= 0:
        raise ValueError("expected checkpoint staging step must be positive")
    allowed_published_step_set = frozenset(allowed_published_steps)
    if any(type(step) is not int or step <= 0 for step in allowed_published_step_set):
        raise ValueError("allowed published checkpoint steps are invalid")
    checkpoint_root = Path(os.path.abspath(checkpoint_root))
    output_dir = Path(os.path.abspath(output_dir))
    if checkpoint_root.parent != output_dir:
        raise ValueError("zero-step checkpoint root must be directly under output_dir")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if output_descriptor is None:
        output_snapshot = _snapshot_directory(
            output_dir,
            label="zero-step output directory",
        )
        bound_output = os.open(output_dir, directory_flags)
    else:
        if expected_output_snapshot is None:
            raise ValueError("bound zero-step replay requires an output snapshot")
        if output_dir != expected_output_snapshot.path:
            raise ValueError("zero-step output differs from its bound directory")
        output_snapshot = expected_output_snapshot
        bound_output = os.dup(output_descriptor)
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        output_identity = os.fstat(bound_output)
        _, output_device, output_inode = output_snapshot.components[-1]
        if (output_identity.st_dev, output_identity.st_ino) != (
            output_device,
            output_inode,
        ):
            raise RuntimeError("zero-step output descriptor changed")
        existing_recoveries = _zero_step_checkpoint_recovery_inventory(
            output_dir,
            output_descriptor=bound_output,
            expected_output_snapshot=output_snapshot,
        )
        if existing_recoveries is None:
            return None
        if checkpoint_root_descriptor is None:
            try:
                checkpoint_identity = os.stat(
                    checkpoint_root.name,
                    dir_fd=bound_output,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return existing_recoveries
            if not stat.S_ISDIR(checkpoint_identity.st_mode):
                return None
            source_descriptor = os.open(
                checkpoint_root.name,
                directory_flags,
                dir_fd=bound_output,
            )
            source_identity = os.fstat(source_descriptor)
            if (source_identity.st_dev, source_identity.st_ino) != (
                checkpoint_identity.st_dev,
                checkpoint_identity.st_ino,
            ):
                raise RuntimeError("zero-step checkpoint root changed during open")
            checkpoint_root = _descriptor_path(source_descriptor)
            root_snapshot = _snapshot_directory(
                checkpoint_root,
                label="zero-step checkpoint root",
            )
        else:
            if expected_checkpoint_root_snapshot is None:
                raise ValueError("bound zero-step replay requires a root snapshot")
            source_descriptor = os.dup(checkpoint_root_descriptor)
            source_identity = os.fstat(source_descriptor)
            _, root_device, root_inode = (
                expected_checkpoint_root_snapshot.components[-1]
            )
            if (source_identity.st_dev, source_identity.st_ino) != (
                root_device,
                root_inode,
            ):
                raise RuntimeError("zero-step checkpoint root descriptor changed")
            checkpoint_root = _descriptor_path(source_descriptor)
            root_snapshot = expected_checkpoint_root_snapshot
        transactions: list[
            tuple[
                str,
                str,
                int,
                int,
                bool,
                _DirectorySnapshot | None,
                _StableFileSnapshot | None,
            ]
        ] = []
        has_published_entries = False
        scheduled_published_checkpoint = False
        canonical_published_pointer: _StableFileSnapshot | None = None
        recovery_names = {Path(value).name for value in existing_recoveries}
        recovered_published_checkpoint = any(
            re.fullmatch(
                rf"step-{expected_staging_step:06d}-published-[0-9]{{6}}",
                name,
            )
            is not None
            for name in recovery_names
        )
        for name in sorted(os.listdir(source_descriptor)):
            try:
                entry_identity = os.stat(
                    name,
                    dir_fd=source_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return None
            partial_match = re.fullmatch(r"\.step-(\d{6})\.([0-9a-f]{24})", name)
            discarded_match = re.fullmatch(
                r"\.discarded-step-(\d{6})-(\d{6})", name
            )
            replaced_match = re.fullmatch(
                r"\.recovery-step-(\d{6})-(\d{6})", name
            )
            pruned_match = re.fullmatch(r"\.pruned-step-(\d{6})-(\d{6})", name)
            pointer_match = re.fullmatch(r"\.latest\.json\.([0-9a-f]{24})", name)
            publication_failed_match = re.fullmatch(
                r"\.step-(\d{6})\.publication-failed-([0-9a-f]{24})",
                name,
            )
            pointer_previous_match = re.fullmatch(
                r"\.latest\.json\.previous-([0-9a-f]{24})",
                name,
            )
            pointer_failed_match = re.fullmatch(
                r"\.latest\.json\.publication-failed-([0-9a-f]{24})",
                name,
            )
            recovery_kind: str | None = None
            recovery_step: int | None = None
            expected_directory: bool | None = None
            if partial_match is not None:
                recovery_kind = "partial"
                recovery_step = int(partial_match.group(1))
                expected_directory = True
                if recovery_step != expected_staging_step:
                    return None
            elif discarded_match is not None:
                recovery_kind = "discarded"
                recovery_step = int(discarded_match.group(1))
                expected_directory = True
                if recovery_step != expected_staging_step:
                    return None
            elif replaced_match is not None:
                recovery_kind = "replaced"
                recovery_step = int(replaced_match.group(1))
                expected_directory = stat.S_ISDIR(entry_identity.st_mode)
                if recovery_step != expected_staging_step or not (
                    expected_directory or stat.S_ISREG(entry_identity.st_mode)
                ):
                    return None
            elif pruned_match is not None:
                recovery_kind = "pruned"
                recovery_step = int(pruned_match.group(1))
                expected_directory = True
                if recovery_step <= 0 or recovery_step >= expected_staging_step:
                    return None
            elif pointer_match is not None:
                recovery_kind = "pointer"
                expected_directory = False
            elif publication_failed_match is not None:
                recovery_kind = "publication-failed"
                recovery_step = int(publication_failed_match.group(1))
                expected_directory = True
                if recovery_step != expected_staging_step:
                    return None
            elif pointer_previous_match is not None:
                recovery_kind = "pointer-previous"
                expected_directory = False
            elif pointer_failed_match is not None:
                recovery_kind = "pointer-publication-failed"
                expected_directory = False
            else:
                published_step = re.fullmatch(r"step-(\d{6})", name)
                if published_step is not None and not allow_published_entries:
                    recovery_step = int(published_step.group(1))
                    if recovery_step != expected_staging_step:
                        return None
                    recovery_kind = "published"
                    expected_directory = True
                    directory_snapshot = _snapshot_directory(
                        checkpoint_root / name,
                        label="interrupted published checkpoint",
                    )
                    expected_device = directory_snapshot.components[-1][1]
                    expected_inode = directory_snapshot.components[-1][2]
                    transactions.append(
                        (
                            name,
                            f"step-{recovery_step:06d}-published-",
                            expected_device,
                            expected_inode,
                            True,
                            directory_snapshot,
                            None,
                        )
                    )
                    scheduled_published_checkpoint = True
                    continue
                if (
                    name == "latest.json"
                    and stat.S_ISREG(entry_identity.st_mode)
                    and not allow_published_entries
                ):
                    pointer_snapshot = _snapshot_regular_file_at(
                        source_descriptor,
                        name,
                        label="unrecorded published checkpoint pointer",
                        capture_payload=True,
                    )
                    pointer_document = _json_from_stable_snapshot(
                        pointer_snapshot,
                        label="unrecorded published checkpoint pointer",
                    )
                    if (
                        not isinstance(pointer_document, Mapping)
                        or set(pointer_document)
                        != {
                            "format_version",
                            "checkpoint",
                            "completed_step",
                            "metadata_sha256",
                        }
                        or pointer_document["format_version"] != 1
                        or pointer_document["checkpoint"]
                        != f"step-{expected_staging_step:06d}"
                        or pointer_document["completed_step"]
                        != expected_staging_step
                        or not _is_lowercase_sha256(
                            pointer_document["metadata_sha256"]
                        )
                    ):
                        return None
                    canonical_published_pointer = pointer_snapshot
                    continue
                if (
                    (name == "latest.json" and stat.S_ISREG(entry_identity.st_mode))
                    or (
                        published_step is not None
                        and stat.S_ISDIR(entry_identity.st_mode)
                    )
                ):
                    if (
                        published_step is not None
                        and allow_published_entries
                        and allowed_published_step_set
                        and int(published_step.group(1))
                        not in allowed_published_step_set
                    ):
                        return None
                    has_published_entries = True
                    if allow_published_entries:
                        continue
                    continue
                return None
            assert recovery_kind is not None and expected_directory is not None
            if stat.S_ISDIR(entry_identity.st_mode) != expected_directory:
                return None
            directory_snapshot = (
                _snapshot_directory(
                    checkpoint_root / name,
                    label="interrupted checkpoint transaction",
                )
                if expected_directory
                else None
            )
            file_snapshot: _StableFileSnapshot | None = None
            if not expected_directory:
                file_snapshot = _snapshot_regular_file_at(
                    source_descriptor,
                    name,
                    label="interrupted checkpoint transaction",
                )
                expected_device = file_snapshot.device
                expected_inode = file_snapshot.inode
            else:
                assert directory_snapshot is not None
                expected_device = directory_snapshot.components[-1][1]
                expected_inode = directory_snapshot.components[-1][2]
            destination_prefix = (
                "latest-pointer-partial-"
                if recovery_kind == "pointer"
                else (
                    "latest-pointer-previous-"
                    if recovery_kind == "pointer-previous"
                    else (
                        "latest-pointer-publication-failed-"
                        if recovery_kind == "pointer-publication-failed"
                        else f"step-{recovery_step:06d}-{recovery_kind}-"
                    )
                )
            )
            transactions.append(
                (
                    name,
                    destination_prefix,
                    expected_device,
                    expected_inode,
                    expected_directory,
                    directory_snapshot,
                    file_snapshot,
                )
            )
        if canonical_published_pointer is not None:
            if not (
                scheduled_published_checkpoint or recovered_published_checkpoint
            ):
                return None
            transactions.append(
                (
                    "latest.json",
                    "latest-pointer-published-",
                    canonical_published_pointer.device,
                    canonical_published_pointer.inode,
                    False,
                    None,
                    canonical_published_pointer,
                )
            )
        _revalidate_directory_snapshot(
            root_snapshot,
            label="zero-step checkpoint root",
        )
        if not transactions:
            return (
                None
                if has_published_entries and not allow_published_entries
                else existing_recoveries
            )

        try:
            os.mkdir("checkpoint-recoveries", mode=0o700, dir_fd=bound_output)
        except FileExistsError:
            pass
        recovery_identity = os.stat(
            "checkpoint-recoveries",
            dir_fd=bound_output,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(recovery_identity.st_mode):
            return None
        destination_descriptor = os.open(
            "checkpoint-recoveries",
            directory_flags,
            dir_fd=bound_output,
        )
        destination_identity = os.fstat(destination_descriptor)
        if (destination_identity.st_dev, destination_identity.st_ino) != (
            recovery_identity.st_dev,
            recovery_identity.st_ino,
        ):
            raise RuntimeError("checkpoint recovery root changed during open")
        recovery_root = _descriptor_path(destination_descriptor)
        recovery_root_snapshot = _snapshot_directory(
            recovery_root,
            label="checkpoint recovery root",
        )
        for (
            entry_name,
            destination_prefix,
            entry_device,
            entry_inode,
            expected_directory,
            entry_snapshot,
            entry_file_snapshot,
        ) in transactions:
            if entry_snapshot is not None:
                _revalidate_directory_snapshot(
                    entry_snapshot,
                    label="interrupted checkpoint transaction",
                )
            if entry_file_snapshot is not None:
                current_file_snapshot = _snapshot_regular_file_at(
                    source_descriptor,
                    entry_name,
                    label="interrupted checkpoint transaction",
                    capture_payload=entry_file_snapshot.payload is not None,
                )
                if not _same_bound_file_snapshot(
                    entry_file_snapshot,
                    current_file_snapshot,
                ):
                    raise RuntimeError(
                        "checkpoint transaction changed before quarantine"
                    )
            destination_name = _move_entry_to_unique_recovery_at(
                source_descriptor=source_descriptor,
                source_name=entry_name,
                destination_descriptor=destination_descriptor,
                destination_prefix=destination_prefix,
                expected_device=entry_device,
                expected_inode=entry_inode,
                expected_directory=expected_directory,
            )
            moved = os.stat(
                destination_name,
                dir_fd=destination_descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISDIR(moved.st_mode) != expected_directory
                or (moved.st_dev, moved.st_ino) != (entry_device, entry_inode)
            ):
                raise RuntimeError("checkpoint transaction changed while quarantined")
            os.fsync(destination_descriptor)
            os.fsync(source_descriptor)
        _revalidate_directory_snapshot(
            root_snapshot,
            label="zero-step checkpoint root",
        )
        _revalidate_directory_snapshot(
            recovery_root_snapshot,
            label="checkpoint recovery root",
        )
        recovered = _zero_step_checkpoint_recovery_inventory(
            output_dir,
            output_descriptor=bound_output,
            expected_output_snapshot=output_snapshot,
        )
        return recovered
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)
        os.close(bound_output)


def _next_checkpoint_staging_step(
    run_document: Mapping[str, object],
    *,
    selected_steps: int,
    checkpoint_interval: int,
) -> int | None:
    """Derive the only checkpoint step that an interrupted resume may stage."""

    if selected_steps <= 0 or checkpoint_interval <= 0:
        raise ValueError("checkpoint staging trajectory is invalid")
    last_checkpoint = run_document.get("last_checkpoint")
    checkpoint_count = run_document.get("checkpoint_count")
    if last_checkpoint is None:
        if checkpoint_count != 0:
            raise ValueError("checkpoint count exists without a last checkpoint")
        return 1
    if not isinstance(last_checkpoint, Mapping):
        raise ValueError("last checkpoint metadata must be an object")
    step = last_checkpoint.get("step")
    if type(step) is not int or not 0 < step <= selected_steps:
        raise ValueError("last checkpoint step is invalid")
    if step == selected_steps:
        return None
    if step != 1 and step % checkpoint_interval != 0:
        raise ValueError("last checkpoint step is outside the save cadence")
    next_step = checkpoint_interval if step == 1 else step + checkpoint_interval
    return min(next_step, selected_steps)


def _allowed_t3b_published_checkpoint_steps(
    run_document: Mapping[str, object],
    *,
    selected_steps: int,
    checkpoint_interval: int,
) -> frozenset[int]:
    """Return retained recorded steps plus the sole permitted crash candidate."""

    last_checkpoint = run_document.get("last_checkpoint")
    if not isinstance(last_checkpoint, Mapping):
        return frozenset()
    last_step = last_checkpoint.get("step")
    if type(last_step) is not int:
        raise ValueError("last checkpoint step is invalid")
    cadence = {1}
    cadence.update(range(checkpoint_interval, last_step + 1, checkpoint_interval))
    if last_step == selected_steps:
        cadence.add(last_step)
    if last_step not in cadence:
        raise ValueError("last checkpoint step is outside the save cadence")
    retained = tuple(sorted(cadence)[-3:])
    next_step = _next_checkpoint_staging_step(
        run_document,
        selected_steps=selected_steps,
        checkpoint_interval=checkpoint_interval,
    )
    return frozenset(retained if next_step is None else (*retained, next_step))


def prune_training_checkpoints(
    checkpoint_root: str | Path,
    *,
    keep_last: int = 3,
    expected_run_config_sha256: str,
    trainable_names: tuple[str, ...],
    expected_model_tensors: Mapping[str, mx.array],
    expected_optimizer_tensors: Mapping[str, mx.array],
    expected_checkpoint_root_snapshot: _DirectorySnapshot | None = None,
    checkpoint_root_descriptor: int | None = None,
    expected_root_inventory: Collection[str] | None = None,
    allowed_recovery_entries: Collection[str] = (),
) -> tuple[str, ...]:
    """Remove only older complete checkpoints belonging to the active run."""

    if keep_last <= 0:
        raise ValueError("checkpoint retention must be positive")
    checkpoint_root = Path(os.path.abspath(Path(checkpoint_root).expanduser()))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if expected_checkpoint_root_snapshot is None:
        checkpoint_root_snapshot = _snapshot_directory(
            checkpoint_root,
            label="training checkpoint root",
        )
    else:
        checkpoint_root_snapshot = expected_checkpoint_root_snapshot
    root_descriptor = (
        os.open(checkpoint_root, directory_flags)
        if checkpoint_root_descriptor is None
        else os.dup(checkpoint_root_descriptor)
    )
    try:
        root_identity = os.fstat(root_descriptor)
        _, root_device, root_inode = checkpoint_root_snapshot.components[-1]
        if (root_identity.st_dev, root_identity.st_ino) != (
            root_device,
            root_inode,
        ):
            raise RuntimeError("training checkpoint root changed before pruning")
        allowed_recoveries = frozenset(allowed_recovery_entries)
        if any(
            re.fullmatch(r"\.recovery-step-[0-9]{6}-[0-9]{6}", name) is None
            for name in allowed_recoveries
        ):
            raise ValueError("allowed checkpoint recovery namespace is invalid")
        root_inventory = (
            set(os.listdir(root_descriptor))
            if expected_root_inventory is None
            else set(expected_root_inventory)
        )
        _require_exact_directory_inventory_at(
            root_descriptor,
            root_inventory,
            label="checkpoint retention root",
        )
        checkpoint_root = _descriptor_path(root_descriptor)
        bound_root_snapshot = _snapshot_directory(
            checkpoint_root,
            label="bound training checkpoint root",
        )
        complete: list[tuple[int, Path, _DirectorySnapshot]] = []
        for name in os.listdir(root_descriptor):
            if Path(name).name != name:
                continue
            path = checkpoint_root / name
            step = _checkpoint_directory_step(path)
            if step is None:
                if name == "latest.json":
                    _snapshot_regular_file_at(
                        root_descriptor,
                        name,
                        label="checkpoint latest pointer",
                    )
                    continue
                if name in allowed_recoveries:
                    recovery = os.stat(
                        name,
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(recovery.st_mode):
                        raise ValueError(
                            f"checkpoint recovery candidate is unsafe: {name}"
                        )
                    continue
                raise ValueError(
                    f"checkpoint namespace contains an unsafe candidate: {name}"
                )
            try:
                path_snapshot = _snapshot_directory(
                    path,
                    label="checkpoint selected for retention",
                )
                _read_checkpoint_directory(
                    path,
                    expected_run_config_sha256=expected_run_config_sha256,
                    trainable_names=trainable_names,
                    expected_model_tensors=expected_model_tensors,
                    expected_optimizer_tensors=expected_optimizer_tensors,
                    checkpoint_root_descriptor=root_descriptor,
                    expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                )
                _revalidate_directory_snapshot(
                    path_snapshot,
                    label="checkpoint selected for retention",
                )
            except Exception as error:
                raise ValueError(
                    f"checkpoint namespace contains an invalid candidate: {name}"
                ) from error
            complete.append((step, path, path_snapshot))
        complete.sort()
        removed: list[str] = []
        for _, path, path_snapshot in complete[:-keep_last]:
            _require_exact_directory_inventory_at(
                root_descriptor,
                root_inventory,
                label="checkpoint retention root",
            )
            _revalidate_directory_snapshot(
                path_snapshot,
                label="checkpoint selected for retention",
            )
            quarantine_name = _move_entry_to_unique_recovery_at(
                source_descriptor=root_descriptor,
                source_name=path.name,
                destination_descriptor=root_descriptor,
                destination_prefix=f".pruned-{path.name}-",
                expected_device=path_snapshot.components[-1][1],
                expected_inode=path_snapshot.components[-1][2],
                expected_directory=True,
            )
            root_inventory.remove(path.name)
            root_inventory.add(quarantine_name)
            _require_exact_directory_inventory_at(
                root_descriptor,
                root_inventory,
                label="checkpoint retention root",
            )
            os.fsync(root_descriptor)
            quarantine = checkpoint_root / quarantine_name
            quarantine_snapshot = _snapshot_directory(
                quarantine,
                label="quarantined pruned checkpoint",
            )
            if quarantine_snapshot.components[-1][1:] != (
                path_snapshot.components[-1][1],
                path_snapshot.components[-1][2],
            ):
                raise RuntimeError("checkpoint changed while it was quarantined")
            _revalidate_directory_snapshot(
                quarantine_snapshot,
                label="quarantined pruned checkpoint",
            )
            quarantine_identity = os.stat(
                quarantine_name,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(quarantine_identity.st_mode)
                or (quarantine_identity.st_dev, quarantine_identity.st_ino)
                != (
                    quarantine_snapshot.components[-1][1],
                    quarantine_snapshot.components[-1][2],
                )
            ):
                raise RuntimeError("quarantined checkpoint changed before removal")
            if not shutil.rmtree.avoids_symlink_attacks:
                raise RuntimeError("safe checkpoint pruning requires fd-based rmtree")
            shutil.rmtree(quarantine_name, dir_fd=root_descriptor)
            root_inventory.remove(quarantine_name)
            _require_exact_directory_inventory_at(
                root_descriptor,
                root_inventory,
                label="checkpoint retention root",
            )
            removed.append(path.name)
        if removed:
            os.fsync(root_descriptor)
        root_after = os.fstat(root_descriptor)
        if (root_after.st_dev, root_after.st_ino) != (root_device, root_inode):
            raise RuntimeError("training checkpoint root changed during pruning")
        _require_exact_directory_inventory_at(
            root_descriptor,
            root_inventory,
            label="checkpoint retention root",
        )
        _revalidate_directory_snapshot(
            bound_root_snapshot,
            label="bound training checkpoint root",
        )
        return tuple(removed)
    finally:
        os.close(root_descriptor)


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
    update_fields = {
        "loss",
        "learning_rate",
        "gradient_norm",
        "clip_coefficient",
        "seconds",
    }
    if not isinstance(last_update, Mapping) or set(last_update) != update_fields:
        raise ValueError("checkpoint last update must be an object")
    integer_fields = (
        "completed_step",
        "selected_steps",
        "peak_memory_bytes",
        "samples_consumed",
        "flow_draw_count",
    )
    if any(type(value[name]) is not int for name in integer_fields):
        raise ValueError("checkpoint integer state fields must be JSON integers")
    float_fields = (
        "smoothed_loss",
        "elapsed_training_seconds",
    )
    if any(type(value[name]) is not float for name in float_fields) or any(
        type(last_update[name]) is not float for name in update_fields
    ):
        raise ValueError("checkpoint scalar state fields must be JSON floats")
    if not _is_lowercase_sha256(value["run_config_sha256"]):
        raise ValueError("checkpoint run-config digest is invalid")
    state = CheckpointState(
        completed_step=value["completed_step"],
        selected_steps=value["selected_steps"],
        smoothed_loss=value["smoothed_loss"],
        elapsed_training_seconds=value["elapsed_training_seconds"],
        peak_memory_bytes=value["peak_memory_bytes"],
        samples_consumed=value["samples_consumed"],
        flow_draw_count=value["flow_draw_count"],
        last_update=UpdateResult(
            loss=last_update["loss"],
            learning_rate=last_update["learning_rate"],
            gradient_norm=last_update["gradient_norm"],
            clip_coefficient=last_update["clip_coefficient"],
            seconds=last_update["seconds"],
        ),
        run_config_sha256=value["run_config_sha256"],
    )
    if not 0 < state.completed_step <= state.selected_steps:
        raise ValueError("checkpoint completed step is outside its training horizon")
    if state.elapsed_training_seconds <= 0:
        raise ValueError("checkpoint elapsed training time must be positive")
    if (
        state.peak_memory_bytes < 0
        or state.samples_consumed < 0
        or state.flow_draw_count < 0
    ):
        raise ValueError("checkpoint memory/draw/sample counts must be nonnegative")
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
    if (
        state.smoothed_loss < 0
        or state.last_update.loss < 0
        or state.last_update.learning_rate < 0
        or state.last_update.gradient_norm < 0
        or not 0 <= state.last_update.clip_coefficient <= 1
        or state.last_update.seconds <= 0
    ):
        raise ValueError("checkpoint scalar state is physically invalid")
    return state


def _require_exact_checkpoint_tensor_values(
    actual: Mapping[str, mx.array],
    expected: Mapping[str, mx.array],
    *,
    kind: str,
) -> None:
    """Require exact tensor values without weakening resume discovery semantics."""

    if set(actual) != set(expected):
        raise RuntimeError(f"checkpoint {kind} tensor set changed")
    mx.eval(actual, expected)
    for name in expected:
        observed = actual[name]
        wanted = expected[name]
        if (
            observed.shape != wanted.shape
            or observed.dtype != wanted.dtype
            or not bool(mx.array_equal(observed, wanted))
        ):
            raise RuntimeError(
                f"checkpoint {kind} tensor value changed for {name}"
            )


def _validate_checkpoint_tensor_file_values(
    path: Path,
    expected: Mapping[str, mx.array],
    *,
    kind: str,
) -> None:
    """Load stable serialized bytes and compare them exactly with live values."""

    with _private_stable_file(
        path,
        label=f"staged checkpoint {kind} tensor file",
    ) as (private_file, _):
        with private_file.open_reader() as handle:
            loaded = mx.load(handle, format="safetensors")
            mx.eval(loaded)
        _require_exact_checkpoint_tensor_values(loaded, expected, kind=kind)


def _read_checkpoint_directory(
    path: Path,
    *,
    expected_run_config_sha256: str | None = None,
    trainable_names: tuple[str, ...] | None = None,
    expected_model_tensors: Mapping[str, mx.array] | None = None,
    expected_optimizer_tensors: Mapping[str, mx.array] | None = None,
    checkpoint_root_descriptor: int | None = None,
    expected_checkpoint_root_snapshot: _DirectorySnapshot | None = None,
) -> tuple[TrainingCheckpoint, dict[str, mx.array], dict[str, mx.array]]:
    """Validate one complete checkpoint directory without mutating live state."""

    path = Path(path)
    directory_step = _checkpoint_directory_step(path)
    if directory_step is None:
        raise ValueError(f"not a complete checkpoint directory name: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if checkpoint_root_descriptor is None:
        directory_snapshot = _snapshot_directory(path, label="checkpoint directory")
        directory_descriptor = os.open(path, directory_flags)
        named_directory = None
    else:
        if expected_checkpoint_root_snapshot is None:
            raise ValueError("bound checkpoint read requires a root snapshot")
        root_identity = os.fstat(checkpoint_root_descriptor)
        _, root_device, root_inode = expected_checkpoint_root_snapshot.components[-1]
        if (root_identity.st_dev, root_identity.st_ino) != (
            root_device,
            root_inode,
        ):
            raise RuntimeError("checkpoint root descriptor changed before candidate read")
        named_directory = os.stat(
            path.name,
            dir_fd=checkpoint_root_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(named_directory.st_mode):
            raise ValueError(f"checkpoint candidate is not a directory: {path.name}")
        directory_descriptor = os.open(
            path.name,
            directory_flags,
            dir_fd=checkpoint_root_descriptor,
        )
        directory_snapshot = _snapshot_directory(
            _descriptor_path(directory_descriptor),
            label="checkpoint directory",
        )
    try:
        opened_directory = os.fstat(directory_descriptor)
        _, directory_device, directory_inode = directory_snapshot.components[-1]
        if (opened_directory.st_dev, opened_directory.st_ino) != (
            directory_device,
            directory_inode,
        ):
            raise RuntimeError("checkpoint directory changed during open")
        if named_directory is not None and (
            opened_directory.st_dev,
            opened_directory.st_ino,
        ) != (named_directory.st_dev, named_directory.st_ino):
            raise RuntimeError("checkpoint candidate changed during bound open")
        expected_inventory = {
            "metadata.json",
            "model.safetensors",
            "optimizer.safetensors",
        }
        actual_inventory = set(os.listdir(directory_descriptor))
        if actual_inventory != expected_inventory:
            raise ValueError(
                "checkpoint directory inventory differs from the frozen schema: "
                f"{sorted(actual_inventory)}"
            )
        for name in expected_inventory:
            child = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(child.st_mode):
                raise ValueError(f"checkpoint child is not a regular file: {name}")
        metadata_snapshot = _snapshot_regular_file_at(
            directory_descriptor,
            "metadata.json",
            label="checkpoint metadata",
            capture_payload=True,
        )
        metadata = _json_from_stable_snapshot(
            metadata_snapshot,
            label="checkpoint metadata",
        )
        metadata_sha256 = metadata_snapshot.sha256
        if not isinstance(metadata, Mapping):
            raise ValueError("checkpoint metadata must be an object")
        if set(metadata) != {
            "format_version",
            "artifact_type",
            "state",
            "model",
            "optimizer",
        }:
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
            raise ValueError(
                "checkpoint was produced by a different training configuration"
            )

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
                raise ValueError(
                    f"checkpoint {kind} metadata fields differ from the schema"
                )
            if section["file"] != expected_file:
                raise ValueError(f"checkpoint {kind} filename is invalid")
            names = tuple(str(name) for name in section["tensor_names"])
            specs = section["tensor_specs"]
            if not isinstance(specs, Mapping) or set(specs) != set(names):
                raise ValueError(
                    f"checkpoint {kind} tensor specs differ from its names"
                )
            with _private_stable_file(
                Path(expected_file),
                label=f"checkpoint {kind} tensor file",
                source_parent_descriptor=directory_descriptor,
            ) as (private_file, digest):
                if digest != section["sha256"]:
                    raise ValueError(f"checkpoint {kind} tensor digest is invalid")
                with private_file.open_reader() as handle:
                    loaded = mx.load(handle, format="safetensors")
                    mx.eval(loaded)
                if set(loaded) != set(names):
                    raise ValueError(
                        f"checkpoint {kind} tensor set differs from its metadata"
                    )
                loaded = {name: loaded[name] for name in names}
                if expected_names is not None and names != expected_names:
                    raise ValueError(
                        f"checkpoint {kind} tensor names differ from the current schema"
                    )
                if expected_tensors is not None and set(loaded) != set(
                    expected_tensors
                ):
                    raise ValueError(
                        f"checkpoint {kind} tensor set differs from the current schema"
                    )
                for name in names:
                    spec = specs[name]
                    if not isinstance(spec, Mapping) or set(spec) != {
                        "shape",
                        "dtype",
                    }:
                        raise ValueError(
                            f"checkpoint {kind} tensor spec is invalid for {name}"
                        )
                    if list(loaded[name].shape) != spec["shape"] or str(
                        loaded[name].dtype
                    ) != spec["dtype"]:
                        raise ValueError(
                            f"checkpoint {kind} tensor differs from metadata for {name}"
                        )
                    if expected_tensors is not None:
                        expected = expected_tensors[name]
                        if (
                            loaded[name].shape != expected.shape
                            or loaded[name].dtype != expected.dtype
                        ):
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
        if "step" not in loaded_optimizer or int(
            loaded_optimizer["step"]
        ) != state.completed_step:
            raise ValueError(
                "checkpoint optimizer internal step differs from completed step"
            )
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
        if set(os.listdir(directory_descriptor)) != expected_inventory:
            raise RuntimeError("checkpoint directory inventory changed while in use")
        opened_after = os.fstat(directory_descriptor)
        if (opened_after.st_dev, opened_after.st_ino) != (
            directory_device,
            directory_inode,
        ):
            raise RuntimeError("checkpoint directory descriptor changed while in use")
        if checkpoint_root_descriptor is None:
            _revalidate_directory_snapshot(
                directory_snapshot,
                label="checkpoint directory",
            )
        else:
            named_after = os.stat(
                path.name,
                dir_fd=checkpoint_root_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(named_after.st_mode)
                or (named_after.st_dev, named_after.st_ino)
                != (directory_device, directory_inode)
            ):
                raise RuntimeError("checkpoint candidate changed while it was in use")
        return checkpoint, loaded_model, loaded_optimizer
    finally:
        os.close(directory_descriptor)


def _revalidate_bound_checkpoint_candidate(
    authority: _BoundCheckpointCandidate,
    *,
    checkpoint_root_descriptor: int,
    expected_checkpoint_root_snapshot: _DirectorySnapshot,
    verify_bytes: bool,
    entry_name: str | None = None,
) -> None:
    """Require the selected checkpoint to remain the same named directory tree."""

    bound_name = authority.name if entry_name is None else entry_name
    root = os.fstat(checkpoint_root_descriptor)
    _, root_device, root_inode = expected_checkpoint_root_snapshot.components[-1]
    if (root.st_dev, root.st_ino) != (root_device, root_inode):
        raise RuntimeError("checkpoint root changed while candidate was bound")
    try:
        named = os.stat(
            bound_name,
            dir_fd=checkpoint_root_descriptor,
            follow_symlinks=False,
        )
        opened = os.fstat(authority.directory_descriptor)
    except OSError as error:
        raise RuntimeError("selected checkpoint changed while bound") from error
    expected_directory = (authority.directory_device, authority.directory_inode)
    if (
        not stat.S_ISDIR(named.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or (named.st_dev, named.st_ino) != expected_directory
        or (opened.st_dev, opened.st_ino) != expected_directory
    ):
        raise RuntimeError("selected checkpoint changed while bound")
    if set(os.listdir(authority.directory_descriptor)) != set(
        authority.file_bindings
    ):
        raise RuntimeError("selected checkpoint inventory changed while bound")
    for name, (snapshot, descriptor) in authority.file_bindings.items():
        _revalidate_bound_regular_file_at(
            authority.directory_descriptor,
            name,
            expected=snapshot,
            descriptor=descriptor,
            label="selected checkpoint file",
            verify_bytes=verify_bytes,
        )
    try:
        named_after = os.stat(
            bound_name,
            dir_fd=checkpoint_root_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise RuntimeError("selected checkpoint changed while bound") from error
    if (
        not stat.S_ISDIR(named_after.st_mode)
        or (named_after.st_dev, named_after.st_ino) != expected_directory
    ):
        raise RuntimeError("selected checkpoint changed while bound")


@contextmanager
def _bound_checkpoint_namespace_candidate(
    name: str,
    *,
    checkpoint_root_descriptor: int,
    expected_checkpoint_root_snapshot: _DirectorySnapshot,
) -> Iterator[_BoundCheckpointCandidate]:
    """Hold one canonical checkpoint-shaped tree without trusting its contents."""

    if re.fullmatch(r"step-[0-9]{6}", name) is None:
        raise ValueError("checkpoint authority name is invalid")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    named = os.stat(
        name,
        dir_fd=checkpoint_root_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(named.st_mode):
        raise RuntimeError("selected checkpoint is not a directory")
    directory_descriptor = os.open(
        name,
        directory_flags,
        dir_fd=checkpoint_root_descriptor,
    )
    bindings: dict[str, tuple[_StableFileSnapshot, int]] = {}
    try:
        opened = os.fstat(directory_descriptor)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("selected checkpoint changed during authority binding")
        expected_inventory = {
            "metadata.json",
            "model.safetensors",
            "optimizer.safetensors",
        }
        if set(os.listdir(directory_descriptor)) != expected_inventory:
            raise RuntimeError("selected checkpoint inventory changed before binding")
        for child_name in sorted(expected_inventory):
            bindings[child_name] = _open_bound_regular_file_at(
                directory_descriptor,
                child_name,
                label="selected checkpoint file",
            )
        authority = _BoundCheckpointCandidate(
            name=name,
            directory_descriptor=directory_descriptor,
            directory_device=opened.st_dev,
            directory_inode=opened.st_ino,
            file_bindings=bindings,
        )
        _revalidate_bound_checkpoint_candidate(
            authority,
            checkpoint_root_descriptor=checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=expected_checkpoint_root_snapshot,
            verify_bytes=True,
        )
        yield authority
    finally:
        for _, descriptor in bindings.values():
            os.close(descriptor)
        os.close(directory_descriptor)


@contextmanager
def _bound_checkpoint_candidate(
    checkpoint: TrainingCheckpoint,
    *,
    checkpoint_root_descriptor: int,
    expected_checkpoint_root_snapshot: _DirectorySnapshot,
) -> Iterator[_BoundCheckpointCandidate]:
    """Hold the selected checkpoint and all three files through restoration."""

    name = checkpoint.path.name
    if name != f"step-{checkpoint.state.completed_step:06d}":
        raise ValueError("selected checkpoint name differs from its state")
    with _bound_checkpoint_namespace_candidate(
        name,
        checkpoint_root_descriptor=checkpoint_root_descriptor,
        expected_checkpoint_root_snapshot=expected_checkpoint_root_snapshot,
    ) as authority:
        expected_hashes = {
            "metadata.json": checkpoint.metadata_sha256,
            "model.safetensors": checkpoint.model_sha256,
            "optimizer.safetensors": checkpoint.optimizer_sha256,
        }
        if {
            child_name: snapshot.sha256
            for child_name, (snapshot, _) in authority.file_bindings.items()
        } != expected_hashes:
            raise RuntimeError("selected checkpoint bytes changed before binding")
        yield authority


def _capture_checkpoint_namespace_evidence(
    *,
    checkpoint_root_descriptor: int,
    expected_checkpoint_root_snapshot: _DirectorySnapshot,
    expected_checkpoints: Mapping[str, TrainingCheckpoint],
    latest_checkpoint: TrainingCheckpoint,
    allowed_other_entries: Collection[str] = (),
) -> _CheckpointNamespaceEvidence:
    """Digest-bind every semantically validated retained checkpoint and pointer."""

    expected_names = frozenset(expected_checkpoints)
    if not expected_names or latest_checkpoint.path.name not in expected_names:
        raise ValueError("checkpoint namespace evidence has no latest checkpoint")
    if any(re.fullmatch(r"step-[0-9]{6}", name) is None for name in expected_names):
        raise ValueError("checkpoint namespace evidence has invalid step names")
    allowed = frozenset(allowed_other_entries)
    actual_inventory = frozenset(os.listdir(checkpoint_root_descriptor))
    canonical_inventory = expected_names | {"latest.json"}
    if actual_inventory != canonical_inventory | allowed:
        raise RuntimeError(
            "checkpoint namespace differs from its validated retained trajectory"
        )
    directory_identities: dict[str, tuple[int, int]] = {}
    file_snapshots: dict[str, dict[str, _StableFileSnapshot]] = {}
    for name in sorted(expected_names):
        with _bound_checkpoint_namespace_candidate(
            name,
            checkpoint_root_descriptor=checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=expected_checkpoint_root_snapshot,
        ) as authority:
            checkpoint = expected_checkpoints[name]
            expected_hashes = {
                "metadata.json": checkpoint.metadata_sha256,
                "model.safetensors": checkpoint.model_sha256,
                "optimizer.safetensors": checkpoint.optimizer_sha256,
            }
            snapshots = {
                child_name: snapshot
                for child_name, (snapshot, _) in authority.file_bindings.items()
            }
            if {
                child_name: snapshot.sha256
                for child_name, snapshot in snapshots.items()
            } != expected_hashes:
                raise RuntimeError(
                    f"retained checkpoint bytes changed after validation: {name}"
                )
            directory_identities[name] = (
                authority.directory_device,
                authority.directory_inode,
            )
            file_snapshots[name] = snapshots
    pointer_snapshot = _snapshot_regular_file_at(
        checkpoint_root_descriptor,
        "latest.json",
        label="checkpoint latest pointer",
        capture_payload=True,
    )
    pointer_document = _json_from_stable_snapshot(
        pointer_snapshot,
        label="checkpoint latest pointer",
    )
    if pointer_document != {
        "format_version": 1,
        "checkpoint": latest_checkpoint.path.name,
        "completed_step": latest_checkpoint.state.completed_step,
        "metadata_sha256": latest_checkpoint.metadata_sha256,
    }:
        raise ValueError("checkpoint latest pointer differs from retained trajectory")
    if frozenset(os.listdir(checkpoint_root_descriptor)) != actual_inventory:
        raise RuntimeError("checkpoint namespace changed during evidence capture")
    return _CheckpointNamespaceEvidence(
        inventory=canonical_inventory,
        directory_identities=directory_identities,
        file_snapshots=file_snapshots,
        pointer_snapshot=pointer_snapshot,
    )


def _require_checkpoint_namespace_evidence(
    evidence: _CheckpointNamespaceEvidence,
    *,
    checkpoint_root_descriptor: int,
    expected_checkpoint_root_snapshot: _DirectorySnapshot,
    verify_bytes: bool,
) -> None:
    """Require the named root to still match exact previously captured evidence."""

    _require_exact_directory_inventory_at(
        checkpoint_root_descriptor,
        evidence.inventory,
        label="checkpoint retained namespace",
    )
    for name in sorted(evidence.directory_identities):
        with _bound_checkpoint_namespace_candidate(
            name,
            checkpoint_root_descriptor=checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=expected_checkpoint_root_snapshot,
        ) as authority:
            if (
                authority.directory_device,
                authority.directory_inode,
            ) != evidence.directory_identities[name]:
                raise RuntimeError(f"retained checkpoint changed after save: {name}")
            expected_files = evidence.file_snapshots[name]
            for child_name, (snapshot, descriptor) in authority.file_bindings.items():
                if not _same_bound_file_snapshot(snapshot, expected_files[child_name]):
                    raise RuntimeError(
                        f"retained checkpoint file changed after save: {name}/{child_name}"
                    )
                _revalidate_bound_regular_file_at(
                    authority.directory_descriptor,
                    child_name,
                    expected=expected_files[child_name],
                    descriptor=descriptor,
                    label="retained checkpoint file",
                    verify_bytes=verify_bytes,
                )
    pointer_snapshot, pointer_descriptor = _open_bound_regular_file_at(
        checkpoint_root_descriptor,
        "latest.json",
        label="checkpoint latest pointer",
    )
    try:
        if not _same_bound_file_snapshot(
            pointer_snapshot,
            evidence.pointer_snapshot,
        ):
            raise RuntimeError("checkpoint latest pointer changed after save")
        _revalidate_bound_regular_file_at(
            checkpoint_root_descriptor,
            "latest.json",
            expected=evidence.pointer_snapshot,
            descriptor=pointer_descriptor,
            label="checkpoint latest pointer",
            verify_bytes=verify_bytes,
        )
    finally:
        os.close(pointer_descriptor)


def _persist_run_state_with_checkpoint_binding(
    *,
    checkpoint: TrainingCheckpoint,
    checkpoint_root_descriptor: int,
    expected_checkpoint_root_snapshot: _DirectorySnapshot,
    persist: Callable[[Path, object], str],
    path: Path,
    value: object,
) -> str:
    """Keep the exact saved checkpoint authoritative across its run-state CAS."""

    namespace_evidence = checkpoint.namespace_evidence
    if namespace_evidence is None:
        raise RuntimeError("saved checkpoint has no retained namespace evidence")
    _require_checkpoint_namespace_evidence(
        namespace_evidence,
        checkpoint_root_descriptor=checkpoint_root_descriptor,
        expected_checkpoint_root_snapshot=expected_checkpoint_root_snapshot,
        verify_bytes=True,
    )
    root_inventory = frozenset(os.listdir(checkpoint_root_descriptor))
    for name in root_inventory:
        if name == "latest.json":
            _snapshot_regular_file_at(
                checkpoint_root_descriptor,
                name,
                label="checkpoint latest pointer",
            )
            continue
        if re.fullmatch(r"step-[0-9]{6}", name) is None:
            raise RuntimeError(
                f"checkpoint namespace is unsafe before run-state publication: {name}"
            )
    with _bound_checkpoint_candidate(
        checkpoint,
        checkpoint_root_descriptor=checkpoint_root_descriptor,
        expected_checkpoint_root_snapshot=expected_checkpoint_root_snapshot,
    ) as authority:
        _require_exact_directory_inventory_at(
            checkpoint_root_descriptor,
            root_inventory,
            label="checkpoint run-state publication root",
        )
        _revalidate_bound_checkpoint_candidate(
            authority,
            checkpoint_root_descriptor=checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=expected_checkpoint_root_snapshot,
            verify_bytes=True,
        )
        digest = persist(path, value)
        _require_checkpoint_namespace_evidence(
            namespace_evidence,
            checkpoint_root_descriptor=checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=expected_checkpoint_root_snapshot,
            verify_bytes=True,
        )
        _revalidate_bound_checkpoint_candidate(
            authority,
            checkpoint_root_descriptor=checkpoint_root_descriptor,
            expected_checkpoint_root_snapshot=expected_checkpoint_root_snapshot,
            verify_bytes=True,
        )
        _require_exact_directory_inventory_at(
            checkpoint_root_descriptor,
            root_inventory,
            label="checkpoint run-state publication root",
        )
        return digest


def _revalidate_t3b_final_checkpoint_namespace(
    lease: _T3BTrainingLease,
    *,
    verify_bytes: bool = False,
) -> None:
    """Keep every retained checkpoint and pointer fixed through final publication."""

    root_descriptor = lease.checkpoint_root_descriptor
    root_snapshot = lease.checkpoint_root_snapshot
    inventory = lease.final_checkpoint_inventory
    authorities = lease.final_checkpoint_bindings
    pointer_binding = lease.final_checkpoint_pointer_binding
    if (
        root_descriptor is None
        or root_snapshot is None
        or inventory is None
        or authorities is None
        or pointer_binding is None
        or lease.final_checkpoint_binding_stack is None
    ):
        raise RuntimeError("final checkpoint namespace is not bound")
    _revalidate_t3b_checkpoint_root(lease)
    try:
        _require_exact_directory_inventory_at(
            root_descriptor,
            inventory,
            label="final checkpoint namespace",
        )
        for authority in authorities.values():
            _revalidate_bound_checkpoint_candidate(
                authority,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=root_snapshot,
                verify_bytes=verify_bytes,
            )
        pointer_snapshot, pointer_descriptor = pointer_binding
        _revalidate_bound_regular_file_at(
            root_descriptor,
            "latest.json",
            expected=pointer_snapshot,
            descriptor=pointer_descriptor,
            label="final checkpoint pointer",
            verify_bytes=verify_bytes,
        )
        _require_exact_directory_inventory_at(
            root_descriptor,
            inventory,
            label="final checkpoint namespace",
        )
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise RuntimeError("final checkpoint namespace changed while bound") from error


def _bind_t3b_final_checkpoint_namespace(
    lease: _T3BTrainingLease,
    *,
    checkpoint: TrainingCheckpoint,
) -> None:
    """Retain all final checkpoint children and the pointer until run.json commits."""

    if lease.final_checkpoint_binding_stack is not None:
        raise RuntimeError("final checkpoint namespace is already bound")
    root_descriptor = lease.checkpoint_root_descriptor
    root_snapshot = lease.checkpoint_root_snapshot
    if root_descriptor is None or root_snapshot is None:
        raise RuntimeError("T3B checkpoint root is not bound")
    evidence = checkpoint.namespace_evidence
    if evidence is None:
        raise RuntimeError("final checkpoint has no retained namespace evidence")
    _revalidate_t3b_checkpoint_root(lease)
    _require_checkpoint_namespace_evidence(
        evidence,
        checkpoint_root_descriptor=root_descriptor,
        expected_checkpoint_root_snapshot=root_snapshot,
        verify_bytes=True,
    )
    inventory = evidence.inventory
    step_names = tuple(sorted(evidence.directory_identities))
    if (
        not step_names
        or len(step_names) > 3
        or any(re.fullmatch(r"step-[0-9]{6}", name) is None for name in step_names)
        or checkpoint.path.name not in step_names
        or inventory != frozenset((*step_names, "latest.json"))
        or frozenset(evidence.file_snapshots) != frozenset(step_names)
    ):
        raise RuntimeError("final checkpoint retained trajectory is invalid")

    stack = ExitStack()
    authorities: dict[str, _BoundCheckpointCandidate] = {}
    pointer_binding: tuple[_StableFileSnapshot, int] | None = None
    try:
        for name in step_names:
            authority = stack.enter_context(
                _bound_checkpoint_namespace_candidate(
                    name,
                    checkpoint_root_descriptor=root_descriptor,
                    expected_checkpoint_root_snapshot=root_snapshot,
                )
            )
            if (
                authority.directory_device,
                authority.directory_inode,
            ) != evidence.directory_identities[name]:
                raise RuntimeError(
                    f"retained checkpoint changed before final binding: {name}"
                )
            expected_files = evidence.file_snapshots[name]
            if frozenset(expected_files) != frozenset(authority.file_bindings):
                raise RuntimeError(
                    f"retained checkpoint inventory changed before final binding: {name}"
                )
            if any(
                not _same_bound_file_snapshot(snapshot, expected_files[child_name])
                for child_name, (snapshot, _) in authority.file_bindings.items()
            ):
                raise RuntimeError(
                    f"retained checkpoint file changed before final binding: {name}"
                )
            authorities[name] = authority
        pointer_binding = _open_bound_regular_file_at(
            root_descriptor,
            "latest.json",
            label="final checkpoint pointer",
            capture_payload=True,
        )
        stack.callback(os.close, pointer_binding[1])
        if not _same_bound_file_snapshot(
            pointer_binding[0],
            evidence.pointer_snapshot,
        ):
            raise RuntimeError("final checkpoint pointer changed before binding")
        final_authority = authorities[checkpoint.path.name]
        expected_hashes = {
            "metadata.json": checkpoint.metadata_sha256,
            "model.safetensors": checkpoint.model_sha256,
            "optimizer.safetensors": checkpoint.optimizer_sha256,
        }
        actual_hashes = {
            name: snapshot.sha256
            for name, (snapshot, _) in final_authority.file_bindings.items()
        }
        if actual_hashes != expected_hashes:
            raise RuntimeError("final checkpoint bytes changed before binding")
        pointer_snapshot, _ = pointer_binding
        pointer_document = _json_from_stable_snapshot(
            pointer_snapshot,
            label="final checkpoint pointer",
        )
        if pointer_document != {
            "format_version": 1,
            "checkpoint": checkpoint.path.name,
            "completed_step": checkpoint.state.completed_step,
            "metadata_sha256": checkpoint.metadata_sha256,
        }:
            raise ValueError("final checkpoint pointer differs from the checkpoint")
        lease.final_checkpoint_binding_stack = stack
        lease.final_checkpoint_bindings = authorities
        lease.final_checkpoint_pointer_binding = pointer_binding
        lease.final_checkpoint_inventory = inventory
        _revalidate_t3b_final_checkpoint_namespace(lease, verify_bytes=True)
    except BaseException:
        if lease.final_checkpoint_binding_stack is stack:
            lease.final_checkpoint_binding_stack = None
            lease.final_checkpoint_bindings = None
            lease.final_checkpoint_pointer_binding = None
            lease.final_checkpoint_inventory = None
        stack.close()
        raise


def _write_latest_checkpoint_pointer(
    checkpoint_root: Path,
    checkpoint: TrainingCheckpoint,
    *,
    expected_checkpoint_root_snapshot: _DirectorySnapshot | None = None,
    checkpoint_root_descriptor: int | None = None,
    expected_pointer_snapshot: _StableFileSnapshot | None | object = (
        _UNSPECIFIED_STATE_DESTINATION
    ),
    expected_root_inventory: Collection[str] | None = None,
) -> None:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if checkpoint_root_descriptor is None:
        root_snapshot = (
            _snapshot_directory(checkpoint_root, label="training checkpoint root")
            if expected_checkpoint_root_snapshot is None
            else expected_checkpoint_root_snapshot
        )
        root_descriptor = os.open(checkpoint_root, directory_flags)
    else:
        if expected_checkpoint_root_snapshot is None:
            raise ValueError("bound checkpoint pointer requires a root snapshot")
        root_snapshot = expected_checkpoint_root_snapshot
        root_descriptor = os.dup(checkpoint_root_descriptor)
    try:
        opened = os.fstat(root_descriptor)
        _, expected_device, expected_inode = root_snapshot.components[-1]
        if (opened.st_dev, opened.st_ino) != (expected_device, expected_inode):
            raise RuntimeError("training checkpoint root changed before pointer write")
        pointer_root_inventory = (
            None
            if expected_root_inventory is None
            else set(expected_root_inventory)
        )
        if pointer_root_inventory is not None:
            _require_exact_directory_inventory_at(
                root_descriptor,
                pointer_root_inventory,
                label="checkpoint pointer root",
            )
        bound_root = _descriptor_path(root_descriptor)
        bound_snapshot = _snapshot_directory(
            bound_root,
            label="bound training checkpoint root",
        )
        if expected_pointer_snapshot is _UNSPECIFIED_STATE_DESTINATION:
            try:
                pointer_binding: _StableFileSnapshot | None = (
                    _snapshot_regular_file_at(
                        root_descriptor,
                        "latest.json",
                        label="checkpoint latest pointer",
                    )
                )
            except FileNotFoundError:
                pointer_binding = None
        else:
            assert expected_pointer_snapshot is None or isinstance(
                expected_pointer_snapshot,
                _StableFileSnapshot,
            )
            pointer_binding = expected_pointer_snapshot
        _write_run_state_with_binding(
            bound_root / "latest.json",
            {
                "format_version": 1,
                "checkpoint": checkpoint.path.name,
                "completed_step": checkpoint.state.completed_step,
                "metadata_sha256": checkpoint.metadata_sha256,
            },
            parent_descriptor=root_descriptor,
            expected_parent_snapshot=bound_snapshot,
            expected_destination_snapshot=pointer_binding,
        )
        os.fsync(root_descriptor)
        if pointer_root_inventory is not None:
            pointer_root_inventory.add("latest.json")
            _require_exact_directory_inventory_at(
                root_descriptor,
                pointer_root_inventory,
                label="checkpoint pointer root",
            )
        after = os.fstat(root_descriptor)
        if (after.st_dev, after.st_ino) != (expected_device, expected_inode):
            raise RuntimeError("training checkpoint root changed during pointer write")
    finally:
        os.close(root_descriptor)


def save_training_checkpoint(
    *,
    model: nn.Module,
    optimizer: SmolVLAAdamW,
    checkpoint_root: str | Path,
    state: CheckpointState,
    trainable_names: tuple[str, ...],
    keep_last: int = 3,
    checkpoint_parent_descriptor: int | None = None,
    expected_checkpoint_parent_snapshot: _DirectorySnapshot | None = None,
    checkpoint_root_descriptor: int | None = None,
    expected_checkpoint_root_snapshot: _DirectorySnapshot | None = None,
    expected_existing_checkpoint_steps: Collection[int] | None = None,
) -> TrainingCheckpoint:
    """Atomically publish model, optimizer, and exact continuation state."""

    if optimizer.step_index != state.completed_step:
        raise ValueError(
            f"optimizer/checkpoint step mismatch: {optimizer.step_index} != {state.completed_step}"
        )
    checkpoint_root = Path(os.path.abspath(Path(checkpoint_root).expanduser()))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if checkpoint_root_descriptor is not None:
        if checkpoint_parent_descriptor is not None:
            raise ValueError("checkpoint save received both root and parent descriptors")
        if expected_checkpoint_root_snapshot is None:
            raise ValueError("bound checkpoint save requires a root snapshot")
        root_descriptor = os.dup(checkpoint_root_descriptor)
        opened_root = os.fstat(root_descriptor)
        _, root_device, root_inode = expected_checkpoint_root_snapshot.components[-1]
        if (opened_root.st_dev, opened_root.st_ino) != (
            root_device,
            root_inode,
        ):
            os.close(root_descriptor)
            raise RuntimeError("training checkpoint root descriptor changed")
        checkpoint_root = _descriptor_path(root_descriptor)
        checkpoint_root_snapshot = expected_checkpoint_root_snapshot
    elif checkpoint_parent_descriptor is None:
        checkpoint_root_snapshot = _ensure_safe_directory(
            checkpoint_root,
            label="training checkpoint root",
        )
        root_descriptor = os.open(checkpoint_root, directory_flags)
    else:
        if expected_checkpoint_parent_snapshot is None:
            raise ValueError("bound checkpoint save requires a parent snapshot")
        if checkpoint_root.parent != expected_checkpoint_parent_snapshot.path:
            raise ValueError("checkpoint root differs from its bound parent")
        parent_identity = os.fstat(checkpoint_parent_descriptor)
        _, parent_device, parent_inode = (
            expected_checkpoint_parent_snapshot.components[-1]
        )
        if (parent_identity.st_dev, parent_identity.st_ino) != (
            parent_device,
            parent_inode,
        ):
            raise RuntimeError("checkpoint parent descriptor changed")
        try:
            os.mkdir(
                checkpoint_root.name,
                mode=0o700,
                dir_fd=checkpoint_parent_descriptor,
            )
        except FileExistsError:
            pass
        named_root = os.stat(
            checkpoint_root.name,
            dir_fd=checkpoint_parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(named_root.st_mode):
            raise FileExistsError("training checkpoint root is unsafe")
        root_descriptor = os.open(
            checkpoint_root.name,
            directory_flags,
            dir_fd=checkpoint_parent_descriptor,
        )
        opened_root = os.fstat(root_descriptor)
        if (opened_root.st_dev, opened_root.st_ino) != (
            named_root.st_dev,
            named_root.st_ino,
        ):
            os.close(root_descriptor)
            raise RuntimeError("training checkpoint root changed during open")
        checkpoint_root = _descriptor_path(root_descriptor)
        checkpoint_root_snapshot = _snapshot_directory(
            checkpoint_root,
            label="training checkpoint root",
        )
    target = checkpoint_root / f"step-{state.completed_step:06d}"
    model_tensors = dict(tree_flatten(model.trainable_parameters()))
    if tuple(model_tensors) != trainable_names:
        raise ValueError("checkpoint model tensor names differ from the trainable contract")
    optimizer_tensors = dict(tree_flatten(optimizer.state))
    if not optimizer_tensors:
        raise ValueError("cannot checkpoint an uninitialized optimizer")
    optimizer.validate_state_for(model.trainable_parameters())
    mx.eval(model_tensors, optimizer_tensors)

    root_identity = os.fstat(root_descriptor)
    _, root_device, root_inode = checkpoint_root_snapshot.components[-1]
    if (root_identity.st_dev, root_identity.st_ino) != (
        root_device,
        root_inode,
    ):
        os.close(root_descriptor)
        raise RuntimeError("training checkpoint root changed before save")
    root_inventory = set(os.listdir(root_descriptor))
    existing_checkpoint_steps: set[int] = set()
    existing_checkpoints: dict[str, TrainingCheckpoint] = {}
    for existing_name in sorted(root_inventory):
        if existing_name == "latest.json":
            _snapshot_regular_file_at(
                root_descriptor,
                existing_name,
                label="checkpoint latest pointer",
            )
            continue
        existing_path = checkpoint_root / existing_name
        existing_step = _checkpoint_directory_step(existing_path)
        if existing_step is None:
            raise ValueError(
                "checkpoint namespace contains an unexpected entry before save: "
                f"{existing_name}"
            )
        try:
            existing_checkpoint = _read_checkpoint_directory(
                existing_path,
                expected_run_config_sha256=state.run_config_sha256,
                trainable_names=trainable_names,
                expected_model_tensors=model_tensors,
                expected_optimizer_tensors=optimizer_tensors,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
            )[0]
        except Exception as error:
            raise ValueError(
                "checkpoint namespace contains an invalid candidate before save: "
                f"{existing_name}"
            ) from error
        existing_checkpoint_steps.add(existing_step)
        existing_checkpoints[existing_name] = existing_checkpoint
    if expected_existing_checkpoint_steps is not None:
        expected_existing = frozenset(expected_existing_checkpoint_steps)
        if any(type(step) is not int or step <= 0 for step in expected_existing):
            raise ValueError("expected existing checkpoint trajectory is invalid")
        if frozenset(existing_checkpoint_steps) != expected_existing:
            raise ValueError(
                "checkpoint namespace differs from the expected retained trajectory "
                f"before save: actual={sorted(existing_checkpoint_steps)}, "
                f"expected={sorted(expected_existing)}"
            )
    _require_exact_directory_inventory_at(
        root_descriptor,
        root_inventory,
        label="training checkpoint root",
    )
    namespace_evidence: _CheckpointNamespaceEvidence | None = None
    try:
        latest_pointer_snapshot = _snapshot_regular_file_at(
            root_descriptor,
            "latest.json",
            label="checkpoint latest pointer",
        )
    except FileNotFoundError:
        latest_pointer_snapshot = None
    try:
        temporary_descriptor, temporary_name, temporary = _create_staged_directory_at(
            root_descriptor,
            prefix=f".{target.name}.",
            expected_parent_inventory=root_inventory,
        )
    except BaseException:
        os.close(root_descriptor)
        raise
    root_inventory.add(temporary_name)
    temporary_snapshot = _snapshot_directory(
        temporary,
        label="staged training checkpoint",
    )
    try:
        _save_safetensors_child_at(
            temporary_descriptor,
            "model.safetensors",
            model_tensors,
        )
        _save_safetensors_child_at(
            temporary_descriptor,
            "optimizer.safetensors",
            optimizer_tensors,
        )
        temporary = _descriptor_path(temporary_descriptor)
        model_path = temporary / "model.safetensors"
        optimizer_path = temporary / "optimizer.safetensors"
        _validate_checkpoint_tensor_file_values(
            model_path,
            model_tensors,
            kind="model",
        )
        _validate_checkpoint_tensor_file_values(
            optimizer_path,
            optimizer_tensors,
            kind="optimizer",
        )
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
        metadata_sha256 = write_run_state(
            temporary / "metadata.json",
            metadata,
            parent_descriptor=temporary_descriptor,
            expected_parent_snapshot=temporary_snapshot,
        )
        os.fsync(temporary_descriptor)
        _require_exact_directory_inventory_at(
            root_descriptor,
            root_inventory,
            label="training checkpoint root",
        )
        candidate = TrainingCheckpoint(
            path=target,
            state=state,
            metadata_sha256=metadata_sha256,
            model_sha256=model_sha256,
            optimizer_sha256=optimizer_sha256,
        )
        _require_exact_directory_inventory_at(
            root_descriptor,
            root_inventory,
            label="training checkpoint root",
        )
        if target.exists() or target.is_symlink():
            target_value = os.lstat(target)
            target_snapshot = (
                _snapshot_directory(target, label="existing training checkpoint")
                if stat.S_ISDIR(target_value.st_mode)
                and not stat.S_ISLNK(target_value.st_mode)
                else None
            )
            try:
                existing, existing_model, existing_optimizer = _read_checkpoint_directory(
                    target,
                    expected_run_config_sha256=state.run_config_sha256,
                    trainable_names=trainable_names,
                    expected_model_tensors=model_tensors,
                    expected_optimizer_tensors=optimizer_tensors,
                    checkpoint_root_descriptor=root_descriptor,
                    expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                )
            except Exception:
                existing = None
            if (
                existing is not None
                and existing.state == state
                and existing.model_sha256 == model_sha256
                and existing.optimizer_sha256 == optimizer_sha256
            ):
                _require_exact_checkpoint_tensor_values(
                    existing_model,
                    model_tensors,
                    kind="model",
                )
                _require_exact_checkpoint_tensor_values(
                    existing_optimizer,
                    optimizer_tensors,
                    kind="optimizer",
                )
                with _bound_checkpoint_candidate(
                    existing,
                    checkpoint_root_descriptor=root_descriptor,
                    expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                ) as existing_authority:
                    _revalidate_directory_snapshot(
                        temporary_snapshot,
                        label="staged training checkpoint",
                    )
                    discarded_name = _move_entry_to_unique_recovery_at(
                        source_descriptor=root_descriptor,
                        source_name=temporary.name,
                        destination_descriptor=root_descriptor,
                        destination_prefix=f".discarded-{target.name}-",
                        expected_device=temporary_snapshot.components[-1][1],
                        expected_inode=temporary_snapshot.components[-1][2],
                        expected_directory=True,
                    )
                    root_inventory.remove(temporary_name)
                    root_inventory.add(discarded_name)
                    _require_exact_directory_inventory_at(
                        root_descriptor,
                        root_inventory,
                        label="training checkpoint root",
                    )
                    discarded = checkpoint_root / discarded_name
                    discarded_snapshot = _snapshot_directory(
                        discarded,
                        label="discarded duplicate checkpoint",
                    )
                    _revalidate_directory_snapshot(
                        discarded_snapshot,
                        label="discarded duplicate checkpoint",
                    )
                    discarded_identity = os.stat(
                        discarded_name,
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISDIR(discarded_identity.st_mode)
                        or (discarded_identity.st_dev, discarded_identity.st_ino)
                        != (
                            discarded_snapshot.components[-1][1],
                            discarded_snapshot.components[-1][2],
                        )
                    ):
                        raise RuntimeError("discarded checkpoint changed before removal")
                    if not shutil.rmtree.avoids_symlink_attacks:
                        raise RuntimeError("safe checkpoint cleanup requires fd-based rmtree")
                    shutil.rmtree(discarded_name, dir_fd=root_descriptor)
                    root_inventory.remove(discarded_name)
                    os.fsync(root_descriptor)
                    _require_exact_directory_inventory_at(
                        root_descriptor,
                        root_inventory,
                        label="training checkpoint root",
                    )
                    _revalidate_bound_checkpoint_candidate(
                        existing_authority,
                        checkpoint_root_descriptor=root_descriptor,
                        expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                        verify_bytes=True,
                    )
                    _write_latest_checkpoint_pointer(
                        checkpoint_root,
                        existing,
                        expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                        checkpoint_root_descriptor=root_descriptor,
                        expected_pointer_snapshot=latest_pointer_snapshot,
                        expected_root_inventory=root_inventory,
                    )
                    root_inventory.add("latest.json")
                    _revalidate_bound_checkpoint_candidate(
                        existing_authority,
                        checkpoint_root_descriptor=root_descriptor,
                        expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                        verify_bytes=True,
                    )
                    pruned_checkpoints = prune_training_checkpoints(
                        checkpoint_root,
                        keep_last=keep_last,
                        expected_run_config_sha256=state.run_config_sha256,
                        trainable_names=trainable_names,
                        expected_model_tensors=model_tensors,
                        expected_optimizer_tensors=optimizer_tensors,
                        expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                        checkpoint_root_descriptor=root_descriptor,
                        expected_root_inventory=root_inventory,
                    )
                    root_inventory.difference_update(pruned_checkpoints)
                    _require_exact_directory_inventory_at(
                        root_descriptor,
                        root_inventory,
                        label="training checkpoint root",
                    )
                    _revalidate_bound_checkpoint_candidate(
                        existing_authority,
                        checkpoint_root_descriptor=root_descriptor,
                        expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                        verify_bytes=True,
                    )
                    retained_checkpoints = {
                        name: retained
                        for name, retained in existing_checkpoints.items()
                        if name not in pruned_checkpoints
                    }
                    retained_checkpoints[target.name] = existing
                    namespace_evidence = _capture_checkpoint_namespace_evidence(
                        checkpoint_root_descriptor=root_descriptor,
                        expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                        expected_checkpoints=retained_checkpoints,
                        latest_checkpoint=existing,
                    )
                    result = TrainingCheckpoint(
                        path=existing.path,
                        state=existing.state,
                        metadata_sha256=existing.metadata_sha256,
                        model_sha256=existing.model_sha256,
                        optimizer_sha256=existing.optimizer_sha256,
                        pruned_checkpoints=pruned_checkpoints,
                        namespace_evidence=namespace_evidence,
                    )
                _revalidate_directory_snapshot(
                    checkpoint_root_snapshot,
                    label="training checkpoint root",
                )
                return result
            if stat.S_ISLNK(target_value.st_mode):
                raise FileExistsError(f"checkpoint target is an unsafe symlink: {target}")
            if target_snapshot is not None:
                _revalidate_directory_snapshot(
                    target_snapshot,
                    label="existing training checkpoint",
                )
            recovery_name = _move_entry_to_unique_recovery_at(
                source_descriptor=root_descriptor,
                source_name=target.name,
                destination_descriptor=root_descriptor,
                destination_prefix=f".recovery-{target.name}-",
                expected_device=target_value.st_dev,
                expected_inode=target_value.st_ino,
                expected_directory=stat.S_ISDIR(target_value.st_mode),
            )
            root_inventory.remove(target.name)
            root_inventory.add(recovery_name)
            os.fsync(root_descriptor)
            _require_exact_directory_inventory_at(
                root_descriptor,
                root_inventory,
                label="training checkpoint root",
            )
        _require_exact_directory_inventory_at(
            root_descriptor,
            root_inventory,
            label="training checkpoint root",
        )
        _revalidate_directory_snapshot(
            temporary_snapshot,
            label="staged training checkpoint",
        )
        _rename_entry_no_clobber_at(
            source_descriptor=root_descriptor,
            source_name=temporary_name,
            destination_descriptor=root_descriptor,
            destination_name=target.name,
            expected_device=temporary_snapshot.components[-1][1],
            expected_inode=temporary_snapshot.components[-1][2],
            expected_directory=True,
        )
        root_inventory.remove(temporary_name)
        root_inventory.add(target.name)
        os.fsync(root_descriptor)
        _require_exact_directory_inventory_at(
            root_descriptor,
            root_inventory,
            label="training checkpoint root",
        )
        published_snapshot = _snapshot_directory(
            target,
            label="published training checkpoint",
        )
        if published_snapshot.components[-1][1:] != (
            temporary_snapshot.components[-1][1],
            temporary_snapshot.components[-1][2],
        ):
            raise RuntimeError("published checkpoint differs from staged checkpoint")
        try:
            (
                published_checkpoint,
                published_model,
                published_optimizer,
            ) = _read_checkpoint_directory(
                target,
                expected_run_config_sha256=state.run_config_sha256,
                trainable_names=trainable_names,
                expected_model_tensors=model_tensors,
                expected_optimizer_tensors=optimizer_tensors,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
            )
        except Exception as error:
            raise ValueError("published checkpoint failed final validation") from error
        if (
            published_checkpoint.state != candidate.state
            or published_checkpoint.metadata_sha256 != candidate.metadata_sha256
            or published_checkpoint.model_sha256 != candidate.model_sha256
            or published_checkpoint.optimizer_sha256 != candidate.optimizer_sha256
        ):
            raise RuntimeError("published checkpoint differs from staged checkpoint")
        _require_exact_checkpoint_tensor_values(
            published_model,
            model_tensors,
            kind="model",
        )
        _require_exact_checkpoint_tensor_values(
            published_optimizer,
            optimizer_tensors,
            kind="optimizer",
        )
        candidate = published_checkpoint
        with _bound_checkpoint_candidate(
            candidate,
            checkpoint_root_descriptor=root_descriptor,
            expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
        ) as published_authority:
            _write_latest_checkpoint_pointer(
                checkpoint_root,
                candidate,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                checkpoint_root_descriptor=root_descriptor,
                expected_pointer_snapshot=latest_pointer_snapshot,
                expected_root_inventory=root_inventory,
            )
            root_inventory.add("latest.json")
            _revalidate_bound_checkpoint_candidate(
                published_authority,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                verify_bytes=True,
            )
            pruned_checkpoints = prune_training_checkpoints(
                checkpoint_root,
                keep_last=keep_last,
                expected_run_config_sha256=state.run_config_sha256,
                trainable_names=trainable_names,
                expected_model_tensors=model_tensors,
                expected_optimizer_tensors=optimizer_tensors,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                checkpoint_root_descriptor=root_descriptor,
                expected_root_inventory=root_inventory,
            )
            root_inventory.difference_update(pruned_checkpoints)
            _require_exact_directory_inventory_at(
                root_descriptor,
                root_inventory,
                label="training checkpoint root",
            )
            _revalidate_bound_checkpoint_candidate(
                published_authority,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                verify_bytes=True,
            )
            retained_checkpoints = {
                name: retained
                for name, retained in existing_checkpoints.items()
                if name not in pruned_checkpoints and name != target.name
            }
            retained_checkpoints[target.name] = candidate
            namespace_evidence = _capture_checkpoint_namespace_evidence(
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                expected_checkpoints=retained_checkpoints,
                latest_checkpoint=candidate,
            )
    finally:
        os.close(temporary_descriptor)
        os.close(root_descriptor)
    _revalidate_directory_snapshot(
        checkpoint_root_snapshot,
        label="training checkpoint root",
    )
    if namespace_evidence is None:
        raise AssertionError("checkpoint namespace evidence was not captured")
    return TrainingCheckpoint(
        path=target,
        state=state,
        metadata_sha256=metadata_sha256,
        model_sha256=model_sha256,
        optimizer_sha256=optimizer_sha256,
        pruned_checkpoints=pruned_checkpoints,
        namespace_evidence=namespace_evidence,
    )


def load_latest_training_checkpoint(
    *,
    model: nn.Module,
    optimizer: SmolVLAAdamW,
    checkpoint_root: str | Path,
    trainable_names: tuple[str, ...],
    expected_run_config_sha256: str,
    checkpoint_parent_descriptor: int | None = None,
    expected_checkpoint_parent_snapshot: _DirectorySnapshot | None = None,
    checkpoint_root_descriptor: int | None = None,
    expected_checkpoint_root_snapshot: _DirectorySnapshot | None = None,
    expected_selected_steps: int | None = None,
    expected_effective_batch_size: int | None = None,
    expected_checkpoint_interval: int | None = None,
    metrics_parent_descriptor: int | None = None,
    expected_metrics_snapshot: _StableFileSnapshot | None = None,
    expected_last_checkpoint: Mapping[str, object] | None = None,
    allowed_uncommitted_step: int | None = None,
) -> TrainingCheckpoint:
    """Discover, repair the pointer to, and restore the newest valid checkpoint."""

    checkpoint_root = Path(os.path.abspath(Path(checkpoint_root).expanduser()))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if checkpoint_root_descriptor is not None:
        if checkpoint_parent_descriptor is not None:
            raise ValueError("checkpoint load received both root and parent descriptors")
        if expected_checkpoint_root_snapshot is None:
            raise ValueError("bound checkpoint load requires a root snapshot")
        root_descriptor = os.dup(checkpoint_root_descriptor)
        opened_root = os.fstat(root_descriptor)
        _, root_device, root_inode = expected_checkpoint_root_snapshot.components[-1]
        if (opened_root.st_dev, opened_root.st_ino) != (
            root_device,
            root_inode,
        ):
            os.close(root_descriptor)
            raise RuntimeError("training checkpoint root descriptor changed")
        checkpoint_root = _descriptor_path(root_descriptor)
        checkpoint_root_snapshot = expected_checkpoint_root_snapshot
    elif checkpoint_parent_descriptor is None:
        checkpoint_root_snapshot = _snapshot_directory(
            checkpoint_root,
            label="training checkpoint directory",
        )
        root_descriptor = os.open(checkpoint_root, directory_flags)
    else:
        if expected_checkpoint_parent_snapshot is None:
            raise ValueError("bound checkpoint load requires a parent snapshot")
        if checkpoint_root.parent != expected_checkpoint_parent_snapshot.path:
            raise ValueError("checkpoint root differs from its bound parent")
        parent_identity = os.fstat(checkpoint_parent_descriptor)
        _, parent_device, parent_inode = (
            expected_checkpoint_parent_snapshot.components[-1]
        )
        if (parent_identity.st_dev, parent_identity.st_ino) != (
            parent_device,
            parent_inode,
        ):
            raise RuntimeError("checkpoint parent descriptor changed")
        named_root = os.stat(
            checkpoint_root.name,
            dir_fd=checkpoint_parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(named_root.st_mode):
            raise FileExistsError("training checkpoint root is unsafe")
        root_descriptor = os.open(
            checkpoint_root.name,
            directory_flags,
            dir_fd=checkpoint_parent_descriptor,
        )
        opened_root = os.fstat(root_descriptor)
        if (opened_root.st_dev, opened_root.st_ino) != (
            named_root.st_dev,
            named_root.st_ino,
        ):
            os.close(root_descriptor)
            raise RuntimeError("training checkpoint root changed during open")
        checkpoint_root = _descriptor_path(root_descriptor)
        checkpoint_root_snapshot = _snapshot_directory(
            checkpoint_root,
            label="training checkpoint directory",
        )
    recovery_authority_stack = ExitStack()
    try:
        root_identity = os.fstat(root_descriptor)
        _, root_device, root_inode = checkpoint_root_snapshot.components[-1]
        if (root_identity.st_dev, root_identity.st_ino) != (
            root_device,
            root_inode,
        ):
            raise RuntimeError("training checkpoint root changed before load")
        try:
            latest_pointer_snapshot = _snapshot_regular_file_at(
                root_descriptor,
                "latest.json",
                label="checkpoint latest pointer",
            )
        except FileNotFoundError:
            latest_pointer_snapshot = None
        current_tensors = dict(tree_flatten(model.trainable_parameters()))
        if tuple(current_tensors) != trainable_names:
            raise ValueError("current model trainable names differ from the resume contract")
        schema_optimizer = SmolVLAAdamW(optimizer.config)
        schema_optimizer.initialize(model.trainable_parameters())
        expected_optimizer_tensors = dict(tree_flatten(schema_optimizer.state))

        recorded_checkpoint_name: str | None = None
        recorded_checkpoint_found = False
        if expected_last_checkpoint is not None:
            recorded_step = expected_last_checkpoint.get("step")
            recorded_path = expected_last_checkpoint.get("path")
            if (
                type(recorded_step) is not int
                or not isinstance(recorded_path, str)
                or Path(recorded_path).name != f"step-{recorded_step:06d}"
                or not all(
                    _is_lowercase_sha256(expected_last_checkpoint.get(field))
                    for field in (
                        "metadata_sha256",
                        "model_sha256",
                        "optimizer_sha256",
                    )
                )
            ):
                raise ValueError("recorded checkpoint binding is invalid")
            recorded_checkpoint_name = Path(recorded_path).name
            if allowed_uncommitted_step is not None and (
                type(allowed_uncommitted_step) is not int
                or allowed_uncommitted_step <= recorded_step
            ):
                raise ValueError("allowed uncommitted checkpoint step is invalid")

        valid: list[
            tuple[TrainingCheckpoint, dict[str, mx.array], dict[str, mx.array]]
        ] = []
        invalid: list[str] = []
        namespace_steps: set[int] = set()
        recoverable_invalid_uncommitted: tuple[
            str, int, _BoundCheckpointCandidate
        ] | None = None
        recovered_invalid_destination: str | None = None
        namespace_inventory = tuple(sorted(os.listdir(root_descriptor)))
        for name in namespace_inventory:
            if Path(name).name != name:
                continue
            path = checkpoint_root / name
            namespace_step = _checkpoint_directory_step(path)
            if namespace_step is None:
                if name != "latest.json":
                    invalid.append(f"{name}: unsafe checkpoint namespace entry")
                continue
            namespace_steps.add(namespace_step)
            try:
                candidate = _read_checkpoint_directory(
                    path,
                    expected_run_config_sha256=expected_run_config_sha256,
                    trainable_names=trainable_names,
                    expected_model_tensors=current_tensors,
                    expected_optimizer_tensors=expected_optimizer_tensors,
                    checkpoint_root_descriptor=root_descriptor,
                    expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                )
                candidate_state = candidate[0].state
                if expected_selected_steps is not None and (
                    candidate_state.selected_steps != expected_selected_steps
                ):
                    raise ValueError("checkpoint training horizon differs")
                if expected_effective_batch_size is not None:
                    expected_draws = (
                        candidate_state.completed_step
                        * expected_effective_batch_size
                    )
                    if (
                        candidate_state.samples_consumed != expected_draws
                        or candidate_state.flow_draw_count != expected_draws
                    ):
                        raise ValueError("checkpoint trajectory counters differ")
                if expected_checkpoint_interval is not None:
                    on_cadence = (
                        candidate_state.completed_step == 1
                        or candidate_state.completed_step
                        % expected_checkpoint_interval
                        == 0
                        or (
                            expected_selected_steps is not None
                            and candidate_state.completed_step
                            == expected_selected_steps
                        )
                    )
                    if not on_cadence:
                        raise ValueError("checkpoint step is outside the save cadence")
                if expected_metrics_snapshot is not None:
                    _validate_metrics_checkpoint_snapshot(
                        expected_metrics_snapshot,
                        candidate_state,
                    )
                if recorded_checkpoint_name == name:
                    recorded_checkpoint_found = True
                    checkpoint_binding = {
                        "metadata_sha256": candidate[0].metadata_sha256,
                        "model_sha256": candidate[0].model_sha256,
                        "optimizer_sha256": candidate[0].optimizer_sha256,
                    }
                    if any(
                        checkpoint_binding[field] != expected_last_checkpoint[field]
                        for field in checkpoint_binding
                    ):
                        raise ValueError(
                            "recorded checkpoint identity differs from run metadata"
                        )
                if expected_last_checkpoint is not None and (
                    candidate_state.completed_step > expected_last_checkpoint["step"]
                    and candidate_state.completed_step != allowed_uncommitted_step
                ):
                    raise ValueError(
                        "checkpoint is outside the single allowed crash window"
                    )
                if (
                    expected_last_checkpoint is None
                    and allowed_uncommitted_step is not None
                    and candidate_state.completed_step != allowed_uncommitted_step
                ):
                    raise ValueError(
                        "checkpoint is outside the initial allowed crash window"
                    )
                valid.append(candidate)
            except Exception as error:
                if recorded_checkpoint_name == name:
                    raise ValueError(
                        "recorded checkpoint differs from run metadata"
                    ) from error
                if (
                    namespace_step == allowed_uncommitted_step
                    and recoverable_invalid_uncommitted is None
                ):
                    try:
                        invalid_authority = recovery_authority_stack.enter_context(
                            _bound_checkpoint_namespace_candidate(
                                name,
                                checkpoint_root_descriptor=root_descriptor,
                                expected_checkpoint_root_snapshot=(
                                    checkpoint_root_snapshot
                                ),
                            )
                        )
                    except Exception as authority_error:
                        invalid.append(
                            f"{path.name}: {type(error).__name__}: {error}; "
                            "candidate could not be bound for recovery: "
                            f"{type(authority_error).__name__}: {authority_error}"
                        )
                    else:
                        recoverable_invalid_uncommitted = (
                            name,
                            namespace_step,
                            invalid_authority,
                        )
                        namespace_steps.discard(namespace_step)
                        continue
                invalid.append(f"{path.name}: {type(error).__name__}: {error}")
        if invalid:
            raise ValueError(
                "checkpoint namespace contains invalid candidates: "
                + "; ".join(sorted(invalid))
            )
        if expected_checkpoint_interval is not None and (
            expected_last_checkpoint is not None
            or allowed_uncommitted_step is not None
        ):
            if expected_selected_steps is None:
                raise ValueError(
                    "exact checkpoint trajectory requires the selected-step horizon"
                )

            def retained_steps_through(step: int) -> tuple[int, ...]:
                cadence = {1}
                cadence.update(
                    range(expected_checkpoint_interval, step + 1, expected_checkpoint_interval)
                )
                if step == expected_selected_steps:
                    cadence.add(step)
                if step not in cadence:
                    raise ValueError(
                        "checkpoint retained trajectory ends outside the save cadence"
                    )
                return tuple(sorted(cadence)[-3:])

            allowed_namespaces: set[frozenset[int]] = set()
            if expected_last_checkpoint is None:
                assert allowed_uncommitted_step is not None
                allowed_namespaces.add(frozenset((allowed_uncommitted_step,)))
            else:
                recorded_step = expected_last_checkpoint["step"]
                assert isinstance(recorded_step, int)
                recorded_retained = retained_steps_through(recorded_step)
                allowed_namespaces.add(frozenset(recorded_retained))
                if allowed_uncommitted_step is not None:
                    with_candidate = (*recorded_retained, allowed_uncommitted_step)
                    allowed_namespaces.add(frozenset(with_candidate))
                    allowed_namespaces.add(frozenset(with_candidate[-3:]))
            if frozenset(namespace_steps) not in allowed_namespaces:
                raise ValueError(
                    "checkpoint namespace differs from the recorded retained trajectory: "
                    f"actual={sorted(namespace_steps)}, "
                    f"allowed={sorted(sorted(value) for value in allowed_namespaces)}"
                )
        if recorded_checkpoint_name is not None and not recorded_checkpoint_found:
            raise FileNotFoundError(
                f"recorded checkpoint is missing: {recorded_checkpoint_name}"
            )
        if not valid:
            detail = "; ".join(sorted(invalid)) if invalid else "no step directories"
            raise FileNotFoundError(
                f"no valid training checkpoint found in {checkpoint_root}: {detail}"
            )
        if recoverable_invalid_uncommitted is not None:
            recovery_name, _, recovery_authority = recoverable_invalid_uncommitted
            if tuple(sorted(os.listdir(root_descriptor))) != namespace_inventory:
                raise RuntimeError(
                    "checkpoint namespace changed before invalid-candidate recovery"
                )
            _revalidate_bound_checkpoint_candidate(
                recovery_authority,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                verify_bytes=True,
            )
            destination_name = _move_entry_to_unique_recovery_at(
                source_descriptor=root_descriptor,
                source_name=recovery_name,
                destination_descriptor=root_descriptor,
                destination_prefix=f".recovery-{recovery_name}-",
                expected_device=recovery_authority.directory_device,
                expected_inode=recovery_authority.directory_inode,
                expected_directory=True,
            )
            recovered_invalid_destination = destination_name
            os.fsync(root_descriptor)
            _revalidate_bound_checkpoint_candidate(
                recovery_authority,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                verify_bytes=True,
                entry_name=destination_name,
            )
        valid.sort(key=lambda item: item[0].state.completed_step)
        checkpoint, loaded_model, loaded_optimizer = valid[-1]
        def require_metrics_boundary() -> None:
            if expected_metrics_snapshot is None:
                return
            if metrics_parent_descriptor is None:
                raise ValueError(
                    "checkpoint metrics binding requires a parent descriptor"
                )
            current_metrics = _snapshot_regular_file_at(
                metrics_parent_descriptor,
                "metrics.csv",
                label="fine-tune metrics",
                capture_payload=True,
            )
            if not _same_bound_file_snapshot(
                current_metrics,
                expected_metrics_snapshot,
            ):
                raise RuntimeError(
                    "fine-tune metrics changed before checkpoint restoration"
                )
            _validate_metrics_checkpoint_snapshot(current_metrics, checkpoint.state)

        with _bound_checkpoint_candidate(
            checkpoint,
            checkpoint_root_descriptor=root_descriptor,
            expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
        ) as candidate_authority:
            require_metrics_boundary()
            _revalidate_bound_checkpoint_candidate(
                candidate_authority,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                verify_bytes=True,
            )
            _write_latest_checkpoint_pointer(
                checkpoint_root,
                checkpoint,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                checkpoint_root_descriptor=root_descriptor,
                expected_pointer_snapshot=latest_pointer_snapshot,
            )
            _revalidate_bound_checkpoint_candidate(
                candidate_authority,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                verify_bytes=True,
            )
            require_metrics_boundary()
            pruned_checkpoints = prune_training_checkpoints(
                checkpoint_root,
                keep_last=3,
                expected_run_config_sha256=expected_run_config_sha256,
                trainable_names=trainable_names,
                expected_model_tensors=current_tensors,
                expected_optimizer_tensors=expected_optimizer_tensors,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                checkpoint_root_descriptor=root_descriptor,
                allowed_recovery_entries=(
                    ()
                    if recovered_invalid_destination is None
                    else (recovered_invalid_destination,)
                ),
            )
            _revalidate_bound_checkpoint_candidate(
                candidate_authority,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                verify_bytes=True,
            )
            require_metrics_boundary()
            model.update(
                tree_unflatten([(name, loaded_model[name]) for name in trainable_names])
            )
            optimizer_names = tuple(loaded_optimizer)
            optimizer.load_state(
                tree_unflatten(
                    [(name, loaded_optimizer[name]) for name in optimizer_names]
                ),
                step_index=checkpoint.state.completed_step,
            )
            mx.eval(model.trainable_parameters(), optimizer.state)
            _revalidate_bound_checkpoint_candidate(
                candidate_authority,
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                verify_bytes=True,
            )
            require_metrics_boundary()
            retained_checkpoints = {
                retained.path.name: retained
                for retained, _, _ in valid
                if retained.path.name not in pruned_checkpoints
            }
            namespace_evidence = _capture_checkpoint_namespace_evidence(
                checkpoint_root_descriptor=root_descriptor,
                expected_checkpoint_root_snapshot=checkpoint_root_snapshot,
                expected_checkpoints=retained_checkpoints,
                latest_checkpoint=checkpoint,
                allowed_other_entries=(
                    ()
                    if recovered_invalid_destination is None
                    else (recovered_invalid_destination,)
                ),
            )
        root_after = os.fstat(root_descriptor)
        if (root_after.st_dev, root_after.st_ino) != (root_device, root_inode):
            raise RuntimeError("training checkpoint root changed during load")
        _revalidate_directory_snapshot(
            checkpoint_root_snapshot,
            label="training checkpoint directory",
        )
        return TrainingCheckpoint(
            path=checkpoint.path,
            state=checkpoint.state,
            metadata_sha256=checkpoint.metadata_sha256,
            model_sha256=checkpoint.model_sha256,
            optimizer_sha256=checkpoint.optimizer_sha256,
            pruned_checkpoints=pruned_checkpoints,
            namespace_evidence=namespace_evidence,
        )
    finally:
        recovery_authority_stack.close()
        os.close(root_descriptor)


def _reconcile_loaded_checkpoint_run_document(
    run_document: Mapping[str, object],
    *,
    checkpoint: TrainingCheckpoint,
    selected_steps: int,
    checkpoint_interval: int,
) -> dict[str, object]:
    """Bind stale run metadata to the checkpoint actually restored for resume."""

    completed_step = checkpoint.state.completed_step
    cadence = {1}
    cadence.update(range(checkpoint_interval, completed_step + 1, checkpoint_interval))
    if completed_step == selected_steps:
        cadence.add(completed_step)
    if completed_step not in cadence:
        raise ValueError("loaded checkpoint step is outside the save cadence")
    return {
        **run_document,
        "checkpoint_count": len(cadence),
        "last_checkpoint": {
            "step": completed_step,
            "path": str(checkpoint.path),
            "metadata_sha256": checkpoint.metadata_sha256,
            "model_sha256": checkpoint.model_sha256,
            "optimizer_sha256": checkpoint.optimizer_sha256,
        },
    }


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


@dataclass(frozen=True)
class ResolvedTrainingBudget:
    """One validated budget artifact and its optional timing benchmark."""

    selected_steps: int
    artifact_name: str
    artifact: dict[str, object]
    benchmark: BenchmarkResult | None


def resolve_training_budget(
    config: FineTuneConfig,
    *,
    persisted: Mapping[str, object] | None = None,
) -> ResolvedTrainingBudget:
    """Resolve either legacy adaptive timing or the T3B fixed-step budget."""

    if config.budget_mode == FIXED_BUDGET_MODE:
        artifact = (
            fixed_step_budget(config)
            if persisted is None
            else validate_fixed_step_budget(persisted, config=config)
        )
        return ResolvedTrainingBudget(
            selected_steps=config.nominal_steps,
            artifact_name="budget.json",
            artifact=artifact,
            benchmark=None,
        )

    benchmark = (
        benchmark_lora_updates(config)
        if persisted is None
        else BenchmarkResult.from_dict(persisted)
    )
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
        raise ValueError(
            "benchmark artifact differs from the requested training configuration"
        )
    return ResolvedTrainingBudget(
        selected_steps=selected_steps,
        artifact_name="benchmark.json",
        artifact=benchmark.as_dict(),
        benchmark=benchmark,
    )


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
        LoRAConfig(
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            scope=config.lora_scope,
        ),
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


def prepare_lora_finetune_launch(
    config: FineTuneConfig,
    *,
    output_path: str | Path | None = None,
) -> tuple[dict[str, object], str]:
    """Freeze and install the exact T3B launch configuration before updates."""

    _validate_t3b_frozen_config(config)
    _require_t3b_runtime_provenance(allow_unfrozen=True)
    output_dir = _safe_t3b_output_path(
        config.output_dir,
        must_exist=False,
        label="T3B output path",
    )
    launch_path = (
        output_dir / "launch.json"
        if output_path is None
        else Path(os.path.abspath(Path(output_path).expanduser()))
    )
    if launch_path != output_dir / "launch.json":
        raise ValueError("T3B launch configuration must be output_dir/launch.json")
    if shutil.disk_usage(output_dir.parent).free < _MINIMUM_FREE_BYTES:
        raise RuntimeError("T3B launch preparation requires at least 40 GiB free")

    budget = resolve_training_budget(config)
    prebuild_inputs = collect_t3b_frozen_input_evidence(
        config,
        None,
        validate_runtime_model=False,
    )
    split, stats, model, lora_report, bridge, optimizer = _build_training_components(
        config,
        training_horizon=budget.selected_steps,
    )
    try:
        if (
            lora_report.scope != EXPERT_ONLY_SCOPE
            or lora_report.adapter_count != 112
            or lora_report.trainable_tensor_count != 224
            or lora_report.trainable_scalar_count != 1_708_032
        ):
            raise RuntimeError("T3B expert-only LoRA topology changed")
        frozen_inputs = collect_t3b_frozen_input_evidence(
            config,
            model,
            runtime_statistics=stats,
        )
        _require_unchanged_t3b_inputs(
            prebuild_inputs,
            frozen_inputs,
            context="training-component construction",
        )
        training_bridge_evidence = _validate_t3b_training_bridge_semantics(
            config=config,
            bridge=bridge,
            stats=stats,
            expected_frozen_inputs=frozen_inputs,
        )
        base_artifact = training_base_artifact_identity(model)
        document = assemble_finetune_launch_config(
            config=config,
            budget=budget.artifact,
            train_statistics_sha256=stats.sha256,
            train_episodes=split.train_episodes,
            holdout_episodes=split.holdout_episodes,
            base_artifact=base_artifact,
            optimizer_config=optimizer.config,
            lora_report=lora_report,
            reference_freeze_policy=reference_freeze_policy_evidence(),
            implementation_sha256=finetune_implementation_hashes(),
            frozen_inputs=frozen_inputs,
            training_bridge=training_bridge_evidence,
            created_at_ns=time.time_ns(),
        )
        prepared_snapshot, prepared_descriptor = _create_safe_directory_no_clobber(
            output_dir,
            label="T3B prepared output directory",
        )
        try:
            digest = write_finetune_launch_config(
                launch_path,
                document,
                parent_descriptor=prepared_descriptor,
                expected_parent_snapshot=prepared_snapshot,
            )
        finally:
            os.close(prepared_descriptor)
        return document, digest
    finally:
        del model, bridge, optimizer
        gc.collect()
        mx.clear_cache()


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
    parent_descriptor: int | None = None,
    expected_parent_snapshot: _DirectorySnapshot | None = None,
) -> str:
    """Publish or exactly reuse the final adapter and its bound metadata."""

    tensors = {
        name: value.astype(mx.float32)
        for name, value in tree_flatten(model.trainable_parameters())
    }
    if tuple(tensors) != lora_report.trainable_names:
        raise RuntimeError("final adapter tensor names changed during training")
    mx.eval(tensors)
    path = Path(os.path.abspath(path))
    metadata_path = path.with_suffix(".json")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if parent_descriptor is None:
        parent_snapshot = _ensure_safe_directory(
            path.parent,
            label="adapter output directory",
        )
        bound_parent = os.open(path.parent, directory_flags)
    else:
        if expected_parent_snapshot is None:
            raise ValueError("bound adapter publication requires a parent snapshot")
        if path.parent != expected_parent_snapshot.path:
            raise ValueError("adapter path differs from its bound parent")
        parent_snapshot = expected_parent_snapshot
        bound_parent = os.dup(parent_descriptor)
    opened_parent = os.fstat(bound_parent)
    _, parent_device, parent_inode = parent_snapshot.components[-1]
    if (opened_parent.st_dev, opened_parent.st_ino) != (
        parent_device,
        parent_inode,
    ):
        os.close(bound_parent)
        raise RuntimeError("adapter parent descriptor changed")
    scope = getattr(lora_report, "scope", LEGACY_FULL_SCOPE)

    def expected_metadata(digest: str) -> dict[str, object]:
        metadata: dict[str, object] = {
            "format_version": 1,
            "rank": lora_report.rank,
            "alpha": lora_report.alpha,
            "dropout": lora_report.dropout,
            "adapter_count": lora_report.adapter_count,
            "tensor_count": len(tensors),
            "scalar_count": lora_report.trainable_scalar_count,
            "sha256": digest,
        }
        if scope != LEGACY_FULL_SCOPE:
            metadata["scope"] = scope
        return metadata

    def validate_adapter() -> tuple[str, os.stat_result]:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=bound_parent,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("adapter checkpoint is not a regular file")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            loaded = mx.load(str(_descriptor_path(descriptor)))
            if set(loaded) != set(tensors):
                raise ValueError("adapter tensor names differ from the trained adapter")
            for name, expected in tensors.items():
                actual = loaded[name]
                if (
                    actual.shape != expected.shape
                    or actual.dtype != expected.dtype
                    or not np.array_equal(np.asarray(actual), np.asarray(expected))
                ):
                    raise ValueError(
                        f"adapter tensor differs from the trained adapter: {name}"
                    )
            after = os.fstat(descriptor)
            named = os.stat(
                path.name,
                dir_fd=bound_parent,
                follow_symlinks=False,
            )
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise RuntimeError("adapter checkpoint changed during validation")
            return digest.hexdigest(), before
        finally:
            os.close(descriptor)

    def read_metadata() -> dict[str, object]:
        descriptor = os.open(
            metadata_path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=bound_parent,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("adapter metadata is not a regular file")
            payload = bytearray()
            while chunk := os.read(descriptor, 1024 * 1024):
                payload.extend(chunk)
            after = os.fstat(descriptor)
            named = os.stat(
                metadata_path.name,
                dir_fd=bound_parent,
                follow_symlinks=False,
            )
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise RuntimeError("adapter metadata changed during validation")
            value = json.loads(bytes(payload))
            if not isinstance(value, dict):
                raise ValueError("adapter metadata must be an object")
            return value
        finally:
            os.close(descriptor)

    try:
        adapter_named = os.stat(
            path.name,
            dir_fd=bound_parent,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        adapter_named = None
    try:
        metadata_named = os.stat(
            metadata_path.name,
            dir_fd=bound_parent,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        metadata_named = None
    if metadata_named is not None and adapter_named is None:
        os.close(bound_parent)
        raise FileExistsError("adapter metadata exists without its adapter checkpoint")
    if adapter_named is not None:
        if not stat.S_ISREG(adapter_named.st_mode):
            os.close(bound_parent)
            raise FileExistsError("adapter checkpoint path is unsafe")
        digest, _ = validate_adapter()
        if metadata_named is not None:
            if not stat.S_ISREG(metadata_named.st_mode):
                os.close(bound_parent)
                raise FileExistsError("adapter metadata path is unsafe")
            if read_metadata() != expected_metadata(digest):
                os.close(bound_parent)
                raise ValueError("existing adapter metadata differs from the trained adapter")
            os.close(bound_parent)
            return digest

    staging_descriptor: int | None = None
    staging_name: str | None = None
    staging_device: int | None = None
    staging_inode: int | None = None
    try:
        staging_descriptor, staging_name, staging_path = _create_staged_directory_at(
            bound_parent,
            prefix=".adapter-stage-",
        )
        staging_identity = os.fstat(staging_descriptor)
        staging_device = staging_identity.st_dev
        staging_inode = staging_identity.st_ino
        _save_safetensors_child_at(
            staging_descriptor,
            path.name,
            tensors,
        )
        named_staging = os.stat(
            staging_name,
            dir_fd=bound_parent,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named_staging.st_mode)
            or (named_staging.st_dev, named_staging.st_ino)
            != (staging_device, staging_inode)
        ):
            raise RuntimeError("adapter staging directory changed while in use")
        staging_path = _descriptor_path(staging_descriptor)
        staged_adapter = staging_path / path.name
        staged_digest = _file_sha256(staged_adapter)
        staged_loaded = mx.load(str(staged_adapter))
        if set(staged_loaded) != set(tensors):
            raise RuntimeError("staged adapter tensor names changed")
        for name, expected in tensors.items():
            actual = staged_loaded[name]
            if (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or not np.array_equal(np.asarray(actual), np.asarray(expected))
            ):
                raise RuntimeError(f"staged adapter tensor changed: {name}")
        staged_metadata = expected_metadata(staged_digest)
        write_run_state(
            staging_path / metadata_path.name,
            staged_metadata,
            parent_descriptor=staging_descriptor,
            expected_parent_snapshot=_snapshot_directory(
                staging_path,
                label="adapter staging directory",
            ),
        )
        os.fsync(staging_descriptor)
        if adapter_named is None:
            staged_adapter_identity = os.stat(
                path.name,
                dir_fd=staging_descriptor,
                follow_symlinks=False,
            )
            _rename_entry_no_clobber_at(
                source_descriptor=staging_descriptor,
                source_name=path.name,
                destination_descriptor=bound_parent,
                destination_name=path.name,
                expected_device=staged_adapter_identity.st_dev,
                expected_inode=staged_adapter_identity.st_ino,
                expected_directory=False,
            )
            os.fsync(bound_parent)
        else:
            existing_digest, _ = validate_adapter()
            if existing_digest != staged_digest:
                raise ValueError("existing adapter differs from the staged adapter")
        staged_metadata_identity = os.stat(
            metadata_path.name,
            dir_fd=staging_descriptor,
            follow_symlinks=False,
        )
        _rename_entry_no_clobber_at(
            source_descriptor=staging_descriptor,
            source_name=metadata_path.name,
            destination_descriptor=bound_parent,
            destination_name=metadata_path.name,
            expected_device=staged_metadata_identity.st_dev,
            expected_inode=staged_metadata_identity.st_ino,
            expected_directory=False,
        )
        os.fsync(bound_parent)
        digest, _ = validate_adapter()
        if digest != staged_digest or read_metadata() != expected_metadata(digest):
            raise RuntimeError("published adapter pair differs from its staged values")
        return digest
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if staging_name is not None and staging_device is not None and staging_inode is not None:
            try:
                remaining = os.stat(
                    staging_name,
                    dir_fd=bound_parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                remaining = None
            if remaining is not None and (remaining.st_dev, remaining.st_ino) == (
                staging_device,
                staging_inode,
            ):
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise RuntimeError("safe adapter cleanup requires fd-based rmtree")
                shutil.rmtree(staging_name, dir_fd=bound_parent)
                os.fsync(bound_parent)
        os.close(bound_parent)


@dataclass(frozen=True)
class FineTuneResult:
    """Local artifacts and measurements from one completed training/export run."""

    selected_steps: int
    benchmark: BenchmarkResult | None
    training_seconds: float
    final_loss: float
    final_smoothed_loss: float
    peak_memory_bytes: int
    adapter_sha256: str
    export_dir: Path
    run_state_sha256: str


def _run_lora_finetune_impl(
    config: FineTuneConfig,
    *,
    launch_config_path: str | Path | None = None,
    progress: Callable[[int, int, UpdateResult], None] | None = None,
    training_lease: _T3BTrainingLease | None = None,
    training_log_lease: _T3BTrainingLogLease | None = None,
    training_log_identity: Mapping[str, object] | None = None,
    startup_recoveries: tuple[str, ...] = (),
    startup_bindings: _T3BStartupBindings | None = None,
) -> FineTuneResult:
    """Benchmark, train or exactly resume, save adapters, merge, and export."""

    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"LoRA fine-tuning requires Metal GPU, got {mx.default_device()}")
    requires_launch = _requires_t3b_launch_config(config)
    if requires_launch and training_lease is None:
        raise RuntimeError("T3B fine-tuning requires a bound output-directory lease")
    if requires_launch and startup_bindings is None:
        raise RuntimeError("T3B fine-tuning requires bound startup state")
    state_bindings: dict[Path, _StableFileSnapshot | None] = {}

    def require_state_bindings() -> None:
        if training_lease is None:
            return
        for path, expected in state_bindings.items():
            if path.parent != training_lease.output_snapshot.path:
                continue
            try:
                current = _snapshot_regular_file_at(
                    training_lease.output_descriptor,
                    path.name,
                    label=f"bound training state {path.name}",
                )
            except FileNotFoundError:
                current = None
            if expected is None:
                if current is not None:
                    raise RuntimeError(
                        f"bound training state appeared unexpectedly: {path.name}"
                    )
            elif current is None or not _same_bound_file_snapshot(current, expected):
                raise RuntimeError(
                    f"bound training state changed while in use: {path.name}"
                )

    def require_output_lease() -> None:
        if training_lease is not None:
            _revalidate_t3b_training_lock(training_lease)
            if training_lease.checkpoint_root_descriptor is not None:
                _revalidate_t3b_checkpoint_root(training_lease)
            if training_lease.export_descriptor is not None:
                _revalidate_t3b_export_root(training_lease)
            if training_lease.adapter_file_bindings is not None:
                _revalidate_t3b_adapter_files(training_lease)
            if training_lease.metrics_file_binding is not None:
                _revalidate_t3b_metrics_file(training_lease)
            if training_lease.final_checkpoint_binding_stack is not None:
                _revalidate_t3b_final_checkpoint_namespace(training_lease)
            if training_log_lease is not None:
                _revalidate_t3b_training_log(training_log_lease, training_lease)
            require_state_bindings()

    def persist_run_state(path: Path, value: object) -> str:
        absolute_path = Path(os.path.abspath(path))
        require_output_lease()
        if (
            training_lease is not None
            and absolute_path.parent == training_lease.output_snapshot.path
        ):
            expected_destination = state_bindings.get(
                absolute_path,
                _UNSPECIFIED_STATE_DESTINATION,
            )
            digest, published = _write_run_state_with_binding(
                path,
                value,
                parent_descriptor=training_lease.output_descriptor,
                expected_parent_snapshot=training_lease.output_snapshot,
                expected_destination_snapshot=expected_destination,
            )
            state_bindings[absolute_path] = published
        else:
            expected_destination = state_bindings.get(
                absolute_path,
                _UNSPECIFIED_STATE_DESTINATION,
            )
            digest, published = _write_run_state_with_binding(
                path,
                value,
                expected_destination_snapshot=expected_destination,
            )
            state_bindings[absolute_path] = published
        require_output_lease()
        return digest

    require_output_lease()
    if requires_launch:
        assert training_lease is not None
        output_dir = training_lease.output_snapshot.path
        configured_output = Path(os.path.abspath(config.output_dir.expanduser()))
        if configured_output != output_dir:
            raise ValueError("T3B configuration output differs from the locked directory")
    else:
        if config.output_dir.is_symlink():
            raise FileExistsError(f"fine-tune output path is unsafe: {config.output_dir}")
        output_dir = config.output_dir.resolve()
    run_path = output_dir / "run.json"
    if startup_bindings is not None:
        is_resume = startup_bindings.run is not None
        state_bindings[run_path] = startup_bindings.run
    else:
        if run_path.is_symlink():
            raise FileExistsError(f"fine-tune run metadata is unsafe: {run_path}")
        is_resume = run_path.is_file()
        state_bindings[run_path] = (
            _snapshot_regular_file(
                run_path,
                label="fine-tune run metadata",
            )
            if is_resume
            else None
        )
    process_identity_path = output_dir / "training.pid"
    if startup_bindings is not None:
        state_bindings[process_identity_path] = startup_bindings.process_identity
    elif is_resume and process_identity_path.is_file() and not process_identity_path.is_symlink():
        state_bindings[process_identity_path] = _snapshot_regular_file(
            process_identity_path,
            label="fine-tune process identity",
        )
    else:
        state_bindings[process_identity_path] = None
    if requires_launch:
        _validate_t3b_frozen_config(config)
        expected_launch_path = output_dir / "launch.json"
        launch_path = (
            expected_launch_path
            if launch_config_path is None
            else Path(os.path.abspath(Path(launch_config_path).expanduser()))
        )
        if launch_path != expected_launch_path:
            raise ValueError("T3B launch configuration must be output_dir/launch.json")
    elif launch_config_path is not None:
        raise ValueError("launch configuration is only valid for fixed expert-only LoRA")
    else:
        launch_path = None
    if is_resume and not config.resume:
        raise FileExistsError(f"refusing to overwrite existing fine-tune run {output_dir}")
    if config.resume and not is_resume:
        raise FileNotFoundError(f"fine-tune run has no resumable metadata: {output_dir}")
    if requires_launch and not is_resume:
        assert training_lease is not None
        _revalidate_t3b_training_lock(training_lease)
    elif output_dir.exists() and not is_resume:
        raise FileExistsError(f"refusing to overwrite existing fine-tune path {output_dir}")
    launch_document: dict[str, object] | None = None
    launch_file_sha256: str | None = None
    if launch_path is not None:
        assert startup_bindings is not None
        launch_document, launch_file_sha256 = _validate_finetune_launch_snapshot(
            startup_bindings.launch
        )
        state_bindings[launch_path] = startup_bindings.launch
        process_identity = {
            "format_version": 1,
            "artifact_type": "smolvla-mlx-training-process",
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "started_at_ns": time.time_ns(),
            "working_directory": str(Path.cwd().resolve()),
            "executable": str(Path(sys.executable).resolve()),
            "launch_config": str(launch_path),
            "launch_file_sha256": launch_file_sha256,
            "configuration_sha256": launch_document["configuration_sha256"],
            "run_config_sha256": launch_document["run_config_sha256"],
        }
        if training_log_identity is None:
            raise RuntimeError("T3B process identity requires a bound training log")
        process_identity["training_log"] = dict(training_log_identity)
        process_identity_sha256 = None
    else:
        process_identity = None
        process_identity_sha256 = None
    disk_free_before = shutil.disk_usage(output_dir.parent).free
    if disk_free_before < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"LoRA fine-tuning requires {_MINIMUM_FREE_BYTES} free bytes, got {disk_free_before}"
        )

    if is_resume:
        artifact_name = (
            "budget.json"
            if config.budget_mode == FIXED_BUDGET_MODE
            else "benchmark.json"
        )
        artifact_path = output_dir / artifact_name
        if startup_bindings is not None:
            run_snapshot = startup_bindings.run
            artifact_snapshot = startup_bindings.budget
            if artifact_name != "budget.json" or artifact_snapshot is None:
                raise FileNotFoundError(
                    "fine-tune output exists without resumable run metadata"
                )
        else:
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    "fine-tune output exists without resumable run metadata"
                )
            run_snapshot = _snapshot_regular_file(
                run_path,
                label="fine-tune run metadata",
                capture_payload=True,
            )
            artifact_snapshot = _snapshot_regular_file(
                artifact_path,
                label="fine-tune budget artifact",
                capture_payload=True,
            )
        if run_snapshot is None or not _same_bound_file_snapshot(
            state_bindings[run_path], run_snapshot
        ):
            raise RuntimeError("fine-tune run metadata changed during resume setup")
        run_document = _json_from_stable_snapshot(
            run_snapshot,
            label="fine-tune run metadata",
        )
        if not isinstance(run_document, Mapping):
            raise ValueError("fine-tune run metadata must be an object")
        run_document = dict(run_document)
        if run_document.get("status") == "trained_and_exported":
            raise FileExistsError(f"fine-tune run is already complete: {output_dir}")
        if run_document.get("status") not in {"running", "interrupted", "exporting"}:
            raise ValueError(f"fine-tune run status is not resumable: {run_document.get('status')!r}")
        state_bindings[artifact_path] = artifact_snapshot
        persisted_budget = _json_from_stable_snapshot(
            artifact_snapshot,
            label="fine-tune budget artifact",
        )
        budget = resolve_training_budget(config, persisted=persisted_budget)
    else:
        budget = resolve_training_budget(config)
        artifact_path = output_dir / budget.artifact_name
    selected_steps = budget.selected_steps
    benchmark = budget.benchmark
    if not is_resume:
        if not requires_launch:
            output_dir.mkdir(parents=True)
        budget_artifact_sha256 = None
    else:
        budget_artifact_sha256 = state_bindings[artifact_path].sha256
        if config.budget_mode == FIXED_BUDGET_MODE and (
            run_document.get("budget_mode") != FIXED_BUDGET_MODE
            or run_document.get("budget") != budget.artifact
            or run_document.get("budget_sha256") != budget_artifact_sha256
        ):
            raise ValueError("existing run fixed budget binding is invalid")
    prebuild_inputs = (
        collect_t3b_frozen_input_evidence(
            config,
            None,
            validate_runtime_model=False,
        )
        if launch_document is not None
        else None
    )
    split, stats, model, lora_report, bridge, optimizer = _build_training_components(
        config,
        training_horizon=selected_steps,
    )
    base_artifact = training_base_artifact_identity(model)
    frozen_inputs = (
        collect_t3b_frozen_input_evidence(
            config,
            model,
            runtime_statistics=stats,
        )
        if launch_document is not None
        else None
    )
    if prebuild_inputs is not None:
        assert frozen_inputs is not None
        _require_unchanged_t3b_inputs(
            prebuild_inputs,
            frozen_inputs,
            context="training-component construction",
        )
    training_bridge_evidence: dict[str, object] | None = None
    if launch_document is not None:
        assert frozen_inputs is not None
        training_bridge_evidence = _validate_t3b_training_bridge_semantics(
            config=config,
            bridge=bridge,
            stats=stats,
            expected_frozen_inputs=frozen_inputs,
        )
    run_config_sha256 = training_run_config_sha256(
        config,
        selected_steps=selected_steps,
        train_statistics_sha256=stats.sha256,
        train_episodes=split.train_episodes,
        holdout_episodes=split.holdout_episodes,
        base_artifact=base_artifact,
        optimizer_config=optimizer.config,
    )
    launch_binding: dict[str, object] | None = None
    if launch_document is not None:
        launch_document = validate_finetune_launch_runtime_binding(
            launch_document,
            config=config,
            budget=budget.artifact,
            train_statistics_sha256=stats.sha256,
            train_episodes=split.train_episodes,
            holdout_episodes=split.holdout_episodes,
            base_artifact=base_artifact,
            optimizer_config=optimizer.config,
            lora_report=lora_report,
            reference_freeze_policy=reference_freeze_policy_evidence(),
            implementation_sha256=finetune_implementation_hashes(),
            frozen_inputs=frozen_inputs,
            training_bridge=training_bridge_evidence,
        )
        if launch_document["run_config_sha256"] != run_config_sha256:
            raise ValueError("launch and runtime run-configuration digests differ")
        assert launch_path is not None and launch_file_sha256 is not None
        launch_binding = {
            "file": launch_path.name,
            "file_sha256": launch_file_sha256,
            "configuration_sha256": launch_document["configuration_sha256"],
            "run_config_sha256": launch_document["run_config_sha256"],
        }
    if not is_resume:
        expected_budget_payload = (
            json.dumps(budget.artifact, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if startup_bindings is not None:
            existing_budget = startup_bindings.budget
        else:
            try:
                existing_budget = _snapshot_regular_file(
                    artifact_path,
                    label="fine-tune budget artifact",
                    capture_payload=True,
                )
            except FileNotFoundError:
                existing_budget = None
        if existing_budget is not None:
            if (
                existing_budget.payload != expected_budget_payload
                or existing_budget.sha256
                != hashlib.sha256(expected_budget_payload).hexdigest()
            ):
                raise FileExistsError(
                    "fine-tune budget artifact appeared with different bytes"
                )
            state_bindings[artifact_path] = existing_budget
            budget_artifact_sha256 = existing_budget.sha256
        else:
            state_bindings[artifact_path] = None
            budget_artifact_sha256 = persist_run_state(
                artifact_path,
                budget.artifact,
            )
    assert budget_artifact_sha256 is not None
    run_immutable_fields: dict[str, object] = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-lora-run",
        "seed": config.seed,
        "sampler_seed": config.sampler_seed,
        "nominal_steps": config.nominal_steps,
        "selected_steps": selected_steps,
        "effective_batch_size": config.effective_batch_size,
        "training_seconds_budget": config.training_seconds,
        "lora": {
            "scope": lora_report.scope,
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
    }
    if launch_binding is not None:
        run_immutable_fields["launch_configuration"] = launch_binding
        run_immutable_fields["training_bridge"] = training_bridge_evidence
    if benchmark is not None:
        run_immutable_fields["benchmark"] = benchmark.as_dict()
    else:
        run_immutable_fields.update(
            {
                "budget_mode": FIXED_BUDGET_MODE,
                "budget": budget.artifact,
                "budget_sha256": budget_artifact_sha256,
            }
        )
    trainable_names = lora_report.trainable_names
    checkpoint_root = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.csv"
    resume_metrics_snapshot: _StableFileSnapshot | None = None
    start_step = 0
    elapsed_before = 0.0
    previous_peak_memory = 0
    smoothed_loss: float | None = None
    final_update: UpdateResult | None = None
    resume_checkpoint_state: CheckpointState | None = None
    latest_checkpoint: TrainingCheckpoint | None = None
    zero_step_resume = False
    checkpoint_startup_recoveries: tuple[str, ...] = ()
    if is_resume:
        if requires_launch:
            _validate_t3b_resume_run_document(
                run_document,
                expected_immutable=run_immutable_fields,
                selected_steps=selected_steps,
                checkpoint_interval=config.checkpoint_interval,
            )
            assert training_lease is not None
            _validate_recorded_t3b_recovery_inventories(
                run_document,
                output_dir=output_dir,
                lease=training_lease,
            )
            metrics_recovery_inventory = _metrics_recovery_inventory(training_lease)
            recorded_metrics_recoveries = run_document.get("metrics_recoveries", [])
            assert isinstance(recorded_metrics_recoveries, list)
            if any(
                re.fullmatch(r"metrics\.recovery-[0-9]{6}\.csv", name) is None
                for name in recorded_metrics_recoveries
            ):
                raise ValueError("resumable run metrics recovery path is invalid")
            missing_metrics_recoveries = set(recorded_metrics_recoveries).difference(
                metrics_recovery_inventory
            )
            if missing_metrics_recoveries:
                raise FileNotFoundError(
                    "recorded metrics recoveries are missing: "
                    f"{sorted(missing_metrics_recoveries)}"
                )
            run_document = {
                **run_document,
                "metrics_recoveries": list(
                    dict.fromkeys(
                        [
                            *recorded_metrics_recoveries,
                            *metrics_recovery_inventory,
                        ]
                    )
                ),
            }
            bound_run_snapshot = state_bindings[run_path]
            if bound_run_snapshot is None:
                raise RuntimeError("resumable run lost its metadata binding")
            process_recoveries = _reconcile_resume_process_identity(
                training_lease,
                bound_run_snapshot,
            )
            restored_process = _snapshot_regular_file_at(
                training_lease.output_descriptor,
                process_identity_path.name,
                label="restored fine-tune process identity",
                capture_payload=True,
            )
            state_bindings[process_identity_path] = restored_process
            _validate_resume_process_identity(run_document, restored_process)
            output_recoveries = _reconcile_t3b_resume_output_staging(
                training_lease
            )
            startup_recoveries = tuple(
                dict.fromkeys(
                    [*startup_recoveries, *process_recoveries, *output_recoveries]
                )
            )
            require_output_lease()
        if run_document.get("run_config_sha256") != run_config_sha256:
            raise ValueError("existing run metadata differs from the requested configuration")
        if requires_launch and run_document.get("launch_configuration") != launch_binding:
            raise ValueError("existing run launch-configuration binding is invalid")
        if requires_launch and run_document.get("training_bridge") != (
            training_bridge_evidence
        ):
            raise ValueError("existing run training-bridge binding is invalid")
        if training_lease is not None:
            _bind_t3b_checkpoint_root(training_lease, allow_existing=True)
            require_output_lease()
        zero_step_artifacts = (
            output_dir / "adapter.safetensors",
            output_dir / "adapter.json",
            output_dir / "export",
        )
        zero_step_metadata_eligible = (
            run_document.get("status") in {"interrupted", "running"}
            and run_document.get("last_completed_step") in {None, 0, 1}
            and run_document.get("checkpoint_count") == 0
            and "last_checkpoint" not in run_document
            and all(
                not path.exists() and not path.is_symlink()
                for path in zero_step_artifacts
            )
        )
        if zero_step_metadata_eligible:
            require_output_lease()
            prepared_checkpoint_root = _prepare_zero_step_checkpoint_replay(
                checkpoint_root,
                output_dir=output_dir,
                output_descriptor=(
                    None
                    if training_lease is None
                    else training_lease.output_descriptor
                ),
                expected_output_snapshot=(
                    None if training_lease is None else training_lease.output_snapshot
                ),
                checkpoint_root_descriptor=(
                    None
                    if training_lease is None
                    else training_lease.checkpoint_root_descriptor
                ),
                expected_checkpoint_root_snapshot=(
                    None
                    if training_lease is None
                    else training_lease.checkpoint_root_snapshot
                ),
            )
            require_output_lease()
            if prepared_checkpoint_root is not None:
                zero_step_resume = True
                checkpoint_startup_recoveries = prepared_checkpoint_root
            elif training_lease is not None:
                recovered_inventory = _zero_step_checkpoint_recovery_inventory(
                    output_dir,
                    output_descriptor=training_lease.output_descriptor,
                    expected_output_snapshot=training_lease.output_snapshot,
                )
                if recovered_inventory is None:
                    raise ValueError("checkpoint recovery inventory is unsafe")
                checkpoint_startup_recoveries = recovered_inventory
        else:
            expected_staging_step = _next_checkpoint_staging_step(
                run_document,
                selected_steps=selected_steps,
                checkpoint_interval=config.checkpoint_interval,
            )
            prepared_checkpoint_root = _prepare_zero_step_checkpoint_replay(
                checkpoint_root,
                output_dir=output_dir,
                output_descriptor=(
                    None
                    if training_lease is None
                    else training_lease.output_descriptor
                ),
                expected_output_snapshot=(
                    None
                    if training_lease is None
                    else training_lease.output_snapshot
                ),
                checkpoint_root_descriptor=(
                    None
                    if training_lease is None
                    else training_lease.checkpoint_root_descriptor
                ),
                expected_checkpoint_root_snapshot=(
                    None
                    if training_lease is None
                    else training_lease.checkpoint_root_snapshot
                ),
                expected_staging_step=(
                    selected_steps
                    if expected_staging_step is None
                    else expected_staging_step
                ),
                allow_published_entries=True,
                allowed_published_steps=_allowed_t3b_published_checkpoint_steps(
                    run_document,
                    selected_steps=selected_steps,
                    checkpoint_interval=config.checkpoint_interval,
                ),
            )
            if prepared_checkpoint_root is None:
                raise ValueError(
                    "checkpoint root contains an unrecognized interrupted save"
                )
            checkpoint_startup_recoveries = prepared_checkpoint_root
        if not zero_step_resume:
            require_output_lease()
            if training_lease is not None:
                resume_metrics_snapshot = _snapshot_regular_file_at(
                    training_lease.output_descriptor,
                    metrics_path.name,
                    label="fine-tune metrics",
                    capture_payload=True,
                )
                state_bindings[metrics_path] = resume_metrics_snapshot
            checkpoint = load_latest_training_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_root=checkpoint_root,
                trainable_names=trainable_names,
                expected_run_config_sha256=run_config_sha256,
                checkpoint_parent_descriptor=(
                    None
                ),
                expected_checkpoint_parent_snapshot=(
                    None
                ),
                checkpoint_root_descriptor=(
                    None
                    if training_lease is None
                    else training_lease.checkpoint_root_descriptor
                ),
                expected_checkpoint_root_snapshot=(
                    None
                    if training_lease is None
                    else training_lease.checkpoint_root_snapshot
                ),
                expected_selected_steps=selected_steps,
                expected_effective_batch_size=config.effective_batch_size,
                expected_checkpoint_interval=config.checkpoint_interval,
                metrics_parent_descriptor=(
                    None
                    if training_lease is None
                    else training_lease.output_descriptor
                ),
                expected_metrics_snapshot=resume_metrics_snapshot,
                expected_last_checkpoint=run_document.get("last_checkpoint"),
                allowed_uncommitted_step=_next_checkpoint_staging_step(
                    run_document,
                    selected_steps=selected_steps,
                    checkpoint_interval=config.checkpoint_interval,
                ),
            )
            if run_document.get("metrics") is not None:
                if resume_metrics_snapshot is None:
                    raise FileNotFoundError(
                        "resumable final metrics evidence has no bound metrics file"
                    )
                _validate_t3b_resume_metrics_evidence(
                    run_document,
                    snapshot=resume_metrics_snapshot,
                    checkpoint_state=checkpoint.state,
                )
                require_output_lease()
            if training_lease is not None:
                post_load_recoveries = _prepare_zero_step_checkpoint_replay(
                    checkpoint_root,
                    output_dir=output_dir,
                    output_descriptor=training_lease.output_descriptor,
                    expected_output_snapshot=training_lease.output_snapshot,
                    checkpoint_root_descriptor=training_lease.checkpoint_root_descriptor,
                    expected_checkpoint_root_snapshot=(
                        training_lease.checkpoint_root_snapshot
                    ),
                    expected_staging_step=_next_checkpoint_staging_step(
                        run_document,
                        selected_steps=selected_steps,
                        checkpoint_interval=config.checkpoint_interval,
                    )
                    or selected_steps,
                    allow_published_entries=True,
                    allowed_published_steps=_allowed_t3b_published_checkpoint_steps(
                        run_document,
                        selected_steps=selected_steps,
                        checkpoint_interval=config.checkpoint_interval,
                    ),
                )
                if post_load_recoveries is None:
                    raise RuntimeError(
                        "checkpoint recovery inventory changed after selection"
                    )
                checkpoint_startup_recoveries = tuple(
                    dict.fromkeys(
                        [*checkpoint_startup_recoveries, *post_load_recoveries]
                    )
                )
            require_output_lease()
            checkpoint_state = checkpoint.state
            resume_checkpoint_state = checkpoint_state
            latest_checkpoint = checkpoint
            if (
                training_lease is not None
                and checkpoint.state.completed_step == selected_steps
            ):
                _bind_t3b_final_checkpoint_namespace(
                    training_lease,
                    checkpoint=checkpoint,
                )
                require_output_lease()
            expected_draws = checkpoint_state.completed_step * config.effective_batch_size
            if (
                checkpoint_state.selected_steps != selected_steps
                or checkpoint_state.samples_consumed != expected_draws
                or checkpoint_state.flow_draw_count != expected_draws
            ):
                raise ValueError(
                    "checkpoint counters differ from the requested training trajectory"
                )
            run_document = _reconcile_loaded_checkpoint_run_document(
                run_document,
                checkpoint=checkpoint,
                selected_steps=selected_steps,
                checkpoint_interval=config.checkpoint_interval,
            )
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
            "process": {
                "pid": os.getpid(),
                "started_at_ns": time.time_ns(),
            },
            "resume_count": int(run_document.get("resume_count", 0)) + 1,
            "resumed_from_step": start_step,
            "last_interruption": run_document.get("interruption"),
            "startup_recoveries": list(
                dict.fromkeys(
                    [
                        *run_document.get("startup_recoveries", []),
                        *startup_recoveries,
                    ]
                )
            ),
            "checkpoint_recoveries": list(
                dict.fromkeys(
                    [
                        *run_document.get("checkpoint_recoveries", []),
                        *checkpoint_startup_recoveries,
                    ]
                )
            ),
        }
    else:
        run_document = {
            **run_immutable_fields,
            "status": "running",
            "checkpoint_count": 0,
            "resume_count": 0,
            "metrics_recoveries": [],
            "checkpoint_recoveries": [],
            "startup_recoveries": list(startup_recoveries),
            "disk_free_before_bytes": disk_free_before,
            "process": {
                "pid": os.getpid(),
                "started_at_ns": time.time_ns(),
            },
        }
    if process_identity is not None:
        if is_resume and training_lease is not None:
            prior_process_identity = state_bindings[process_identity_path]
            if prior_process_identity is None:
                raise RuntimeError("resumable run lost its process identity binding")
            require_output_lease()
            previous_process_name = _backup_resume_process_identity(
                training_lease,
                prior_process_identity,
            )
            previous_process_path = (
                f"process-identity-recoveries/{previous_process_name}"
            )
            run_document["startup_recoveries"] = list(
                dict.fromkeys(
                    [
                        *run_document.get("startup_recoveries", []),
                        previous_process_path,
                    ]
                )
            )
            state_bindings[process_identity_path] = None
            require_output_lease()
        process_identity_sha256 = persist_run_state(
            process_identity_path,
            process_identity,
        )
        run_document["process"] = {
            **process_identity,
            "identity_file": "training.pid",
            "identity_sha256": process_identity_sha256,
        }
    if requires_launch:
        assert training_lease is not None
        _validate_recorded_t3b_recovery_inventories(
            run_document,
            output_dir=output_dir,
            lease=training_lease,
        )
    persist_run_state(run_path, run_document)
    if not is_resume and training_lease is not None:
        _bind_t3b_checkpoint_root(training_lease, allow_existing=False)
        require_output_lease()

    train_start = time.perf_counter()
    mx.reset_peak_memory()
    completed_step = start_step
    try:
        if zero_step_resume and metrics_path.is_symlink():
            raise FileExistsError(
                f"zero-step resume metrics path is unsafe: {metrics_path}"
            )
        metrics_resume_step = (
            start_step
            if is_resume and (not zero_step_resume or metrics_path.is_file())
            else None
        )
        require_output_lease()
        with MetricsWriter(
            metrics_path,
            resume_from_step=metrics_resume_step,
            checkpoint_state=resume_checkpoint_state,
            reinitialize_zero_step=(
                zero_step_resume and metrics_resume_step == 0
            ),
            parent_descriptor=(
                None if training_lease is None else training_lease.output_descriptor
            ),
            expected_parent_snapshot=(
                None if training_lease is None else training_lease.output_snapshot
            ),
            expected_source_snapshot=resume_metrics_snapshot,
        ) as metrics:
            state_bindings.pop(metrics_path, None)
            require_output_lease()
            if metrics.recovery_path is not None:
                run_document = {
                    **run_document,
                    "metrics_recoveries": list(
                        dict.fromkeys(
                            [
                                *run_document.get("metrics_recoveries", []),
                                metrics.recovery_path.name,
                            ]
                        )
                    ),
                }
                if requires_launch:
                    assert training_lease is not None
                    _validate_recorded_t3b_recovery_inventories(
                        run_document,
                        output_dir=output_dir,
                        lease=training_lease,
                    )
                persist_run_state(run_path, run_document)
            if launch_path is not None:
                current_frozen_inputs = collect_t3b_frozen_input_evidence(
                    config,
                    model,
                    runtime_statistics=stats,
                )
                assert training_lease is not None
                current_launch_snapshot = _snapshot_regular_file_at(
                    training_lease.output_descriptor,
                    "launch.json",
                    label="T3B launch configuration",
                    capture_payload=True,
                )
                current_launch, current_launch_file_sha256 = (
                    _validate_finetune_launch_snapshot(current_launch_snapshot)
                )
                if (
                    not _same_bound_file_snapshot(
                        current_launch_snapshot,
                        state_bindings[launch_path],
                    )
                    or
                    current_launch != launch_document
                    or current_launch_file_sha256 != launch_file_sha256
                ):
                    raise RuntimeError(
                        "launch configuration changed before the first optimizer update"
                    )
                validate_finetune_launch_runtime_binding(
                    current_launch,
                    config=config,
                    budget=budget.artifact,
                    train_statistics_sha256=stats.sha256,
                    train_episodes=split.train_episodes,
                    holdout_episodes=split.holdout_episodes,
                    base_artifact=base_artifact,
                    optimizer_config=optimizer.config,
                    lora_report=lora_report,
                    reference_freeze_policy=reference_freeze_policy_evidence(),
                    implementation_sha256=finetune_implementation_hashes(),
                    frozen_inputs=current_frozen_inputs,
                    training_bridge=training_bridge_evidence,
                )
            for step_index in range(start_step, selected_steps):
                require_output_lease()
                update = _optimizer_update(
                    model=model,
                    bridge=bridge,
                    optimizer=optimizer,
                    effective_batch_size=config.effective_batch_size,
                )
                require_output_lease()
                completed_step = step_index + 1
                final_update = update
                smoothed_loss = (
                    update.loss
                    if smoothed_loss is None
                    else 0.98 * smoothed_loss + 0.02 * update.loss
                )
                elapsed = elapsed_before + time.perf_counter() - train_start
                peak_memory = max(previous_peak_memory, int(mx.get_peak_memory()))
                require_output_lease()
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
                require_output_lease()
                if (
                    completed_step == 1
                    or completed_step % config.checkpoint_interval == 0
                    or completed_step == selected_steps
                ):
                    bridge_checkpoint_state = bridge.state_dict()
                    require_output_lease()
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
                        checkpoint_parent_descriptor=(
                            None
                        ),
                        expected_checkpoint_parent_snapshot=(
                            None
                        ),
                        checkpoint_root_descriptor=(
                            None
                            if training_lease is None
                            else training_lease.checkpoint_root_descriptor
                        ),
                        expected_checkpoint_root_snapshot=(
                            None
                            if training_lease is None
                            else training_lease.checkpoint_root_snapshot
                        ),
                        expected_existing_checkpoint_steps=(
                            None
                            if training_lease is None
                            else _allowed_t3b_published_checkpoint_steps(
                                run_document,
                                selected_steps=selected_steps,
                                checkpoint_interval=config.checkpoint_interval,
                            ).difference({completed_step})
                        ),
                    )
                    latest_checkpoint = checkpoint
                    require_output_lease()
                    next_run_document = {
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
                    if requires_launch:
                        assert training_lease is not None
                        _validate_recorded_t3b_recovery_inventories(
                            next_run_document,
                            output_dir=output_dir,
                            lease=training_lease,
                        )
                    if training_lease is None:
                        persist_run_state(run_path, next_run_document)
                    else:
                        assert training_lease.checkpoint_root_descriptor is not None
                        assert training_lease.checkpoint_root_snapshot is not None
                        _persist_run_state_with_checkpoint_binding(
                            checkpoint=checkpoint,
                            checkpoint_root_descriptor=(
                                training_lease.checkpoint_root_descriptor
                            ),
                            expected_checkpoint_root_snapshot=(
                                training_lease.checkpoint_root_snapshot
                            ),
                            persist=persist_run_state,
                            path=run_path,
                            value=next_run_document,
                        )
                        if completed_step == selected_steps:
                            _bind_t3b_final_checkpoint_namespace(
                                training_lease,
                                checkpoint=checkpoint,
                            )
                            require_output_lease()
                    run_document = next_run_document
                if progress is not None:
                    progress(completed_step, selected_steps, update)
        require_output_lease()
    except BaseException as error:
        interrupted_document = {
            **run_document,
            "status": "interrupted",
            "last_completed_step": completed_step,
            "interruption": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        if requires_launch:
            assert training_lease is not None
            _validate_recorded_t3b_recovery_inventories(
                interrupted_document,
                output_dir=output_dir,
                lease=training_lease,
            )
        persist_run_state(run_path, interrupted_document)
        raise
    if final_update is None or smoothed_loss is None:
        raise RuntimeError("fine-tune loop completed without an optimizer update")
    if (
        latest_checkpoint is None
        or latest_checkpoint.state.completed_step != completed_step
        or completed_step != selected_steps
    ):
        raise RuntimeError("fine-tune completion lacks its final checkpoint binding")
    if training_lease is None:
        final_metrics_snapshot = _snapshot_regular_file(
            metrics_path,
            label="final fine-tune metrics",
            capture_payload=True,
        )
        metrics_evidence = _final_metrics_evidence(
            final_metrics_snapshot,
            checkpoint_state=latest_checkpoint.state,
        )
    else:
        metrics_evidence = _bind_t3b_metrics_file(
            training_lease,
            checkpoint_state=latest_checkpoint.state,
        )
        assert training_lease.metrics_file_binding is not None
        state_bindings[metrics_path] = training_lease.metrics_file_binding[0]
        _revalidate_t3b_metrics_file(training_lease, verify_bytes=True)
    actual_training_seconds = elapsed_before + time.perf_counter() - train_start
    peak_memory = max(previous_peak_memory, int(mx.get_peak_memory()))
    run_document = {
        **run_document,
        "status": "exporting",
        "last_completed_step": completed_step,
        "actual_training_seconds": actual_training_seconds,
        "peak_memory_bytes": peak_memory,
        "metrics": metrics_evidence,
    }
    if requires_launch:
        assert training_lease is not None
        _validate_recorded_t3b_recovery_inventories(
            run_document,
            output_dir=output_dir,
            lease=training_lease,
        )
    persist_run_state(run_path, run_document)
    require_output_lease()
    adapter_sha256 = _save_adapter_checkpoint(
        model,
        output_dir / "adapter.safetensors",
        lora_report=lora_report,
        parent_descriptor=(
            None if training_lease is None else training_lease.output_descriptor
        ),
        expected_parent_snapshot=(
            None if training_lease is None else training_lease.output_snapshot
        ),
    )
    if training_lease is not None:
        _bind_t3b_adapter_files(
            training_lease,
            expected_sha256=adapter_sha256,
            lora_report=lora_report,
        )
    require_output_lease()
    merge_report = merge_lora(model, dtype=mx.float32)
    if merge_report.scope != lora_report.scope:
        raise RuntimeError("merged LoRA scope differs from the installed topology")
    if launch_document is not None:
        export_frozen_inputs = collect_t3b_frozen_input_evidence(
            config,
            model,
            runtime_statistics=stats,
            validate_runtime_model=False,
        )
        validate_finetune_launch_runtime_binding(
            launch_document,
            config=config,
            budget=budget.artifact,
            train_statistics_sha256=stats.sha256,
            train_episodes=split.train_episodes,
            holdout_episodes=split.holdout_episodes,
            base_artifact=base_artifact,
            optimizer_config=optimizer.config,
            lora_report=lora_report,
            reference_freeze_policy=reference_freeze_policy_evidence(),
            implementation_sha256=finetune_implementation_hashes(),
            frozen_inputs=export_frozen_inputs,
            training_bridge=training_bridge_evidence,
        )
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
    if config.lora_scope != LEGACY_FULL_SCOPE:
        export_metadata["lora_scope"] = config.lora_scope
    export_dir = output_dir / "export"
    if launch_document is not None:
        require_output_lease()
        with (
            _private_t3b_source_checkpoint(
                config=config,
                expected_evidence=launch_document["frozen_inputs"],
                output_descriptor=(
                    None
                    if training_lease is None
                    else training_lease.output_descriptor
                ),
                expected_output_snapshot=(
                    None if training_lease is None else training_lease.output_snapshot
                ),
            ) as source_checkpoint,
            _private_t3b_tokenizer_snapshot(
                config=config,
                expected_evidence=launch_document["frozen_inputs"],
                output_descriptor=(
                    None
                    if training_lease is None
                    else training_lease.output_descriptor
                ),
                expected_output_snapshot=(
                    None if training_lease is None else training_lease.output_snapshot
                ),
            ) as tokenizer_snapshot,
        ):
            require_output_lease()
            support_sha256 = expected_merged_checkpoint_support_file_sha256(
                source_checkpoint_dir=source_checkpoint,
                processor_stats=stats.processor_stats,
                tokenizer_dir=tokenizer_snapshot,
            )
            export_metadata["support_file_sha256"] = support_sha256
            assert training_lease is not None
            try:
                existing_export = os.stat(
                    export_dir.name,
                    dir_fd=training_lease.output_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                existing_export = None
            if existing_export is not None and not stat.S_ISDIR(
                existing_export.st_mode
            ):
                raise FileExistsError("T3B export path is unsafe")
            if existing_export is None:
                require_output_lease()
                export_merged_checkpoint(
                    model=model,
                    source_checkpoint_dir=source_checkpoint,
                    output_dir=export_dir,
                    processor_stats=stats.processor_stats,
                    metadata=export_metadata,
                    tokenizer_dir=tokenizer_snapshot,
                    output_parent_descriptor=(
                        None
                        if training_lease is None
                        else training_lease.output_descriptor
                    ),
                    expected_output_parent=(
                        None
                        if training_lease is None
                        else (
                            training_lease.output_snapshot.path,
                            training_lease.output_snapshot.components[-1][1],
                            training_lease.output_snapshot.components[-1][2],
                        )
                    ),
                )
                require_output_lease()
            _bind_t3b_export_root(training_lease)
            require_output_lease()
            assert training_lease.export_descriptor is not None
            export_report = validate_bound_merged_checkpoint_export(
                output_descriptor=training_lease.export_descriptor,
                source_checkpoint_descriptor=source_checkpoint.descriptor,
                expected_metadata=export_metadata,
                expected_support_sha256=support_sha256,
                model=model,
            )
            _bind_t3b_export_files(
                training_lease,
                expected_report=export_report,
                expected_metadata=export_metadata,
            )
            require_output_lease()
        require_output_lease()
    elif export_dir.exists() or export_dir.is_symlink():
        require_output_lease()
        export_report = validate_merged_checkpoint_export(
            export_dir,
            expected_metadata=export_metadata,
        )
        require_output_lease()
    else:
        source_checkpoint = resolve_base_checkpoint(config.cache_dir)
        require_output_lease()
        export_report = export_merged_checkpoint(
            model=model,
            source_checkpoint_dir=source_checkpoint,
            output_dir=export_dir,
            processor_stats=stats.processor_stats,
            metadata=export_metadata,
        )
        require_output_lease()
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
    if training_lease is not None:
        _revalidate_t3b_final_checkpoint_namespace(
            training_lease,
            verify_bytes=True,
        )
        _revalidate_t3b_metrics_file(training_lease, verify_bytes=True)
        _revalidate_t3b_adapter_files(training_lease, verify_bytes=True)
        _revalidate_t3b_export_root(training_lease, verify_bytes=True)
    if requires_launch:
        assert training_lease is not None
        _validate_recorded_t3b_recovery_inventories(
            final_state,
            output_dir=output_dir,
            lease=training_lease,
        )
    run_state_sha256 = persist_run_state(output_dir / "run.json", final_state)
    if training_lease is not None:
        _revalidate_t3b_final_checkpoint_namespace(
            training_lease,
            verify_bytes=True,
        )
        _revalidate_t3b_metrics_file(training_lease, verify_bytes=True)
        _revalidate_t3b_adapter_files(training_lease, verify_bytes=True)
        _revalidate_t3b_export_root(training_lease, verify_bytes=True)
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


def run_lora_finetune(
    config: FineTuneConfig,
    *,
    launch_config_path: str | Path | None = None,
    progress: Callable[[int, int, UpdateResult], None] | None = None,
    training_log_path: str | Path | None = None,
) -> FineTuneResult:
    """Run fine-tuning while holding the mandatory T3B single-owner lock."""

    if not _requires_t3b_launch_config(config):
        if training_log_path is not None:
            raise ValueError("bound training logs are only valid for a T3B launch")
        return _run_lora_finetune_impl(
            config,
            launch_config_path=launch_config_path,
            progress=progress,
        )
    _validate_t3b_frozen_config(config)
    _require_t3b_runtime_provenance(allow_unfrozen=True)
    output_dir = _safe_t3b_output_path(
        config.output_dir,
        must_exist=True,
        label="T3B fine-tune output path",
    )
    lease = _acquire_t3b_training_lock(output_dir)
    log_lease: _T3BTrainingLogLease | None = None
    try:
        _revalidate_t3b_training_lock(lease)
        _restore_missing_t3b_previous_generations(lease)
        _revalidate_t3b_training_lock(lease)
        if training_log_path is None:
            raise ValueError(
                "T3B fine-tuning requires OUTPUT/training.log"
            )
        training_log_identity: Mapping[str, object] | None = None
        initial_run_snapshot = _optional_t3b_startup_file(
            lease,
            "run.json",
            label="T3B run metadata",
            capture_payload=True,
        )
        run_exists = initial_run_snapshot is not None
        if config.resume and not run_exists:
            raise FileNotFoundError(
                f"fine-tune run has no resumable metadata: {output_dir}"
            )
        if not config.resume and run_exists:
            raise FileExistsError(
                f"refusing to overwrite existing fine-tune run {output_dir}"
            )
        if training_log_path is not None:
            expected_log_identity: Mapping[str, object] | None = None
            if config.resume:
                assert initial_run_snapshot is not None
                prior_run = _json_from_stable_snapshot(
                    initial_run_snapshot,
                    label="T3B run metadata",
                )
                if not isinstance(prior_run, Mapping):
                    raise ValueError("T3B run metadata must be an object")
                prior_process = prior_run.get("process")
                if not isinstance(prior_process, Mapping) or not isinstance(
                    prior_process.get("training_log"), Mapping
                ):
                    raise ValueError("resumable run has no bound training-log identity")
                expected_log_identity = prior_process["training_log"]
            log_lease = _open_t3b_training_log(
                lease,
                training_log_path,
                resume=config.resume,
                expected_identity=expected_log_identity,
            )
            training_log_identity = log_lease.identity

        def execute() -> FineTuneResult:
            startup_recoveries = (
                _reconcile_t3b_prestart_output(
                    output_dir,
                    config=config,
                    lease=lease,
                )
                if not run_exists
                else ()
            )
            if log_lease is not None:
                _revalidate_t3b_training_log(log_lease, lease)
            startup_bindings = _capture_t3b_startup_bindings(lease)
            if initial_run_snapshot is None:
                if startup_bindings.run is not None:
                    raise RuntimeError(
                        "T3B run metadata appeared during startup binding"
                    )
            elif startup_bindings.run is None or not _same_bound_file_snapshot(
                initial_run_snapshot,
                startup_bindings.run,
            ):
                raise RuntimeError("T3B run metadata changed during startup binding")
            result = _run_lora_finetune_impl(
                config,
                launch_config_path=launch_config_path,
                progress=progress,
                training_lease=lease,
                training_log_lease=log_lease,
                training_log_identity=training_log_identity,
                startup_recoveries=startup_recoveries,
                startup_bindings=startup_bindings,
            )
            _revalidate_t3b_training_lock(lease)
            if log_lease is not None:
                _revalidate_t3b_training_log(log_lease, lease)
            return result

        if log_lease is None:
            return execute()
        with _redirect_standard_streams_to_log(log_lease.descriptor):
            try:
                return execute()
            except BaseException:
                traceback.print_exc()
                raise
    finally:
        if log_lease is not None:
            os.close(log_lease.descriptor)
        _release_t3b_training_lock(lease)
