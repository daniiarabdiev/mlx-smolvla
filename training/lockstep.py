"""Native MLX 25-step SmolVLA optimizer lockstep against PyTorch goldens."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from mlx_smolvla._lab.training.data import TrainingArtifact
from mlx_smolvla._lab.training.differentiable import differentiable_cpu_primitives
from mlx_smolvla._lab.training.gradients import (
    GradientComparison,
    canonical_parameter_name,
    compare_gradient_arrays,
    configure_reference_trainable,
    relative_loss_difference,
)
from mlx_smolvla._lab.training.model import SmolVLATrainingModel, TrainingBatch, training_loss
from mlx_smolvla._lab.training.optimizer import (
    SmolVLAAdamW,
    SmolVLAOptimizerConfig,
    clip_gradients_by_global_norm,
    cosine_decay_with_warmup_lr,
)
from mlx_smolvla._lab.training.parity import (
    load_serialized_training_batch,
    validate_checkpoint_parameter_identity,
)


LOSS_RELATIVE_TOLERANCE = 1e-3
PARAMETER_RELATIVE_L2_TOLERANCE = 5e-3

_EXPECTED_STEPS = 25
_EXPECTED_TRAINABLE_TENSORS = 155
_EXPECTED_TRAINABLE_SCALARS = 99_880_992
_EXPECTED_OPTIMIZER_TENSORS = 330
_MINIMUM_FREE_BYTES = 40 * 1024**3


@dataclass(frozen=True)
class StepLossComparison:
    """Reference and MLX metrics immediately before one optimizer update."""

    step: int
    reference_loss: float
    mlx_loss: float
    relative_difference: float
    learning_rate: float
    reference_gradient_norm: float
    mlx_gradient_norm: float
    reference_clip_coefficient: float
    mlx_clip_coefficient: float


@dataclass(frozen=True)
class OptimizerLockstepResult:
    """Complete machine-readable evidence for the immutable T2 gate."""

    passed: bool
    device: str
    dtype: str
    python_version: str
    macos_version: str
    mlx_version: str
    checkpoint: dict[str, str]
    dataset: dict[str, str]
    step_count: int
    training_horizon: int
    t1_manifest_sha256: str
    optimizer_manifest_sha256: str
    parameter_match_count: int
    trainable_scalar_count: int
    loss_comparisons: tuple[StepLossComparison, ...]
    parameter_comparisons: tuple[GradientComparison, ...]
    worst_loss_steps: tuple[StepLossComparison, ...]
    worst_parameters: tuple[GradientComparison, ...]
    maximum_loss_relative_difference: float
    maximum_parameter_relative_l2: float
    maximum_gradient_norm_relative_difference: float
    maximum_clip_coefficient_absolute_difference: float
    converted_weights_path: str
    update_seconds: float
    total_seconds: float
    active_memory_bytes: int
    peak_memory_bytes: int
    disk_free_before_bytes: int
    disk_free_after_bytes: int

    def as_dict(self) -> dict[str, object]:
        """Return the full JSON-compatible lockstep report."""

        return {
            "active_memory_bytes": self.active_memory_bytes,
            "checkpoint": dict(self.checkpoint),
            "converted_weights_path": self.converted_weights_path,
            "dataset": dict(self.dataset),
            "device": self.device,
            "disk_free_after_bytes": self.disk_free_after_bytes,
            "disk_free_before_bytes": self.disk_free_before_bytes,
            "dtype": self.dtype,
            "loss_comparisons": [asdict(item) for item in self.loss_comparisons],
            "macos_version": self.macos_version,
            "maximum_clip_coefficient_absolute_difference": (
                self.maximum_clip_coefficient_absolute_difference
            ),
            "maximum_gradient_norm_relative_difference": (
                self.maximum_gradient_norm_relative_difference
            ),
            "maximum_loss_relative_difference": self.maximum_loss_relative_difference,
            "maximum_parameter_relative_l2": self.maximum_parameter_relative_l2,
            "mlx_version": self.mlx_version,
            "optimizer_manifest_sha256": self.optimizer_manifest_sha256,
            "parameter_comparisons": [
                asdict(item) for item in self.parameter_comparisons
            ],
            "parameter_match_count": self.parameter_match_count,
            "passed": self.passed,
            "peak_memory_bytes": self.peak_memory_bytes,
            "python_version": self.python_version,
            "step_count": self.step_count,
            "t1_manifest_sha256": self.t1_manifest_sha256,
            "thresholds": {
                "per_step_loss_relative_difference_maximum": LOSS_RELATIVE_TOLERANCE,
                "final_parameter_relative_l2_maximum": PARAMETER_RELATIVE_L2_TOLERANCE,
            },
            "total_seconds": self.total_seconds,
            "trainable_scalar_count": self.trainable_scalar_count,
            "training_horizon": self.training_horizon,
            "update_seconds": self.update_seconds,
            "worst_loss_steps": [asdict(item) for item in self.worst_loss_steps],
            "worst_parameters": [asdict(item) for item in self.worst_parameters],
        }


def validate_optimizer_artifact_link(
    t1_artifact: TrainingArtifact,
    optimizer_artifact: TrainingArtifact,
) -> tuple[str, str]:
    """Require complete, pinned, one-to-one T1 and T2 artifact linkage."""

    t1_names = t1_artifact.verify_all()
    optimizer_names = optimizer_artifact.verify_all()
    t1_metadata = t1_artifact.metadata
    optimizer_metadata = optimizer_artifact.metadata
    if len(t1_names) != 324 or t1_metadata.get("artifact_type") != "smolvla-gradient-golden":
        raise ValueError("T1 gradient artifact contract changed")
    expected_optimizer_fields = {
        "artifact_type": "smolvla-optimizer-golden",
        "device": "cpu",
        "dtype": "float32",
        "step_count": _EXPECTED_STEPS,
        "training_horizon": 100_000,
        "trainable_tensor_count": _EXPECTED_TRAINABLE_TENSORS,
        "trainable_scalar_count": _EXPECTED_TRAINABLE_SCALARS,
        "tensor_count": _EXPECTED_OPTIMIZER_TENSORS,
        "t1_batch_verified": True,
        "initial_parameters_verified": _EXPECTED_TRAINABLE_TENSORS,
    }
    for key, expected in expected_optimizer_fields.items():
        if optimizer_metadata.get(key) != expected:
            raise ValueError(
                f"optimizer artifact {key} mismatch: "
                f"{optimizer_metadata.get(key)!r} != {expected!r}"
            )
    if len(optimizer_names) != _EXPECTED_OPTIMIZER_TENSORS:
        raise ValueError("optimizer artifact payload count changed")
    t1_hash = str(t1_metadata["manifest_sha256"])
    optimizer_hash = str(optimizer_metadata["manifest_sha256"])
    if optimizer_metadata.get("t1_manifest_sha256") != t1_hash:
        raise ValueError("optimizer artifact does not bind the loaded T1 manifest")
    for source_key in ("checkpoint", "base_vlm", "dataset"):
        if optimizer_metadata.get(source_key) != t1_metadata.get(source_key):
            raise ValueError(f"optimizer artifact {source_key} pin differs from T1")
    config = SmolVLAOptimizerConfig()
    expected_optimizer = {
        "type": "torch.optim.AdamW",
        "lr": config.lr,
        "betas": list(config.betas),
        "eps": config.eps,
        "weight_decay": config.weight_decay,
        "grad_clip_norm": config.grad_clip_norm,
        "bias_correction": True,
        "weight_decay_semantics": "decoupled multiplicative before moment update",
        "epsilon_placement": "after bias-corrected sqrt(second moment)",
    }
    if optimizer_metadata.get("optimizer") != expected_optimizer:
        raise ValueError("optimizer artifact AdamW semantics differ from the audited preset")
    return t1_hash, optimizer_hash


def load_lockstep_training_batch(
    t1_artifact: TrainingArtifact,
    optimizer_artifact: TrainingArtifact,
    *,
    step: int,
    base_batch: TrainingBatch | None = None,
) -> TrainingBatch:
    """Load the exact T1 batch with one T2 step's serialized flow draws."""

    if not 0 <= step < _EXPECTED_STEPS:
        raise IndexError(f"optimizer lockstep step must be in [0, 24], got {step}")
    batch = load_serialized_training_batch(t1_artifact) if base_batch is None else base_batch
    return replace(
        batch,
        noise=mx.array(optimizer_artifact.load(f"draws/{step:03d}/noise")),
        timesteps=mx.array(optimizer_artifact.load(f"draws/{step:03d}/timesteps")),
    )


def _selected_parameters(model: SmolVLATrainingModel) -> dict[str, mx.array]:
    selected_names = configure_reference_trainable(model)
    flat_parameters = tuple(tree_flatten(model.trainable_parameters()))
    if tuple(name for name, _ in flat_parameters) != selected_names:
        raise RuntimeError("selected parameter traversal changed during optimizer lockstep")
    canonical: dict[str, mx.array] = {}
    for name, parameter in flat_parameters:
        canonical_name = canonical_parameter_name(name)
        if canonical_name in canonical:
            raise RuntimeError(f"duplicate canonical parameter: {canonical_name}")
        canonical[canonical_name] = parameter
    return canonical


def _assert_gradient_names(gradients: dict, expected_names: tuple[str, ...]) -> None:
    names = tuple(
        sorted(canonical_parameter_name(name) for name, _ in tree_flatten(gradients))
    )
    if names != expected_names:
        missing = sorted(set(expected_names) - set(names))
        unexpected = sorted(set(names) - set(expected_names))
        raise RuntimeError(
            f"optimizer lockstep gradient names differ; missing={missing}, "
            f"unexpected={unexpected}"
        )


def run_optimizer_lockstep(
    *,
    t1_dir: str | Path,
    optimizer_golden_dir: str | Path,
    native_cache: str | Path,
) -> OptimizerLockstepResult:
    """Execute and compare the complete 25-step native MLX optimizer window."""

    repository_root = Path(__file__).resolve().parents[1]
    disk_free_before = shutil.disk_usage(repository_root).free
    if disk_free_before < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"optimizer lockstep requires at least {_MINIMUM_FREE_BYTES} free bytes, "
            f"got {disk_free_before}"
        )
    total_start = time.perf_counter()
    t1_artifact = TrainingArtifact(Path(t1_dir))
    optimizer_artifact = TrainingArtifact(Path(optimizer_golden_dir))
    t1_hash, optimizer_hash = validate_optimizer_artifact_link(
        t1_artifact,
        optimizer_artifact,
    )
    config = SmolVLAOptimizerConfig()

    with mx.stream(mx.cpu):
        if mx.default_device() != mx.cpu:
            raise RuntimeError(f"optimizer lockstep selected {mx.default_device()}, expected CPU")
        model = SmolVLATrainingModel.from_pretrained(
            cache_dir=native_cache,
            dtype=mx.float32,
        )
        expected_names = validate_checkpoint_parameter_identity(model, t1_artifact)
        base_batch = load_serialized_training_batch(t1_artifact)
        optimizer = SmolVLAAdamW(config)
        loss_comparisons: list[StepLossComparison] = []
        gradient_norm_relative_differences: list[float] = []
        clip_coefficient_differences: list[float] = []
        mx.reset_peak_memory()
        update_start = time.perf_counter()

        with differentiable_cpu_primitives():
            for step in range(_EXPECTED_STEPS):
                batch = load_lockstep_training_batch(
                    t1_artifact,
                    optimizer_artifact,
                    step=step,
                    base_batch=base_batch,
                )
                value_and_grad = nn.value_and_grad(
                    model,
                    lambda: training_loss(model, batch),
                )
                loss, gradients = value_and_grad()
                _assert_gradient_names(gradients, expected_names)
                mx.eval(loss, gradients)
                clip_result = clip_gradients_by_global_norm(
                    gradients,
                    config.grad_clip_norm,
                )
                used_learning_rate = optimizer.update(model, clip_result.gradients)
                mx.eval(model.trainable_parameters(), optimizer.state)

                reference_learning_rate = float(
                    optimizer_artifact.load(f"steps/{step:03d}/lr_used")
                )
                expected_learning_rate = cosine_decay_with_warmup_lr(step, config)
                if used_learning_rate != expected_learning_rate:
                    raise RuntimeError(
                        f"MLX optimizer LR differs at step {step}: "
                        f"{used_learning_rate} != {expected_learning_rate}"
                    )
                if reference_learning_rate != expected_learning_rate:
                    raise RuntimeError(
                        f"reference artifact LR differs at step {step}: "
                        f"{reference_learning_rate} != {expected_learning_rate}"
                    )

                reference_loss = float(
                    optimizer_artifact.load(f"steps/{step:03d}/loss")
                )
                mlx_loss = float(loss)
                reference_gradient_norm = float(
                    optimizer_artifact.load(f"steps/{step:03d}/gradient_norm")
                )
                mlx_gradient_norm = float(clip_result.total_norm)
                reference_clip = float(
                    optimizer_artifact.load(f"steps/{step:03d}/clip_coefficient")
                )
                mlx_clip = float(clip_result.coefficient)
                loss_comparisons.append(
                    StepLossComparison(
                        step=step,
                        reference_loss=reference_loss,
                        mlx_loss=mlx_loss,
                        relative_difference=relative_loss_difference(
                            reference_loss,
                            mlx_loss,
                        ),
                        learning_rate=used_learning_rate,
                        reference_gradient_norm=reference_gradient_norm,
                        mlx_gradient_norm=mlx_gradient_norm,
                        reference_clip_coefficient=reference_clip,
                        mlx_clip_coefficient=mlx_clip,
                    )
                )
                gradient_norm_relative_differences.append(
                    relative_loss_difference(reference_gradient_norm, mlx_gradient_norm)
                )
                clip_coefficient_differences.append(abs(reference_clip - mlx_clip))

        update_seconds = time.perf_counter() - update_start
        final_parameters = _selected_parameters(model)
        if tuple(sorted(final_parameters)) != expected_names:
            raise RuntimeError("final MLX parameter names differ from the initial identity set")
        mx.eval(*final_parameters.values())
        parameter_comparisons = tuple(
            sorted(
                (
                    compare_gradient_arrays(
                        name,
                        optimizer_artifact.load(f"final_parameters/{name}"),
                        np.asarray(final_parameters[name]),
                    )
                    for name in expected_names
                ),
                key=lambda item: item.name,
            )
        )
        active_memory_bytes = int(mx.get_active_memory())
        peak_memory_bytes = int(max(active_memory_bytes, mx.get_peak_memory()))

    ordered_losses = tuple(sorted(loss_comparisons, key=lambda item: item.step))
    worst_loss_steps = tuple(
        sorted(
            ordered_losses,
            key=lambda item: (-item.relative_difference, item.step),
        )[:5]
    )
    worst_parameters = tuple(
        sorted(
            parameter_comparisons,
            key=lambda item: (-item.relative_l2, item.name),
        )[:5]
    )
    maximum_loss_difference = max(item.relative_difference for item in ordered_losses)
    maximum_parameter_difference = max(item.relative_l2 for item in parameter_comparisons)
    passed = (
        all(
            item.relative_difference <= LOSS_RELATIVE_TOLERANCE
            for item in ordered_losses
        )
        and all(
            item.relative_l2 <= PARAMETER_RELATIVE_L2_TOLERANCE
            for item in parameter_comparisons
        )
    )
    disk_free_after = shutil.disk_usage(repository_root).free
    if disk_free_after < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"optimizer lockstep left less than {_MINIMUM_FREE_BYTES} free bytes: "
            f"{disk_free_after}"
        )
    total_seconds = time.perf_counter() - total_start
    if not math.isfinite(total_seconds) or not math.isfinite(update_seconds):
        raise RuntimeError("optimizer lockstep timing is non-finite")
    metadata = optimizer_artifact.metadata
    return OptimizerLockstepResult(
        passed=passed,
        device="cpu",
        dtype="float32",
        python_version=platform.python_version(),
        macos_version=platform.mac_ver()[0],
        mlx_version=version("mlx"),
        checkpoint=dict(metadata["checkpoint"]),
        dataset=dict(metadata["dataset"]),
        step_count=_EXPECTED_STEPS,
        training_horizon=int(metadata["training_horizon"]),
        t1_manifest_sha256=t1_hash,
        optimizer_manifest_sha256=optimizer_hash,
        parameter_match_count=len(expected_names),
        trainable_scalar_count=int(metadata["trainable_scalar_count"]),
        loss_comparisons=ordered_losses,
        parameter_comparisons=parameter_comparisons,
        worst_loss_steps=worst_loss_steps,
        worst_parameters=worst_parameters,
        maximum_loss_relative_difference=maximum_loss_difference,
        maximum_parameter_relative_l2=maximum_parameter_difference,
        maximum_gradient_norm_relative_difference=max(
            gradient_norm_relative_differences
        ),
        maximum_clip_coefficient_absolute_difference=max(
            clip_coefficient_differences
        ),
        converted_weights_path=str(model.converted_weights_path),
        update_seconds=update_seconds,
        total_seconds=total_seconds,
        active_memory_bytes=active_memory_bytes,
        peak_memory_bytes=peak_memory_bytes,
        disk_free_before_bytes=disk_free_before,
        disk_free_after_bytes=disk_free_after,
    )


def write_optimizer_lockstep_report(
    result: OptimizerLockstepResult,
    output_path: str | Path,
) -> str:
    """Atomically write the complete lockstep report and return its SHA-256."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()
