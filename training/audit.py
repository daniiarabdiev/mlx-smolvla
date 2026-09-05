"""Measured full-architecture differentiability audit for Stage T0."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import math
from pathlib import Path
import platform
import shutil
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_smolvla._lab.training.gradients import canonical_parameter_name, configure_reference_trainable
from mlx_smolvla._lab.training.model import SmolVLATrainingModel, make_random_audit_batch, training_loss


_MINIMUM_FREE_BYTES = 40 * 1024**3


def installed_mlx_version() -> str:
    """Return the installed MLX distribution version from package metadata."""

    return version("mlx")


@dataclass(frozen=True)
class GradientSummary:
    """Evaluated correspondence and numerical state of one gradient tree."""

    tensor_count: int
    scalar_count: int
    all_finite: bool
    zero_norm_names: tuple[str, ...]
    maximum_absolute_value: float


@dataclass(frozen=True)
class TrainingAuditResult:
    """JSON-compatible measurements from one full random-weight train step."""

    device: str
    dtype: str
    seed: int
    python_version: str
    macos_version: str
    mlx_version: str
    microbatch: int
    camera_count: int
    image_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    physical_action_dim: int
    trainable_tensor_count: int
    trainable_scalar_count: int
    gradient_tensor_count: int
    selected_parameter_names: tuple[str, ...]
    canonical_parameter_names: tuple[str, ...]
    all_gradients_finite: bool
    zero_norm_gradient_tensors: tuple[str, ...]
    maximum_absolute_gradient: float
    loss: float
    forward_ms: float
    forward_backward_ms: float
    active_memory_bytes: int
    peak_memory_bytes: int
    disk_free_before_bytes: int
    disk_free_after_bytes: int

    def as_dict(self) -> dict[str, object]:
        """Return a stable machine-readable representation."""

        return {
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "python_version": self.python_version,
            "macos_version": self.macos_version,
            "mlx_version": self.mlx_version,
            "microbatch": self.microbatch,
            "camera_count": self.camera_count,
            "image_shape": list(self.image_shape),
            "action_shape": list(self.action_shape),
            "physical_action_dim": self.physical_action_dim,
            "trainable_tensor_count": self.trainable_tensor_count,
            "trainable_scalar_count": self.trainable_scalar_count,
            "gradient_tensor_count": self.gradient_tensor_count,
            "selected_parameter_names": list(self.selected_parameter_names),
            "canonical_parameter_names": list(self.canonical_parameter_names),
            "all_gradients_finite": self.all_gradients_finite,
            "zero_norm_gradient_tensors": list(self.zero_norm_gradient_tensors),
            "maximum_absolute_gradient": self.maximum_absolute_gradient,
            "loss": self.loss,
            "forward_ms": self.forward_ms,
            "forward_backward_ms": self.forward_backward_ms,
            "active_memory_bytes": self.active_memory_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "disk_free_before_bytes": self.disk_free_before_bytes,
            "disk_free_after_bytes": self.disk_free_after_bytes,
        }


def summarize_gradients(
    parameters: dict,
    gradients: dict,
) -> GradientSummary:
    """Validate one-to-one gradient coverage and summarize evaluated tensors."""

    flat_parameters = tuple(tree_flatten(parameters))
    flat_gradients = tuple(tree_flatten(gradients))
    parameter_names = tuple(name for name, _ in flat_parameters)
    gradient_names = tuple(name for name, _ in flat_gradients)
    if gradient_names != parameter_names:
        raise RuntimeError(
            "gradient names do not match trainable parameter names; "
            f"parameters={parameter_names}, gradients={gradient_names}"
        )

    scalar_count = 0
    all_finite = True
    zero_norm_names: list[str] = []
    maximum_absolute_value = 0.0
    for (name, parameter), (_, gradient) in zip(flat_parameters, flat_gradients, strict=True):
        if gradient.shape != parameter.shape:
            raise RuntimeError(
                f"gradient shape {gradient.shape} does not match parameter {name} shape {parameter.shape}"
            )
        scalar_count += math.prod(parameter.shape)
        finite = bool(mx.all(mx.isfinite(gradient)))
        all_finite = all_finite and finite
        gradient_norm = float(mx.linalg.norm(gradient.astype(mx.float32)))
        if gradient_norm == 0.0:
            zero_norm_names.append(name)
        maximum_absolute_value = max(
            maximum_absolute_value,
            float(mx.max(mx.abs(gradient.astype(mx.float32)))),
        )

    return GradientSummary(
        tensor_count=len(flat_gradients),
        scalar_count=scalar_count,
        all_finite=all_finite,
        zero_norm_names=tuple(zero_norm_names),
        maximum_absolute_value=maximum_absolute_value,
    )


def run_training_readiness_audit(seed: int = 0) -> TrainingAuditResult:
    """Run and measure one full bf16-storage forward/backward training step."""

    repository_root = Path(__file__).resolve().parents[1]
    disk_free_before = shutil.disk_usage(repository_root).free
    if disk_free_before < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"training audit requires at least {_MINIMUM_FREE_BYTES} free bytes, got {disk_free_before}"
        )

    mx.random.seed(seed)
    model = SmolVLATrainingModel()
    model.set_dtype(mx.bfloat16)
    selected_names = configure_reference_trainable(model)
    canonical_names = tuple(canonical_parameter_name(name) for name in selected_names)
    if len(set(canonical_names)) != len(canonical_names):
        raise RuntimeError("canonical training parameter names are not one-to-one")
    batch = make_random_audit_batch(seed)
    mx.eval(*[value for _, value in tree_flatten(model.parameters())])

    forward_start = time.perf_counter()
    forward_loss = training_loss(model, batch)
    mx.eval(forward_loss)
    forward_ms = (time.perf_counter() - forward_start) * 1_000.0

    mx.reset_peak_memory()
    value_and_grad = nn.value_and_grad(model, lambda: training_loss(model, batch))
    step_start = time.perf_counter()
    loss, gradients = value_and_grad()
    flat_gradients = tuple(tree_flatten(gradients))
    mx.eval(loss, *[gradient for _, gradient in flat_gradients])
    forward_backward_ms = (time.perf_counter() - step_start) * 1_000.0

    summary = summarize_gradients(model.trainable_parameters(), gradients)
    if not summary.all_finite:
        raise RuntimeError("training audit produced at least one non-finite gradient tensor")
    if summary.zero_norm_names:
        raise RuntimeError(
            f"training audit produced zero-norm gradients for {summary.zero_norm_names}"
        )

    active_memory_bytes = int(mx.get_active_memory())
    peak_memory_bytes = int(max(active_memory_bytes, mx.get_peak_memory()))
    disk_free_after = shutil.disk_usage(repository_root).free
    if disk_free_after < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"training audit left less than {_MINIMUM_FREE_BYTES} free bytes: {disk_free_after}"
        )

    return TrainingAuditResult(
        device=str(mx.default_device()),
        dtype="bfloat16",
        seed=seed,
        python_version=platform.python_version(),
        macos_version=platform.mac_ver()[0],
        mlx_version=installed_mlx_version(),
        microbatch=batch.actions.shape[0],
        camera_count=batch.processed.pixel_values.shape[0],
        image_shape=tuple(batch.processed.pixel_values.shape),
        action_shape=tuple(batch.actions.shape),
        physical_action_dim=batch.action_dim,
        trainable_tensor_count=len(selected_names),
        trainable_scalar_count=summary.scalar_count,
        gradient_tensor_count=summary.tensor_count,
        selected_parameter_names=selected_names,
        canonical_parameter_names=canonical_names,
        all_gradients_finite=summary.all_finite,
        zero_norm_gradient_tensors=summary.zero_norm_names,
        maximum_absolute_gradient=summary.maximum_absolute_value,
        loss=float(loss),
        forward_ms=forward_ms,
        forward_backward_ms=forward_backward_ms,
        active_memory_bytes=active_memory_bytes,
        peak_memory_bytes=peak_memory_bytes,
        disk_free_before_bytes=disk_free_before,
        disk_free_after_bytes=disk_free_after,
    )
