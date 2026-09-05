"""Strict checkpoint-backed MLX/Torch step-zero gradient parity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

from mlx_smolvla.types import ProcessedObservation
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


LOSS_RELATIVE_TOLERANCE = 1e-4
GRADIENT_RELATIVE_L2_TOLERANCE = 1e-2
GRADIENT_COSINE_MINIMUM = 0.999

_EXPECTED_CHECKPOINT = {
    "id": "lerobot/smolvla_base",
    "revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
}
_EXPECTED_BASE_VLM = {
    "id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
}
_EXPECTED_DATASET = {
    "id": "lerobot/svla_so101_pickplace",
    "revision": "f641879e22172be7e8161d5e6c1503c2d2feb657",
}
_EXPECTED_TRAINABLE_TENSORS = 155
_EXPECTED_TRAINABLE_SCALARS = 99_880_992
_EXPECTED_ARTIFACT_TENSORS = 324
_MINIMUM_FREE_BYTES = 40 * 1024**3


@dataclass(frozen=True)
class GradientParityResult:
    """Complete machine-readable evidence for the immutable T1 gate."""

    passed: bool
    device: str
    dtype: str
    python_version: str
    macos_version: str
    mlx_version: str
    checkpoint: dict[str, str]
    base_vlm: dict[str, str]
    dataset: dict[str, str]
    seed: int
    episode: int
    frame_index: int
    absolute_index: int
    artifact_manifest_sha256: str
    artifact_tensor_count: int
    parameter_match_count: int
    trainable_scalar_count: int
    gradient_count: int
    reference_loss: float
    mlx_loss: float
    loss_relative_difference: float
    maximum_gradient_relative_l2: float
    minimum_gradient_cosine: float
    comparisons: tuple[GradientComparison, ...]
    worst_relative_l2: tuple[GradientComparison, ...]
    worst_cosine: tuple[GradientComparison, ...]
    converted_weights_path: str
    forward_backward_seconds: float
    total_seconds: float
    active_memory_bytes: int
    peak_memory_bytes: int
    disk_free_before_bytes: int
    disk_free_after_bytes: int

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible report containing every comparison."""

        return {
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "artifact_tensor_count": self.artifact_tensor_count,
            "base_vlm": dict(self.base_vlm),
            "checkpoint": dict(self.checkpoint),
            "comparisons": [asdict(comparison) for comparison in self.comparisons],
            "converted_weights_path": self.converted_weights_path,
            "dataset": dict(self.dataset),
            "device": self.device,
            "disk_free_after_bytes": self.disk_free_after_bytes,
            "disk_free_before_bytes": self.disk_free_before_bytes,
            "dtype": self.dtype,
            "episode": self.episode,
            "forward_backward_seconds": self.forward_backward_seconds,
            "frame_index": self.frame_index,
            "gradient_count": self.gradient_count,
            "loss_relative_difference": self.loss_relative_difference,
            "macos_version": self.macos_version,
            "maximum_gradient_relative_l2": self.maximum_gradient_relative_l2,
            "minimum_gradient_cosine": self.minimum_gradient_cosine,
            "mlx_loss": self.mlx_loss,
            "mlx_version": self.mlx_version,
            "parameter_match_count": self.parameter_match_count,
            "passed": self.passed,
            "python_version": self.python_version,
            "reference_loss": self.reference_loss,
            "seed": self.seed,
            "absolute_index": self.absolute_index,
            "thresholds": {
                "loss_relative_difference_maximum": LOSS_RELATIVE_TOLERANCE,
                "gradient_relative_l2_maximum": GRADIENT_RELATIVE_L2_TOLERANCE,
                "gradient_cosine_minimum": GRADIENT_COSINE_MINIMUM,
            },
            "total_seconds": self.total_seconds,
            "trainable_scalar_count": self.trainable_scalar_count,
            "worst_cosine": [asdict(comparison) for comparison in self.worst_cosine],
            "worst_relative_l2": [
                asdict(comparison) for comparison in self.worst_relative_l2
            ],
            "active_memory_bytes": self.active_memory_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


def _assert_artifact_contract(artifact: TrainingArtifact) -> tuple[str, ...]:
    names = artifact.verify_all()
    metadata = artifact.metadata
    required_equalities = {
        "artifact_type": "smolvla-gradient-golden",
        "device": "cpu",
        "dtype": "float32",
        "checkpoint": _EXPECTED_CHECKPOINT,
        "base_vlm": _EXPECTED_BASE_VLM,
        "dataset": _EXPECTED_DATASET,
        "trainable_tensor_count": _EXPECTED_TRAINABLE_TENSORS,
        "trainable_scalar_count": _EXPECTED_TRAINABLE_SCALARS,
        "tensor_count": _EXPECTED_ARTIFACT_TENSORS,
        "physical_action_dim": 6,
    }
    for key, expected in required_equalities.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"gradient artifact {key} mismatch: {metadata.get(key)!r} != {expected!r}"
            )
    if len(names) != _EXPECTED_ARTIFACT_TENSORS:
        raise ValueError(
            f"gradient artifact has {len(names)} tensors, expected {_EXPECTED_ARTIFACT_TENSORS}"
        )
    parameter_map = metadata.get("parameter_map")
    if not isinstance(parameter_map, list) or len(parameter_map) != _EXPECTED_TRAINABLE_TENSORS:
        raise ValueError("gradient artifact parameter map is incomplete")
    canonical_names = [item.get("canonical") for item in parameter_map]
    if len(set(canonical_names)) != _EXPECTED_TRAINABLE_TENSORS:
        raise ValueError("gradient artifact canonical parameter names are not a bijection")
    return names


def load_serialized_training_batch(artifact: TrainingArtifact) -> TrainingBatch:
    """Build a model-ready MLX batch using only saved values and draws."""

    return TrainingBatch(
        processed=ProcessedObservation(
            pixel_values=mx.array(artifact.load("batch/pixel_values")),
            pixel_attention_mask=mx.array(artifact.load("batch/pixel_attention_mask")),
            input_ids=mx.array(artifact.load("batch/input_ids")),
            text_attention_mask=mx.array(artifact.load("batch/text_attention_mask")),
            state=mx.array(artifact.load("batch/state")),
        ),
        actions=mx.array(artifact.load("batch/actions")),
        action_is_pad=mx.array(artifact.load("batch/action_is_pad")),
        noise=mx.array(artifact.load("draws/noise")),
        timesteps=mx.array(artifact.load("draws/timesteps")),
        action_dim=int(artifact.metadata["physical_action_dim"]),
    )


def _selected_parameters(
    model: SmolVLATrainingModel,
) -> dict[str, mx.array]:
    selected_names = configure_reference_trainable(model)
    flat_parameters = tuple(tree_flatten(model.trainable_parameters()))
    if tuple(name for name, _ in flat_parameters) != selected_names:
        raise RuntimeError("selected training parameter traversal changed unexpectedly")
    selected: dict[str, mx.array] = {}
    for name, parameter in flat_parameters:
        canonical = canonical_parameter_name(name)
        if canonical in selected:
            raise RuntimeError(f"duplicate canonical MLX parameter name: {canonical}")
        selected[canonical] = parameter
    return selected


def validate_checkpoint_parameter_identity(
    model: SmolVLATrainingModel,
    artifact: TrainingArtifact,
) -> tuple[str, ...]:
    """Require exact fp32 equality for all selected checkpoint parameters."""

    selected = _selected_parameters(model)
    expected_names = tuple(
        sorted(item["canonical"] for item in artifact.metadata["parameter_map"])
    )
    if tuple(sorted(selected)) != expected_names:
        missing = sorted(set(expected_names) - set(selected))
        unexpected = sorted(set(selected) - set(expected_names))
        raise RuntimeError(
            f"selected checkpoint parameter names differ; missing={missing}, unexpected={unexpected}"
        )
    mx.eval(*selected.values())
    for name in expected_names:
        parameter = selected[name]
        if parameter.dtype != mx.float32:
            raise TypeError(f"selected MLX parameter {name} is {parameter.dtype}, expected float32")
        candidate = np.asarray(parameter)
        reference = artifact.load(f"parameters/{name}")
        if reference.dtype != np.float32:
            raise TypeError(
                f"serialized reference parameter {name} is {reference.dtype}, expected float32"
            )
        if candidate.shape != reference.shape:
            raise RuntimeError(
                f"selected checkpoint parameter shape differs for {name}: "
                f"{candidate.shape} != {reference.shape}"
            )
        if not np.array_equal(candidate, reference):
            maximum = float(
                np.max(
                    np.abs(
                        candidate.astype(np.float64) - reference.astype(np.float64)
                    )
                )
            )
            raise RuntimeError(
                f"selected checkpoint parameter values differ for {name}; max_abs={maximum}"
            )
    return expected_names


def _canonical_gradients(gradients: dict) -> dict[str, mx.array]:
    canonical: dict[str, mx.array] = {}
    for name, gradient in tree_flatten(gradients):
        canonical_name = canonical_parameter_name(name)
        if canonical_name in canonical:
            raise RuntimeError(f"duplicate canonical MLX gradient name: {canonical_name}")
        canonical[canonical_name] = gradient
    return canonical


def run_gradient_parity(
    *,
    golden_dir: str | Path,
    native_cache: str | Path,
) -> GradientParityResult:
    """Run one identical-draw CPU/fp32 MLX step and compare all gradients."""

    repository_root = Path(__file__).resolve().parents[1]
    disk_free_before = shutil.disk_usage(repository_root).free
    if disk_free_before < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"gradient parity requires at least {_MINIMUM_FREE_BYTES} free bytes, "
            f"got {disk_free_before}"
        )
    total_start = time.perf_counter()
    artifact = TrainingArtifact(Path(golden_dir))
    artifact_names = _assert_artifact_contract(artifact)

    with mx.stream(mx.cpu):
        if mx.default_device() != mx.cpu:
            raise RuntimeError(f"gradient parity selected {mx.default_device()}, expected MLX CPU")
        model = SmolVLATrainingModel.from_pretrained(
            cache_dir=native_cache,
            dtype=mx.float32,
        )
        expected_names = validate_checkpoint_parameter_identity(model, artifact)
        batch = load_serialized_training_batch(artifact)
        mx.eval(
            batch.processed.pixel_values,
            batch.processed.pixel_attention_mask,
            batch.processed.input_ids,
            batch.processed.text_attention_mask,
            batch.processed.state,
            batch.actions,
            batch.action_is_pad,
            batch.noise,
            batch.timesteps,
        )
        mx.reset_peak_memory()
        step_start = time.perf_counter()
        with differentiable_cpu_primitives():
            value_and_grad = nn.value_and_grad(
                model,
                lambda: training_loss(model, batch),
            )
            loss, gradients = value_and_grad()
            candidate_gradients = _canonical_gradients(gradients)
            if tuple(sorted(candidate_gradients)) != expected_names:
                missing = sorted(set(expected_names) - set(candidate_gradients))
                unexpected = sorted(set(candidate_gradients) - set(expected_names))
                raise RuntimeError(
                    f"MLX gradient names differ; missing={missing}, unexpected={unexpected}"
                )
            mx.eval(loss, *candidate_gradients.values())
        forward_backward_seconds = time.perf_counter() - step_start

        comparisons: list[GradientComparison] = []
        for name in expected_names:
            candidate = candidate_gradients[name]
            if candidate.dtype != mx.float32:
                raise TypeError(f"MLX gradient {name} is {candidate.dtype}, expected float32")
            comparisons.append(
                compare_gradient_arrays(
                    name,
                    artifact.load(f"gradients/{name}"),
                    np.asarray(candidate),
                )
            )
        mlx_loss = float(loss)
        active_memory_bytes = int(mx.get_active_memory())
        peak_memory_bytes = int(max(active_memory_bytes, mx.get_peak_memory()))

    ordered_comparisons = tuple(sorted(comparisons, key=lambda item: item.name))
    worst_relative_l2 = tuple(
        sorted(
            ordered_comparisons,
            key=lambda item: (-item.relative_l2, item.name),
        )[:5]
    )
    worst_cosine = tuple(
        sorted(
            ordered_comparisons,
            key=lambda item: (item.cosine_similarity, item.name),
        )[:5]
    )
    maximum_relative_l2 = max(item.relative_l2 for item in ordered_comparisons)
    minimum_cosine = min(item.cosine_similarity for item in ordered_comparisons)
    reference_loss = float(artifact.load("flow/loss"))
    loss_difference = relative_loss_difference(reference_loss, mlx_loss)
    passed = (
        loss_difference <= LOSS_RELATIVE_TOLERANCE
        and all(
            item.relative_l2 <= GRADIENT_RELATIVE_L2_TOLERANCE
            and item.cosine_similarity >= GRADIENT_COSINE_MINIMUM
            for item in ordered_comparisons
        )
    )
    disk_free_after = shutil.disk_usage(repository_root).free
    if disk_free_after < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"gradient parity left less than {_MINIMUM_FREE_BYTES} free bytes: "
            f"{disk_free_after}"
        )
    total_seconds = time.perf_counter() - total_start
    if not math.isfinite(total_seconds) or not math.isfinite(forward_backward_seconds):
        raise RuntimeError("gradient parity timing is non-finite")
    metadata = artifact.metadata
    return GradientParityResult(
        passed=passed,
        device="cpu",
        dtype="float32",
        python_version=platform.python_version(),
        macos_version=platform.mac_ver()[0],
        mlx_version=version("mlx"),
        checkpoint=dict(metadata["checkpoint"]),
        base_vlm=dict(metadata["base_vlm"]),
        dataset=dict(metadata["dataset"]),
        seed=int(metadata["seed"]),
        episode=int(metadata["episode"]),
        frame_index=int(metadata["frame_index"]),
        absolute_index=int(metadata["absolute_index"]),
        artifact_manifest_sha256=str(metadata["manifest_sha256"]),
        artifact_tensor_count=len(artifact_names),
        parameter_match_count=len(expected_names),
        trainable_scalar_count=int(metadata["trainable_scalar_count"]),
        gradient_count=len(ordered_comparisons),
        reference_loss=reference_loss,
        mlx_loss=mlx_loss,
        loss_relative_difference=loss_difference,
        maximum_gradient_relative_l2=maximum_relative_l2,
        minimum_gradient_cosine=minimum_cosine,
        comparisons=ordered_comparisons,
        worst_relative_l2=worst_relative_l2,
        worst_cosine=worst_cosine,
        converted_weights_path=str(model.converted_weights_path),
        forward_backward_seconds=forward_backward_seconds,
        total_seconds=total_seconds,
        active_memory_bytes=active_memory_bytes,
        peak_memory_bytes=peak_memory_bytes,
        disk_free_before_bytes=disk_free_before,
        disk_free_after_bytes=disk_free_after,
    )


def write_gradient_parity_report(
    result: GradientParityResult,
    output_path: str | Path,
) -> str:
    """Atomically write the full parity report and return its SHA-256."""

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
