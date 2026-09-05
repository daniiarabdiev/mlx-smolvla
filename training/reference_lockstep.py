"""Pinned PyTorch reference capture for 25-step SmolVLA optimizer lockstep."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import math
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from mlx_smolvla._lab.training.data import TrainingArtifact, TrainingArtifactWriter
from mlx_smolvla._lab.training.optimizer import SmolVLAOptimizerConfig, cosine_decay_with_warmup_lr
from mlx_smolvla._lab.training.reference import prepare_reference_training_case


OPTIMIZER_LOCKSTEP_STEPS = 25
OPTIMIZER_TRAINING_HORIZON = 100_000
OPTIMIZER_LOCKSTEP_SEED = 20_260_831

_EXPECTED_TRAINABLE_TENSORS = 155
_EXPECTED_TRAINABLE_SCALARS = 99_880_992
_EXPECTED_TENSOR_COUNT = 330
_MINIMUM_FREE_BYTES = 40 * 1024**3


def _torch_to_numpy(value: torch.Tensor) -> np.ndarray:
    values = value.detach().cpu()
    if values.dtype == torch.bfloat16:
        values = values.float()
    return values.numpy()


def _assert_array_equal(name: str, candidate: np.ndarray, reference: np.ndarray) -> None:
    if candidate.dtype != reference.dtype:
        raise TypeError(
            f"T1 equality dtype mismatch for {name}: {candidate.dtype} != {reference.dtype}"
        )
    if candidate.shape != reference.shape:
        raise RuntimeError(
            f"T1 equality shape mismatch for {name}: {candidate.shape} != {reference.shape}"
        )
    if not np.array_equal(candidate, reference):
        raise RuntimeError(f"T1 equality values differ for {name}")


def _validate_t1_identity(case: object, t1_artifact: TrainingArtifact) -> int:
    t1_artifact.verify_all()
    batch_values = {
        "batch/pixel_values": case.pixel_values,
        "batch/pixel_attention_mask": case.pixel_attention_mask,
        "batch/input_ids": case.input_ids,
        "batch/text_attention_mask": case.text_attention_mask,
        "batch/state": case.state,
        "batch/actions": case.actions,
        "batch/action_is_pad": case.action_is_pad,
    }
    for name, value in batch_values.items():
        _assert_array_equal(name, _torch_to_numpy(value), t1_artifact.load(name))

    named_parameters = dict(case.reference.policy.named_parameters())
    for spec in case.parameter_specs:
        _assert_array_equal(
            f"parameters/{spec.canonical_name}",
            _torch_to_numpy(named_parameters[spec.source_name]),
            t1_artifact.load(f"parameters/{spec.canonical_name}"),
        )
    return len(case.parameter_specs)


def _validate_reference_presets(case: object) -> tuple[object, object, SmolVLAOptimizerConfig]:
    expected = SmolVLAOptimizerConfig(training_horizon=OPTIMIZER_TRAINING_HORIZON)
    optimizer_config = case.reference.config.get_optimizer_preset()
    scheduler_config = case.reference.config.get_scheduler_preset()
    actual_optimizer = {
        "lr": optimizer_config.lr,
        "betas": tuple(optimizer_config.betas),
        "eps": optimizer_config.eps,
        "weight_decay": optimizer_config.weight_decay,
        "grad_clip_norm": optimizer_config.grad_clip_norm,
    }
    expected_optimizer = {
        "lr": expected.lr,
        "betas": expected.betas,
        "eps": expected.eps,
        "weight_decay": expected.weight_decay,
        "grad_clip_norm": expected.grad_clip_norm,
    }
    if actual_optimizer != expected_optimizer:
        raise RuntimeError(
            f"installed SmolVLA optimizer preset changed: {actual_optimizer} != {expected_optimizer}"
        )
    actual_scheduler = {
        "num_warmup_steps": scheduler_config.num_warmup_steps,
        "num_decay_steps": scheduler_config.num_decay_steps,
        "peak_lr": scheduler_config.peak_lr,
        "decay_lr": scheduler_config.decay_lr,
    }
    expected_scheduler = {
        "num_warmup_steps": expected.warmup_steps,
        "num_decay_steps": expected.decay_steps,
        "peak_lr": expected.lr,
        "decay_lr": expected.decay_lr,
    }
    if actual_scheduler != expected_scheduler:
        raise RuntimeError(
            f"installed SmolVLA scheduler preset changed: {actual_scheduler} != {expected_scheduler}"
        )
    return optimizer_config, scheduler_config, expected


def capture_reference_optimizer_golden(
    cache_dir: str | Path,
    t1_dir: str | Path,
    output_dir: str | Path,
    *,
    step_count: int = OPTIMIZER_LOCKSTEP_STEPS,
    seed: int = OPTIMIZER_LOCKSTEP_SEED,
) -> dict[str, object]:
    """Run and serialize the actual reference AdamW/scheduler evolution."""

    if step_count != OPTIMIZER_LOCKSTEP_STEPS:
        raise ValueError(
            f"official optimizer golden requires {OPTIMIZER_LOCKSTEP_STEPS} steps, "
            f"got {step_count}"
        )
    repository_root = Path(__file__).resolve().parents[1]
    disk_free_before = shutil.disk_usage(repository_root).free
    if disk_free_before < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"optimizer capture requires at least {_MINIMUM_FREE_BYTES} free bytes, "
            f"got {disk_free_before}"
        )
    capture_start = time.perf_counter()
    t1_artifact = TrainingArtifact(Path(t1_dir))
    case = prepare_reference_training_case(Path(cache_dir))
    initial_parameters_verified = _validate_t1_identity(case, t1_artifact)
    optimizer_config, scheduler_config, expected_config = _validate_reference_presets(case)

    policy = case.reference.policy
    policy.train()
    optimizer = optimizer_config.build(policy.get_optim_params())
    scheduler = scheduler_config.build(optimizer, OPTIMIZER_TRAINING_HORIZON)
    if scheduler is None:
        raise RuntimeError("reference SmolVLA scheduler unexpectedly resolved to None")
    policy.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    writer = TrainingArtifactWriter(Path(output_dir))
    losses: list[float] = []
    gradient_norms: list[float] = []

    for step in range(step_count):
        with torch.no_grad():
            noise = policy.model.sample_noise(case.actions.shape, torch.device("cpu"))
            timesteps = policy.model.sample_time(case.actions.shape[0], torch.device("cpu"))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        expected_learning_rate = cosine_decay_with_warmup_lr(step, expected_config)
        if learning_rate != expected_learning_rate:
            raise RuntimeError(
                f"reference LR differs at step {step}: "
                f"{learning_rate} != {expected_learning_rate}"
            )

        loss, _ = policy(case.batch, noise=noise, time=timesteps)
        loss.backward()
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            optimizer_config.grad_clip_norm,
        )
        loss_value = float(loss.detach())
        gradient_norm = float(gradient_norm_tensor)
        if not math.isfinite(loss_value) or not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite reference optimizer metric at step {step}")
        clip_coefficient = min(
            1.0,
            optimizer_config.grad_clip_norm / (gradient_norm + 1e-6),
        )

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        expected_next = cosine_decay_with_warmup_lr(step + 1, expected_config)
        if next_learning_rate != expected_next:
            raise RuntimeError(
                f"reference next LR differs at step {step}: "
                f"{next_learning_rate} != {expected_next}"
            )

        writer.add(f"draws/{step:03d}/noise", _torch_to_numpy(noise))
        writer.add(f"draws/{step:03d}/timesteps", _torch_to_numpy(timesteps))
        writer.add(f"steps/{step:03d}/loss", loss_value)
        writer.add(f"steps/{step:03d}/lr_used", learning_rate)
        writer.add(f"steps/{step:03d}/lr_next", next_learning_rate)
        writer.add(f"steps/{step:03d}/gradient_norm", gradient_norm)
        writer.add(f"steps/{step:03d}/clip_coefficient", clip_coefficient)
        losses.append(loss_value)
        gradient_norms.append(gradient_norm)

    named_parameters = dict(policy.named_parameters())
    for spec in case.parameter_specs:
        parameter = _torch_to_numpy(named_parameters[spec.source_name])
        if parameter.dtype != np.float32:
            raise TypeError(
                f"final reference parameter {spec.canonical_name} is {parameter.dtype}, "
                "expected float32"
            )
        if not np.all(np.isfinite(parameter)):
            raise RuntimeError(f"final reference parameter is non-finite: {spec.canonical_name}")
        writer.add(f"final_parameters/{spec.canonical_name}", parameter)

    runtime_before_finalize = time.perf_counter() - capture_start
    t1_metadata = t1_artifact.metadata
    metadata = writer.finalize(
        {
            "format_version": 1,
            "artifact_type": "smolvla-optimizer-golden",
            "checkpoint": t1_metadata["checkpoint"],
            "base_vlm": t1_metadata["base_vlm"],
            "dataset": t1_metadata["dataset"],
            "device": "cpu",
            "dtype": "float32",
            "seed": seed,
            "step_count": step_count,
            "training_horizon": OPTIMIZER_TRAINING_HORIZON,
            "batch_schedule": "repeat fixed T1 batch for every step",
            "episode": case.episode,
            "frame_index": case.frame_index,
            "absolute_index": case.absolute_index,
            "t1_manifest_sha256": t1_metadata["manifest_sha256"],
            "t1_batch_verified": True,
            "initial_parameters_verified": initial_parameters_verified,
            "trainable_tensor_count": len(case.parameter_specs),
            "trainable_scalar_count": sum(spec.scalar_count for spec in case.parameter_specs),
            "parameter_map": [
                {
                    "source": spec.source_name,
                    "canonical": spec.canonical_name,
                    "shape": list(spec.shape),
                    "scalar_count": spec.scalar_count,
                }
                for spec in case.parameter_specs
            ],
            "optimizer": {
                "type": "torch.optim.AdamW",
                "lr": expected_config.lr,
                "betas": list(expected_config.betas),
                "eps": expected_config.eps,
                "weight_decay": expected_config.weight_decay,
                "grad_clip_norm": expected_config.grad_clip_norm,
                "bias_correction": True,
                "weight_decay_semantics": "decoupled multiplicative before moment update",
                "epsilon_placement": "after bias-corrected sqrt(second moment)",
            },
            "scheduler": {
                "type": "CosineDecayWithWarmupSchedulerConfig",
                "warmup_steps": expected_config.warmup_steps,
                "decay_steps": expected_config.decay_steps,
                "decay_lr": expected_config.decay_lr,
                "training_horizon": expected_config.training_horizon,
                "order": "optimizer step then scheduler step",
            },
            "update_order": [
                "forward",
                "backward",
                "global norm clip",
                "AdamW step",
                "zero gradients",
                "scheduler step",
            ],
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "maximum_gradient_norm": max(gradient_norms),
            "minimum_gradient_norm": min(gradient_norms),
            "python_version": platform.python_version(),
            "torch_version": version("torch"),
            "lerobot_version": version("lerobot"),
            "runtime_before_finalize_seconds": runtime_before_finalize,
            "disk_free_before_bytes": disk_free_before,
        }
    )
    if metadata["trainable_tensor_count"] != _EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError("reference optimizer artifact trainable tensor count changed")
    if metadata["trainable_scalar_count"] != _EXPECTED_TRAINABLE_SCALARS:
        raise RuntimeError("reference optimizer artifact trainable scalar count changed")
    verified_names = TrainingArtifact(Path(output_dir)).verify_all()
    if len(verified_names) != _EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"reference optimizer artifact has {len(verified_names)} tensors, "
            f"expected {_EXPECTED_TENSOR_COUNT}"
        )
    disk_free_after = shutil.disk_usage(repository_root).free
    if disk_free_after < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"optimizer capture left less than {_MINIMUM_FREE_BYTES} free bytes: "
            f"{disk_free_after}"
        )
    result = dict(metadata)
    result["capture_seconds"] = time.perf_counter() - capture_start
    result["disk_free_after_bytes"] = disk_free_after
    result["artifact_bytes"] = sum(
        path.stat().st_size for path in Path(output_dir).rglob("*") if path.is_file()
    )
    result["t1_metadata_sha256"] = hashlib.sha256(
        (Path(t1_dir) / "metadata.json").read_bytes()
    ).hexdigest()
    return result
