"""Torch/LeRobot bridge for the pinned step-zero training gradient golden."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import math
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Mapping

import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

from mlx_smolvla._lab.reference.discovery import (
    BASE_VLM_ID,
    BASE_VLM_REVISION,
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    DATASET_ID,
    DATASET_REVISION,
)
from mlx_smolvla._lab.reference.policy import ReferencePolicy
from mlx_smolvla.convert import target_name_for_source
from mlx_smolvla._lab.training.data import TrainingArtifact, TrainingArtifactWriter


GRADIENT_GOLDEN_SEED = 20_260_831
GRADIENT_GOLDEN_EPISODE = 0
GRADIENT_GOLDEN_FRAME = 100
_EXPECTED_TRAINABLE_TENSORS = 155
_EXPECTED_TRAINABLE_SCALARS = 99_880_992
_MINIMUM_FREE_BYTES = 40 * 1024**3
_CAMERA_RENAME_MAP = {
    "observation.images.side": "observation.images.camera1",
    "observation.images.up": "observation.images.camera2",
}


@dataclass(frozen=True)
class ReferenceParameterSpec:
    """One selected Torch parameter and its canonical MLX identity."""

    source_name: str
    canonical_name: str
    shape: tuple[int, ...]
    scalar_count: int


@dataclass(frozen=True)
class ReferenceTrainingCase:
    """One actual LeRobot batch plus exact model-ready training inputs."""

    reference: ReferencePolicy
    batch: dict[str, Any]
    dataset_stats: Mapping[str, Mapping[str, object]]
    dataset_stats_sha256: str
    dataset_fps: int
    episode: int
    frame_index: int
    absolute_index: int
    task: str
    raw_state: torch.Tensor
    raw_actions: torch.Tensor
    pixel_values: torch.Tensor
    pixel_attention_mask: torch.Tensor
    input_ids: torch.Tensor
    text_attention_mask: torch.Tensor
    state: torch.Tensor
    actions: torch.Tensor
    action_is_pad: torch.Tensor
    physical_action_dim: int
    parameter_specs: tuple[ReferenceParameterSpec, ...]


def _parameter_specs(policy: torch.nn.Module) -> tuple[ReferenceParameterSpec, ...]:
    specs = tuple(
        ReferenceParameterSpec(
            source_name=name,
            canonical_name=target_name_for_source(name),
            shape=tuple(parameter.shape),
            scalar_count=parameter.numel(),
        )
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
    canonical_names = tuple(spec.canonical_name for spec in specs)
    source_names = tuple(spec.source_name for spec in specs)
    scalar_count = sum(spec.scalar_count for spec in specs)
    if len(specs) != _EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError(
            f"reference selected {len(specs)} trainable tensors, "
            f"expected {_EXPECTED_TRAINABLE_TENSORS}"
        )
    if scalar_count != _EXPECTED_TRAINABLE_SCALARS:
        raise RuntimeError(
            f"reference selected {scalar_count} trainable scalars, "
            f"expected {_EXPECTED_TRAINABLE_SCALARS}"
        )
    if len(set(source_names)) != len(specs) or len(set(canonical_names)) != len(specs):
        raise RuntimeError("reference trainable parameter mapping is not a strict bijection")
    if not all(
        name.startswith(("expert.", "state_proj.", "action_"))
        for name in canonical_names
    ):
        raise RuntimeError(f"reference trainable set has unexpected canonical names: {canonical_names}")
    return specs


def prepare_reference_training_case(
    cache_dir: Path,
    *,
    episode: int = GRADIENT_GOLDEN_EPISODE,
    frame_index: int = GRADIENT_GOLDEN_FRAME,
) -> ReferenceTrainingCase:
    """Mirror LeRobot's actual non-resume train-loop preparation on CPU."""

    cache_dir = Path(cache_dir)
    reference = ReferencePolicy.load(cache_dir)
    dataset_root = cache_dir / "datasets" / "svla_so101_pickplace"
    dataset_metadata = LeRobotDatasetMetadata(
        DATASET_ID,
        root=dataset_root,
        revision=DATASET_REVISION,
    )
    delta_timestamps = resolve_delta_timestamps(reference.config, dataset_metadata)
    dataset = LeRobotDataset(
        DATASET_ID,
        root=dataset_root,
        episodes=[episode],
        delta_timestamps=delta_timestamps,
        revision=DATASET_REVISION,
        video_backend="pyav",
        return_uint8=True,
    )
    if frame_index < 0 or frame_index >= len(dataset):
        raise IndexError(
            f"frame index {frame_index} is outside episode {episode} with {len(dataset)} frames"
        )

    raw_item = dataset[frame_index]
    batch = default_collate([raw_item])
    raw_state = batch["observation.state"].detach().cpu().float().clone()
    raw_actions = batch["action"].detach().cpu().float().clone()
    for camera_key in dataset.meta.camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=reference.config,
        pretrained_path=CHECKPOINT_ID,
        pretrained_revision=CHECKPOINT_REVISION,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {
                "features": {
                    **reference.config.input_features,
                    **reference.config.output_features,
                },
                "norm_map": reference.config.normalization_mapping,
                "stats": dataset.meta.stats,
            },
            "rename_observations_processor": {"rename_map": _CAMERA_RENAME_MAP},
            "tokenizer_processor": {"tokenizer_name": str(reference.vlm_snapshot)},
        },
    )
    processed_batch = preprocessor(batch)
    images, image_masks = reference.policy.prepare_images(processed_batch)
    pixel_values = torch.cat(images, dim=0).detach().cpu().float().contiguous()
    pixel_attention_mask = torch.cat(
        [mask.reshape(-1, 1) for mask in image_masks],
        dim=0,
    ).detach().cpu().bool().contiguous()
    state = reference.policy.prepare_state(processed_batch).detach().cpu().float().contiguous()
    actions = reference.policy.prepare_action(processed_batch).detach().cpu().float().contiguous()
    action_is_pad = processed_batch["action_is_pad"].detach().cpu().bool().contiguous()
    input_ids = processed_batch["observation.language.tokens"].detach().cpu().contiguous()
    text_attention_mask = (
        processed_batch["observation.language.attention_mask"].detach().cpu().bool().contiguous()
    )

    actual_episode = int(raw_item["episode_index"])
    actual_frame = int(raw_item["frame_index"])
    absolute_index = int(raw_item["index"])
    if (actual_episode, actual_frame) != (episode, frame_index):
        raise RuntimeError(
            "dataset identity mismatch: "
            f"requested {(episode, frame_index)}, got {(actual_episode, actual_frame)}"
        )
    task_value = processed_batch["task"]
    if not isinstance(task_value, list) or len(task_value) != 1 or not isinstance(task_value[0], str):
        raise RuntimeError(f"unexpected processed task value: {task_value!r}")

    stats_path = dataset.root / "meta" / "stats.json"
    stats_sha256 = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    physical_action_dim = reference.config.action_feature.shape[0]
    return ReferenceTrainingCase(
        reference=reference,
        batch=processed_batch,
        dataset_stats=dataset.meta.stats,
        dataset_stats_sha256=stats_sha256,
        dataset_fps=dataset.meta.fps,
        episode=actual_episode,
        frame_index=actual_frame,
        absolute_index=absolute_index,
        task=task_value[0],
        raw_state=raw_state,
        raw_actions=raw_actions,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
        input_ids=input_ids,
        text_attention_mask=text_attention_mask,
        state=state,
        actions=actions,
        action_is_pad=action_is_pad,
        physical_action_dim=physical_action_dim,
        parameter_specs=_parameter_specs(reference.policy),
    )


def _torch_to_numpy(value: torch.Tensor) -> object:
    value = value.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy()


def capture_reference_gradient_golden(
    cache_dir: Path,
    output_dir: Path,
    *,
    seed: int = GRADIENT_GOLDEN_SEED,
    episode: int = GRADIENT_GOLDEN_EPISODE,
    frame_index: int = GRADIENT_GOLDEN_FRAME,
) -> dict[str, object]:
    """Capture the actual CPU/fp32 loss and every selected Torch gradient."""

    repository_root = Path(__file__).resolve().parents[1]
    disk_free_before = shutil.disk_usage(repository_root).free
    if disk_free_before < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"gradient capture requires at least {_MINIMUM_FREE_BYTES} free bytes, "
            f"got {disk_free_before}"
        )
    capture_start = time.perf_counter()
    case = prepare_reference_training_case(
        cache_dir,
        episode=episode,
        frame_index=frame_index,
    )
    policy = case.reference.policy
    policy.train()
    policy.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    with torch.no_grad():
        noise = policy.model.sample_noise(case.actions.shape, torch.device("cpu"))
        timesteps = policy.model.sample_time(case.actions.shape[0], torch.device("cpu"))

    predicted_outputs: list[torch.Tensor] = []

    def capture_prediction(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        predicted_outputs.append(output.detach().cpu().float())

    handle = policy.model.action_out_proj.register_forward_hook(capture_prediction)
    forward_start = time.perf_counter()
    try:
        loss, loss_details = policy(case.batch, noise=noise, time=timesteps)
    finally:
        handle.remove()
    forward_seconds = time.perf_counter() - forward_start
    if len(predicted_outputs) != 1:
        raise RuntimeError(f"expected one predicted velocity capture, got {len(predicted_outputs)}")
    predicted_velocity = predicted_outputs[0]

    backward_start = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - backward_start

    time_expanded = timesteps[:, None, None]
    noisy_actions = time_expanded * noise + (1.0 - time_expanded) * case.actions
    target_velocity = noise - case.actions
    squared_error = (target_velocity - predicted_velocity) ** 2
    valid = (~case.action_is_pad).unsqueeze(-1)
    physical_error = squared_error[:, :, : case.physical_action_dim] * valid
    denominator = ((~case.action_is_pad).sum() * case.physical_action_dim).clamp_min(1)
    reconstructed_loss = physical_error.sum() / denominator
    torch.testing.assert_close(reconstructed_loss, loss.detach(), rtol=0, atol=0)

    named_parameters = dict(policy.named_parameters())
    writer = TrainingArtifactWriter(Path(output_dir))
    writer.add("batch/pixel_values", _torch_to_numpy(case.pixel_values))
    writer.add("batch/pixel_attention_mask", _torch_to_numpy(case.pixel_attention_mask))
    writer.add("batch/input_ids", _torch_to_numpy(case.input_ids))
    writer.add("batch/text_attention_mask", _torch_to_numpy(case.text_attention_mask))
    writer.add("batch/state", _torch_to_numpy(case.state))
    writer.add("batch/actions", _torch_to_numpy(case.actions))
    writer.add("batch/action_is_pad", _torch_to_numpy(case.action_is_pad))
    writer.add("draws/noise", _torch_to_numpy(noise))
    writer.add("draws/timesteps", _torch_to_numpy(timesteps))
    writer.add("flow/noisy_actions", _torch_to_numpy(noisy_actions))
    writer.add("flow/target_velocity", _torch_to_numpy(target_velocity))
    writer.add("flow/predicted_velocity", _torch_to_numpy(predicted_velocity))
    writer.add("flow/squared_error", _torch_to_numpy(squared_error))
    writer.add("flow/loss", float(loss.detach()))

    for spec in case.parameter_specs:
        parameter = named_parameters[spec.source_name]
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"reference gradient is missing for {spec.canonical_name}")
        if gradient.shape != parameter.shape:
            raise RuntimeError(
                f"reference gradient shape mismatch for {spec.canonical_name}: "
                f"{gradient.shape} != {parameter.shape}"
            )
        gradient_values = gradient.detach().cpu().float()
        if not bool(torch.all(torch.isfinite(gradient_values))):
            raise RuntimeError(f"reference gradient is non-finite for {spec.canonical_name}")
        if float(torch.linalg.vector_norm(gradient_values)) == 0.0:
            raise RuntimeError(f"reference gradient has zero norm for {spec.canonical_name}")
        writer.add(f"parameters/{spec.canonical_name}", _torch_to_numpy(parameter))
        writer.add(f"gradients/{spec.canonical_name}", _torch_to_numpy(gradient_values))

    metadata = writer.finalize(
        {
            "format_version": 1,
            "artifact_type": "smolvla-gradient-golden",
            "checkpoint": {"id": CHECKPOINT_ID, "revision": CHECKPOINT_REVISION},
            "base_vlm": {"id": BASE_VLM_ID, "revision": BASE_VLM_REVISION},
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "device": "cpu",
            "dtype": "float32",
            "seed": seed,
            "episode": case.episode,
            "frame_index": case.frame_index,
            "absolute_index": case.absolute_index,
            "task": case.task,
            "dataset_fps": case.dataset_fps,
            "dataset_stats_sha256": case.dataset_stats_sha256,
            "normalization": "public dataset statistics override",
            "camera_rename_map": _CAMERA_RENAME_MAP,
            "physical_action_dim": case.physical_action_dim,
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
            "reference_loss": float(loss.detach()),
            "sampled_timestep": float(timesteps[0]),
            "loss_details": loss_details,
            "python_version": platform.python_version(),
            "lerobot_version": version("lerobot"),
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "disk_free_before_bytes": disk_free_before,
        }
    )
    verified_names = TrainingArtifact(Path(output_dir)).verify_all()
    if len(verified_names) != metadata["tensor_count"]:
        raise RuntimeError("completed gradient artifact failed its full verification")
    disk_free_after = shutil.disk_usage(repository_root).free
    if disk_free_after < _MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"gradient capture left less than {_MINIMUM_FREE_BYTES} free bytes: {disk_free_after}"
        )
    result = dict(metadata)
    result["disk_free_after_bytes"] = disk_free_after
    result["capture_seconds"] = time.perf_counter() - capture_start
    result["artifact_bytes"] = sum(
        path.stat().st_size for path in Path(output_dir).rglob("*") if path.is_file()
    )
    if not math.isfinite(float(result["capture_seconds"])):
        raise RuntimeError("gradient capture timing is non-finite")
    return result
