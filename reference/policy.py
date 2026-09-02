"""Pinned CPU/fp32 adapter around LeRobot's SmolVLA reference policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from reference.discovery import (
    BASE_VLM_ID,
    BASE_VLM_REVISION,
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    DATASET_ID,
    DATASET_REVISION,
)


_BASE_VLM_PROCESSOR_FILES = (
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
)


@dataclass(frozen=True)
class ReferenceSample:
    """One real dataset observation mapped to the checkpoint feature names."""

    observation: Mapping[str, torch.Tensor | str]
    action: torch.Tensor


@dataclass(frozen=True)
class ReferencePrediction:
    """Normalized model output and checkpoint-un-normalized actions."""

    normalized_actions: torch.Tensor
    actions: torch.Tensor


def load_dataset_observation(cache_dir: Path, index: int, episode: int = 0) -> ReferenceSample:
    """Load one real SO-101 frame and map its two cameras deterministically."""

    dataset = LeRobotDataset(
        DATASET_ID,
        root=cache_dir / "datasets" / "svla_so101_pickplace",
        episodes=[episode],
        revision=DATASET_REVISION,
        video_backend="pyav",
    )
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Frame index {index} is outside episode {episode} with {len(dataset)} frames")
    item = dataset[index]
    observation = {
        "observation.images.camera1": item["observation.images.side"].cpu().float(),
        "observation.images.camera2": item["observation.images.up"].cpu().float(),
        "observation.state": item["observation.state"].cpu().float(),
        "task": item["task"],
    }
    return ReferenceSample(observation=observation, action=item["action"].cpu().float())


def load_checkpoint_dataset_observation(
    *,
    dataset_id: str,
    dataset_revision: str,
    dataset_root: Path,
    checkpoint_camera_keys: tuple[str, ...],
    index: int,
    episode: int,
    camera_mapping: Mapping[str, str] | None = None,
) -> ReferenceSample:
    """Load a frame and map dataset cameras to arbitrary checkpoint feature keys."""

    dataset = LeRobotDataset(
        dataset_id,
        root=dataset_root,
        episodes=[episode],
        revision=dataset_revision,
        video_backend="pyav",
    )
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Frame index {index} is outside episode {episode} with {len(dataset)} frames")
    item = dataset[index]
    mapping = dict(camera_mapping or {})
    observation: dict[str, torch.Tensor | str] = {}
    for checkpoint_key in checkpoint_camera_keys:
        dataset_key = mapping.get(checkpoint_key, checkpoint_key)
        if dataset_key not in item:
            raise KeyError(
                f"Dataset observation is missing {dataset_key!r} required for checkpoint key {checkpoint_key!r}"
            )
        observation[checkpoint_key] = item[dataset_key].cpu().float()
    observation["observation.state"] = item["observation.state"].cpu().float()
    observation["task"] = item["task"]
    return ReferenceSample(observation=observation, action=item["action"].cpu().float())


@dataclass(frozen=True)
class ReferencePolicy:
    """Loaded reference policy with explicit deterministic runtime properties."""

    policy: SmolVLAPolicy
    preprocessor: object
    postprocessor: object
    vlm_snapshot: Path

    @classmethod
    def load(
        cls,
        cache_dir: Path,
        *,
        checkpoint_dir: Path | None = None,
        device: str = "cpu",
    ) -> "ReferencePolicy":
        """Load a pinned/matching checkpoint in fp32; CPU remains the default."""

        if device not in {"cpu", "mps"}:
            raise ValueError("reference device must be 'cpu' or 'mps'")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is not available on this machine")

        vlm_snapshot = Path(
            snapshot_download(
                BASE_VLM_ID,
                revision=BASE_VLM_REVISION,
                cache_dir=cache_dir,
                allow_patterns=list(_BASE_VLM_PROCESSOR_FILES),
            )
        )
        pretrained_path: str | Path = CHECKPOINT_ID
        pretrained_revision: str | None = CHECKPOINT_REVISION
        if checkpoint_dir is not None:
            pretrained_path = checkpoint_dir.resolve()
            pretrained_revision = None
        config = SmolVLAConfig.from_pretrained(
            pretrained_path,
            revision=pretrained_revision,
            cache_dir=cache_dir,
        )
        config.device = device
        config.load_vlm_weights = False
        config.push_to_hub = False
        config.vlm_model_name = str(vlm_snapshot)
        policy = SmolVLAPolicy.from_pretrained(
            pretrained_path,
            revision=pretrained_revision,
            cache_dir=cache_dir,
            config=config,
            strict=True,
        )
        policy.to(device=device, dtype=torch.float32)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=pretrained_path,
            pretrained_revision=pretrained_revision,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {"tokenizer_name": str(vlm_snapshot)},
            },
        )
        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            vlm_snapshot=vlm_snapshot,
        )

    def prepare(self, observation: Mapping[str, torch.Tensor | str]) -> dict[str, torch.Tensor]:
        """Apply the checkpoint's saved input pipeline on CPU."""

        return self.preprocessor(dict(observation))

    def predict(
        self,
        observation: Mapping[str, torch.Tensor | str],
        noise: torch.Tensor,
    ) -> ReferencePrediction:
        """Run one deterministic action chunk from explicit Gaussian noise."""

        batch = self.prepare(observation)
        with torch.inference_mode():
            normalized_actions = self.policy.predict_action_chunk(batch, noise=noise)
            actions = self.postprocessor(normalized_actions)
        return ReferencePrediction(
            normalized_actions=normalized_actions.cpu().float(),
            actions=actions.cpu().float(),
        )

    @property
    def config(self) -> SmolVLAConfig:
        return self.policy.config

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.policy.parameters()).dtype

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.policy.parameters())
