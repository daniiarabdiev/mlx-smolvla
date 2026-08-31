"""Pinned Torch/LeRobot strict loader for a local merged T3 export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from reference.discovery import BASE_VLM_ID, BASE_VLM_REVISION


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
class TorchExportPolicy:
    """A strict fp32 CPU policy and its stats-active saved processors."""

    policy: SmolVLAPolicy
    preprocessor: object
    postprocessor: object
    tokenizer_snapshot: Path

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        *,
        cache_dir: str | Path,
    ) -> "TorchExportPolicy":
        checkpoint_dir = Path(checkpoint_dir).resolve()
        cache_dir = Path(cache_dir)
        tokenizer_snapshot = Path(
            snapshot_download(
                BASE_VLM_ID,
                revision=BASE_VLM_REVISION,
                cache_dir=cache_dir,
                allow_patterns=list(_BASE_VLM_PROCESSOR_FILES),
            )
        )
        config = SmolVLAConfig.from_pretrained(checkpoint_dir)
        config.device = "cpu"
        config.load_vlm_weights = False
        config.push_to_hub = False
        config.vlm_model_name = str(tokenizer_snapshot)
        policy = SmolVLAPolicy.from_pretrained(
            checkpoint_dir,
            config=config,
            strict=True,
        )
        policy.to(device="cpu", dtype=torch.float32)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=checkpoint_dir,
            preprocessor_overrides={
                "device_processor": {"device": "cpu"},
                "tokenizer_processor": {"tokenizer_name": str(tokenizer_snapshot)},
            },
        )
        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            tokenizer_snapshot=tokenizer_snapshot,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.policy.parameters())

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.policy.parameters()).dtype
