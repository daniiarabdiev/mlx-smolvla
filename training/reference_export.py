"""Pinned Torch/LeRobot strict loader for a local merged T3 export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from mlx_smolvla._lab.reference.discovery import BASE_VLM_ID, BASE_VLM_REVISION


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


def resolve_tokenizer_snapshot(cache_dir: str | Path) -> Path:
    """Resolve the exact pinned tokenizer files used by exported checkpoints."""

    return Path(
        snapshot_download(
            BASE_VLM_ID,
            revision=BASE_VLM_REVISION,
            cache_dir=Path(cache_dir),
            allow_patterns=list(_BASE_VLM_PROCESSOR_FILES),
            local_files_only=True,
        )
    )


@dataclass(frozen=True)
class TorchExportPolicy:
    """A strict policy and its stats-active saved processors."""

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
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "TorchExportPolicy":
        checkpoint_dir = Path(checkpoint_dir).resolve()
        cache_dir = Path(cache_dir)
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("Torch export evaluation supports only float32 or float64")
        if device not in {"cpu", "mps"}:
            raise ValueError("Torch export evaluation supports only cpu or mps")
        if device == "mps" and dtype != torch.float32:
            raise ValueError("MPS export evaluation supports only float32")
        tokenizer_snapshot = resolve_tokenizer_snapshot(cache_dir)
        config = SmolVLAConfig.from_pretrained(checkpoint_dir)
        config.device = device
        config.load_vlm_weights = False
        config.push_to_hub = False
        config.vlm_model_name = str(tokenizer_snapshot)
        # The nested expert inherits bf16 from the backbone config. Loading
        # first would round fp32 trained weights, even with strict=True; casting
        # afterward cannot restore them. Establish the destination dtype before
        # copying any saved values, then transfer the exact model to its device.
        policy = SmolVLAPolicy(config)
        policy.to(device="cpu", dtype=dtype)
        load_model(
            policy,
            str(checkpoint_dir / "model.safetensors"),
            strict=True,
            device="cpu",
        )
        policy.to(device=device, dtype=dtype)
        policy.eval()
        dtype_name = str(dtype).removeprefix("torch.")
        device_override = {"device": device, "float_dtype": dtype_name}
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=checkpoint_dir,
            preprocessor_overrides={
                "device_processor": device_override,
                "tokenizer_processor": {"tokenizer_name": str(tokenizer_snapshot)},
            },
            postprocessor_overrides={"device_processor": device_override},
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
