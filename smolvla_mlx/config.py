"""Checkpoint-derived configuration for the pinned SmolVLA inference model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SUPPORTED_VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
_AUDITED_TEXT_HIDDEN_SIZE = 960
_AUDITED_ARCHITECTURE = {
    "chunk_size": 50,
    "max_action_dim": 32,
    "max_state_dim": 32,
    "resize_imgs_with_padding": (512, 512),
    "num_vlm_layers": 16,
    "expert_width_multiplier": 0.75,
    "self_attn_every_n_layers": 2,
}


def _read_safetensors_names(path: Path) -> set[str]:
    """Read names from a safetensors header without materializing its tensors."""

    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), byteorder="little")
        header = json.loads(handle.read(header_size))
    return {name for name in header if name != "__metadata__"}


def _effective_normalization(checkpoint_dir: Path, key: str, processor_file: str) -> str:
    """Resolve the exact key-match behavior of a saved LeRobot stats processor."""

    stats_file = checkpoint_dir / processor_file
    if not stats_file.is_file():
        return "identity"
    names = _read_safetensors_names(stats_file)
    prefix = f"{key}."
    return "mean_std" if any(name.startswith(prefix) for name in names) else "identity"


@dataclass(frozen=True)
class SmolVLAConfig:
    """The policy settings required by the native v0.1 inference implementation."""

    checkpoint_dir: Path
    vlm_model_name: str
    chunk_size: int
    n_action_steps: int
    action_dim: int
    state_dim: int
    max_action_dim: int
    max_state_dim: int
    image_size: tuple[int, int]
    image_keys: tuple[str, ...]
    image_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    state_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    empty_cameras: int
    tokenizer_max_length: int
    tokenizer_padding: str
    num_steps: int
    min_period: float
    max_period: float
    vlm_layers: int
    expert_width_multiplier: float
    expert_hidden_size: int
    self_attn_every_n_layers: int
    prefix_length: int
    state_normalization: str
    action_normalization: str

    @property
    def input_contract(self) -> str:
        """Render checkpoint-derived observation keys and shapes for user errors."""

        cameras = ", ".join(f"{name} {shape}" for name, shape in self.image_shapes)
        return (
            f"at least one camera from [{cameras}]; "
            f"observation.state {self.state_shape}; task non-empty string"
        )

    @classmethod
    def from_pretrained_files(cls, checkpoint_dir: Path) -> "SmolVLAConfig":
        """Parse the pinned policy config and saved processor-key behavior locally."""

        checkpoint_dir = checkpoint_dir.resolve()
        config_path = checkpoint_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"SmolVLA config not found at {config_path}")
        raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        vlm_model_name = raw["vlm_model_name"]
        if vlm_model_name != _SUPPORTED_VLM:
            raise ValueError(f"Unsupported base VLM {vlm_model_name!r}; expected {_SUPPORTED_VLM!r}")

        input_features = raw["input_features"]
        output_features = raw["output_features"]
        image_keys = tuple(
            name for name, feature in input_features.items() if feature.get("type") == "VISUAL"
        )
        if not image_keys:
            raise ValueError("SmolVLA checkpoint config contains no visual input features")
        state_feature = input_features["observation.state"]
        action_feature = output_features["action"]
        image_shapes = tuple(
            (name, tuple(int(dimension) for dimension in input_features[name]["shape"]))
            for name in image_keys
        )
        state_shape = tuple(int(dimension) for dimension in state_feature["shape"])
        action_shape = tuple(int(dimension) for dimension in action_feature["shape"])
        if len(state_shape) != 1 or not state_shape or state_shape[0] > int(raw["max_state_dim"]):
            raise ValueError(f"Unsupported observation.state shape {state_shape}")
        if len(action_shape) != 1 or not action_shape or action_shape[0] > int(raw["max_action_dim"]):
            raise ValueError(f"Unsupported action shape {action_shape}")
        image_size = tuple(raw["resize_imgs_with_padding"])
        if len(image_size) != 2:
            raise ValueError(f"resize_imgs_with_padding must contain width and height, got {image_size!r}")
        expert_hidden_size = int(_AUDITED_TEXT_HIDDEN_SIZE * float(raw["expert_width_multiplier"]))
        architecture_values = {
            "chunk_size": int(raw["chunk_size"]),
            "max_action_dim": int(raw["max_action_dim"]),
            "max_state_dim": int(raw["max_state_dim"]),
            "resize_imgs_with_padding": tuple(int(value) for value in image_size),
            "num_vlm_layers": int(raw["num_vlm_layers"]),
            "expert_width_multiplier": float(raw["expert_width_multiplier"]),
            "self_attn_every_n_layers": int(raw["self_attn_every_n_layers"]),
        }
        mismatches = [
            f"{name}={architecture_values[name]!r} (supported {expected!r})"
            for name, expected in _AUDITED_ARCHITECTURE.items()
            if architecture_values[name] != expected
        ]
        if mismatches:
            cameras = ", ".join(f"{name} {shape}" for name, shape in image_shapes)
            contract = (
                f"at least one camera from [{cameras}]; "
                f"observation.state {state_shape}; action {action_shape}"
            )
            raise ValueError(
                "Unsupported SmolVLA architecture: "
                + "; ".join(mismatches)
                + f". Checkpoint contract: {contract}"
            )

        return cls(
            checkpoint_dir=checkpoint_dir,
            vlm_model_name=vlm_model_name,
            chunk_size=int(raw["chunk_size"]),
            n_action_steps=int(raw["n_action_steps"]),
            action_dim=action_shape[0],
            state_dim=state_shape[0],
            max_action_dim=int(raw["max_action_dim"]),
            max_state_dim=int(raw["max_state_dim"]),
            image_size=(int(image_size[0]), int(image_size[1])),
            image_keys=image_keys,
            image_shapes=image_shapes,
            state_shape=state_shape,
            action_shape=action_shape,
            empty_cameras=int(raw["empty_cameras"]),
            tokenizer_max_length=int(raw["tokenizer_max_length"]),
            tokenizer_padding=str(raw["pad_language_to"]),
            num_steps=int(raw["num_steps"]),
            min_period=float(raw["min_period"]),
            max_period=float(raw["max_period"]),
            vlm_layers=int(raw["num_vlm_layers"]),
            expert_width_multiplier=float(raw["expert_width_multiplier"]),
            expert_hidden_size=expert_hidden_size,
            self_attn_every_n_layers=int(raw["self_attn_every_n_layers"]),
            prefix_length=int(raw["prefix_length"]),
            state_normalization=_effective_normalization(
                checkpoint_dir,
                "observation.state",
                "policy_preprocessor_step_5_normalizer_processor.safetensors",
            ),
            action_normalization=_effective_normalization(
                checkpoint_dir,
                "action",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            ),
        )
