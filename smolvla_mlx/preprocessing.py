"""Dependency-light SmolVLA observation preprocessing with reference parity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from smolvla_mlx.config import SmolVLAConfig
from smolvla_mlx.types import ProcessedObservation


def _as_float_chw(value: object, key: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{key} must have three dimensions, got {array.shape}")
    if array.shape[0] == 3:
        chw = array
    elif array.shape[-1] == 3:
        chw = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"{key} must be RGB CHW or HWC, got {array.shape}")
    if chw.dtype == np.uint8:
        chw = chw.astype(np.float32) / np.float32(255.0)
    else:
        chw = chw.astype(np.float32, copy=False)
    if not np.isfinite(chw).all():
        raise ValueError(f"{key} contains non-finite values")
    if float(chw.min()) < 0.0 or float(chw.max()) > 1.0:
        raise ValueError(f"{key} must be uint8 or float values in [0, 1]")
    return np.ascontiguousarray(chw)


def _resize_bilinear_align_corners_false(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match PyTorch bilinear `interpolate(..., align_corners=False)` for CHW input."""

    channels, input_height, input_width = image.shape
    if (input_height, input_width) == (height, width):
        return image

    y = (np.arange(height, dtype=np.float32) + np.float32(0.5)) * (
        np.float32(input_height) / np.float32(height)
    ) - np.float32(0.5)
    x = (np.arange(width, dtype=np.float32) + np.float32(0.5)) * (
        np.float32(input_width) / np.float32(width)
    ) - np.float32(0.5)
    y0_unclipped = np.floor(y).astype(np.int64)
    x0_unclipped = np.floor(x).astype(np.int64)
    y1_unclipped = y0_unclipped + 1
    x1_unclipped = x0_unclipped + 1
    y0 = np.clip(y0_unclipped, 0, input_height - 1)
    y1 = np.clip(y1_unclipped, 0, input_height - 1)
    x0 = np.clip(x0_unclipped, 0, input_width - 1)
    x1 = np.clip(x1_unclipped, 0, input_width - 1)
    wy = (y - y0_unclipped.astype(np.float32)).reshape(1, height, 1)
    wx = (x - x0_unclipped.astype(np.float32)).reshape(1, 1, width)

    top = image[:, y0, :]
    bottom = image[:, y1, :]
    resized_height = top * (np.float32(1.0) - wy) + bottom * wy
    left = resized_height[:, :, x0]
    right = resized_height[:, :, x1]
    return np.ascontiguousarray(left * (np.float32(1.0) - wx) + right * wx)


def resize_with_top_left_padding(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Resize aspect-preservingly, then pad left/top exactly like SmolVLA."""

    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got {image.shape}")
    _, current_height, current_width = image.shape
    if (current_height, current_width) == (height, width):
        return image
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = _resize_bilinear_align_corners_false(image, resized_height, resized_width)
    padded = np.zeros((image.shape[0], height, width), dtype=np.float32)
    padded[:, height - resized_height :, width - resized_width :] = resized
    return padded


def _pad_id(tokenizer: Tokenizer, tokenizer_dir: Path) -> int:
    config = json.loads((tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    token = config.get("pad_token")
    if not isinstance(token, str):
        raise ValueError("Tokenizer config has no string pad_token")
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer pad token {token!r} is absent from tokenizer.json")
    return token_id


@dataclass
class SmolVLAPreprocessor:
    """A native processor for camera frames, instruction text, and robot state."""

    config: SmolVLAConfig
    tokenizer: Tokenizer
    pad_token_id: int

    @classmethod
    def from_pretrained_files(
        cls,
        checkpoint_dir: Path,
        tokenizer_dir: Path,
    ) -> "SmolVLAPreprocessor":
        config = SmolVLAConfig.from_pretrained_files(checkpoint_dir)
        tokenizer_path = tokenizer_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return cls(config=config, tokenizer=tokenizer, pad_token_id=_pad_id(tokenizer, tokenizer_dir))

    def _tokenize(self, task: object) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(task, str) or not task:
            raise ValueError("task must be a non-empty string")
        prompt = task if task.endswith("\n") else f"{task}\n"
        encoding = self.tokenizer.encode(prompt, add_special_tokens=True)
        token_ids = np.asarray(encoding.ids[: self.config.tokenizer_max_length], dtype=np.int64)
        input_ids = np.full((1, self.config.tokenizer_max_length), self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((1, self.config.tokenizer_max_length), dtype=bool)
        input_ids[0, : token_ids.size] = token_ids
        attention_mask[0, : token_ids.size] = True
        return input_ids, attention_mask

    def _state(self, value: object) -> np.ndarray:
        state = np.asarray(value, dtype=np.float32)
        if state.ndim != 1 or state.shape[0] != self.config.state_dim:
            raise ValueError(f"observation.state must have shape ({self.config.state_dim},), got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("observation.state contains non-finite values")
        # The saved processor's statistics do not match observation.state for this checkpoint.
        return state.reshape(1, -1)

    def __call__(self, observation: Mapping[str, object]) -> ProcessedObservation:
        present_keys = [key for key in self.config.image_keys if key in observation]
        if not present_keys:
            raise ValueError(f"At least one configured camera is required; expected one of {self.config.image_keys}")
        width, height = self.config.image_size
        images = [
            resize_with_top_left_padding(_as_float_chw(observation[key], key), height=height, width=width)
            for key in present_keys
        ]
        pixel_masks = [True] * len(images)
        for _ in range(min(len(self.config.image_keys) - len(present_keys), self.config.empty_cameras)):
            images.append(np.full_like(images[-1], -1.0, dtype=np.float32))
            pixel_masks.append(False)
        pixel_values = np.stack(images, axis=0) * np.float32(2.0) - np.float32(1.0)
        input_ids, text_attention_mask = self._tokenize(observation.get("task"))
        state = self._state(observation.get("observation.state"))
        return ProcessedObservation(
            pixel_values=mx.array(pixel_values),
            pixel_attention_mask=mx.array(np.asarray(pixel_masks, dtype=bool).reshape(-1, 1)),
            input_ids=mx.array(input_ids),
            text_attention_mask=mx.array(text_attention_mask),
            state=mx.array(state),
        )

    def normalize_actions(self, actions: mx.array) -> mx.array:
        """The pinned checkpoint's saved action stats do not match the `action` key."""

        return actions

    def unnormalize_actions(self, actions: mx.array) -> mx.array:
        """Mirror the reference postprocessor's effective identity transform."""

        return actions
