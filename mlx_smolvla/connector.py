# Adapted from mlx-vlm 0.6.4:
# mlx_vlm/models/idefics3/idefics3.py
# Copyright © 2025 Prince Canuma
# SPDX-License-Identifier: MIT
"""The focused scale-4 image-token connector used by SmolVLA."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


_VISION_HIDDEN_SIZE = 768
_TEXT_HIDDEN_SIZE = 960
_SCALE_FACTOR = 4


class ModalityProjection(nn.Module):
    """Project each scale-4 pixel-shuffled visual token into text space."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(
            _VISION_HIDDEN_SIZE * (_SCALE_FACTOR**2),
            _TEXT_HIDDEN_SIZE,
            bias=False,
        )

    def __call__(self, image_hidden_states: mx.array) -> mx.array:
        return self.proj(image_hidden_states)


class Connector(nn.Module):
    """Reshape 32x32 vision tokens into 8x8 tokens and project to 960 dims."""

    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = _SCALE_FACTOR
        self.modality_projection = ModalityProjection()

    def pixel_shuffle(self, image_hidden_states: mx.array) -> mx.array:
        """Apply the checked Idefics3 layout-preserving pixel shuffle."""

        if image_hidden_states.ndim != 3:
            raise ValueError(
                "image_hidden_states must have [batch, sequence, hidden] shape, "
                f"got {image_hidden_states.shape}"
            )
        batch_size, sequence_length, hidden_size = image_hidden_states.shape
        side = math.isqrt(sequence_length)
        if side * side != sequence_length or side % self.scale_factor:
            raise ValueError(
                "connector requires a square patch sequence whose side is divisible by "
                f"{self.scale_factor}, got sequence length {sequence_length}"
            )
        if hidden_size != _VISION_HIDDEN_SIZE:
            raise ValueError(
                f"connector requires {_VISION_HIDDEN_SIZE}-wide vision tokens, got {hidden_size}"
            )

        scale = self.scale_factor
        hidden_states = image_hidden_states.reshape(batch_size, side, side, hidden_size)
        hidden_states = hidden_states.reshape(batch_size, side, side // scale, hidden_size * scale)
        hidden_states = hidden_states.transpose(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(
            batch_size,
            side // scale,
            side // scale,
            hidden_size * (scale**2),
        )
        hidden_states = hidden_states.transpose(0, 2, 1, 3)
        return hidden_states.reshape(
            batch_size,
            sequence_length // (scale**2),
            hidden_size * (scale**2),
        )

    def __call__(self, image_hidden_states: mx.array) -> mx.array:
        # See VisionEncoder: bf16 is a compact parameter-storage option while
        # activation math remains fp32 for the CPU/fp32 golden contract.
        return self.modality_projection(self.pixel_shuffle(image_hidden_states.astype(mx.float32)))
