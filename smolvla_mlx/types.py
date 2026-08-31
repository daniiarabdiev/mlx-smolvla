"""Typed values exchanged between native SmolVLA runtime stages."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class ProcessedObservation:
    """Exact native representation of one batched policy observation."""

    pixel_values: mx.array
    pixel_attention_mask: mx.array
    input_ids: mx.array
    text_attention_mask: mx.array
    state: mx.array
