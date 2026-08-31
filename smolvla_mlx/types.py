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


@dataclass(frozen=True)
class PrefixInputs:
    """The fully assembled VLM prefix before decoder prefill."""

    embeddings: mx.array
    pad_mask: mx.array
    attention_flags: mx.array
    attention_mask: mx.array
    position_ids: mx.array


@dataclass(frozen=True)
class PrefixCache:
    """The frozen prefix decoder output and post-RoPE K/V tensors by layer."""

    hidden: mx.array
    keys: tuple[mx.array, ...]
    values: tuple[mx.array, ...]
    mask: mx.array
    layer_outputs: tuple[mx.array, ...] = ()
