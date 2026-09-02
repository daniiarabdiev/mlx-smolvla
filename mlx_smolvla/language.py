# Adapted in part from mlx-vlm 0.6.4:
# mlx_vlm/models/idefics3/language.py
# Copyright © 2025 Prince Canuma
# SPDX-License-Identifier: MIT
#
# The prefix/cache execution boundary follows behavior verified from LeRobot
# 0.6.1 `policies/smolvla/smolvlm_with_expert.py`.
# Copyright 2025 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-isolated truncated SmolVLM decoder and prefix construction."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_smolvla.rmsnorm import ReferenceRMSNorm, reference_rope, reference_silu, reference_softmax
from mlx_smolvla.types import PrefixCache, PrefixInputs, ProcessedObservation


_HIDDEN_SIZE = 960
_INTERMEDIATE_SIZE = 2560
_VOCAB_SIZE = 49280
_USED_LAYERS = 16
_NUM_HEADS = 15
_NUM_KV_HEADS = 5
_HEAD_DIM = 64
_RMS_NORM_EPS = 1e-5
_PREFIX_LENGTH = 177
_ROPE_BASE = 10_000.0


def pad_state_to_width(state: mx.array, *, width: int) -> mx.array:
    """Right-pad state to the checkpoint's 32-wide state-projection input."""

    if state.ndim != 2:
        raise ValueError(f"state must have [batch, features] shape, got {state.shape}")
    if state.shape[1] > width:
        raise ValueError(f"state width {state.shape[1]} exceeds requested width {width}")
    state = state.astype(mx.float32)
    if state.shape[1] == width:
        return state
    padding = mx.zeros((state.shape[0], width - state.shape[1]), dtype=mx.float32)
    return mx.concatenate((state, padding), axis=1)


def _make_attention_mask(pad_mask: mx.array, attention_flags: mx.array) -> mx.array:
    """Reproduce LeRobot's cumulative prefix-LM boolean attention mask."""

    if pad_mask.ndim != 2 or attention_flags.ndim != 2:
        raise ValueError("pad_mask and attention_flags must both have [batch, sequence] shape")
    if pad_mask.shape != attention_flags.shape:
        raise ValueError(f"mask shapes differ: {pad_mask.shape} != {attention_flags.shape}")
    cumulative = mx.cumsum(attention_flags.astype(mx.int32), axis=1)
    causal_boundary = cumulative[:, None, :] <= cumulative[:, :, None]
    valid_pair = mx.logical_and(pad_mask[:, None, :], pad_mask[:, :, None])
    return mx.logical_and(causal_boundary, valid_pair)


def _apply_reference_rope(states: mx.array, position_ids: mx.array) -> mx.array:
    """Apply the LeRobot split-half RoPE with its hard-coded 10,000 base."""

    if states.ndim != 4:
        raise ValueError(f"RoPE states must be [batch, sequence, heads, dimension], got {states.shape}")
    if position_ids.shape != states.shape[:2]:
        raise ValueError(f"RoPE positions {position_ids.shape} do not match states {states.shape[:2]}")
    dimension = states.shape[-1]
    if dimension % 2:
        raise ValueError(f"RoPE dimension must be even, got {dimension}")

    half = dimension // 2
    states = states.astype(mx.float32)
    if mx.default_device() == mx.cpu:
        return reference_rope(states, position_ids.astype(mx.int32))
    exponents = (2.0 / dimension) * mx.arange(half, dtype=mx.float32)
    timescale = mx.power(mx.array(_ROPE_BASE, dtype=mx.float32), exponents)
    radians = position_ids.astype(mx.float32)[..., None] / timescale[None, None, :]
    sin = mx.sin(radians)[..., None, :]
    cos = mx.cos(radians)[..., None, :]
    first, second = states[..., :half], states[..., half:]
    return mx.concatenate((first * cos - second * sin, second * cos + first * sin), axis=-1)


class LanguageAttention(nn.Module):
    """Llama-style grouped-query attention with explicit reference cache export."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = _HEAD_DIM**-0.5
        self.q_proj = nn.Linear(_HIDDEN_SIZE, _NUM_HEADS * _HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(_HIDDEN_SIZE, _NUM_KV_HEADS * _HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(_HIDDEN_SIZE, _NUM_KV_HEADS * _HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(_NUM_HEADS * _HEAD_DIM, _HIDDEN_SIZE, bias=False)

    @staticmethod
    def _repeat_kv(states: mx.array) -> mx.array:
        """Expand five K/V heads into the reference's 15 query-head layout."""

        batch_size, sequence_length, kv_heads, head_dim = states.shape
        groups = _NUM_HEADS // _NUM_KV_HEADS
        return mx.broadcast_to(
            states[:, :, :, None, :],
            (batch_size, sequence_length, kv_heads, groups, head_dim),
        ).reshape(batch_size, sequence_length, _NUM_HEADS, head_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        position_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != _HIDDEN_SIZE:
            raise ValueError(f"hidden_states must end in {_HIDDEN_SIZE}, got {hidden_states.shape}")
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.astype(mx.float32)
        queries = self.q_proj(hidden_states).reshape(batch_size, sequence_length, _NUM_HEADS, _HEAD_DIM)
        keys = self.k_proj(hidden_states).reshape(batch_size, sequence_length, _NUM_KV_HEADS, _HEAD_DIM)
        values = self.v_proj(hidden_states).reshape(batch_size, sequence_length, _NUM_KV_HEADS, _HEAD_DIM)

        queries = _apply_reference_rope(queries, position_ids)
        keys = _apply_reference_rope(keys, position_ids)
        expanded_keys = self._repeat_kv(keys)
        expanded_values = self._repeat_kv(values)

        queries = queries.transpose(0, 2, 1, 3)
        expanded_keys = expanded_keys.transpose(0, 2, 1, 3)
        expanded_values = expanded_values.transpose(0, 2, 1, 3)
        scores = mx.matmul(queries, expanded_keys.transpose(0, 1, 3, 2)) * self.scale
        scores = mx.where(
            attention_mask[:, None, :, :],
            scores,
            mx.array(-3.4028234663852886e38, dtype=mx.float32),
        )
        scores = scores.astype(mx.float32)
        probabilities = reference_softmax(scores) if mx.default_device() == mx.cpu else mx.softmax(scores, axis=-1)
        output = mx.matmul(probabilities, expanded_values)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, _HIDDEN_SIZE)
        return self.o_proj(output), keys.transpose(0, 2, 1, 3), values.transpose(0, 2, 1, 3)


class LanguageMLP(nn.Module):
    """The bias-free SwiGLU MLP used by every SmolLM decoder layer."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(_INTERMEDIATE_SIZE, _HIDDEN_SIZE, bias=False)
        self.up_proj = nn.Linear(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        gate = self.gate_proj(hidden_states)
        gate = reference_silu(gate) if mx.default_device() == mx.cpu else nn.silu(gate)
        return self.down_proj(gate * self.up_proj(hidden_states))


class LanguageLayer(nn.Module):
    """One pre-normalized VLM decoder layer without generic cache machinery."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = LanguageAttention()
        self.mlp = LanguageMLP()
        self.input_layernorm = ReferenceRMSNorm(_HIDDEN_SIZE, eps=_RMS_NORM_EPS)
        self.post_attention_layernorm = ReferenceRMSNorm(_HIDDEN_SIZE, eps=_RMS_NORM_EPS)

    def __call__(
        self,
        hidden_states: mx.array,
        position_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        attention_output, keys, values = self.self_attn(
            self.input_layernorm(hidden_states.astype(mx.float32)),
            position_ids,
            attention_mask,
        )
        hidden_states = hidden_states.astype(mx.float32) + attention_output
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)), keys, values


class TruncatedLanguageModel(nn.Module):
    """All checkpoint language weights, executing only SmolVLA's first 16 layers."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB_SIZE, _HIDDEN_SIZE)
        # The source base-VLM config advertises 32 layers, but the audited
        # SmolVLA checkpoint stores exactly its used 0..15 subset. Keeping this
        # tree at 16 makes strict converted-weight loading meaningful.
        self.layers = [LanguageLayer() for _ in range(_USED_LAYERS)]
        self.norm = ReferenceRMSNorm(_HIDDEN_SIZE, eps=_RMS_NORM_EPS)
        self.lm_head = nn.Linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False)

    def embed_language_tokens(self, input_ids: mx.array) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have [batch, sequence] shape, got {input_ids.shape}")
        return self.embed_tokens(input_ids.astype(mx.int32))

    def build_prefix(
        self,
        processed: ProcessedObservation,
        image_tokens: mx.array,
        state_embedding: mx.array,
    ) -> PrefixInputs:
        """Assemble the exact image/language/state prefix before decoder prefill."""

        if image_tokens.ndim != 3 or image_tokens.shape[-1] != _HIDDEN_SIZE:
            raise ValueError(f"image_tokens must have [images, tokens, {_HIDDEN_SIZE}] shape, got {image_tokens.shape}")
        if processed.pixel_attention_mask.ndim != 2 or processed.pixel_attention_mask.shape[1] != 1:
            raise ValueError("pixel_attention_mask must have [images, 1] shape")
        if image_tokens.shape[0] != processed.pixel_attention_mask.shape[0]:
            raise ValueError("image token count must equal pixel_attention_mask image count")
        if state_embedding.ndim != 3 or state_embedding.shape[1:] != (1, _HIDDEN_SIZE):
            raise ValueError(f"state_embedding must have [batch, 1, {_HIDDEN_SIZE}] shape, got {state_embedding.shape}")
        batch_size = state_embedding.shape[0]
        if batch_size != processed.input_ids.shape[0] or batch_size != processed.state.shape[0]:
            raise ValueError("state, language tokens, and state_embedding must share a batch size")
        if batch_size != 1:
            raise ValueError("v0.1 prefix assembly currently supports one observation at a time")

        image_tokens = image_tokens.astype(mx.float32) * math.sqrt(_HIDDEN_SIZE)
        image_tokens = image_tokens.reshape(1, -1, _HIDDEN_SIZE)
        image_mask = mx.broadcast_to(
            processed.pixel_attention_mask.astype(mx.bool_),
            (processed.pixel_attention_mask.shape[0], image_tokens.shape[1] // processed.pixel_attention_mask.shape[0]),
        ).reshape(1, -1)
        language_tokens = self.embed_language_tokens(processed.input_ids).astype(mx.float32) * math.sqrt(_HIDDEN_SIZE)
        language_mask = processed.text_attention_mask.astype(mx.bool_)
        state_embedding = state_embedding.astype(mx.float32)
        state_mask = mx.ones((batch_size, 1), dtype=mx.bool_)

        embeddings = mx.concatenate((image_tokens, language_tokens, state_embedding), axis=1)
        pad_mask = mx.concatenate((image_mask, language_mask, state_mask), axis=1)
        attention_flags = mx.concatenate(
            (
                mx.zeros((batch_size, image_tokens.shape[1] + language_tokens.shape[1]), dtype=mx.bool_),
                mx.ones((batch_size, 1), dtype=mx.bool_),
            ),
            axis=1,
        )
        if embeddings.shape[1] > _PREFIX_LENGTH:
            raise ValueError(f"assembled prefix length {embeddings.shape[1]} exceeds {_PREFIX_LENGTH}")
        if embeddings.shape[1] < _PREFIX_LENGTH:
            padding = _PREFIX_LENGTH - embeddings.shape[1]
            embeddings = mx.concatenate(
                (embeddings, mx.zeros((batch_size, padding, _HIDDEN_SIZE), dtype=mx.float32)),
                axis=1,
            )
            pad_mask = mx.concatenate((pad_mask, mx.zeros((batch_size, padding), dtype=mx.bool_)), axis=1)
            attention_flags = mx.concatenate(
                (attention_flags, mx.zeros((batch_size, padding), dtype=mx.bool_)), axis=1
            )
        attention_mask = _make_attention_mask(pad_mask, attention_flags)
        position_ids = mx.cumsum(pad_mask.astype(mx.int32), axis=1) - 1
        return PrefixInputs(
            embeddings=embeddings,
            pad_mask=pad_mask,
            attention_flags=attention_flags,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def encode_prefix(
        self,
        prefix: PrefixInputs,
        *,
        stop_after: int | None = None,
        collect_layer_outputs: bool = False,
    ) -> PrefixCache:
        """Run the audited decoder subset and export post-RoPE K/V for every layer."""

        layer_count = _USED_LAYERS if stop_after is None else stop_after
        if not 1 <= layer_count <= _USED_LAYERS:
            raise ValueError(f"stop_after must be in [1, {_USED_LAYERS}], got {layer_count}")
        if prefix.embeddings.ndim != 3 or prefix.embeddings.shape[-1] != _HIDDEN_SIZE:
            raise ValueError(f"prefix embeddings must have [batch, sequence, {_HIDDEN_SIZE}] shape")
        if prefix.attention_mask.shape != (
            prefix.embeddings.shape[0],
            prefix.embeddings.shape[1],
            prefix.embeddings.shape[1],
        ):
            raise ValueError("prefix attention mask must be square over the prefix sequence")

        hidden_states = prefix.embeddings.astype(mx.float32)
        keys: list[mx.array] = []
        values: list[mx.array] = []
        layer_outputs: list[mx.array] = []
        for layer in self.layers[:layer_count]:
            hidden_states, key_states, value_states = layer(
                hidden_states,
                prefix.position_ids,
                prefix.attention_mask,
            )
            keys.append(key_states)
            values.append(value_states)
            if collect_layer_outputs:
                layer_outputs.append(hidden_states)

        output = self.norm(hidden_states) if layer_count == _USED_LAYERS else hidden_states
        return PrefixCache(
            hidden=output,
            keys=tuple(keys),
            values=tuple(values),
            mask=prefix.attention_mask,
            layer_outputs=tuple(layer_outputs),
        )
