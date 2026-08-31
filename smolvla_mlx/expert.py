"""Native MLX action expert used by SmolVLA's flow-matching sampler.

The execution order matches the audited LeRobot implementation: even expert
layers append action K/V to the frozen VLM cache for causal self-attention,
while odd layers project that frozen VLM K/V cache for cross-attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from smolvla_mlx.language import _apply_reference_rope
from smolvla_mlx.rmsnorm import ReferenceRMSNorm, reference_silu, reference_softmax
from smolvla_mlx.types import PrefixCache


_ACTION_DIM = 32
_HIDDEN_SIZE = 720
_INTERMEDIATE_SIZE = 2048
_NUM_LAYERS = 16
_NUM_HEADS = 15
_NUM_KV_HEADS = 5
_HEAD_DIM = 64
_CACHE_WIDTH = _NUM_KV_HEADS * _HEAD_DIM
_RMS_NORM_EPS = 1e-5
_MIN_PERIOD = 0.004
_MAX_PERIOD = 4.0


@dataclass(frozen=True)
class DenoiseResult:
    """Named action-expert boundaries captured during one denoising call."""

    suffix_embeddings: mx.array
    hidden: mx.array
    velocity: mx.array
    layer_outputs: tuple[mx.array, ...] = ()


def timestep_embedding(
    timestep: mx.array,
    *,
    dimension: int = _HIDDEN_SIZE,
    min_period: float = _MIN_PERIOD,
    max_period: float = _MAX_PERIOD,
) -> mx.array:
    """Match the reference's float64 sinusoid construction before its fp32 cast."""

    if timestep.ndim != 1:
        raise ValueError(f"timestep must have [batch] shape, got {timestep.shape}")
    if dimension % 2:
        raise ValueError(f"dimension must be even, got {dimension}")
    fraction = mx.linspace(0.0, 1.0, dimension // 2, dtype=mx.float64)
    period = mx.array(min_period, dtype=mx.float64) * mx.power(
        mx.array(max_period / min_period, dtype=mx.float64), fraction
    )
    scaling = (mx.array(2.0 * math.pi, dtype=mx.float64) / period)[None, :]
    radians = timestep.astype(mx.float64)[:, None] * scaling
    return mx.concatenate((mx.sin(radians), mx.cos(radians)), axis=1).astype(mx.float32)


def _repeat_kv(states: mx.array) -> mx.array:
    """Expand five K/V heads to the reference's fifteen query-head layout."""

    batch_size, sequence_length, key_value_heads, head_dimension = states.shape
    if key_value_heads != _NUM_KV_HEADS or head_dimension != _HEAD_DIM:
        raise ValueError(f"expected [batch, sequence, {_NUM_KV_HEADS}, {_HEAD_DIM}] K/V states, got {states.shape}")
    groups = _NUM_HEADS // _NUM_KV_HEADS
    return mx.broadcast_to(
        states[:, :, :, None, :],
        (batch_size, sequence_length, key_value_heads, groups, head_dimension),
    ).reshape(batch_size, sequence_length, _NUM_HEADS, _HEAD_DIM)


def _prefix_valid_tokens(cache: PrefixCache) -> mx.array:
    """Recover the cached prefix's key-valid mask from its audited 2-D mask."""

    if cache.mask.ndim != 3 or cache.mask.shape[1] != cache.mask.shape[2]:
        raise ValueError(f"cache mask must be [batch, prefix, prefix], got {cache.mask.shape}")
    return mx.any(cache.mask.astype(mx.bool_), axis=1)


def _suffix_position_ids(cache: PrefixCache, suffix_length: int) -> mx.array:
    prefix_valid = _prefix_valid_tokens(cache)
    prefix_offsets = mx.sum(prefix_valid.astype(mx.int32), axis=-1)[:, None]
    suffix_offsets = mx.arange(suffix_length, dtype=mx.int32)[None, :]
    return prefix_offsets + suffix_offsets


def _self_attention_mask(cache: PrefixCache, suffix_length: int) -> mx.array:
    """Build the reference [prefix-valid; causal-action] denoising mask."""

    prefix_valid = _prefix_valid_tokens(cache)
    batch_size, prefix_length = prefix_valid.shape
    prefix_mask = mx.broadcast_to(prefix_valid[:, None, :], (batch_size, suffix_length, prefix_length))
    positions = mx.arange(suffix_length, dtype=mx.int32)
    suffix_mask = positions[None, :] <= positions[:, None]
    suffix_mask = mx.broadcast_to(suffix_mask[None, :, :], (batch_size, suffix_length, suffix_length))
    return mx.concatenate((prefix_mask, suffix_mask), axis=2)


class ExpertAttention(nn.Module):
    """Grouped-query attention with separate query and K/V input widths."""

    def __init__(self, key_value_input_width: int) -> None:
        super().__init__()
        self.scale = _HEAD_DIM**-0.5
        self.q_proj = nn.Linear(_HIDDEN_SIZE, _NUM_HEADS * _HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(key_value_input_width, _NUM_KV_HEADS * _HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(key_value_input_width, _NUM_KV_HEADS * _HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(_NUM_HEADS * _HEAD_DIM, _HIDDEN_SIZE, bias=False)

    def project_query(self, hidden_states: mx.array, position_ids: mx.array) -> mx.array:
        batch_size, sequence_length, width = hidden_states.shape
        if width != _HIDDEN_SIZE:
            raise ValueError(f"expert queries must end in {_HIDDEN_SIZE}, got {hidden_states.shape}")
        query = self.q_proj(hidden_states.astype(mx.float32)).reshape(
            batch_size, sequence_length, _NUM_HEADS, _HEAD_DIM
        )
        return _apply_reference_rope(query, position_ids)

    def project_key_value(
        self,
        inputs: mx.array,
        *,
        position_ids: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        if inputs.ndim != 3:
            raise ValueError(f"expert K/V inputs must be [batch, sequence, width], got {inputs.shape}")
        batch_size, sequence_length, _ = inputs.shape
        inputs = inputs.astype(mx.float32)
        keys = self.k_proj(inputs).reshape(batch_size, sequence_length, _NUM_KV_HEADS, _HEAD_DIM)
        values = self.v_proj(inputs).reshape(batch_size, sequence_length, _NUM_KV_HEADS, _HEAD_DIM)
        if position_ids is not None:
            keys = _apply_reference_rope(keys, position_ids)
        return keys, values

    def attend(
        self,
        query_states: mx.array,
        key_states: mx.array,
        value_states: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        batch_size, query_length, query_heads, head_dimension = query_states.shape
        if query_heads != _NUM_HEADS or head_dimension != _HEAD_DIM:
            raise ValueError(f"unexpected expert query shape {query_states.shape}")
        if attention_mask.shape != (batch_size, query_length, key_states.shape[1]):
            raise ValueError(
                "attention mask must have [batch, query, key] shape matching projected K/V; "
                f"got {attention_mask.shape} for queries {query_states.shape} and keys {key_states.shape}"
            )

        keys = _repeat_kv(key_states).transpose(0, 2, 1, 3)
        values = _repeat_kv(value_states).transpose(0, 2, 1, 3)
        queries = query_states.astype(mx.float32).transpose(0, 2, 1, 3)
        scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * self.scale
        scores = mx.where(
            attention_mask[:, None, :, :],
            scores,
            mx.array(-3.4028234663852886e38, dtype=mx.float32),
        ).astype(mx.float32)
        probabilities = reference_softmax(scores) if mx.default_device() == mx.cpu else mx.softmax(scores, axis=-1)
        output = mx.matmul(probabilities, values)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_length, _NUM_HEADS * _HEAD_DIM)
        return self.o_proj(output)


class ExpertMLP(nn.Module):
    """The action expert's bias-free SwiGLU MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(_INTERMEDIATE_SIZE, _HIDDEN_SIZE, bias=False)
        self.up_proj = nn.Linear(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        gate = self.gate_proj(hidden_states.astype(mx.float32))
        gate = reference_silu(gate) if mx.default_device() == mx.cpu else nn.silu(gate)
        return self.down_proj(gate * self.up_proj(hidden_states.astype(mx.float32)))


class ExpertLayer(nn.Module):
    """One self- or VLM-cache-cross-attending expert residual block."""

    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.is_self_attention = layer_index % 2 == 0
        self.self_attn = ExpertAttention(_HIDDEN_SIZE if self.is_self_attention else _CACHE_WIDTH)
        self.mlp = ExpertMLP()
        self.input_layernorm = ReferenceRMSNorm(_HIDDEN_SIZE, eps=_RMS_NORM_EPS)
        self.post_attention_layernorm = ReferenceRMSNorm(_HIDDEN_SIZE, eps=_RMS_NORM_EPS)

    def __call__(self, hidden_states: mx.array, cache: PrefixCache) -> mx.array:
        if self.layer_index >= len(cache.keys) or self.layer_index >= len(cache.values):
            raise ValueError(f"cache is missing VLM K/V for expert layer {self.layer_index}")
        normalized = self.input_layernorm(hidden_states.astype(mx.float32))
        suffix_length = hidden_states.shape[1]
        suffix_positions = _suffix_position_ids(cache, suffix_length)
        attention_mask = _self_attention_mask(cache, suffix_length)

        if self.is_self_attention:
            query = self.self_attn.project_query(normalized, suffix_positions)
            suffix_keys, suffix_values = self.self_attn.project_key_value(
                normalized,
                position_ids=suffix_positions,
            )
            prefix_keys = cache.keys[self.layer_index].transpose(0, 2, 1, 3)
            prefix_values = cache.values[self.layer_index].transpose(0, 2, 1, 3)
            keys = mx.concatenate((prefix_keys, suffix_keys), axis=1)
            values = mx.concatenate((prefix_values, suffix_values), axis=1)
            attention_output = self.self_attn.attend(query, keys, values, attention_mask)
        else:
            query_positions = suffix_positions - mx.min(suffix_positions, axis=1, keepdims=True)
            query = self.self_attn.project_query(normalized, query_positions)
            prefix_keys = cache.keys[self.layer_index].transpose(0, 2, 1, 3).reshape(
                cache.keys[self.layer_index].shape[0], -1, _CACHE_WIDTH
            )
            prefix_values = cache.values[self.layer_index].transpose(0, 2, 1, 3).reshape(
                cache.values[self.layer_index].shape[0], -1, _CACHE_WIDTH
            )
            keys, values = self.self_attn.project_key_value(
                prefix_keys,
                position_ids=None,
            )
            # Source cross-attention projects values separately from the frozen cache.
            _, values = self.self_attn.project_key_value(prefix_values, position_ids=None)
            attention_output = self.self_attn.attend(
                query,
                keys,
                values,
                attention_mask[:, :, : keys.shape[1]],
            )

        after_attention = hidden_states.astype(mx.float32) + attention_output
        return after_attention + self.mlp(self.post_attention_layernorm(after_attention))


class ActionExpert(nn.Module):
    """The checkpoint's action/timestep projections, 16 blocks, norm, and velocity head."""

    def __init__(self) -> None:
        super().__init__()
        self.action_in_proj = nn.Linear(_ACTION_DIM, _HIDDEN_SIZE, bias=True)
        self.action_out_proj = nn.Linear(_HIDDEN_SIZE, _ACTION_DIM, bias=True)
        self.action_time_mlp_in = nn.Linear(_HIDDEN_SIZE * 2, _HIDDEN_SIZE, bias=True)
        self.action_time_mlp_out = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=True)
        self.layers = [ExpertLayer(layer_index) for layer_index in range(_NUM_LAYERS)]
        self.norm = ReferenceRMSNorm(_HIDDEN_SIZE, eps=_RMS_NORM_EPS)

    def embed_suffix(self, noisy_actions: mx.array, timestep: mx.array) -> mx.array:
        """Fuse each padded action token with its reference sinusoidal timestep."""

        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != _ACTION_DIM:
            raise ValueError(f"noisy_actions must have [batch, chunk, {_ACTION_DIM}] shape, got {noisy_actions.shape}")
        if timestep.shape != (noisy_actions.shape[0],):
            raise ValueError(f"timestep shape {timestep.shape} does not match batch {noisy_actions.shape[0]}")
        action_embeddings = self.action_in_proj(noisy_actions.astype(mx.float32))
        time_embeddings = timestep_embedding(timestep).astype(action_embeddings.dtype)
        time_embeddings = mx.broadcast_to(
            time_embeddings[:, None, :],
            action_embeddings.shape,
        )
        hidden_states = self.action_time_mlp_in(mx.concatenate((action_embeddings, time_embeddings), axis=2))
        hidden_states = reference_silu(hidden_states) if mx.default_device() == mx.cpu else nn.silu(hidden_states)
        return self.action_time_mlp_out(hidden_states).astype(mx.float32)

    def denoise(
        self,
        cache: PrefixCache,
        noisy_actions: mx.array,
        timestep: mx.array,
        *,
        collect_layer_outputs: bool = False,
    ) -> DenoiseResult:
        """Run one velocity-field evaluation without mutating the frozen prefix cache."""

        hidden_states = self.embed_suffix(noisy_actions, timestep)
        suffix_embeddings = hidden_states
        layer_outputs: list[mx.array] = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, cache)
            if collect_layer_outputs:
                layer_outputs.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        velocity = self.action_out_proj(hidden_states.astype(mx.float32))
        return DenoiseResult(
            suffix_embeddings=suffix_embeddings,
            hidden=hidden_states,
            velocity=velocity,
            layer_outputs=tuple(layer_outputs),
        )
