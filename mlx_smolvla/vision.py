# Adapted from mlx-vlm 0.6.4:
# mlx_vlm/models/idefics3/vision.py
# Copyright © 2025 Prince Canuma
# SPDX-License-Identifier: MIT
"""The dependency-isolated SigLIP-style vision tower used by SmolVLA.

This is a deliberately focused adaptation of mlx-vlm's Idefics3 vision model.
It accepts the NCHW image batches emitted by :mod:`mlx_smolvla.preprocessing`
and retains the converted checkpoint's canonical ``vision.*`` parameter tree.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


_IMAGE_SIZE = 512
_PATCH_SIZE = 16
_HIDDEN_SIZE = 768
_INTERMEDIATE_SIZE = 3072
_NUM_HEADS = 12
_NUM_LAYERS = 12
_LAYER_NORM_EPS = 1e-6


def _gelu_pytorch_tanh(x: mx.array) -> mx.array:
    """Match ``torch.nn.functional.gelu(..., approximate='tanh')`` exactly."""

    coefficient = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + mx.tanh(coefficient * (x + 0.044715 * x * x * x)))


class VisionAttention(nn.Module):
    """Unmasked 12-head self-attention over the 32x32 patch grid."""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = _NUM_HEADS
        self.scale = (_HIDDEN_SIZE // _NUM_HEADS) ** -0.5
        self.q_proj = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=True)
        self.k_proj = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=True)
        self.v_proj = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=True)
        self.out_proj = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        batch_size, query_length, _ = queries.shape
        key_length = keys.shape[1]
        queries = queries.reshape(batch_size, query_length, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, key_length, self.num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, key_length, self.num_heads, -1).transpose(0, 2, 1, 3)

        attention_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale
        )
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)
        return self.out_proj(attention_output)


class VisionMLP(nn.Module):
    """SmolVLM's tanh-GELU MLP, retaining the checkpoint's parameter names."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, bias=True)
        self.fc2 = nn.Linear(_INTERMEDIATE_SIZE, _HIDDEN_SIZE, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.fc2(_gelu_pytorch_tanh(self.fc1(hidden_states)))


class VisionEncoderLayer(nn.Module):
    """Pre-normalized residual vision-transformer layer."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = VisionAttention()
        self.layer_norm1 = nn.LayerNorm(_HIDDEN_SIZE, eps=_LAYER_NORM_EPS)
        self.mlp = VisionMLP()
        self.layer_norm2 = nn.LayerNorm(_HIDDEN_SIZE, eps=_LAYER_NORM_EPS)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class VisionEmbeddings(nn.Module):
    """Patch convolution and fixed 32x32 position embedding table."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=_HIDDEN_SIZE,
            kernel_size=_PATCH_SIZE,
            stride=_PATCH_SIZE,
            bias=True,
        )
        self.position_embedding = nn.Embedding(
            (_IMAGE_SIZE // _PATCH_SIZE) ** 2,
            _HIDDEN_SIZE,
        )

    def __call__(self, pixels_nhwc: mx.array) -> mx.array:
        patches = self.patch_embedding(pixels_nhwc)
        batch_size = patches.shape[0]
        patches = patches.reshape(batch_size, -1, _HIDDEN_SIZE)
        positions = self.position_embedding(mx.arange(patches.shape[1], dtype=mx.int32))
        return patches + positions


class VisionTransformerEncoder(nn.Module):
    """Container retained solely to preserve ``encoder.layers.*`` checkpoint names."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [VisionEncoderLayer() for _ in range(_NUM_LAYERS)]


class VisionEncoder(nn.Module):
    """SmolVLA's complete 12-layer 512px vision encoder.

    The reference policy always routes a ``None`` patch-attention mask to the
    visual tower. ``pixel_mask`` remains an accepted interface argument so the
    native policy can pass its preprocessing result without altering the proven
    unmasked reference behavior.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = VisionEmbeddings()
        self.encoder = VisionTransformerEncoder()
        self.post_layernorm = nn.LayerNorm(_HIDDEN_SIZE, eps=_LAYER_NORM_EPS)

    def __call__(self, pixel_values: mx.array, pixel_mask: mx.array | None = None) -> mx.array:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(
                "pixel_values must have NCHW shape [batch, 3, height, width], "
                f"got {pixel_values.shape}"
            )
        if pixel_values.shape[2:] != (_IMAGE_SIZE, _IMAGE_SIZE):
            raise ValueError(
                f"SmolVLA vision requires {_IMAGE_SIZE}x{_IMAGE_SIZE} inputs, got {pixel_values.shape[2:]}"
            )
        if pixel_mask is not None and pixel_mask.shape[0] != pixel_values.shape[0]:
            raise ValueError("pixel_mask batch size must equal pixel_values batch size")

        # Keep activation math in fp32 even when checkpoint weights are stored
        # in bf16. The reference itself upcasts its bf16 checkpoint to CPU fp32
        # before producing goldens, and MLX promotes bf16 weights accordingly.
        # MLX Conv2d uses NHWC input and OHWI converted weights.
        hidden_states = self.embeddings(pixel_values.astype(mx.float32).transpose(0, 2, 3, 1))
        for layer in self.encoder.layers:
            hidden_states = layer(hidden_states)
        return self.post_layernorm(hidden_states)
