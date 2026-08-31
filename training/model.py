"""Differentiable composition of the existing native SmolVLA modules."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from smolvla_mlx.connector import Connector
from smolvla_mlx.expert import ActionExpert
from smolvla_mlx.language import TruncatedLanguageModel, pad_state_to_width
from smolvla_mlx.types import ProcessedObservation
from smolvla_mlx.vision import VisionEncoder
from training.objective import flow_matching_inputs, masked_velocity_mse


_CAMERA_COUNT = 2
_IMAGE_SIZE = 512
_TOKEN_COUNT = 48
_STATE_DIM = 6
_STATE_PADDED_DIM = 32
_CHUNK_SIZE = 50
_ACTION_DIM = 6
_ACTION_PADDED_DIM = 32


@dataclass(frozen=True)
class TrainingBatch:
    """One fully prepared batch and its caller-controlled flow draws."""

    processed: ProcessedObservation
    actions: mx.array
    noise: mx.array
    timesteps: mx.array
    action_dim: int


class SmolVLATrainingModel(nn.Module):
    """Full native architecture arranged under a differentiable MLX module."""

    def __init__(self) -> None:
        super().__init__()
        self.vision = VisionEncoder()
        self.connector = Connector()
        self.language = TruncatedLanguageModel()
        self.state_proj = nn.Linear(_STATE_PADDED_DIM, 960, bias=True)
        self.expert = ActionExpert()


def make_random_audit_batch(seed: int) -> TrainingBatch:
    """Return the fixed-shape deterministic synthetic batch required by T0."""

    mx.random.seed(seed)
    processed = ProcessedObservation(
        pixel_values=mx.random.normal(
            (_CAMERA_COUNT, 3, _IMAGE_SIZE, _IMAGE_SIZE)
        ).astype(mx.float32),
        pixel_attention_mask=mx.ones((_CAMERA_COUNT, 1), dtype=mx.bool_),
        input_ids=mx.arange(_TOKEN_COUNT, dtype=mx.int32)[None, :],
        text_attention_mask=mx.ones((1, _TOKEN_COUNT), dtype=mx.bool_),
        state=mx.random.normal((1, _STATE_DIM)).astype(mx.float32),
    )
    return TrainingBatch(
        processed=processed,
        actions=mx.random.normal((1, _CHUNK_SIZE, _ACTION_PADDED_DIM)).astype(mx.float32),
        noise=mx.random.normal((1, _CHUNK_SIZE, _ACTION_PADDED_DIM)).astype(mx.float32),
        timesteps=mx.array([0.5], dtype=mx.float32),
        action_dim=_ACTION_DIM,
    )


def training_loss(model: SmolVLATrainingModel, batch: TrainingBatch) -> mx.array:
    """Run the full prefix/expert path and return physical-action flow MSE."""

    processed = batch.processed
    vision_features = model.vision(
        processed.pixel_values,
        processed.pixel_attention_mask,
    )
    image_tokens = model.connector(vision_features)
    padded_state = pad_state_to_width(processed.state, width=_STATE_PADDED_DIM)
    state_embedding = model.state_proj(padded_state)[:, None, :]
    prefix = model.language.build_prefix(processed, image_tokens, state_embedding)
    cache = model.language.encode_prefix(prefix)
    noisy_actions, target_velocity = flow_matching_inputs(
        batch.actions,
        batch.noise,
        batch.timesteps,
    )
    predicted_velocity = model.expert.denoise(
        cache,
        noisy_actions,
        batch.timesteps,
    ).velocity
    return masked_velocity_mse(
        predicted_velocity,
        target_velocity,
        action_dim=batch.action_dim,
    )
