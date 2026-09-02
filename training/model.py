"""Differentiable composition of the existing native SmolVLA modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_smolvla.connector import Connector
from mlx_smolvla.expert import ActionExpert
from mlx_smolvla.language import TruncatedLanguageModel, pad_state_to_width
from mlx_smolvla.types import ProcessedObservation
from mlx_smolvla.vision import VisionEncoder
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
    action_is_pad: mx.array
    noise: mx.array
    timesteps: mx.array
    action_dim: int


class SmolVLATrainingModel(nn.Module):
    """Full native architecture arranged under a differentiable MLX module."""

    def __init__(
        self,
        *,
        vision: VisionEncoder | None = None,
        connector: Connector | None = None,
        language: TruncatedLanguageModel | None = None,
        state_proj: nn.Linear | None = None,
        expert: ActionExpert | None = None,
    ) -> None:
        super().__init__()
        self.vision = VisionEncoder() if vision is None else vision
        self.connector = Connector() if connector is None else connector
        self.language = TruncatedLanguageModel() if language is None else language
        self.state_proj = (
            nn.Linear(_STATE_PADDED_DIM, 960, bias=True)
            if state_proj is None
            else state_proj
        )
        self.expert = ActionExpert() if expert is None else expert
        self._converted_weights_path: Path | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path = "lerobot/smolvla_base",
        cache_dir: str | Path | None = None,
        dtype: object = mx.float32,
        *,
        tokenizer_dir: str | Path | None = None,
    ) -> "SmolVLATrainingModel":
        """Compose training ownership around the strict native checkpoint load."""

        if dtype not in (mx.float32, mx.bfloat16, "float32", "bfloat16"):
            raise ValueError("training weights must use float32 or bfloat16 storage")
        from mlx_smolvla.policy import SmolVLAMLX

        policy = SmolVLAMLX.from_pretrained(
            model_id=model_id,
            cache_dir=cache_dir,
            dtype=dtype,
            tokenizer_dir=tokenizer_dir,
            execution_mode="strict",
        )
        model = cls(
            vision=policy.vision,
            connector=policy.connector,
            language=policy.language,
            state_proj=policy.state_proj,
            expert=policy.expert,
        )
        model._converted_weights_path = policy.converted_weights_path
        return model

    @property
    def converted_weights_path(self) -> Path | None:
        """Return the strict converted artifact backing a loaded model, if any."""

        return self._converted_weights_path


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
        action_is_pad=mx.zeros((1, _CHUNK_SIZE), dtype=mx.bool_),
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
        action_is_pad=batch.action_is_pad,
    )
