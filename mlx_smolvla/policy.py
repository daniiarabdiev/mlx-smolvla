"""Public dependency-isolated SmolVLA policy API backed by native MLX modules."""

from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path
from typing import Literal, Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from huggingface_hub import snapshot_download
import numpy as np

from mlx_smolvla.cache import resolve_cache_dir
from mlx_smolvla.config import SmolVLAConfig
from mlx_smolvla.connector import Connector
from mlx_smolvla.convert import convert_checkpoint
from mlx_smolvla.expert import ActionExpert
from mlx_smolvla.flow import euler_sample
from mlx_smolvla.language import TruncatedLanguageModel, pad_state_to_width
from mlx_smolvla.preprocessing import SmolVLAPreprocessor
from mlx_smolvla.quantization import (
    QuantizationManifest,
    expected_topology_manifest,
    quantize_vlm_linears,
)
from mlx_smolvla.vision import VisionEncoder


_DEFAULT_CHECKPOINT_ID = "lerobot/smolvla_base"
_DEFAULT_CHECKPOINT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
_AUDITED_VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
_AUDITED_VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
)

ExecutionMode = Literal["production", "strict"]
QuantizationPreset = Literal["vlm-8bit", "vlm-4bit"]


def _normalize_execution_mode(value: object) -> ExecutionMode:
    if value == "production":
        return "production"
    if value == "strict":
        return "strict"
    raise ValueError("execution_mode must be 'production' or 'strict'")


def _execution_device(execution_mode: ExecutionMode):
    return mx.gpu if execution_mode == "production" else mx.cpu


def _normalize_quantization(
    value: object,
    *,
    dtype_name: str,
    execution_mode: ExecutionMode,
) -> QuantizationPreset | None:
    if value is None:
        return None
    if value not in {"vlm-8bit", "vlm-4bit"}:
        raise ValueError("quantization must be None, 'vlm-8bit', or 'vlm-4bit'")
    if dtype_name != "bfloat16":
        raise ValueError("VLM quantization requires the validated bfloat16 base dtype")
    if execution_mode != "production":
        raise ValueError("VLM quantization is validated only for production Metal execution")
    return value


def _dtype_name(dtype: object) -> str:
    if dtype == "float32" or dtype == mx.float32:
        return "float32"
    if dtype == "bfloat16" or dtype == mx.bfloat16:
        return "bfloat16"
    raise ValueError("dtype must be mlx.core.float32, mlx.core.bfloat16, 'float32', or 'bfloat16'")


def _resolve_checkpoint(model_id: str | Path, cache_dir: Path) -> Path:
    candidate = Path(model_id).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    identifier = str(model_id)
    revision = _DEFAULT_CHECKPOINT_REVISION if identifier == _DEFAULT_CHECKPOINT_ID else None
    return Path(
        snapshot_download(
            identifier,
            revision=revision,
            cache_dir=str(cache_dir / "hf"),
            allow_patterns=list(_CHECKPOINT_FILES),
        )
    )


def _resolve_tokenizer(
    config: SmolVLAConfig,
    cache_dir: Path,
    tokenizer_dir: str | Path | None,
) -> Path:
    if tokenizer_dir is not None:
        directory = Path(tokenizer_dir).expanduser().resolve()
        missing = [name for name in _TOKENIZER_FILES if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Tokenizer directory {directory} is missing {missing}")
        return directory
    revision = _AUDITED_VLM_REVISION if config.vlm_model_name == _AUDITED_VLM_ID else None
    return Path(
        snapshot_download(
            config.vlm_model_name,
            revision=revision,
            cache_dir=str(cache_dir / "hf"),
            allow_patterns=list(_TOKENIZER_FILES),
        )
    )


def _component_parameter_names(prefix: str, module: nn.Module) -> set[str]:
    return {f"{prefix}{name}" for name, _ in tree_flatten(module.parameters())}


def _load_component(module: nn.Module, weights: Mapping[str, mx.array], prefix: str) -> set[str]:
    selected = [(name.removeprefix(prefix), value) for name, value in weights.items() if name.startswith(prefix)]
    module.load_weights(selected, strict=True)
    return {f"{prefix}{name}" for name, _ in selected}


class SmolVLAMLX:
    """A native MLX inference policy with a queue-backed one-action interface."""

    def __init__(
        self,
        *,
        config: SmolVLAConfig,
        preprocessor: SmolVLAPreprocessor,
        vision: VisionEncoder,
        connector: Connector,
        language: TruncatedLanguageModel,
        state_proj: nn.Linear,
        expert: ActionExpert,
        converted_weights_path: Path,
        loaded_parameter_names: tuple[str, ...],
        execution_mode: ExecutionMode,
        quantization: QuantizationPreset | None = None,
        quantization_manifest: QuantizationManifest | None = None,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor
        self.vision = vision
        self.connector = connector
        self.language = language
        self.state_proj = state_proj
        self.expert = expert
        self._converted_weights_path = converted_weights_path
        self._loaded_parameter_names = loaded_parameter_names
        self._execution_mode = execution_mode
        self._quantization = quantization
        self._quantization_manifest = quantization_manifest
        self._queue: deque[np.ndarray] = deque()
        self._last_prefix_evaluations = 0

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path = _DEFAULT_CHECKPOINT_ID,
        cache_dir: str | Path | None = None,
        dtype: object = mx.bfloat16,
        *,
        tokenizer_dir: str | Path | None = None,
        execution_mode: ExecutionMode = "production",
        quantization: QuantizationPreset | None = None,
    ) -> "SmolVLAMLX":
        """Load a checkpoint for explicit production-Metal or strict-CPU execution.

        ``tokenizer_dir`` is an optional offline injection point. Typical callers
        need only ``model_id``: the audited SmolVLM tokenizer is downloaded into
        the same native cache automatically. ``production`` is the default and
        owns MLX's Metal device context; ``strict`` owns the CPU context used by
        the immutable PyTorch parity ladder.
        """

        normalized_mode = _normalize_execution_mode(execution_mode)
        dtype_name = _dtype_name(dtype)
        normalized_quantization = _normalize_quantization(
            quantization,
            dtype_name=dtype_name,
            execution_mode=normalized_mode,
        )
        with mx.stream(_execution_device(normalized_mode)):
            return cls._from_pretrained(
                model_id=model_id,
                cache_dir=cache_dir,
                dtype=dtype_name,
                tokenizer_dir=tokenizer_dir,
                execution_mode=normalized_mode,
                quantization=normalized_quantization,
            )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str | Path,
        cache_dir: str | Path | None,
        dtype: object,
        tokenizer_dir: str | Path | None,
        execution_mode: ExecutionMode,
        quantization: QuantizationPreset | None,
    ) -> "SmolVLAMLX":
        resolved_cache = resolve_cache_dir(cache_dir)
        resolved_cache.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = _resolve_checkpoint(model_id, resolved_cache)
        config = SmolVLAConfig.from_pretrained_files(checkpoint_dir)
        tokenizer_path = _resolve_tokenizer(config, resolved_cache, tokenizer_dir)
        dtype_name = _dtype_name(dtype)

        identity = hashlib.sha256(str(checkpoint_dir).encode("utf-8")).hexdigest()[:16]
        conversion_dir = resolved_cache / "converted" / identity / dtype_name
        output_path = conversion_dir / f"model.{dtype_name}.safetensors"
        name_map_path = conversion_dir / "name_map.json"
        if not output_path.is_file() or not name_map_path.is_file():
            report = convert_checkpoint(checkpoint_dir, conversion_dir, dtype=dtype_name)
            output_path = report.output_path

        weights = mx.load(str(output_path))
        vision = VisionEncoder()
        connector = Connector()
        language = TruncatedLanguageModel()
        state_proj = nn.Linear(config.max_state_dim, 960, bias=True)
        expert = ActionExpert()

        loaded_names: set[str] = set()
        loaded_names.update(_load_component(vision, weights, "vision."))
        loaded_names.update(_load_component(connector, weights, "connector."))
        loaded_names.update(_load_component(language, weights, "language."))
        loaded_names.update(_load_component(state_proj, weights, "state_proj."))
        expert_weights = [
            (name.removeprefix("expert."), value)
            for name, value in weights.items()
            if name.startswith("expert.")
        ] + [(name, value) for name, value in weights.items() if name.startswith("action_")]
        expert.load_weights(expert_weights, strict=True)
        loaded_names.update(name for name, _ in expert_weights if name.startswith("action_"))
        loaded_names.update(f"expert.{name}" for name, _ in expert_weights if not name.startswith("action_"))

        if loaded_names != set(weights):
            missing = sorted(set(weights) - loaded_names)
            unexpected = sorted(loaded_names - set(weights))
            raise RuntimeError(f"native loader did not consume the converted tensor set; missing={missing}, unexpected={unexpected}")

        preprocessor = SmolVLAPreprocessor.from_pretrained_files(checkpoint_dir, tokenizer_path)
        policy = cls(
            config=config,
            preprocessor=preprocessor,
            vision=vision,
            connector=connector,
            language=language,
            state_proj=state_proj,
            expert=expert,
            converted_weights_path=output_path,
            loaded_parameter_names=tuple(sorted(loaded_names)),
            execution_mode=execution_mode,
            quantization=None,
            quantization_manifest=None,
        )
        if set(policy._runtime_parameter_names()) != loaded_names:
            missing = sorted(loaded_names - set(policy._runtime_parameter_names()))
            unexpected = sorted(set(policy._runtime_parameter_names()) - loaded_names)
            raise RuntimeError(f"native parameter tree disagrees with converted tensors; missing={missing}, unexpected={unexpected}")
        if quantization is not None:
            bits = 8 if quantization == "vlm-8bit" else 4
            manifest = quantize_vlm_linears(policy, bits=bits)
            if manifest.as_dict() != expected_topology_manifest(quantization):
                raise RuntimeError(
                    "runtime quantization topology differs from the audited Stage Q manifest"
                )
            policy._quantization = quantization
            policy._quantization_manifest = manifest
        return policy

    @property
    def converted_weights_path(self) -> Path:
        """The cached safetensors artifact currently backing this policy."""

        return self._converted_weights_path

    @property
    def loaded_parameter_names(self) -> tuple[str, ...]:
        """The exact canonical tensor names consumed by the strict native loader."""

        return self._loaded_parameter_names

    @property
    def execution_mode(self) -> ExecutionMode:
        """The policy-owned execution contract: production Metal or strict CPU."""

        return self._execution_mode

    @property
    def quantization(self) -> QuantizationPreset | None:
        """The explicitly selected VLM-only preset, or ``None`` for dense bf16."""

        return self._quantization

    @property
    def quantization_manifest(self) -> QuantizationManifest | None:
        """The audited in-memory module topology for the selected preset."""

        return self._quantization_manifest

    @property
    def execution_device(self):
        """The MLX device selected for every public inference call."""

        return _execution_device(self._execution_mode)

    @property
    def queued_actions(self) -> int:
        """The number of postprocessed robot actions awaiting execution."""

        return len(self._queue)

    @property
    def last_prefix_evaluations(self) -> int:
        """Prefix-cache evaluations performed by the most recent chunk prediction."""

        return self._last_prefix_evaluations

    def _runtime_parameter_names(self) -> tuple[str, ...]:
        names = set()
        names.update(_component_parameter_names("vision.", self.vision))
        names.update(_component_parameter_names("connector.", self.connector))
        names.update(_component_parameter_names("language.", self.language))
        names.update(_component_parameter_names("state_proj.", self.state_proj))
        for name, _ in tree_flatten(self.expert.parameters()):
            names.add(name if name.startswith("action_") else f"expert.{name}")
        return tuple(sorted(names))

    def _prepare_prefix_cache(self, observation: Mapping[str, object]):
        processed = self.preprocessor(observation)
        vision_features = self.vision(processed.pixel_values, processed.pixel_attention_mask)
        image_tokens = self.connector(vision_features)
        padded_state = pad_state_to_width(processed.state, width=self.config.max_state_dim)
        state_embedding = self.state_proj(padded_state)[:, None, :]
        prefix = self.language.build_prefix(processed, image_tokens, state_embedding)
        cache = self.language.encode_prefix(prefix)
        self._last_prefix_evaluations += 1
        return cache

    def predict_action_chunk(
        self,
        observation: Mapping[str, object],
        noise: mx.array | np.ndarray | None = None,
    ) -> mx.array:
        """Return one normalized action chunk on this policy's owned device."""

        with mx.stream(self.execution_device):
            return self._predict_action_chunk(observation, noise=noise)

    def _predict_action_chunk(
        self,
        observation: Mapping[str, object],
        noise: mx.array | np.ndarray | None = None,
    ) -> mx.array:
        """Execute a chunk inside the already-selected policy device context."""

        self._last_prefix_evaluations = 0
        cache = self._prepare_prefix_cache(observation)
        if noise is None:
            noisy_actions = mx.random.normal(
                (1, self.config.chunk_size, self.config.max_action_dim)
            ).astype(mx.float32)
        else:
            noisy_actions = mx.array(noise).astype(mx.float32)
        expected_shape = (1, self.config.chunk_size, self.config.max_action_dim)
        if noisy_actions.shape != expected_shape:
            raise ValueError(f"noise must have shape {expected_shape}, got {noisy_actions.shape}")

        padded_actions = euler_sample(
            lambda x_t, timestep: self.expert.denoise(cache, x_t, timestep).velocity,
            noisy_actions,
            num_steps=self.config.num_steps,
        )
        return padded_actions[:, :, : self.config.action_dim]

    def select_action(
        self,
        observation: Mapping[str, object],
        noise: mx.array | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one postprocessed action and refill the FIFO only when it is empty."""

        with mx.stream(self.execution_device):
            if not self._queue:
                normalized = self._predict_action_chunk(observation, noise=noise)
                actions = self.preprocessor.unnormalize_actions(normalized)
                action_array = np.asarray(actions.astype(mx.float32))
                expected_shape = (1, self.config.chunk_size, self.config.action_dim)
                if action_array.shape != expected_shape:
                    raise ValueError(f"action chunk must have shape {expected_shape}, got {action_array.shape}")
                self._queue.extend(action.copy() for action in action_array[0, : self.config.n_action_steps])
            return self._queue.popleft()

    def reset(self) -> None:
        """Clear all queued action state before beginning a new episode."""

        self._queue.clear()
