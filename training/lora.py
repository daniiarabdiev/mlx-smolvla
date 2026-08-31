"""Torch-free LoRA insertion and merge support for native SmolVLA training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_ACTION_PROJECTIONS = (
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)
_EXPECTED_LANGUAGE_LAYERS = 16
_EXPECTED_EXPERT_LAYERS = 16
_EXPECTED_ADAPTERS = 229


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for the Stage T3 SmolVLA adapter topology."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be finite and positive, got {self.alpha}")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")


class LoRALinear(nn.Module):
    """A frozen MLX linear plus an fp32 low-rank additive branch."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        config = LoRAConfig(rank=rank, alpha=alpha, dropout=dropout)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear base must be nn.Linear, got {type(base).__name__}")
        output_width, input_width = base.weight.shape
        self.base = base
        self.rank = config.rank
        self.alpha = config.alpha
        self.scale = config.alpha / config.rank
        self.dropout_probability = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        bound = 1.0 / math.sqrt(input_width)
        self.lora_a = mx.random.uniform(
            low=-bound,
            high=bound,
            shape=(input_width, config.rank),
        ).astype(mx.float32)
        self.lora_b = mx.zeros((config.rank, output_width), dtype=mx.float32)
        self.base.freeze()

    def __call__(self, inputs: mx.array) -> mx.array:
        adapter_inputs = self.dropout(inputs.astype(mx.float32))
        update = mx.matmul(mx.matmul(adapter_inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scale

    def merged_linear(self, *, dtype: object = mx.float32) -> nn.Linear:
        """Return a plain linear containing the base plus fp32 adapter delta."""

        output_width, input_width = self.base.weight.shape
        bias = getattr(self.base, "bias", None)
        # Randomly initialized MLX modules are lazy. Materialize the existing
        # values before constructing another Linear, whose initialization would
        # otherwise advance the same random stream first.
        mx.eval(self.base.parameters(), self.lora_a, self.lora_b)
        merged = nn.Linear(input_width, output_width, bias=bias is not None)
        delta = mx.matmul(self.lora_b.T, self.lora_a.T) * self.scale
        merged.weight = (self.base.weight.astype(mx.float32) + delta).astype(dtype)
        if bias is not None:
            merged.bias = bias.astype(dtype)
        merged.freeze()
        return merged


@dataclass(frozen=True)
class LoRAInstallationReport:
    """Exact topology and trainable-set identity after adapter insertion."""

    adapter_count: int
    target_names: tuple[str, ...]
    trainable_names: tuple[str, ...]
    trainable_tensor_count: int
    trainable_scalar_count: int
    rank: int
    alpha: float
    dropout: float


@dataclass(frozen=True)
class LoRAMergeReport:
    """Exact topology replaced by one merge operation."""

    adapter_count: int
    target_names: tuple[str, ...]
    dtype: str


def _target_slots(model: nn.Module) -> Iterator[tuple[str, nn.Module, str]]:
    """Yield the frozen, audited SmolVLA LoRA target slots in stable order."""

    language_layers = getattr(getattr(model, "language", None), "layers", None)
    expert = getattr(model, "expert", None)
    expert_layers = getattr(expert, "layers", None)
    if language_layers is None or len(language_layers) != _EXPECTED_LANGUAGE_LAYERS:
        raise ValueError("LoRA insertion requires exactly 16 used VLM layers")
    if expert_layers is None or len(expert_layers) != _EXPECTED_EXPERT_LAYERS:
        raise ValueError("LoRA insertion requires exactly 16 action-expert layers")

    for layer_index, layer in enumerate(language_layers):
        for projection in _ATTENTION_PROJECTIONS:
            yield (
                f"language.layers.{layer_index}.self_attn.{projection}",
                layer.self_attn,
                projection,
            )
        for projection in _MLP_PROJECTIONS:
            yield (
                f"language.layers.{layer_index}.mlp.{projection}",
                layer.mlp,
                projection,
            )
    for layer_index, layer in enumerate(expert_layers):
        for projection in _ATTENTION_PROJECTIONS:
            yield (
                f"expert.layers.{layer_index}.self_attn.{projection}",
                layer.self_attn,
                projection,
            )
        for projection in _MLP_PROJECTIONS:
            yield (
                f"expert.layers.{layer_index}.mlp.{projection}",
                layer.mlp,
                projection,
            )
    for projection in _ACTION_PROJECTIONS:
        yield f"expert.{projection}", expert, projection
    yield "state_proj", model, "state_proj"


def iter_lora(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    """Yield installed adapters at only the committed SmolVLA target slots."""

    for name, parent, attribute in _target_slots(model):
        value = getattr(parent, attribute)
        if isinstance(value, LoRALinear):
            yield name, value


def install_lora(
    model: nn.Module,
    config: LoRAConfig | None = None,
) -> LoRAInstallationReport:
    """Freeze a SmolVLA model and install exactly 229 fp32 adapters."""

    config = LoRAConfig() if config is None else config
    slots = tuple(_target_slots(model))
    if len(slots) != _EXPECTED_ADAPTERS:
        raise RuntimeError(
            f"SmolVLA LoRA target count changed: {len(slots)} != {_EXPECTED_ADAPTERS}"
        )
    if any(isinstance(getattr(parent, attribute), LoRALinear) for _, parent, attribute in slots):
        raise ValueError("LoRA is already installed on at least one target")
    invalid = [
        name
        for name, parent, attribute in slots
        if not isinstance(getattr(parent, attribute), nn.Linear)
    ]
    if invalid:
        raise TypeError(f"LoRA targets are not plain MLX linears: {invalid}")

    model.freeze()
    for _, parent, attribute in slots:
        base = getattr(parent, attribute)
        setattr(
            parent,
            attribute,
            LoRALinear(
                base,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )
    # Freeze again after the structural mutation, then expose only adapter
    # matrices. The frozen base still propagates derivatives to its inputs.
    model.freeze()
    for _, adapter in iter_lora(model):
        adapter.unfreeze(
            recurse=False,
            keys=["lora_a", "lora_b"],
            strict=True,
        )

    target_names = tuple(name for name, _ in iter_lora(model))
    if len(target_names) != _EXPECTED_ADAPTERS:
        raise RuntimeError(
            f"installed {len(target_names)} LoRA adapters, expected {_EXPECTED_ADAPTERS}"
        )
    trainable = tuple(tree_flatten(model.trainable_parameters()))
    trainable_names = tuple(name for name, _ in trainable)
    if len(trainable_names) != 2 * _EXPECTED_ADAPTERS:
        raise RuntimeError(
            f"LoRA exposed {len(trainable_names)} tensors, expected {2 * _EXPECTED_ADAPTERS}"
        )
    if not all(name.endswith((".lora_a", ".lora_b")) for name in trainable_names):
        raise RuntimeError(f"LoRA trainable set contains base parameters: {trainable_names}")
    if not all(value.dtype == mx.float32 for _, value in trainable):
        raise RuntimeError("LoRA master parameters must all remain fp32")
    return LoRAInstallationReport(
        adapter_count=len(target_names),
        target_names=target_names,
        trainable_names=trainable_names,
        trainable_tensor_count=len(trainable),
        trainable_scalar_count=sum(value.size for _, value in trainable),
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
    )


def merge_lora(
    model: nn.Module,
    *,
    dtype: object = mx.float32,
) -> LoRAMergeReport:
    """Merge every installed adapter and restore the plain checkpoint tree."""

    adapters = tuple(iter_lora(model))
    if len(adapters) != _EXPECTED_ADAPTERS:
        raise RuntimeError(
            f"merge requires {_EXPECTED_ADAPTERS} adapters, found {len(adapters)}"
        )
    target_names: list[str] = []
    for name, parent, attribute in _target_slots(model):
        adapter = getattr(parent, attribute)
        if not isinstance(adapter, LoRALinear):
            raise RuntimeError(f"LoRA target {name} is not installed")
        setattr(parent, attribute, adapter.merged_linear(dtype=dtype))
        target_names.append(name)
    model.freeze()
    if tuple(iter_lora(model)):
        raise RuntimeError("LoRA merge left adapter wrappers in the model")
    dtype_name = "float32" if dtype == mx.float32 or dtype == "float32" else str(dtype)
    return LoRAMergeReport(
        adapter_count=len(target_names),
        target_names=tuple(target_names),
        dtype=dtype_name,
    )
