"""Fixed Stage T5 native MLX training benchmark matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import math
from pathlib import Path
import statistics
from typing import Callable, Mapping

import mlx.core as mx
import numpy as np

from mlx_smolvla._lab.reference.discovery import DATASET_ID
from mlx_smolvla._lab.training.ux import (
    FullTrainingConfig,
    LoRATrainingConfig,
    _DRAW_CHAIN_INITIAL,
    _perform_update,
    _prepare_training,
)


@dataclass(frozen=True)
class TrainingBenchmarkConfig:
    """Immutable four-cell benchmark protocol."""

    warmup_updates: int = 3
    measured_updates: int = 10
    batch_size: int = 8
    training_horizon: int = 3_000
    learning_rate: float = 1e-4
    modes: tuple[str, ...] = ("lora", "full")
    dtypes: tuple[str, ...] = ("bfloat16", "float32")

    def __post_init__(self) -> None:
        if self.warmup_updates != 3 or self.measured_updates != 10:
            raise ValueError("training benchmark must use exactly 3 warmups + 10 measurements")
        if self.batch_size != 8 or self.training_horizon != 3_000:
            raise ValueError("training benchmark batch size/horizon differs from the fixed protocol")
        if self.learning_rate != 1e-4:
            raise ValueError("training benchmark learning rate differs from the fixed protocol")
        if self.modes != ("lora", "full") or self.dtypes != ("bfloat16", "float32"):
            raise ValueError("training benchmark matrix differs from the fixed protocol")


def summarize_update_seconds(values: tuple[float, ...]) -> dict[str, object]:
    """Summarize synchronized effective-batch update durations."""

    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("training update durations must be finite and positive")
    median = statistics.median(values)
    return {
        "update_seconds": list(values),
        "median_update_seconds": median,
        "mean_update_seconds": statistics.fmean(values),
        "p95_update_seconds": float(np.percentile(values, 95)),
        "steps_per_second": 1.0 / median,
        "wall_seconds_per_1000_steps": 1_000.0 * median,
    }


def benchmark_training_cells(
    *,
    dataset: str | Path = DATASET_ID,
    cache_dir: str | Path = Path(".cache/hf"),
    native_cache: str | Path = Path(".cache/mlx_smolvla/policy-float32"),
    config: TrainingBenchmarkConfig | None = None,
    progress: Callable[[str, str, str, Mapping[str, object] | None], None] | None = None,
) -> list[dict[str, object]]:
    """Measure the fixed LoRA/full by bf16/fp32 Metal matrix."""

    protocol = TrainingBenchmarkConfig() if config is None else config
    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"training benchmark requires Metal GPU, got {mx.default_device()}")
    cells: list[dict[str, object]] = []
    for mode in protocol.modes:
        for dtype in protocol.dtypes:
            if progress is not None:
                progress(mode, dtype, "start", None)
            config_class = FullTrainingConfig if mode == "full" else LoRATrainingConfig
            run_config = config_class(
                dataset=dataset,
                steps=protocol.training_horizon,
                batch_size=protocol.batch_size,
                learning_rate=protocol.learning_rate,
                dtype=dtype,
                output_dir=Path(".cache/training/t5-unused") / f"{mode}-{dtype}",
                cache_dir=Path(cache_dir),
                native_cache=Path(native_cache),
                checkpoint_interval=100,
            )
            components = _prepare_training(run_config)
            draw_chain = _DRAW_CHAIN_INITIAL
            measured: list[float] = []
            total = protocol.warmup_updates + protocol.measured_updates
            try:
                for update_index in range(total):
                    update, draw_chain, _ = _perform_update(
                        model=components.model,
                        bridge=components.bridge,
                        optimizer=components.optimizer,
                        batch_size=protocol.batch_size,
                        draw_chain_sha256=draw_chain,
                    )
                    if update_index + 1 == protocol.warmup_updates:
                        mx.reset_peak_memory()
                    if update_index >= protocol.warmup_updates:
                        measured.append(update.seconds)
                summary = summarize_update_seconds(tuple(measured))
                cell = {
                        "mode": mode,
                        "dtype": dtype,
                        "warmup_updates": protocol.warmup_updates,
                        "measured_updates": protocol.measured_updates,
                        "batch_size": protocol.batch_size,
                        "training_horizon": protocol.training_horizon,
                        **summary,
                        "peak_memory_bytes": int(mx.get_peak_memory()),
                        "trainable_tensor_count": components.topology.trainable_tensor_count,
                        "trainable_scalar_count": components.topology.trainable_scalar_count,
                    }
                cells.append(cell)
                if progress is not None:
                    progress(mode, dtype, "complete", cell)
            finally:
                del components
                gc.collect()
                mx.clear_cache()
    return cells


def validate_training_benchmark(value: object) -> dict[str, object]:
    """Recompute the fixed matrix identities and timing summaries."""

    if not isinstance(value, Mapping):
        raise ValueError("training benchmark must be an object")
    document = dict(value)
    if document.get("format_version") != 1 or document.get("artifact_type") != (
        "smolvla-mlx-training-benchmark"
    ):
        raise ValueError("training benchmark identity is invalid")
    idle = document.get("idle")
    if not isinstance(idle, Mapping) or idle.get("verified") is not True or idle.get(
        "matching_processes"
    ) != []:
        raise ValueError("training benchmark lacks a clean idle declaration")
    expected_protocol = asdict(TrainingBenchmarkConfig())
    expected_protocol["modes"] = list(expected_protocol["modes"])
    expected_protocol["dtypes"] = list(expected_protocol["dtypes"])
    protocol = document.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("training benchmark protocol is absent")
    # Older callers need not redundantly record the fixed learning rate, but if
    # present it is still immutable.
    required_protocol = {
        key: expected_protocol[key]
        for key in (
            "warmup_updates",
            "measured_updates",
            "batch_size",
            "training_horizon",
            "modes",
            "dtypes",
        )
    }
    if any(protocol.get(key) != expected for key, expected in required_protocol.items()):
        raise ValueError("training benchmark protocol differs from the fixed values")
    if "learning_rate" in protocol and protocol["learning_rate"] != 1e-4:
        raise ValueError("training benchmark learning rate differs")
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise ValueError("training benchmark cells are absent")
    expected_matrix = {
        (mode, dtype)
        for mode in ("lora", "full")
        for dtype in ("bfloat16", "float32")
    }
    observed_matrix = {
        (cell.get("mode"), cell.get("dtype"))
        for cell in cells
        if isinstance(cell, Mapping)
    }
    if len(cells) != 4 or observed_matrix != expected_matrix:
        raise ValueError("training benchmark matrix is incomplete or duplicated")
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("training benchmark cell must be an object")
        if any(
            cell.get(field) != expected_protocol[field]
            for field in (
                "warmup_updates",
                "measured_updates",
                "batch_size",
                "training_horizon",
            )
        ):
            raise ValueError("training benchmark cell protocol differs")
        durations = cell.get("update_seconds")
        if not isinstance(durations, list) or len(durations) != 10:
            raise ValueError("training benchmark cell timing count differs")
        summary = summarize_update_seconds(tuple(float(item) for item in durations))
        if any(cell.get(name) != expected for name, expected in summary.items()):
            raise ValueError("training benchmark cell summary was not recomputed")
        if type(cell.get("peak_memory_bytes")) is not int or cell["peak_memory_bytes"] <= 0:
            raise ValueError("training benchmark peak memory is invalid")
        for field in ("trainable_tensor_count", "trainable_scalar_count"):
            if type(cell.get(field)) is not int or cell[field] <= 0:
                raise ValueError("training benchmark topology count is invalid")
    return document
