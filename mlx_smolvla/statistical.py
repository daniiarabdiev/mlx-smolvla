"""Read-only parser for auditable native-vs-reference accuracy evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StatisticalResult:
    """Aggregate MAEs emitted by the reference-lane statistical checker."""

    sample_count: int
    torch_fp32_mae: float
    mlx_fp32_mae: float
    mlx_bf16_mae: float
    mlx_fp32_ratio: float
    mlx_bf16_ratio: float
    execution_mode: str | None = None
    device: str | None = None

    @classmethod
    def from_json(cls, path: Path) -> "StatisticalResult":
        """Load and validate a saved statistical evidence record."""

        if not path.is_file():
            raise FileNotFoundError(f"Statistical evidence is absent at {path}; run scripts/statistical_check.py")
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Statistical evidence must be a JSON object")
        required = (
            "sample_count",
            "torch_fp32_mae",
            "mlx_fp32_mae",
            "mlx_bf16_mae",
            "mlx_fp32_ratio",
            "mlx_bf16_ratio",
        )
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f"Statistical evidence is missing {missing}")
        sample_count = raw["sample_count"]
        if not isinstance(sample_count, int) or sample_count < 0:
            raise ValueError(f"sample_count must be a non-negative integer, got {sample_count!r}")
        numeric = {name: float(raw[name]) for name in required[1:]}
        if not all(math.isfinite(value) and value >= 0.0 for value in numeric.values()):
            raise ValueError("Statistical MAEs and ratios must be finite and non-negative")
        if numeric["torch_fp32_mae"] == 0.0:
            raise ValueError("torch_fp32_mae must be positive so ratios are defined")
        for mae_name, ratio_name in (
            ("mlx_fp32_mae", "mlx_fp32_ratio"),
            ("mlx_bf16_mae", "mlx_bf16_ratio"),
        ):
            computed = numeric[mae_name] / numeric["torch_fp32_mae"]
            if not math.isclose(computed, numeric[ratio_name], rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(f"{ratio_name} does not match the recorded MAEs")
        execution_mode = raw.get("execution_mode")
        device = raw.get("device")
        if (execution_mode is None) != (device is None):
            raise ValueError("Statistical execution_mode and device must appear together")
        if execution_mode is not None:
            if execution_mode not in {"production", "strict"}:
                raise ValueError(f"Unknown statistical execution mode {execution_mode!r}")
            if not isinstance(device, str) or not device:
                raise ValueError("Statistical device must be a non-empty string")
        return cls(
            sample_count=sample_count,
            execution_mode=execution_mode,
            device=device,
            **numeric,
        )
