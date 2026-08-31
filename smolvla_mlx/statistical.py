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
        return cls(sample_count=sample_count, **numeric)
