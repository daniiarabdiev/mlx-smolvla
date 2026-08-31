"""Cache-path resolution shared by conversion and model loading."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_cache_dir(explicit: Path | str | None = None) -> Path:
    """Return the explicit, environment, or default SmolVLA MLX cache path."""

    candidate = explicit if explicit is not None else os.environ.get("SMOLVLA_MLX_CACHE")
    if candidate is not None:
        return Path(candidate).expanduser().resolve()
    return Path.home() / ".cache" / "smolvla_mlx"
