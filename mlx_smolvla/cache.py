"""Cache-path resolution shared by conversion and model loading."""

from __future__ import annotations

import os
from pathlib import Path
import warnings


CURRENT_CACHE_ENV = "MLX_SMOLVLA_CACHE"
LEGACY_CACHE_ENV = "SMOLVLA_MLX_CACHE"


def resolve_cache_dir(explicit: Path | str | None = None) -> Path:
    """Return the explicit, environment, or default SmolVLA MLX cache path."""

    current = os.environ.get(CURRENT_CACHE_ENV)
    legacy = os.environ.get(LEGACY_CACHE_ENV)
    if legacy is not None:
        disposition = "ignored because MLX_SMOLVLA_CACHE is set" if current else "used"
        warnings.warn(
            f"{LEGACY_CACHE_ENV} is deprecated and was {disposition}; use "
            f"{CURRENT_CACHE_ENV} instead. Legacy support will be removed after one release.",
            FutureWarning,
            stacklevel=2,
        )
    candidate = explicit if explicit is not None else current or legacy
    if candidate is not None:
        return Path(candidate).expanduser().resolve()
    return Path.home() / ".cache" / "mlx_smolvla"
