"""Shared accessors for the ignored, locally generated reference goldens."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mlx.core as mx
import numpy as np
import pytest

from reference.goldens import GoldenStore


@dataclass(frozen=True)
class GoldenCase:
    """Typed access to all saved tensors and the original observation for one sample."""

    root: Path
    sample_name: str
    metadata: Mapping[str, object]

    def _store(self) -> GoldenStore:
        return GoldenStore(self.root)

    def array(self, name: str) -> np.ndarray:
        return self._store().load(f"{self.sample_name}/{name}")

    def mx(self, name: str, dtype: mx.Dtype = mx.float32) -> mx.array:
        return mx.array(self.array(name)).astype(dtype)

    def observation(self) -> dict[str, np.ndarray | str]:
        task = self.metadata["task"]
        if not isinstance(task, str):
            raise TypeError(f"Golden task must be a string, got {type(task).__name__}")
        return {
            "observation.images.camera1": self.array("raw/camera1"),
            "observation.images.camera2": self.array("raw/camera2"),
            "observation.state": self.array("raw/state"),
            "task": task,
        }


@pytest.fixture(scope="session")
def golden_root() -> Path:
    root = Path(__file__).parent / "golden"
    if not (root / "manifest.json").is_file() or not (root / "metadata.json").is_file():
        raise FileNotFoundError(f"Reference goldens are absent at {root}; run `make goldens`")
    return root


@pytest.fixture(scope="session")
def golden_metadata(golden_root: Path) -> dict[str, object]:
    return json.loads((golden_root / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture
def golden(request: pytest.FixtureRequest, golden_root: Path, golden_metadata: dict[str, object]) -> GoldenCase:
    sample_index = getattr(request, "param", 0)
    samples = golden_metadata["samples"]
    if not isinstance(samples, list):
        raise TypeError("Golden metadata 'samples' must be a list")
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"Golden sample index {sample_index} is outside [0, {len(samples) - 1}]")
    metadata = samples[sample_index]
    if not isinstance(metadata, dict) or not isinstance(metadata.get("name"), str):
        raise TypeError("Golden sample metadata must contain a string name")
    return GoldenCase(root=golden_root, sample_name=metadata["name"], metadata=metadata)
