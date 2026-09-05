#!/usr/bin/env python3
"""Capture the pinned Torch CPU/fp32 SmolVLA training gradient golden."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from mlx_smolvla._lab.training.reference import capture_reference_gradient_golden


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/training/gradient_goldens"),
    )
    args = parser.parse_args(argv)
    result = capture_reference_gradient_golden(args.cache_dir, args.output)
    summary_keys = (
        "artifact_type",
        "tensor_count",
        "trainable_tensor_count",
        "trainable_scalar_count",
        "reference_loss",
        "sampled_timestep",
        "manifest_sha256",
        "artifact_bytes",
        "capture_seconds",
        "disk_free_before_bytes",
        "disk_free_after_bytes",
    )
    summary = {"output": str(args.output.resolve())}
    summary.update({key: result[key] for key in summary_keys})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
