#!/usr/bin/env python3
"""Create the one-shot real-clock marker immediately before MLX inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mlx_smolvla.training.trained_parity import (  # noqa: E402
    create_comparison_start_marker,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--floor",
        type=Path,
        default=Path(".cache/training/t3b/floor.json"),
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=Path(".cache/training/t3b/floor-variants"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path(".cache/training/t3b/comparison.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/training/t3b/comparison-start.json"),
    )
    args = parser.parse_args(argv)
    marker, digest = create_comparison_start_marker(
        floor_path=args.floor,
        variant_root=args.variants,
        comparison_path=args.comparison,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "created_at_ns": marker["created_at_ns"],
                "floor_sha256": marker["floor_sha256"],
                "floor_bundle_sha256": marker["floor_bundle_sha256"],
                "comparison_path": marker["comparison_path"],
                "output": str(args.output.resolve()),
                "marker_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
