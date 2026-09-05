#!/usr/bin/env python3
"""Freeze Stage T3 held-out cases and base MLX MAE before fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mlx_smolvla._lab.training.evaluation import capture_and_evaluate_base  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    parser.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/mlx_smolvla/policy-float32"),
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path(".cache/training/t3-evaluation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/training/t3-base-evaluation.json"),
    )
    args = parser.parse_args()

    def progress(completed: int, total: int) -> None:
        if completed == 1 or completed % 8 == 0 or completed == total:
            print(f"base evaluation {completed}/{total}", flush=True)

    result, digest = capture_and_evaluate_base(
        cache_dir=args.cache_dir,
        native_cache=args.native_cache,
        evaluation_dir=args.evaluation_dir,
        output_path=args.output,
        progress=progress,
    )
    print(
        json.dumps(
            {
                "sample_count": result.sample_count,
                "element_count": result.element_count,
                "mlx_mae": result.mae,
                "report_sha256": digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
