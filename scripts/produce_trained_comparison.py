#!/usr/bin/env python3
"""Produce the floor-bound T3B MLX/PyTorch comparison after its start marker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mlx_smolvla._lab.training.evaluation import run_trained_comparison_evaluation  # noqa: E402


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
        default=Path(".cache/training/t3b/self-consistency/variants"),
    )
    parser.add_argument(
        "--start-marker",
        type=Path,
        default=Path(".cache/training/t3b/comparison-start.json"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path(".cache/training/t3b/comparison.json"),
    )
    parser.add_argument(
        "--outcome",
        type=Path,
        default=Path(".cache/training/t3b/outcome.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    parser.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/mlx_smolvla/policy-float32"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(".cache/training/t3b"),
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path(".cache/training/t3-evaluation"),
    )
    parser.add_argument(
        "--base-report",
        type=Path,
        default=Path(".cache/training/t3-base-evaluation.json"),
    )
    args = parser.parse_args(argv)

    def progress(framework: str, completed: int, total: int) -> None:
        if completed == 1 or completed % 8 == 0 or completed == total:
            print(f"{framework} evaluation {completed}/{total}", flush=True)

    comparison, comparison_sha256, outcome, outcome_sha256 = (
        run_trained_comparison_evaluation(
            floor_path=args.floor,
            variant_root=args.variants,
            start_marker_path=args.start_marker,
            comparison_path=args.comparison,
            outcome_path=args.outcome,
            cache_dir=args.cache_dir,
            native_cache=args.native_cache,
            run_dir=args.run_dir,
            evaluation_dir=args.evaluation_dir,
            base_report_path=args.base_report,
            progress=progress,
        )
    )
    print(
        json.dumps(
            {
                "comparison": str(args.comparison.resolve()),
                "comparison_sha256": comparison_sha256,
                "created_at_ns": comparison["created_at_ns"],
                "fixed_gates_passed": all(
                    (
                        outcome["gates"]["improvement_passed"],
                        outcome["gates"]["roundtrip_passed"],
                        outcome["stats_active_parity"]["image_preprocessing_max_abs"]
                        <= 1e-5,
                        outcome["stats_active_parity"]["state_preprocessing_max_abs"]
                        <= 1e-6,
                    )
                ),
                "outcome": str(args.outcome.resolve()),
                "outcome_sha256": outcome_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
