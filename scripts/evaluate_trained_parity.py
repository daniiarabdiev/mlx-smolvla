#!/usr/bin/env python3
"""Evaluate the prospectively frozen trained-checkpoint parity procedure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from smolvla_mlx.training.trained_parity import (  # noqa: E402
    evaluate_trained_parity_files,
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
        "--output",
        type=Path,
        default=Path(".cache/training/t3b/parity-evaluation.json"),
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument("--evaluated-at-ns", type=int)
    args = parser.parse_args(argv)

    report, digest = evaluate_trained_parity_files(
        floor_path=args.floor,
        variant_root=args.variants,
        start_marker_path=args.start_marker,
        comparison_path=args.comparison,
        output_path=args.output,
        evidence_root=args.evidence_root,
        evaluated_at_ns=args.evaluated_at_ns,
    )
    print(
        json.dumps(
            {
                "passed": report["gates"]["passed"],
                "fixed_gates_passed": report["gates"]["fixed_gates_passed"],
                "deterministic_parity_passed": report["gates"][
                    "deterministic_parity_passed"
                ],
                "normalized_action_threshold": report["thresholds"][
                    "normalized_action_max_abs"
                ],
                "output": str(args.output.resolve()),
                "report_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
