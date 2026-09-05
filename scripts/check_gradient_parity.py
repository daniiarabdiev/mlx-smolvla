#!/usr/bin/env python3
"""Run and persist the immutable SmolVLA step-zero gradient-parity gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from mlx_smolvla._lab.training.parity import run_gradient_parity, write_gradient_parity_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goldens",
        type=Path,
        default=Path(".cache/training/gradient_goldens"),
    )
    parser.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/mlx_smolvla/policy-float32"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/training/t1-parity.json"),
    )
    args = parser.parse_args(argv)

    result = run_gradient_parity(
        golden_dir=args.goldens,
        native_cache=args.native_cache,
    )
    report_sha256 = write_gradient_parity_report(result, args.output)
    summary = {
        "artifact_manifest_sha256": result.artifact_manifest_sha256,
        "gradient_count": result.gradient_count,
        "loss_relative_difference": result.loss_relative_difference,
        "maximum_gradient_relative_l2": result.maximum_gradient_relative_l2,
        "minimum_gradient_cosine": result.minimum_gradient_cosine,
        "mlx_loss": result.mlx_loss,
        "output": str(args.output.resolve()),
        "parameter_match_count": result.parameter_match_count,
        "passed": result.passed,
        "reference_loss": result.reference_loss,
        "report_sha256": report_sha256,
        "worst_cosine": [asdict(item) for item in result.worst_cosine],
        "worst_relative_l2": [asdict(item) for item in result.worst_relative_l2],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
