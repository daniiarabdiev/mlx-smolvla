#!/usr/bin/env python3
"""Run and persist the immutable SmolVLA 25-step optimizer-lockstep gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from training.lockstep import run_optimizer_lockstep, write_optimizer_lockstep_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t1-goldens",
        type=Path,
        default=Path(".cache/training/gradient_goldens"),
    )
    parser.add_argument(
        "--optimizer-goldens",
        type=Path,
        default=Path(".cache/training/optimizer_goldens"),
    )
    parser.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/mlx_smolvla/policy-float32"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/training/t2-lockstep.json"),
    )
    args = parser.parse_args(argv)
    result = run_optimizer_lockstep(
        t1_dir=args.t1_goldens,
        optimizer_golden_dir=args.optimizer_goldens,
        native_cache=args.native_cache,
    )
    report_sha256 = write_optimizer_lockstep_report(result, args.output)
    summary = {
        "maximum_loss_relative_difference": result.maximum_loss_relative_difference,
        "maximum_parameter_relative_l2": result.maximum_parameter_relative_l2,
        "optimizer_manifest_sha256": result.optimizer_manifest_sha256,
        "output": str(args.output.resolve()),
        "parameter_match_count": result.parameter_match_count,
        "passed": result.passed,
        "report_sha256": report_sha256,
        "step_count": result.step_count,
        "t1_manifest_sha256": result.t1_manifest_sha256,
        "worst_loss_steps": [asdict(item) for item in result.worst_loss_steps],
        "worst_parameters": [asdict(item) for item in result.worst_parameters],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
