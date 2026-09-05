#!/usr/bin/env python3
"""Run and persist the three immutable Stage T3 LoRA outcome gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mlx_smolvla._lab.training.evaluation import run_finetune_outcome_evaluation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    parser.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/mlx_smolvla/policy-float32"),
    )
    parser.add_argument("--run-dir", type=Path, default=Path(".cache/training/t3"))
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/training/t3-outcome.json"),
    )
    args = parser.parse_args(argv)

    def progress(framework: str, completed: int, total: int) -> None:
        if completed == 1 or completed % 8 == 0 or completed == total:
            print(f"{framework} evaluation {completed}/{total}", flush=True)

    report, report_sha256 = run_finetune_outcome_evaluation(
        cache_dir=args.cache_dir,
        native_cache=args.native_cache,
        run_dir=args.run_dir,
        evaluation_dir=args.evaluation_dir,
        base_report_path=args.base_report,
        output_path=args.output,
        progress=progress,
    )
    gates = report["gates"]
    print(
        json.dumps(
            {
                "passed": gates["passed"],
                "fine_to_base_ratio": gates["fine_to_base_ratio"],
                "torch_to_mlx_ratio": gates["torch_to_mlx_ratio"],
                "parity_max_abs": gates["parity_max_abs"],
                "output": str(args.output.resolve()),
                "report_sha256": report_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if gates["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
