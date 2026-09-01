#!/usr/bin/env python3
"""Compute the fixed PyTorch arithmetic self-consistency floor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.floor_runtime import bootstrap_hidden_worker  # noqa: E402


bootstrap_hidden_worker(sys.argv[1:])

from training.self_consistency import (  # noqa: E402
    PROCEDURE_ID,
    _worker_environment,
    assemble_existing_floor,
    perturbation_plan,
    run_reference_variant,
    run_self_consistency_floor,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--list-perturbations",
        action="store_true",
        help="print the prospectively fixed perturbation plan without loading a model",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="aggregate already completed, hash-bound workers without loading a model",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="explicit maximum CPU thread count used by perturbation (c)",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path(".cache/training/t3/export"))
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path(".cache/training/t3-evaluation"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".cache/training/t3/self-consistency"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/training/t3/floor.json"),
    )
    parser.add_argument(
        "--purpose",
        choices=("retrospective_diagnostic", "prospective_gate"),
        default="retrospective_diagnostic",
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--input-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--variant-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected_modes = sum(
        (bool(args.list_perturbations), bool(args.assemble_only), args.worker is not None)
    )
    if selected_modes > 1:
        raise SystemExit(
            "--list-perturbations, --assemble-only, and --worker are mutually exclusive"
        )
    plan = perturbation_plan(max_threads=args.max_threads)
    if args.list_perturbations:
        print(
            json.dumps(
                {
                    "procedure_id": PROCEDURE_ID,
                    "perturbations": [item.as_dict() for item in plan],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.worker is not None:
        by_name = {item.name: item for item in plan}
        if args.worker not in by_name:
            raise SystemExit(f"unknown self-consistency worker: {args.worker}")
        if args.input_sha256 is None or args.variant_output is None:
            raise SystemExit("worker requires --input-sha256 and --variant-output")
        variant = by_name[args.worker]
        fixed_environment = _worker_environment(variant, os.environ)
        os.environ.clear()
        os.environ.update(fixed_environment)
        digest = run_reference_variant(
            variant=variant,
            checkpoint_dir=args.checkpoint,
            evaluation_dir=args.evaluation_dir,
            cache_dir=args.cache_dir,
            work_dir=args.work_dir,
            input_combined_sha256=args.input_sha256,
            output_dir=args.variant_output,
        )
        print(
            json.dumps(
                {
                    "variant": variant.name,
                    "variant_artifact_sha256": digest,
                    "output": str(args.variant_output.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.assemble_only:
        input_manifest = args.input_manifest or args.work_dir / "input_sha256.json"
        report, digest = assemble_existing_floor(
            checkpoint_dir=args.checkpoint,
            evaluation_dir=args.evaluation_dir,
            cache_dir=args.cache_dir,
            work_dir=args.work_dir,
            input_manifest_path=input_manifest,
            output_path=args.output,
            purpose=args.purpose,
            max_threads=args.max_threads,
        )
        print(
            json.dumps(
                {
                    "F": report["F"],
                    "F64": report["F64"],
                    "mode": "assemble_only",
                    "output": str(args.output.resolve()),
                    "report_sha256": digest,
                    "workers_started": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report, digest = run_self_consistency_floor(
        checkpoint_dir=args.checkpoint,
        evaluation_dir=args.evaluation_dir,
        cache_dir=args.cache_dir,
        work_dir=args.work_dir,
        output_path=args.output,
        purpose=args.purpose,
        max_threads=args.max_threads,
    )
    print(
        json.dumps(
            {
                "F": report["F"],
                "F64": report["F64"],
                "created_at_ns": report["created_at_ns"],
                "file_mtime_ns": args.output.stat().st_mtime_ns,
                "output": str(args.output.resolve()),
                "report_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
