#!/usr/bin/env python3
"""Run the fixed T4 100-update versus 50+resume exactness protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

from training.finetune import write_run_state
from training.ux import (
    FullTrainingConfig,
    LoRATrainingConfig,
    evaluate_resume_exactness,
    run_training,
)


STEPS = 100
SPLIT_STEP = 50
CHECKPOINT_INTERVAL = 25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("lora", "full"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    parser.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/smolvla_mlx/policy-float32"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.output_root.resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"refusing to overwrite resume evidence root: {root}")
    direct_dir = root / "uninterrupted"
    resumed_dir = root / "resumed"
    common = {
        "dataset": args.dataset,
        "steps": STEPS,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "dtype": args.dtype,
        "cache_dir": args.cache_dir,
        "native_cache": args.native_cache,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
    }
    config_class = FullTrainingConfig if args.mode == "full" else LoRATrainingConfig
    snapshot_taken = False

    def progress(step: int, total: int, update: object) -> None:
        nonlocal snapshot_taken
        if step == 1 or step % 10 == 0 or step == total:
            print(
                f"{args.mode} direct step={step}/{total} "
                f"loss={getattr(update, 'loss'):.6f}",
                flush=True,
            )
        if step == SPLIT_STEP:
            if snapshot_taken or resumed_dir.exists():
                raise RuntimeError("resume snapshot target already exists")
            shutil.copytree(direct_dir, resumed_dir, copy_function=shutil.copy2)
            snapshot_taken = True

    direct = run_training(config_class(**common, output_dir=direct_dir), progress=progress)
    if not snapshot_taken:
        raise RuntimeError("the step-50 resume snapshot was not captured")

    def resumed_progress(step: int, total: int, update: object) -> None:
        if step % 10 == 0 or step == total:
            print(
                f"{args.mode} resumed step={step}/{total} "
                f"loss={getattr(update, 'loss'):.6f}",
                flush=True,
            )

    resumed = run_training(
        config_class(**common, output_dir=resumed_dir, resume=True),
        progress=resumed_progress,
    )
    report_path = root / "resume_exactness.json"
    report = evaluate_resume_exactness(
        direct_dir,
        resumed_dir,
        output_path=report_path,
    )
    if not report["passed"]:
        raise RuntimeError(f"{args.mode} exact-resume gates failed: {report}")
    if not direct.loss_decreased or not resumed.loss_decreased:
        raise RuntimeError(
            f"{args.mode} smoke loss did not decrease: "
            f"direct={direct.loss_decreased}, resumed={resumed.loss_decreased}"
        )
    evidence = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-t4-training-smoke",
        "classification": "functional-smoke-non-benchmark",
        "mode": args.mode,
        "protocol": {
            "uninterrupted_steps": STEPS,
            "interrupted_steps": SPLIT_STEP,
            "resumed_steps": STEPS - SPLIT_STEP,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "dtype": args.dtype,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "uninterrupted": direct.as_dict(),
        "resumed": resumed.as_dict(),
        "resume_exactness": report,
        "resume_exactness_file": report_path.name,
        "resume_exactness_file_sha256": _sha256(report_path),
    }
    evidence_path = root / "evidence.json"
    digest = write_run_state(evidence_path, evidence)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "passed": True,
                "loss_decreased": True,
                "parameter_max_abs": report["parameter_max_abs"],
                "loss_max_abs": report["loss_max_abs"],
                "optimizer_exact": report["optimizer_exact"],
                "draw_chain_exact": report["draw_chain_exact"],
                "evidence": str(evidence_path),
                "evidence_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
