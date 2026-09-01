#!/usr/bin/env python3
"""Benchmark or execute the fixed Stage T3 native MLX LoRA run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.finetune import (  # noqa: E402
    FineTuneConfig,
    benchmark_lora_updates,
    run_lora_finetune,
    write_run_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    parser.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/smolvla_mlx/policy-float32"),
    )
    parser.add_argument("--output", type=Path, default=Path(".cache/training/t3"))
    parser.add_argument("--nominal-steps", type=int, default=3_000)
    parser.add_argument("--effective-batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing incomplete output from its latest validated checkpoint",
    )
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path(".cache/training/t3-benchmark.json"),
    )
    args = parser.parse_args()
    config = FineTuneConfig(
        cache_dir=args.cache_dir,
        native_cache=args.native_cache,
        output_dir=args.output,
        nominal_steps=args.nominal_steps,
        effective_batch_size=args.effective_batch_size,
        rank=args.rank,
        alpha=args.alpha,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    if args.benchmark_only:
        result = benchmark_lora_updates(config)
        digest = write_run_state(args.benchmark_output, result.as_dict())
        print(json.dumps({**result.as_dict(), "report_sha256": digest}, indent=2))
        return

    def progress(step: int, total: int, update) -> None:
        if step == 1 or step % 25 == 0 or step == total:
            print(
                f"step={step}/{total} loss={update.loss:.6f} "
                f"lr={update.learning_rate:.8g} seconds={update.seconds:.3f}",
                flush=True,
            )

    result = run_lora_finetune(config, progress=progress)
    print(
        json.dumps(
            {
                "selected_steps": result.selected_steps,
                "training_seconds": result.training_seconds,
                "final_loss": result.final_loss,
                "final_smoothed_loss": result.final_smoothed_loss,
                "peak_memory_bytes": result.peak_memory_bytes,
                "adapter_sha256": result.adapter_sha256,
                "export_dir": str(result.export_dir),
                "run_state_sha256": result.run_state_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
