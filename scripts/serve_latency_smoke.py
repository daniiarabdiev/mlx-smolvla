#!/usr/bin/env python3
"""Run the trusted MLX server with no-clobber observation-to-chunk telemetry."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from smolvla_mlx.server import ServeConfig, serve_forever


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new JSONL session log")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--execution-mode",
        choices=("production", "strict"),
        default="production",
    )
    parser.add_argument("--quantization", choices=("vlm-8bit", "vlm-4bit"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--obs-queue-timeout", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-remote", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve_forever(
        ServeConfig(
            host=args.host,
            port=args.port,
            cache_dir=args.cache_dir,
            tokenizer_dir=args.tokenizer_dir,
            dtype=args.dtype,
            execution_mode=args.execution_mode,
            quantization=args.quantization,
            latency_log=args.output,
            fps=args.fps,
            obs_queue_timeout=args.obs_queue_timeout,
            max_workers=args.max_workers,
            seed=args.seed,
            allow_remote=args.allow_remote,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
