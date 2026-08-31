#!/usr/bin/env python
"""Benchmark warmed native SmolVLA MLX inference on a saved real observation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import mlx.core as mx
import numpy as np

from smolvla_mlx.benchmark import BenchmarkResult, run_benchmark
from smolvla_mlx.policy import SmolVLAMLX


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/smolvla_mlx"))
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--sample-root", type=Path, default=Path("tests/golden/sample_000"))
    parser.add_argument("--metadata", type=Path, default=Path("tests/golden/metadata.json"))
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "both"), default="both")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("BENCHMARK.md"))
    return parser.parse_args()


def _observation(sample_root: Path, metadata_path: Path) -> dict[str, object]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Golden metadata is absent at {metadata_path}; run make goldens first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples or not isinstance(samples[0], dict):
        raise ValueError("Golden metadata has no first sample")
    task = samples[0].get("task")
    if not isinstance(task, str):
        raise ValueError("Golden metadata first sample has no task string")
    return {
        "observation.images.camera1": np.load(sample_root / "raw/camera1.npy"),
        "observation.images.camera2": np.load(sample_root / "raw/camera2.npy"),
        "observation.state": np.load(sample_root / "raw/state.npy"),
        "task": task,
    }


def _system_value(*command: str) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _render(results: list[BenchmarkResult]) -> str:
    rows = []
    for result in results:
        stages = result.stages
        rows.append(
            "| {dtype} | {total_median:.2f} | {total_p95:.2f} | {pre:.2f} | {vision:.2f} | {prefix:.2f} | {expert:.2f} | {peak:.2f} |".format(
                dtype=result.dtype,
                total_median=result.total_ms.median,
                total_p95=result.total_ms.p95,
                pre=stages["preprocessing"].median,
                vision=stages["vision"].median,
                prefix=stages["prefix"].median,
                expert=stages["expert"].median,
                peak=result.peak_memory_bytes / (1024**3),
            )
        )
    measured_runs = results[0].measured_runs
    warmup_runs = results[0].warmup_runs
    return "\n".join(
        (
            "# Benchmark",
            "",
            "Native MLX inference measured after warmup on one real SO-101 golden observation.",
            "",
            "## Environment",
            "",
            f"- Device: `{results[0].device}`",
            f"- CPU: `{_system_value('sysctl', '-n', 'machdep.cpu.brand_string')}`",
            f"- Unified memory: `{_system_value('sysctl', '-n', 'hw.memsize')}` bytes",
            f"- macOS: `{_system_value('sw_vers', '-productVersion')}`",
            f"- Python: `{platform.python_version()}`",
            f"- MLX: `{importlib.metadata.version('mlx')}`",
            f"- Commit: `{_system_value('git', 'rev-parse', 'HEAD')}`",
            f"- Measured runs / excluded warmups: `{measured_runs}` / `{warmup_runs}`",
            "",
            "## Results",
            "",
            "| Storage dtype | Total median ms | Total p95 ms | Preprocess median ms | Vision+connector median ms | Prefix median ms | Expert loop median ms | Peak MLX GB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "The original under-200-ms bf16 target remains a target, not a correctness gate.",
            "",
        )
    )


def main() -> int:
    args = _parse_args()
    observation = _observation(args.sample_root, args.metadata)
    dtypes = ("float32", "bfloat16") if args.dtype == "both" else (args.dtype,)
    results = []
    for dtype in dtypes:
        policy = SmolVLAMLX.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            dtype=dtype,
            tokenizer_dir=args.tokenizer_dir,
        )
        result = run_benchmark(
            policy,
            observation,
            measured_runs=args.runs,
            warmup_runs=args.warmups,
        )
        results.append(result)
        print(json.dumps(result.as_dict(), sort_keys=True))
        del policy
        mx.clear_cache()
    args.output.write_text(_render(results), encoding="utf-8")
    print(f"wrote benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
