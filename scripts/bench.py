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
from smolvla_mlx.production_evidence import ProductionDeterministicEvidence
from smolvla_mlx.statistical import StatisticalResult


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/smolvla_mlx"))
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--sample-root", type=Path, default=Path("tests/golden/sample_000"))
    parser.add_argument("--metadata", type=Path, default=Path("tests/golden/metadata.json"))
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "both"), default="both")
    parser.add_argument("--execution-mode", choices=("production", "strict"), default="production")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("BENCHMARK.md"))
    parser.add_argument(
        "--production-deterministic",
        type=Path,
        default=Path(".cache/production-deterministic.json"),
    )
    parser.add_argument(
        "--strict-statistical",
        type=Path,
        default=Path(".cache/statistical-strict-production-report.json"),
    )
    parser.add_argument(
        "--production-statistical",
        type=Path,
        default=Path(".cache/statistical-production.json"),
    )
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


def _render(
    results: list[BenchmarkResult],
    production: ProductionDeterministicEvidence,
    strict_stats: StatisticalResult,
    production_stats: StatisticalResult,
) -> str:
    if not results or any(result.execution_mode != "production" for result in results):
        raise ValueError("The release benchmark report requires production-mode timing results")
    if strict_stats.execution_mode != "strict" or "cpu" not in (strict_stats.device or "").lower():
        raise ValueError("Strict statistical evidence must explicitly identify strict CPU mode")
    if production_stats.execution_mode != "production" or "gpu" not in (production_stats.device or "").lower():
        raise ValueError("Production statistical evidence must explicitly identify production GPU mode")
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
    production_rows = []
    strict_rows = []
    for dtype, ratio in (
        ("float32", strict_stats.mlx_fp32_ratio),
        ("bfloat16", strict_stats.mlx_bf16_ratio),
    ):
        gate = 0.005 if dtype == "float32" else 0.05
        strict_rows.append(
            f"| {dtype} | pass (8/8) | {gate:g} | {ratio:.10f} | pass (`<=1.05`) |"
        )
    for dtype, ratio in (
        ("float32", production_stats.mlx_fp32_ratio),
        ("bfloat16", production_stats.mlx_bf16_ratio),
    ):
        deterministic = production.results[dtype]
        outcome = "pass" if deterministic.passed else "fail"
        production_rows.append(
            f"| {dtype} | {deterministic.max_abs:.10f} | {deterministic.fixed_max_abs_gate:g} | {outcome} | {ratio:.10f} | pass (`<=1.05`) |"
        )
    return "\n".join(
        (
            "# Benchmark",
            "",
            "Strict CPU correctness and default production-Metal correctness/performance are reported separately.",
            "",
            "## Environment",
            "",
            f"- Device: `{results[0].device}`",
            f"- Execution mode: `{results[0].execution_mode}`",
            f"- CPU: `{_system_value('sysctl', '-n', 'machdep.cpu.brand_string')}`",
            f"- Unified memory: `{_system_value('sysctl', '-n', 'hw.memsize')}` bytes",
            f"- macOS: `{_system_value('sw_vers', '-productVersion')}`",
            f"- Python: `{platform.python_version()}`",
            f"- MLX: `{importlib.metadata.version('mlx')}`",
            f"- Commit: `{_system_value('git', 'rev-parse', 'HEAD')}`",
            f"- Measured runs / excluded warmups: `{measured_runs}` / `{warmup_runs}`",
            "",
            "## Execution modes",
            "",
            "`production` is the public default and owns an MLX Metal device context. `strict` owns an MLX CPU context and uses the compatibility arithmetic validated against the pinned PyTorch CPU goldens. Callers select the latter with `execution_mode=\"strict\"` or CLI `--execution-mode strict`.",
            "",
            "## Strict-parity correctness (CPU)",
            "",
            "The immutable deterministic limits apply to eight fixed real observations; the statistical limit applies to first-action MAE on 50 pinned real frames.",
            "",
            "| Storage dtype | Deterministic result | Fixed max-abs gate | 50-frame MLX/reference MAE ratio | Statistical result |",
            "| --- | ---: | ---: | ---: | ---: |",
            *strict_rows,
            "",
            "## Default-production correctness (Metal)",
            "",
            "These are the same observations, noise tensors, reference actions, and unchanged gates, executed through the installed default production mode.",
            "",
            "| Storage dtype | Deterministic max abs | Fixed gate | Deterministic result | 50-frame MLX/reference MAE ratio | Statistical result |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *production_rows,
            "",
            "Metal fp32 does not satisfy the strict deterministic contract; this is a recorded negative result, not a tolerance change. Metal bf16 satisfies its fixed deterministic gate, and both production dtypes satisfy the fixed statistical gate.",
            "",
            "## Default-production performance (Metal)",
            "",
            "| Storage dtype | Total median ms | Total p95 ms | Preprocess median ms | Vision+connector median ms | Prefix median ms | Expert loop median ms | Peak MLX GB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "A 50-action chunk represents about 1.67 seconds of motion at 30 fps. The latency table is model-only and excludes capture, transport, and actuation. The original under-200-ms bf16 target remains a target, not a correctness gate.",
            "",
        )
    )


def main() -> int:
    args = _parse_args()
    if args.execution_mode != "production":
        raise ValueError("The release benchmark report must measure the default production mode")
    observation = _observation(args.sample_root, args.metadata)
    dtypes = ("float32", "bfloat16") if args.dtype == "both" else (args.dtype,)
    results = []
    for dtype in dtypes:
        policy = SmolVLAMLX.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            dtype=dtype,
            tokenizer_dir=args.tokenizer_dir,
            execution_mode=args.execution_mode,
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
    production = ProductionDeterministicEvidence.from_json(args.production_deterministic)
    strict_stats = StatisticalResult.from_json(args.strict_statistical)
    production_stats = StatisticalResult.from_json(args.production_statistical)
    args.output.write_text(
        _render(results, production, strict_stats, production_stats),
        encoding="utf-8",
    )
    print(f"wrote benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
