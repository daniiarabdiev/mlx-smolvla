#!/usr/bin/env python3
"""Collect the fixed idle-machine Stage T5 training benchmark matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import mlx.core as mx

from training.benchmark import (
    TrainingBenchmarkConfig,
    benchmark_training_cells,
    validate_training_benchmark,
)
from training.finetune import write_run_state


_PROHIBITED_PROCESS_PATTERNS = (
    "finetune_lora",
    "check_training_resume",
    "floor_runtime",
    "make_training_goldens",
    "make_lora_evaluation",
    "mlx-smolvla train",
    "pytest",
    "make test",
)
_SOURCE_FILES = (
    "training/benchmark.py",
    "training/ux.py",
    "training/finetune.py",
    "training/dataset.py",
    "training/model.py",
    "training/optimizer.py",
    "training/lora.py",
    "scripts/benchmark_training.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _idle_evidence() -> dict[str, object]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    processes: dict[int, tuple[int, str]] = {}
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        processes[int(fields[0])] = (int(fields[1]), fields[2])
    excluded: set[int] = set()
    current = os.getpid()
    while current > 0 and current not in excluded:
        excluded.add(current)
        current = processes.get(current, (0, ""))[0]
    matches = [
        {"pid": pid, "command": command}
        for pid, (_, command) in sorted(processes.items())
        if pid not in excluded
        and any(pattern in command for pattern in _PROHIBITED_PROCESS_PATTERNS)
    ]
    if matches:
        raise RuntimeError(f"training benchmark requires an idle machine: {matches}")
    return {
        "verified": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "matching_processes": [],
        "prohibited_patterns": list(_PROHIBITED_PROCESS_PATTERNS),
    }


def _sysctl(name: str) -> str:
    completed = subprocess.run(
        ["sysctl", "-n", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    parser.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/mlx_smolvla/policy-float32"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite training benchmark: {output}")
    if shutil.disk_usage(output.parent).free < 40 * 1024**3:
        raise RuntimeError("training benchmark requires at least 40 GiB free")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("training benchmark requires a clean tracked worktree")
    commit = _git("rev-parse", "HEAD")
    idle = _idle_evidence()
    protocol = TrainingBenchmarkConfig()
    started_at = datetime.now(timezone.utc).isoformat()

    def progress(
        mode: str,
        dtype: str,
        state: str,
        cell: dict[str, object] | None,
    ) -> None:
        if state == "start":
            print(f"starting mode={mode} dtype={dtype}", flush=True)
        else:
            assert cell is not None
            print(
                f"complete mode={mode} dtype={dtype} "
                f"median={cell['median_update_seconds']:.6f}s "
                f"peak={cell['peak_memory_bytes']}",
                flush=True,
            )

    cells = benchmark_training_cells(
        dataset=args.dataset,
        cache_dir=args.cache_dir,
        native_cache=args.native_cache,
        config=protocol,
        progress=progress,
    )
    protocol_document = json.loads(json.dumps(protocol.__dict__))
    source_sha256 = {
        name: _sha256(Path(name)) for name in _SOURCE_FILES
    }
    document: dict[str, object] = {
        "format_version": 1,
        "artifact_type": "smolvla-mlx-training-benchmark",
        "classification": "idle-machine-training-performance",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_tracked_clean": True,
        "idle": idle,
        "protocol": protocol_document,
        "dataset": str(args.dataset),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "native_cache": str(Path(args.native_cache).resolve()),
        "device": str(mx.default_device()),
        "environment": {
            "cpu": _sysctl("machdep.cpu.brand_string"),
            "unified_memory_bytes": int(_sysctl("hw.memsize")),
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mlx": version("mlx"),
            "lerobot": version("lerobot"),
            "torch": version("torch"),
        },
        "source_sha256": source_sha256,
        "cells": cells,
    }
    validate_training_benchmark(document)
    document["report_sha256"] = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    digest = write_run_state(output, document)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": digest,
                "report_sha256": document["report_sha256"],
                "cells": len(cells),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
