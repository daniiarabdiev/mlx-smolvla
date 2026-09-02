#!/usr/bin/env python3
"""Collect the frozen Stage Q P2-2 fp32/bf16 component profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from reference.discovery import CHECKPOINT_ID, CHECKPOINT_REVISION
from scripts.benchmark_inference_comparison import (
    _atomic_no_clobber_json,
    _combined_sha256,
    _competing_processes,
    _numpy_observation,
    _run,
    _sha256,
    _system_value,
    _tracked_worktree_is_clean,
)
from mlx_smolvla.profile import (
    ProfileProtocol,
    collect_dtype_profile,
    derive_dtype_analysis,
    validate_profile_document,
)


_RESULT_PREFIX = "SMOLVLA_PROFILE_RESULT="
_MINIMUM_FREE_BYTES = 40 * 1024**3
_SOURCE_FILES = (
    "scripts/benchmark_inference_comparison.py",
    "scripts/profile_inference_dtypes.py",
    "mlx_smolvla/benchmark.py",
    "mlx_smolvla/policy.py",
    "mlx_smolvla/profile.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-cache", type=Path, default=Path(".cache/mlx_smolvla"))
    parser.add_argument("--sample-root", type=Path, default=Path("tests/golden/sample_000"))
    parser.add_argument("--metadata", type=Path, default=Path("tests/golden/metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("BF16_PROFILE.json"))
    parser.add_argument(
        "--worker-dtype",
        choices=("float32", "bfloat16"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _worker(args: argparse.Namespace) -> int:
    import mlx.core as mx
    import numpy as np

    from mlx_smolvla.policy import SmolVLAMLX

    policy = SmolVLAMLX.from_pretrained(
        CHECKPOINT_ID,
        cache_dir=args.native_cache,
        dtype=args.worker_dtype,
        execution_mode="production",
    )
    observation = _numpy_observation(args.sample_root, args.metadata)
    noise = mx.array(np.load(args.sample_root / "noise.npy")).astype(mx.float32)
    result = collect_dtype_profile(policy, observation, noise).as_dict()
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


def _child_result(dtype: str, args: argparse.Namespace) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-dtype",
            dtype,
            "--native-cache",
            str(args.native_cache),
            "--sample-root",
            str(args.sample_root),
            "--metadata",
            str(args.metadata),
        ],
        cwd=_REPOSITORY_ROOT,
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
        timeout=20 * 60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{dtype} profile worker failed with exit {completed.returncode}: "
            f"{completed.stderr[-4000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(f"{dtype} profile worker emitted {len(lines)} result records")
    value = json.loads(lines[0].removeprefix(_RESULT_PREFIX))
    if not isinstance(value, dict):
        raise RuntimeError(f"{dtype} profile worker result is not an object")
    return value


def _coordinator(args: argparse.Namespace) -> int:
    os.chdir(_REPOSITORY_ROOT)
    protocol = ProfileProtocol()
    output = args.output.resolve()
    try:
        output.relative_to(_REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError("profile output must remain inside the repository") from error
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite profile artifact: {output}")
    if not args.native_cache.exists() or not args.sample_root.is_dir() or not args.metadata.is_file():
        raise FileNotFoundError("one or more fixed profile inputs are absent")
    if shutil.disk_usage(_REPOSITORY_ROOT).free < _MINIMUM_FREE_BYTES:
        raise RuntimeError("component profile requires at least 40 GiB free")
    if not _tracked_worktree_is_clean():
        raise RuntimeError("component profile requires a clean tracked worktree")
    matches = _competing_processes()
    if matches:
        raise RuntimeError(f"component profile requires an idle machine: {matches}")
    checked_at = datetime.now(timezone.utc).isoformat()

    sample_files = [
        args.metadata,
        args.sample_root / "raw/camera1.npy",
        args.sample_root / "raw/camera2.npy",
        args.sample_root / "raw/state.npy",
    ]
    noise_path = args.sample_root / "noise.npy"
    for path in (*sample_files, noise_path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"profile input must be a regular file: {path}")

    profiles = [_child_result(dtype, args) for dtype in protocol.dtypes]
    document = {
        "artifact_type": "smolvla-mlx-bf16-component-profile",
        "format_version": 1,
        "protocol": protocol.as_dict(),
        "idle": {
            "verified": True,
            "checked_at_utc": checked_at,
            "matching_processes": matches,
        },
        "environment": {
            "cpu": _system_value("sysctl", "-n", "machdep.cpu.brand_string"),
            "unified_memory_bytes": int(_system_value("sysctl", "-n", "hw.memsize")),
            "macos": _system_value("sw_vers", "-productVersion"),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
        },
        "inputs": {
            "sample": "tests/golden/sample_000",
            "sample_sha256": _combined_sha256(sample_files),
            "noise_sha256": _sha256(noise_path),
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_revision": CHECKPOINT_REVISION,
        },
        "source": {
            "git_commit": _run("git", "rev-parse", "HEAD"),
            "tracked_worktree_clean": True,
            "sha256": {
                name: _sha256(_REPOSITORY_ROOT / name) for name in _SOURCE_FILES
            },
        },
        "profiles": profiles,
        "analysis": derive_dtype_analysis(profiles),
    }
    validated = validate_profile_document(document)
    _atomic_no_clobber_json(output, validated)
    print(json.dumps({"output": str(args.output), "sha256": _sha256(output)}, sort_keys=True))
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return _worker(args) if args.worker_dtype is not None else _coordinator(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"profile_inference_dtypes: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
