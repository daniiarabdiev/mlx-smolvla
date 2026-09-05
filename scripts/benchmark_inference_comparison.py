#!/usr/bin/env python3
"""Collect the frozen Stage Q P2-1 MLX/PyTorch-MPS comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Mapping


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from mlx_smolvla._lab.reference.benchmark import (
    ComparisonProtocol,
    EngineTiming,
    validate_comparison_document,
)
from mlx_smolvla._lab.reference.discovery import CHECKPOINT_ID, CHECKPOINT_REVISION


_RESULT_PREFIX = "SMOLVLA_BENCHMARK_RESULT="
_MINIMUM_FREE_BYTES = 40 * 1024**3
_SOURCE_FILES = (
    "reference/benchmark.py",
    "reference/policy.py",
    "scripts/benchmark_inference_comparison.py",
    "mlx_smolvla/benchmark.py",
    "mlx_smolvla/policy.py",
)
_COMPETING_MARKERS = (
    "pytest",
    "make test",
    "finetune_lora",
    "benchmark_training.py",
    "mlx_smolvla._lab.training.self_consistency",
    "self_consistency.py",
    "scripts/bench.py",
    "profile_inference_dtypes.py",
    "experiment_quantization.py",
)
_MPS_ENVIRONMENT_KEYS = (
    "PYTORCH_DEBUG_MPS_ALLOCATOR",
    "PYTORCH_MPS_LOG_PROFILE_INFO",
    "PYTORCH_MPS_TRACE_SIGNPOSTS",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_PREFER_METAL",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, default=Path(".cache/hf"))
    parser.add_argument("--native-cache", type=Path, default=Path(".cache/mlx_smolvla"))
    parser.add_argument("--sample-root", type=Path, default=Path("tests/golden/sample_000"))
    parser.add_argument("--metadata", type=Path, default=Path("tests/golden/metadata.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/INFERENCE_COMPARISON.json"),
    )
    parser.add_argument(
        "--worker-engine",
        choices=("mlx", "pytorch-mps"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.resolve().relative_to(_REPOSITORY_ROOT.resolve()).as_posix()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _run(*command: str) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _system_value(*command: str) -> str:
    try:
        return _run(*command)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _tracked_worktree_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout == ""


def _ancestor_pids() -> set[int]:
    result = {os.getpid()}
    current = os.getppid()
    while current > 1 and current not in result:
        result.add(current)
        completed = subprocess.run(
            ["ps", "-p", str(current), "-o", "ppid="],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            current = int(completed.stdout.strip())
        except ValueError:
            break
    return result


def _competing_processes() -> list[str]:
    ignored = _ancestor_pids()
    completed = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    matches: list[str] = []
    for raw in completed.stdout.splitlines():
        fields = raw.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        pid_text, command = fields
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        lowered = command.lower()
        if pid not in ignored and any(marker.lower() in lowered for marker in _COMPETING_MARKERS):
            matches.append(f"pid={pid} command={command}")
    return sorted(matches)


def _task(metadata_path: Path, sample_name: str) -> str:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = metadata.get("samples")
    if not isinstance(samples, list):
        raise ValueError("golden metadata samples are absent")
    for sample in samples:
        if isinstance(sample, Mapping) and sample.get("name") == sample_name:
            task = sample.get("task")
            if isinstance(task, str) and task:
                return task
    raise ValueError(f"golden metadata has no task for {sample_name!r}")


def _numpy_observation(sample_root: Path, metadata: Path) -> dict[str, object]:
    import numpy as np

    return {
        "observation.images.camera1": np.load(sample_root / "raw/camera1.npy"),
        "observation.images.camera2": np.load(sample_root / "raw/camera2.npy"),
        "observation.state": np.load(sample_root / "raw/state.npy"),
        "task": _task(metadata, sample_root.name),
    }


def _mlx_worker(args: argparse.Namespace) -> dict[str, object]:
    import mlx.core as mx
    import numpy as np

    from mlx_smolvla.benchmark import _run_staged
    from mlx_smolvla.policy import SmolVLAMLX

    protocol = ComparisonProtocol()
    policy = SmolVLAMLX.from_pretrained(
        CHECKPOINT_ID,
        cache_dir=args.native_cache,
        dtype="float32",
        execution_mode="production",
    )
    observation = _numpy_observation(args.sample_root, args.metadata)
    noise = mx.array(np.load(args.sample_root / "noise.npy")).astype(mx.float32)
    samples: list[float] = []
    with mx.stream(policy.execution_device):
        for _ in range(protocol.warmup_runs):
            _run_staged(policy, observation, noise)
        mx.clear_cache()
        mx.reset_peak_memory()
        for _ in range(protocol.measured_runs):
            samples.append(_run_staged(policy, observation, noise)["total"])
        peak = int(max(mx.get_peak_memory(), mx.get_active_memory()))
    return EngineTiming.from_samples(
        engine="mlx",
        samples_ms=samples,
        warmup_runs=protocol.warmup_runs,
        peak_memory_bytes=peak,
        device=str(policy.execution_device),
        dtype="float32",
        fallback_enabled=False,
    ).as_dict()


def _torch_worker(args: argparse.Namespace) -> dict[str, object]:
    for name in _MPS_ENVIRONMENT_KEYS:
        os.environ.pop(name, None)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    import numpy as np
    import torch

    from mlx_smolvla._lab.reference.policy import ReferencePolicy

    protocol = ComparisonProtocol()
    if not torch.backends.mps.is_available():
        raise RuntimeError("the fixed PyTorch comparison requires an available MPS backend")
    reference = ReferencePolicy.load(args.reference_cache, device="mps")
    raw = _numpy_observation(args.sample_root, args.metadata)
    observation = {
        key: torch.from_numpy(np.ascontiguousarray(value)) if isinstance(value, np.ndarray) else value
        for key, value in raw.items()
    }
    noise = torch.from_numpy(
        np.ascontiguousarray(np.load(args.sample_root / "noise.npy"))
    ).to(device="mps", dtype=torch.float32)
    samples: list[float] = []
    observed_memory: list[int] = []

    def run_once() -> None:
        batch = reference.prepare(observation)
        reference.policy.predict_action_chunk(batch, noise=noise)
        torch.mps.synchronize()

    with torch.inference_mode():
        for _ in range(protocol.warmup_runs):
            run_once()
        torch.mps.empty_cache()
        for _ in range(protocol.measured_runs):
            torch.mps.synchronize()
            started = time.perf_counter()
            run_once()
            samples.append((time.perf_counter() - started) * 1_000.0)
            observed_memory.append(
                max(
                    int(torch.mps.current_allocated_memory()),
                    int(torch.mps.driver_allocated_memory()),
                )
            )
    return EngineTiming.from_samples(
        engine="pytorch-mps",
        samples_ms=samples,
        warmup_runs=protocol.warmup_runs,
        peak_memory_bytes=max(observed_memory),
        device=str(reference.device),
        dtype="float32",
        fallback_enabled=True,
    ).as_dict()


def _worker(args: argparse.Namespace) -> int:
    result = _mlx_worker(args) if args.worker_engine == "mlx" else _torch_worker(args)
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


def _child_result(engine: str, args: argparse.Namespace) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-engine",
        engine,
        "--reference-cache",
        str(args.reference_cache),
        "--native-cache",
        str(args.native_cache),
        "--sample-root",
        str(args.sample_root),
        "--metadata",
        str(args.metadata),
    ]
    environment = dict(os.environ)
    if engine == "pytorch-mps":
        for name in _MPS_ENVIRONMENT_KEYS:
            environment.pop(name, None)
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30 * 60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{engine} benchmark worker failed with exit {completed.returncode}: "
            f"{completed.stderr[-4000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(f"{engine} worker emitted {len(lines)} result records")
    result = json.loads(lines[0].removeprefix(_RESULT_PREFIX))
    if not isinstance(result, dict):
        raise RuntimeError(f"{engine} worker result is not an object")
    return result


def _atomic_no_clobber_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite comparison artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _coordinator(args: argparse.Namespace) -> int:
    os.chdir(_REPOSITORY_ROOT)
    protocol = ComparisonProtocol()
    for path in (args.reference_cache, args.native_cache, args.sample_root, args.metadata):
        if not path.exists():
            raise FileNotFoundError(f"required benchmark input is absent: {path}")
    output = args.output.resolve()
    try:
        output.relative_to(_REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError("comparison output must remain inside the repository") from error
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite comparison artifact: {output}")
    if shutil.disk_usage(_REPOSITORY_ROOT).free < _MINIMUM_FREE_BYTES:
        raise RuntimeError("comparison benchmark requires at least 40 GiB free")
    if not _tracked_worktree_is_clean():
        raise RuntimeError("comparison benchmark requires a clean tracked worktree")
    matches = _competing_processes()
    if matches:
        raise RuntimeError(f"comparison benchmark requires an idle machine: {matches}")
    checked_at = datetime.now(timezone.utc).isoformat()

    required_sample_files = [
        args.metadata,
        args.sample_root / "raw/camera1.npy",
        args.sample_root / "raw/camera2.npy",
        args.sample_root / "raw/state.npy",
    ]
    noise_path = args.sample_root / "noise.npy"
    for path in (*required_sample_files, noise_path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"benchmark input must be a regular file: {path}")

    engines = [_child_result(engine, args) for engine in protocol.engines]
    document = {
        "artifact_type": "smolvla-mlx-inference-comparison",
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
            "torch": importlib.metadata.version("torch"),
            "lerobot": importlib.metadata.version("lerobot"),
        },
        "inputs": {
            "sample": "tests/golden/sample_000",
            "sample_sha256": _combined_sha256(required_sample_files),
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
        "engines": engines,
    }
    validated = validate_comparison_document(document)
    _atomic_no_clobber_json(output, validated)
    print(json.dumps({"output": str(args.output), "sha256": _sha256(output)}, sort_keys=True))
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return _worker(args) if args.worker_engine is not None else _coordinator(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"benchmark_inference_comparison: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
