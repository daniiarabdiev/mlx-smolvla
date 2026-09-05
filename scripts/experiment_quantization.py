#!/usr/bin/env python3
"""Run the frozen Stage Q P2-3 VLM-only 8/4-bit experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Mapping


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from mlx_smolvla._lab.reference.discovery import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    DATASET_ID,
    DATASET_REVISION,
)
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
from mlx_smolvla.quantization import (
    QuantizationAccuracy,
    QuantizationLatency,
    QuantizationProtocol,
    derive_quantization_decision,
    describe_dense_vlm,
    expected_topology_manifest,
    quantize_vlm_linears,
    validate_quantization_document,
)


_RESULT_PREFIX = "SMOLVLA_QUANTIZATION_RESULT="
_MINIMUM_FREE_BYTES = 40 * 1024**3
_DENSE_STATISTICAL_SHA256 = "c506ddcfdde50297e97b9905a299d55f117680a93b257e0af335ae6c9ad5fe07"
_SOURCE_FILES = (
    "scripts/benchmark_inference_comparison.py",
    "scripts/experiment_quantization.py",
    "mlx_smolvla/benchmark.py",
    "mlx_smolvla/policy.py",
    "mlx_smolvla/quantization.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, default=Path(".cache/hf"))
    parser.add_argument("--native-cache", type=Path, default=Path(".cache/mlx_smolvla"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(".cache/hf/datasets/svla_so101_pickplace"),
    )
    parser.add_argument(
        "--dense-statistical",
        type=Path,
        default=Path(".cache/statistical-production.json"),
    )
    parser.add_argument("--sample-root", type=Path, default=Path("tests/golden/sample_000"))
    parser.add_argument("--metadata", type=Path, default=Path("tests/golden/metadata.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/QUANTIZATION_EXPERIMENT.json"),
    )
    parser.add_argument(
        "--worker-kind",
        choices=("accuracy", "latency"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-variant",
        choices=("dense-bf16", "vlm-8bit", "vlm-4bit"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _load_policy(variant: str, native_cache: Path):
    import mlx.core as mx

    from mlx_smolvla.policy import SmolVLAMLX

    policy = SmolVLAMLX.from_pretrained(
        CHECKPOINT_ID,
        cache_dir=native_cache,
        dtype="bfloat16",
        execution_mode="production",
    )
    manifest = (
        describe_dense_vlm(policy)
        if variant == "dense-bf16"
        else quantize_vlm_linears(policy, bits=8 if variant == "vlm-8bit" else 4)
    )
    gc.collect()
    mx.clear_cache()
    topology = manifest.as_dict()
    if topology != expected_topology_manifest(variant):
        raise RuntimeError(f"{variant} runtime topology differs from the audited contract")
    return policy, topology


def _latency_worker(args: argparse.Namespace) -> dict[str, object]:
    import mlx.core as mx
    import numpy as np

    from mlx_smolvla.benchmark import _run_staged

    protocol = QuantizationProtocol()
    policy, topology = _load_policy(args.worker_variant, args.native_cache)
    observation = _numpy_observation(args.sample_root, args.metadata)
    noise = mx.array(np.load(args.sample_root / "noise.npy")).astype(mx.float32)
    samples: list[float] = []
    with mx.stream(policy.execution_device):
        for _ in range(protocol.latency_warmup_runs):
            _run_staged(policy, observation, noise)
        mx.clear_cache()
        mx.reset_peak_memory()
        for _ in range(protocol.latency_measured_runs):
            samples.append(_run_staged(policy, observation, noise)["total"])
        peak = int(max(mx.get_peak_memory(), mx.get_active_memory()))
    latency = QuantizationLatency.from_samples(
        variant=args.worker_variant,
        samples_ms=samples,
        warmup_runs=protocol.latency_warmup_runs,
        peak_memory_bytes=peak,
        device=str(policy.execution_device),
    ).as_dict()
    return {"topology": topology, "latency": latency}


def _dense_statistical_samples(path: Path) -> list[dict[str, object]]:
    if _sha256(path) != _DENSE_STATISTICAL_SHA256:
        raise ValueError("dense statistical artifact differs from the frozen P1-1 evidence")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("dense statistical artifact must be an object")
    if (
        value.get("checkpoint") != CHECKPOINT_ID
        or value.get("execution_mode") != "production"
        or value.get("sample_count") != 50
        or value.get("noise_seed_base") != QuantizationProtocol().noise_seed_base
        or value.get("dataset") != {"id": DATASET_ID, "revision": DATASET_REVISION}
    ):
        raise ValueError("dense statistical artifact identity differs from the fixed population")
    records = value.get("samples")
    if not isinstance(records, list) or len(records) != 50:
        raise ValueError("dense statistical artifact sample records are incomplete")
    result: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("dense statistical sample must be an object")
        if (
            record.get("episode"),
            record.get("frame_index"),
            record.get("seed"),
            record.get("element_count"),
        ) != (index, 0, QuantizationProtocol().noise_seed_base + index, 6):
            raise ValueError("dense statistical sample identity differs from the fixed population")
        result.append(
            {
                "episode": index,
                "frame_index": 0,
                "seed": QuantizationProtocol().noise_seed_base + index,
                "element_count": 6,
                "torch_fp32_abs_error_sum": record["torch_fp32_abs_error_sum"],
                "dense_bf16_abs_error_sum": record["mlx_bf16_abs_error_sum"],
                "candidate_abs_error_sum": record["mlx_bf16_abs_error_sum"],
            }
        )
    accuracy = QuantizationAccuracy.from_samples(
        variant="dense-bf16",
        samples=result,
    )
    if (
        accuracy.torch_fp32_mae != value.get("torch_fp32_mae")
        or accuracy.candidate_mae != value.get("mlx_bf16_mae")
        or accuracy.statistical_ratio != value.get("mlx_bf16_ratio")
    ):
        raise ValueError("dense statistical summary does not recompute from its samples")
    return result


def _accuracy_worker(args: argparse.Namespace) -> dict[str, object]:
    if args.worker_variant == "dense-bf16":
        raise ValueError("dense accuracy is imported from the frozen P1-1 evidence")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import mlx.core as mx
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    policy, topology = _load_policy(args.worker_variant, args.native_cache)
    baseline = _dense_statistical_samples(args.dense_statistical)
    dataset = LeRobotDataset(
        DATASET_ID,
        root=args.dataset_root,
        episodes=list(range(50)),
        revision=DATASET_REVISION,
        video_backend="pyav",
    )
    starts = dataset.meta.episodes["dataset_from_index"]
    if len(starts) < 50:
        raise ValueError("dataset metadata lacks the fixed 50 episode starts")
    records: list[dict[str, object]] = []
    for index in range(50):
        item = dataset[int(starts[index])]
        if int(item["episode_index"]) != index or int(item["frame_index"]) != 0:
            raise ValueError("dataset row differs from the fixed episode-start population")
        observation = {
            "observation.images.camera1": item["observation.images.side"].cpu().float(),
            "observation.images.camera2": item["observation.images.up"].cpu().float(),
            "observation.state": item["observation.state"].cpu().float(),
            "task": item["task"],
        }
        noise = np.random.default_rng(QuantizationProtocol().noise_seed_base + index).standard_normal(
            (1, policy.config.chunk_size, policy.config.max_action_dim),
            dtype=np.float32,
        )
        normalized = policy.predict_action_chunk(observation, noise=mx.array(noise))
        with mx.stream(policy.execution_device):
            physical = policy.preprocessor.unnormalize_actions(normalized)
            mx.eval(physical)
        candidate = np.asarray(physical.astype(mx.float32))[0, 0]
        target = np.asarray(item["action"].cpu().float().numpy(), dtype=np.float32)
        if candidate.shape != target.shape:
            raise ValueError("candidate and ground-truth first action shapes differ")
        absolute_error_sum = float(
            np.sum(np.abs(candidate - target), dtype=np.float64)
        )
        source = baseline[index]
        records.append(
            {
                **source,
                "candidate_abs_error_sum": absolute_error_sum,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"{args.worker_variant}: completed {index + 1}/50", file=sys.stderr)
    accuracy = QuantizationAccuracy.from_samples(
        variant=args.worker_variant,
        samples=records,
    ).as_dict()
    return {"topology": topology, "accuracy": accuracy}


def _worker(args: argparse.Namespace) -> int:
    if args.worker_kind is None or args.worker_variant is None:
        raise ValueError("both hidden worker selectors are required")
    result = (
        _latency_worker(args)
        if args.worker_kind == "latency"
        else _accuracy_worker(args)
    )
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


def _child_result(kind: str, variant: str, args: argparse.Namespace) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-kind",
        kind,
        "--worker-variant",
        variant,
        "--reference-cache",
        str(args.reference_cache),
        "--native-cache",
        str(args.native_cache),
        "--dataset-root",
        str(args.dataset_root),
        "--dense-statistical",
        str(args.dense_statistical),
        "--sample-root",
        str(args.sample_root),
        "--metadata",
        str(args.metadata),
    ]
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30 * 60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{kind}/{variant} worker failed with exit {completed.returncode}: "
            f"{completed.stderr[-5000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(f"{kind}/{variant} worker emitted {len(lines)} result records")
    value = json.loads(lines[0].removeprefix(_RESULT_PREFIX))
    if not isinstance(value, dict):
        raise RuntimeError(f"{kind}/{variant} result is not an object")
    return value


def _coordinator(args: argparse.Namespace) -> int:
    os.chdir(_REPOSITORY_ROOT)
    protocol = QuantizationProtocol()
    output = args.output.resolve()
    try:
        output.relative_to(_REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError("quantization output must remain inside the repository") from error
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite quantization artifact: {output}")
    for path in (
        args.reference_cache,
        args.native_cache,
        args.dataset_root,
        args.sample_root,
        args.metadata,
        args.dense_statistical,
    ):
        if not path.exists():
            raise FileNotFoundError(f"required quantization input is absent: {path}")
    if _sha256(args.dense_statistical) != _DENSE_STATISTICAL_SHA256:
        raise ValueError("dense statistical source differs from the frozen P1-1 artifact")
    if shutil.disk_usage(_REPOSITORY_ROOT).free < _MINIMUM_FREE_BYTES:
        raise RuntimeError("quantization experiment requires at least 40 GiB free")
    if not _tracked_worktree_is_clean():
        raise RuntimeError("quantization experiment requires a clean tracked worktree")
    matches = _competing_processes()
    if matches:
        raise RuntimeError(f"quantization timing requires an idle machine: {matches}")
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
            raise FileNotFoundError(f"quantization latency input must be a regular file: {path}")

    latency_results = {
        variant: _child_result("latency", variant, args)
        for variant in protocol.variants
    }
    dense_accuracy = QuantizationAccuracy.from_samples(
        variant="dense-bf16",
        samples=_dense_statistical_samples(args.dense_statistical),
    ).as_dict()
    accuracy_results = {
        variant: _child_result("accuracy", variant, args)
        for variant in protocol.variants[1:]
    }

    variants: list[dict[str, object]] = []
    for variant in protocol.variants:
        latency_result = latency_results[variant]
        topology = latency_result.get("topology")
        if topology != expected_topology_manifest(variant):
            raise RuntimeError(f"latency/{variant} topology differs from the fixed scope")
        if variant == "dense-bf16":
            accuracy = dense_accuracy
        else:
            accuracy_result = accuracy_results[variant]
            if accuracy_result.get("topology") != topology:
                raise RuntimeError(f"accuracy/{variant} topology differs from latency worker")
            accuracy = accuracy_result.get("accuracy")
        variants.append(
            {
                "variant": variant,
                "topology": topology,
                "accuracy": accuracy,
                "latency": latency_result.get("latency"),
            }
        )

    document = {
        "artifact_type": "smolvla-mlx-vlm-quantization-experiment",
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
            "lerobot": importlib.metadata.version("lerobot"),
        },
        "inputs": {
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "latency_sample": "tests/golden/sample_000",
            "latency_sample_sha256": _combined_sha256(sample_files),
            "latency_noise_sha256": _sha256(noise_path),
            "dense_statistical_sha256": _sha256(args.dense_statistical),
        },
        "source": {
            "git_commit": _run("git", "rev-parse", "HEAD"),
            "tracked_worktree_clean": True,
            "sha256": {
                name: _sha256(_REPOSITORY_ROOT / name) for name in _SOURCE_FILES
            },
        },
        "variants": variants,
        "decision": derive_quantization_decision(variants),
    }
    validated = validate_quantization_document(document)
    _atomic_no_clobber_json(output, validated)
    print(json.dumps({"output": str(args.output), "sha256": _sha256(output)}, sort_keys=True))
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return _worker(args) if args.worker_kind is not None else _coordinator(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"experiment_quantization: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
