#!/usr/bin/env python
"""Record default production-Metal deterministic results against fixed goldens."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from huggingface_hub import snapshot_download
import mlx.core as mx
import numpy as np

from mlx_smolvla._lab.reference.goldens import GoldenStore
from mlx_smolvla.policy import SmolVLAMLX
from mlx_smolvla.production_evidence import ProductionDeterministicEvidence


_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
_FIXED_GATES = {"float32": 0.005, "bfloat16": 0.05}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-root", type=Path, default=Path("tests/golden"))
    parser.add_argument("--reference-cache", type=Path, default=Path(".cache/hf"))
    parser.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/mlx_smolvla/production-evidence"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/production-deterministic.json"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _snapshot(metadata: dict[str, object], cache: Path, *, key: str, patterns: tuple[str, ...]) -> Path:
    record = metadata.get(key)
    if not isinstance(record, dict):
        raise ValueError(f"Golden metadata has no {key} provenance")
    identifier = record.get("id")
    revision = record.get("revision")
    if not isinstance(identifier, str) or not isinstance(revision, str):
        raise ValueError(f"Golden metadata {key} id/revision is invalid")
    return Path(
        snapshot_download(
            identifier,
            revision=revision,
            cache_dir=str(cache),
            allow_patterns=list(patterns),
        )
    )


def _observation(store: GoldenStore, sample: dict[str, object]) -> dict[str, object]:
    name = sample.get("name")
    task = sample.get("task")
    if not isinstance(name, str) or not isinstance(task, str):
        raise ValueError("Golden sample name/task is invalid")
    return {
        "observation.images.camera1": store.load(f"{name}/raw/camera1"),
        "observation.images.camera2": store.load(f"{name}/raw/camera2"),
        "observation.state": store.load(f"{name}/raw/state"),
        "task": task,
    }


def main() -> int:
    args = _parse_args()
    golden_root = args.golden_root.resolve()
    metadata_path = golden_root / "metadata.json"
    manifest_path = golden_root / "manifest.json"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Complete goldens are absent at {golden_root}; run make goldens")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples or not all(isinstance(item, dict) for item in samples):
        raise ValueError("Golden metadata samples are invalid")
    reference_cache = args.reference_cache.resolve()
    reference_cache.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = _snapshot(metadata, reference_cache, key="checkpoint", patterns=_CHECKPOINT_FILES)
    tokenizer_dir = _snapshot(metadata, reference_cache, key="base_vlm", patterns=_TOKENIZER_FILES)
    store = GoldenStore(golden_root)

    dtype_results: dict[str, object] = {}
    device = None
    for dtype, fixed_gate in _FIXED_GATES.items():
        policy = SmolVLAMLX.from_pretrained(
            checkpoint_dir,
            cache_dir=args.native_cache.resolve() / dtype,
            dtype=dtype,
            tokenizer_dir=tokenizer_dir,
            execution_mode="production",
        )
        records = []
        with mx.stream(policy.execution_device):
            current_device = str(mx.default_device())
        if device is None:
            device = current_device
        elif device != current_device:
            raise RuntimeError(f"Production device changed between dtypes: {device} != {current_device}")
        for ordinal, sample in enumerate(samples, start=1):
            name = str(sample["name"])
            actual = policy.predict_action_chunk(
                _observation(store, sample),
                noise=store.load(f"{name}/noise"),
            )
            with mx.stream(policy.execution_device):
                mx.eval(actual)
            expected = store.load(f"{name}/actions/normalized").astype(np.float32, copy=False)
            actual_array = np.asarray(actual.astype(mx.float32))
            if actual_array.shape != expected.shape:
                raise ValueError(f"{name}/{dtype} output shape changed: {actual_array.shape} != {expected.shape}")
            max_abs = float(np.max(np.abs(actual_array - expected)))
            records.append({"name": name, "max_abs": max_abs})
            print(f"{dtype}: completed {ordinal}/{len(samples)} ({name}, max_abs={max_abs:.10f})")
        worst = max(records, key=lambda record: float(record["max_abs"]))
        maximum = float(worst["max_abs"])
        dtype_results[dtype] = {
            "fixed_max_abs_gate": fixed_gate,
            "max_abs": maximum,
            "passed": maximum <= fixed_gate,
            "worst_case": worst["name"],
            "samples": records,
        }
        del policy
        mx.clear_cache()

    checkpoint_record = metadata["checkpoint"]
    report = {
        "format_version": 1,
        "execution_mode": "production",
        "device": device,
        "case_count": len(samples),
        "checkpoint": {
            "id": checkpoint_record["id"],
            "revision": checkpoint_record["revision"],
            "model_sha256": _sha256(checkpoint_dir / "model.safetensors"),
        },
        "golden": {
            "manifest_sha256": _sha256(manifest_path),
            "metadata_sha256": _sha256(metadata_path),
        },
        "environment": {
            "mlx": importlib.metadata.version("mlx"),
            "source_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "results": dtype_results,
    }
    _atomic_json(args.output.resolve(), report)
    ProductionDeterministicEvidence.from_json(args.output.resolve())
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, sort_keys=True))
    for dtype, result in dtype_results.items():
        print(json.dumps({"dtype": dtype, **result, "samples": len(result["samples"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
