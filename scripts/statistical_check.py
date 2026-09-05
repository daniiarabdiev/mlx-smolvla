#!/usr/bin/env python
"""Generate deterministic real-frame MAE evidence for SmolVLA MLX parity.

This script belongs to the reference lane and may import PyTorch/LeRobot. The
runtime package remains dependency-isolated; it consumes only the resulting
JSON record through :class:`mlx_smolvla.statistical.StatisticalResult`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download

from mlx_smolvla._lab.reference.discovery import CHECKPOINT_ID, CHECKPOINT_REVISION
from mlx_smolvla._lab.reference.discovery import DATASET_ID, DATASET_REVISION
from mlx_smolvla._lab.reference.policy import (
    ReferencePolicy,
    load_checkpoint_dataset_observation,
    load_dataset_observation,
)
from mlx_smolvla.policy import SmolVLAMLX


_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
_SEED = 20_260_831
_EPISODE_COUNT = 50


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50, help="Number of deterministic episode-start frames (1-50).")
    parser.add_argument("--output", type=Path, default=Path(".cache/statistical.json"))
    parser.add_argument("--reference-cache", type=Path, default=Path(".cache/hf"))
    parser.add_argument("--native-cache", type=Path, default=Path(".cache/mlx_smolvla"))
    parser.add_argument(
        "--execution-mode",
        choices=("production", "strict"),
        default="strict",
        help="Native inference engine: strict CPU parity or production Metal.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional matching local checkpoint; defaults to the pinned base checkpoint.",
    )
    parser.add_argument("--checkpoint-label", default=CHECKPOINT_ID)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Local LeRobot dataset root for a non-default checkpoint/dataset pair.",
    )
    return parser.parse_args()


def _absolute_error(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}")
    absolute = np.abs(prediction.astype(np.float32, copy=False) - target.astype(np.float32, copy=False))
    return float(np.sum(absolute, dtype=np.float64)), int(absolute.size)


def _native_first_action(
    policy: SmolVLAMLX,
    observation: dict[str, object],
    noise: np.ndarray,
) -> np.ndarray:
    normalized = policy.predict_action_chunk(observation, noise=mx.array(noise))
    with mx.stream(policy.execution_device):
        action = policy.preprocessor.unnormalize_actions(normalized)
        mx.eval(action)
    array = np.asarray(action.astype(mx.float32))
    return array[0, 0]


def main() -> int:
    args = _parse_args()
    if not 1 <= args.samples <= _EPISODE_COUNT:
        raise ValueError(f"--samples must be in [1, {_EPISODE_COUNT}], got {args.samples}")
    reference_cache = args.reference_cache.resolve()
    native_cache = args.native_cache.resolve()
    reference_cache.mkdir(parents=True, exist_ok=True)
    native_cache.mkdir(parents=True, exist_ok=True)

    reference = ReferencePolicy.load(
        cache_dir=reference_cache,
        checkpoint_dir=args.checkpoint,
    )
    if args.checkpoint is None:
        checkpoint_dir = Path(
            snapshot_download(
                CHECKPOINT_ID,
                revision=CHECKPOINT_REVISION,
                cache_dir=str(reference_cache),
                allow_patterns=list(_CHECKPOINT_FILES),
            )
        )
    else:
        checkpoint_dir = args.checkpoint.resolve()
    native_fp32 = SmolVLAMLX.from_pretrained(
        checkpoint_dir,
        cache_dir=native_cache / "statistical-fp32",
        dtype=mx.float32,
        tokenizer_dir=reference.vlm_snapshot,
        execution_mode=args.execution_mode,
    )
    native_bf16 = SmolVLAMLX.from_pretrained(
        checkpoint_dir,
        cache_dir=native_cache / "statistical-bf16",
        dtype=mx.bfloat16,
        tokenizer_dir=reference.vlm_snapshot,
        execution_mode=args.execution_mode,
    )

    totals = {"torch": 0.0, "mlx_fp32": 0.0, "mlx_bf16": 0.0}
    counts = {"torch": 0, "mlx_fp32": 0, "mlx_bf16": 0}
    records = []
    for ordinal in range(args.samples):
        episode = ordinal
        frame_index = 0
        seed = _SEED + ordinal
        generator = np.random.default_rng(seed)
        noise = generator.standard_normal(
            (1, native_fp32.config.chunk_size, native_fp32.config.max_action_dim),
            dtype=np.float32,
        )
        if args.dataset == DATASET_ID and args.dataset_root is None:
            sample = load_dataset_observation(reference_cache, index=frame_index, episode=episode)
        else:
            if args.dataset_root is None:
                raise ValueError("--dataset-root is required for a non-default dataset")
            sample = load_checkpoint_dataset_observation(
                dataset_id=args.dataset,
                dataset_revision=args.dataset_revision,
                dataset_root=args.dataset_root,
                checkpoint_camera_keys=tuple(reference.config.image_features),
                index=frame_index,
                episode=episode,
            )
        target = sample.action.numpy()
        reference_prediction = reference.predict(sample.observation, torch.from_numpy(noise.copy()))
        torch_sum, torch_count = _absolute_error(reference_prediction.actions.numpy()[0, 0], target)
        mlx_fp32_sum, mlx_fp32_count = _absolute_error(
            _native_first_action(native_fp32, dict(sample.observation), noise),
            target,
        )
        mlx_bf16_sum, mlx_bf16_count = _absolute_error(
            _native_first_action(native_bf16, dict(sample.observation), noise),
            target,
        )
        for name, error_sum, count in (
            ("torch", torch_sum, torch_count),
            ("mlx_fp32", mlx_fp32_sum, mlx_fp32_count),
            ("mlx_bf16", mlx_bf16_sum, mlx_bf16_count),
        ):
            totals[name] += error_sum
            counts[name] += count
        records.append(
            {
                "episode": episode,
                "frame_index": frame_index,
                "seed": seed,
                "element_count": torch_count,
                "torch_fp32_abs_error_sum": torch_sum,
                "mlx_fp32_abs_error_sum": mlx_fp32_sum,
                "mlx_bf16_abs_error_sum": mlx_bf16_sum,
            }
        )
        if (ordinal + 1) % 5 == 0 or ordinal + 1 == args.samples:
            print(f"completed {ordinal + 1}/{args.samples} real frames")

    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Metric element counts diverged: {counts}")
    torch_mae = totals["torch"] / counts["torch"]
    mlx_fp32_mae = totals["mlx_fp32"] / counts["mlx_fp32"]
    mlx_bf16_mae = totals["mlx_bf16"] / counts["mlx_bf16"]
    if torch_mae == 0.0:
        raise RuntimeError("Reference MAE is zero; ratio is undefined")
    result = {
        "format_version": 2,
        "checkpoint": args.checkpoint_label,
        "execution_mode": args.execution_mode,
        "device": "Device(gpu, 0)" if args.execution_mode == "production" else "Device(cpu, 0)",
        "dataset": {"id": args.dataset, "revision": args.dataset_revision},
        "sample_count": args.samples,
        "target": "ground-truth current action at deterministic episode-start frame",
        "noise_seed_base": _SEED,
        "torch_fp32_mae": torch_mae,
        "mlx_fp32_mae": mlx_fp32_mae,
        "mlx_bf16_mae": mlx_bf16_mae,
        "mlx_fp32_ratio": mlx_fp32_mae / torch_mae,
        "mlx_bf16_ratio": mlx_bf16_mae / torch_mae,
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "samples"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
