#!/usr/bin/env python3
"""Capture goldens for the pinned public multitask SmolVLA fine-tune."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from huggingface_hub import snapshot_download

from mlx_smolvla._lab.reference.discovery import BASE_VLM_ID, BASE_VLM_REVISION
from mlx_smolvla._lab.reference.goldens import GOLDEN_SAMPLE_SPECS, GoldenWriter, capture_sample
from mlx_smolvla._lab.reference.policy import ReferencePolicy, load_checkpoint_dataset_observation


MODEL_ID = "soonweihong0857/swhfypv3_smolvla_multitask_model"
MODEL_REVISION = "5e2491c809ec892427f54db1eb23bf8c4bbbf770"
DATASET_ID = "soonweihong0857/smolvla_multitask_data"
DATASET_REVISION = "ec0062a53e0ae88d46a4341ab0695dfa9f03111b"
_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    parser.add_argument("--output", type=Path, default=Path("tests/golden-public-finetune"))
    args = parser.parse_args(argv)
    checkpoint = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=args.cache_dir,
            allow_patterns=list(_MODEL_FILES),
        )
    )
    dataset_root = Path(
        snapshot_download(
            DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=args.cache_dir,
        )
    )
    reference = ReferencePolicy.load(args.cache_dir, checkpoint_dir=checkpoint)
    writer = GoldenWriter(args.output)
    samples = []
    for spec in GOLDEN_SAMPLE_SPECS:
        print(f"Capturing {spec.name}: episode {spec.episode}, frame {spec.frame_index}", flush=True)
        sample = load_checkpoint_dataset_observation(
            dataset_id=DATASET_ID,
            dataset_revision=DATASET_REVISION,
            dataset_root=dataset_root,
            checkpoint_camera_keys=tuple(reference.config.image_features),
            index=spec.frame_index,
            episode=spec.episode,
        )
        samples.append(
            capture_sample(
                writer,
                reference,
                sample,
                sample_name=spec.name,
                episode=spec.episode,
                frame_index=spec.frame_index,
                seed=spec.seed,
            )
        )
    manifest = writer.finalize()
    writer.write_metadata(
        {
            "format_version": 1,
            "checkpoint": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "base_vlm": {"id": BASE_VLM_ID, "revision": BASE_VLM_REVISION},
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "samples": samples,
            "tensor_count": len(manifest),
        }
    )
    print(json.dumps({"sample_count": len(samples), "tensor_count": len(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

