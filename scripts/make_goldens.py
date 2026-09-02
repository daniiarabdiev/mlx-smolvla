#!/usr/bin/env python3
"""Capture deterministic CPU/fp32 reference tensors for the native MLX port."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from reference.discovery import (
    BASE_VLM_ID,
    BASE_VLM_REVISION,
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    DATASET_ID,
    DATASET_REVISION,
)
from reference.goldens import GOLDEN_SAMPLE_SPECS, GoldenWriter, capture_sample
from reference.policy import ReferencePolicy, load_dataset_observation


def _selected_specs(indices: list[int] | None):
    if indices is None:
        return GOLDEN_SAMPLE_SPECS
    selected = []
    for index in indices:
        if index < 0 or index >= len(GOLDEN_SAMPLE_SPECS):
            raise ValueError(f"Sample index must be in [0, {len(GOLDEN_SAMPLE_SPECS) - 1}], got {index}")
        selected.append(GOLDEN_SAMPLE_SPECS[index])
    if len({spec.name for spec in selected}) != len(selected):
        raise ValueError("Each --sample-index may appear at most once")
    return tuple(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/golden"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional matching local checkpoint; defaults to the pinned base checkpoint.",
    )
    parser.add_argument("--checkpoint-id", help="Identity recorded for --checkpoint metadata.")
    parser.add_argument("--checkpoint-revision", help="Revision recorded for --checkpoint metadata.")
    parser.add_argument(
        "--sample-index",
        type=int,
        action="append",
        help="Capture only one or more zero-based entries from GOLDEN_SAMPLE_SPECS.",
    )
    args = parser.parse_args(argv)
    selected = _selected_specs(args.sample_index)

    if args.checkpoint is None and (args.checkpoint_id is not None or args.checkpoint_revision is not None):
        raise ValueError("--checkpoint-id/--checkpoint-revision require --checkpoint")
    if args.checkpoint is not None and (not args.checkpoint_id or not args.checkpoint_revision):
        raise ValueError("--checkpoint requires --checkpoint-id and --checkpoint-revision")
    reference = ReferencePolicy.load(args.cache_dir, checkpoint_dir=args.checkpoint)
    writer = GoldenWriter(args.output)
    samples = []
    for spec in selected:
        print(f"Capturing {spec.name}: episode {spec.episode}, frame {spec.frame_index}", flush=True)
        sample = load_dataset_observation(args.cache_dir, spec.frame_index, episode=spec.episode)
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
            "checkpoint": {
                "id": args.checkpoint_id or CHECKPOINT_ID,
                "revision": args.checkpoint_revision or CHECKPOINT_REVISION,
            },
            "base_vlm": {"id": BASE_VLM_ID, "revision": BASE_VLM_REVISION},
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "samples": samples,
            "tensor_count": len(manifest),
        }
    )
    print(f"Wrote {len(manifest)} tensors for {len(samples)} sample(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
