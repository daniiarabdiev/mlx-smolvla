#!/usr/bin/env python3
"""Build the pinned stats-active SmolVLA base reference checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from huggingface_hub import snapshot_download

from reference.discovery import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    DATASET_ID,
    DATASET_REVISION,
)
from reference.stats_active import build_stats_active_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(".cache/hf/datasets/svla_so101_pickplace"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reference/artifacts/stats-active-base"),
    )
    args = parser.parse_args(argv)
    source = Path(
        snapshot_download(
            CHECKPOINT_ID,
            revision=CHECKPOINT_REVISION,
            cache_dir=args.cache_dir,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            ],
        )
    )
    report = build_stats_active_artifact(
        source_checkpoint=source,
        dataset_stats_path=args.dataset_root / "meta" / "stats.json",
        output_dir=args.output,
        checkpoint_id=CHECKPOINT_ID,
        checkpoint_revision=CHECKPOINT_REVISION,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

