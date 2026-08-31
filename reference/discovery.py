"""Discover the installed SmolVLA reference without assuming package paths."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download


CHECKPOINT_ID = "lerobot/smolvla_base"
CHECKPOINT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
DATASET_ID = "lerobot/svla_so101_pickplace"
DATASET_REVISION = "f641879e22172be7e8161d5e6c1503c2d2feb657"


@dataclass(frozen=True)
class ReferenceDiscovery:
    """Exact installed reference locations and identifiers."""

    lerobot_version: str
    torch_version: str
    transformers_version: str
    policy_source: Path
    config_source: Path
    checkpoint_id: str
    checkpoint_revision: str
    checkpoint_config: Path
    tensor_count: int
    parameter_count: int
    dataset_id: str
    dataset_revision: str
    camera_keys: tuple[str, ...]
    state_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    has_language_tasks: bool


def _source_containing(root: Path, symbol: str) -> Path:
    matches = [
        path
        for path in root.rglob("*.py")
        if "smolvla" in str(path).lower() and symbol in path.read_text(encoding="utf-8")
    ]
    if len(matches) != 1:
        rendered = ", ".join(str(path) for path in matches) or "none"
        raise RuntimeError(f"Expected one SmolVLA source containing {symbol!r}; found {rendered}")
    return matches[0]


def discover_reference(cache_dir: Path) -> ReferenceDiscovery:
    """Locate and pin installed sources, checkpoint metadata, and dataset schema."""

    spec = importlib.util.find_spec("lerobot")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("The optional reference dependency 'lerobot' is not installed")
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    checkpoint_config = Path(
        hf_hub_download(
            CHECKPOINT_ID,
            "config.json",
            revision=CHECKPOINT_REVISION,
            cache_dir=cache_dir,
        )
    )
    safetensors = get_safetensors_metadata(
        CHECKPOINT_ID,
        revision=CHECKPOINT_REVISION,
    )
    dataset_api_info = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION)
    dataset_info_path = Path(
        hf_hub_download(
            DATASET_ID,
            "meta/info.json",
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=cache_dir,
        )
    )
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    features = dataset_info["features"]
    camera_keys = tuple(
        sorted(
            name
            for name, feature in features.items()
            if name.startswith("observation.images.") and feature["dtype"] in {"image", "video"}
        )
    )
    sibling_names = {sibling.rfilename for sibling in dataset_api_info.siblings}
    return ReferenceDiscovery(
        lerobot_version=version("lerobot"),
        torch_version=version("torch"),
        transformers_version=version("transformers"),
        policy_source=_source_containing(root, "class SmolVLAPolicy"),
        config_source=_source_containing(root, "class SmolVLAConfig"),
        checkpoint_id=CHECKPOINT_ID,
        checkpoint_revision=CHECKPOINT_REVISION,
        checkpoint_config=checkpoint_config,
        tensor_count=len(safetensors.weight_map),
        parameter_count=sum(safetensors.parameter_count.values()),
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        camera_keys=camera_keys,
        state_shape=tuple(features["observation.state"]["shape"]),
        action_shape=tuple(features["action"]["shape"]),
        has_language_tasks="meta/tasks.parquet" in sibling_names,
    )


def render_architecture_evidence(discovery: ReferenceDiscovery) -> str:
    """Render the immutable reference pins established during Phase 0."""

    cameras = ", ".join(f"`{key}`" for key in discovery.camera_keys)
    return f"""# Architecture

## Reference pins

- LeRobot {discovery.lerobot_version}
- PyTorch {discovery.torch_version} (CPU fp32 golden reference)
- Transformers {discovery.transformers_version}
- Policy source: `{discovery.policy_source}`
- Configuration source: `{discovery.config_source}`

## Checkpoint inventory

- Repository: `{discovery.checkpoint_id}`
- Revision: `{discovery.checkpoint_revision}`
- Inventory: {discovery.tensor_count:,} tensors; {discovery.parameter_count:,} parameters
- Configuration cache path: `{discovery.checkpoint_config}`

## Golden dataset

- Repository: `{discovery.dataset_id}`
- Revision: `{discovery.dataset_revision}`
- Cameras ({len(discovery.camera_keys)}): {cameras}
- State shape: `{list(discovery.state_shape)}`
- Action shape: `{list(discovery.action_shape)}`
- Language task table present: `{discovery.has_language_tasks}`

The remaining architecture hypotheses from `BRIEF.md` Section 3 are resolved
by the source and runtime audit in Phase 1.
"""


def main(argv: list[str] | None = None) -> int:
    """Write pinned discovery evidence for the architecture audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args(argv)
    discovery = discover_reference(args.cache_dir)
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(render_architecture_evidence(discovery), encoding="utf-8")
    print(args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
