"""Pinned LeRobot dataset split and Torch-to-NumPy training bridge."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from reference.discovery import (
    BASE_VLM_ID,
    BASE_VLM_REVISION,
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    DATASET_ID,
    DATASET_REVISION,
)


SPLIT_SEED = 20_260_901
SAMPLER_SEED = 20_260_901
HOLDOUT_FRACTION = 0.15
_CAMERA_RENAME_MAP = {
    "observation.images.side": "observation.images.camera1",
    "observation.images.up": "observation.images.camera2",
}
_BASE_VLM_PROCESSOR_FILES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class EpisodeSplit:
    """A deterministic whole-episode train/held-out partition."""

    seed: int
    holdout_fraction: float
    train_episodes: tuple[int, ...]
    holdout_episodes: tuple[int, ...]


def make_episode_split(
    *,
    num_episodes: int,
    seed: int = SPLIT_SEED,
    minimum_holdout_fraction: float = HOLDOUT_FRACTION,
) -> EpisodeSplit:
    """Select the first seeded permutation entries as whole held-out episodes."""

    if num_episodes < 2:
        raise ValueError("episode splitting requires at least two episodes")
    if not 0.0 < minimum_holdout_fraction < 1.0:
        raise ValueError("minimum holdout fraction must be between zero and one")
    holdout_count = math.ceil(num_episodes * minimum_holdout_fraction)
    holdout = tuple(
        sorted(
            int(index)
            for index in np.random.default_rng(seed).permutation(num_episodes)[:holdout_count]
        )
    )
    holdout_set = set(holdout)
    train = tuple(index for index in range(num_episodes) if index not in holdout_set)
    return EpisodeSplit(
        seed=seed,
        holdout_fraction=len(holdout) / num_episodes,
        train_episodes=train,
        holdout_episodes=holdout,
    )


def _stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _semantic_value(value: object) -> object:
    """Convert materialized bridge state into deterministic JSON evidence."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("bridge semantic state contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _semantic_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _semantic_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(_semantic_value(key)): _semantic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    module_name = type(value).__module__
    if module_name.startswith("torch") and hasattr(value, "detach"):
        tensor = value.detach().cpu()
        dtype = str(tensor.dtype)
        if dtype == "torch.bfloat16":
            array = tensor.float().numpy()
        else:
            array = tensor.numpy()
        return {"torch_dtype": dtype, **_semantic_value(array)}
    if module_name.startswith("pandas") and hasattr(value, "to_dict"):
        return _semantic_value(value.to_dict(orient="split"))
    if module_name.startswith("datasets") and hasattr(value, "column_names"):
        return {
            "columns": list(value.column_names),
            "values": {
                name: _semantic_value(list(value[name])) for name in value.column_names
            },
        }
    if callable(value):
        callable_name = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", type(value).__qualname__),
        )
        return {
            "callable": f"{getattr(value, '__module__', type(value).__module__)}.{callable_name}"
        }
    raise TypeError(f"unsupported bridge semantic value: {type(value)!r}")


def _materialized_hf_dataset_evidence(value: object) -> dict[str, object]:
    """Hash the Arrow schema and every materialized behavior-row value."""

    import pyarrow as pa

    data = getattr(value, "data", None)
    table = getattr(data, "table", None)
    if not isinstance(table, pa.Table):
        raise RuntimeError("training bridge has no materialized Arrow behavior table")
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    payload = sink.getvalue().to_pybytes()
    schema_payload = table.schema.serialize().to_pybytes()
    return {
        "row_count": table.num_rows,
        "column_count": table.num_columns,
        "column_names": list(table.column_names),
        "schema_sha256": hashlib.sha256(schema_payload).hexdigest(),
        "rows_sha256": hashlib.sha256(payload).hexdigest(),
        "serialized_bytes": len(payload),
    }


@dataclass(frozen=True)
class TrainStatistics:
    """Train-row statistics plus a complete processor-ready stats mapping."""

    episode_indices: tuple[int, ...]
    row_count: int
    excluded_row_count: int
    total_row_count: int
    stats: Mapping[str, Mapping[str, list[float] | list[int]]]
    processor_stats: Mapping[str, Mapping[str, object]]
    sha256: str


def _feature_statistics(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"statistics require a nonempty [rows, features] array, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("statistics source contains non-finite values")
    return {
        "count": [int(values.shape[0])],
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
    }


def compute_train_statistics(
    dataset_root: str | Path,
    train_episodes: tuple[int, ...],
) -> TrainStatistics:
    """Recompute state/action statistics from only the selected Parquet rows."""

    import pyarrow.parquet as parquet

    dataset_root = Path(dataset_root)
    data_paths = tuple(sorted((dataset_root / "data").glob("chunk-*/file-*.parquet")))
    stats_path = dataset_root / "meta" / "stats.json"
    if not data_paths or not all(path.is_file() for path in data_paths) or not stats_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset is incomplete under {dataset_root}")
    episodes = tuple(sorted(int(index) for index in train_episodes))
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("training episodes must be a nonempty unique tuple")

    import pyarrow as pa

    table = pa.concat_tables(
        [
            parquet.read_table(
                path,
                columns=["episode_index", "observation.state", "action"],
            )
            for path in data_paths
        ]
    )
    episode_values = np.asarray(table["episode_index"], dtype=np.int64)
    selected = np.isin(episode_values, np.asarray(episodes, dtype=np.int64))
    row_count = int(selected.sum())
    feature_stats: dict[str, dict[str, list[float] | list[int]]] = {}
    for key in ("observation.state", "action"):
        values = np.asarray(table[key].to_pylist(), dtype=np.float64)
        feature_stats[key] = _feature_statistics(values[selected])

    processor_stats = deepcopy(json.loads(stats_path.read_text(encoding="utf-8")))
    processor_stats.update(deepcopy(feature_stats))
    return TrainStatistics(
        episode_indices=episodes,
        row_count=row_count,
        excluded_row_count=int(table.num_rows - row_count),
        total_row_count=int(table.num_rows),
        stats=feature_stats,
        processor_stats=processor_stats,
        sha256=_stable_json_sha256(feature_stats),
    )


@dataclass(frozen=True)
class HeldOutCaseSpec:
    """One immutable evaluation identity from an unseen episode."""

    episode: int
    frame_index: int
    absolute_index: int


def make_heldout_case_specs(
    dataset_root: str | Path,
    heldout_episodes: tuple[int, ...],
    *,
    samples_per_episode: int = 7,
    chunk_size: int = 50,
) -> tuple[HeldOutCaseSpec, ...]:
    """Spread full-action-chunk evaluation cases through every held-out episode."""

    import pyarrow.parquet as parquet

    if samples_per_episode <= 0 or chunk_size <= 0:
        raise ValueError("evaluation sample count and chunk size must be positive")
    data_path = Path(dataset_root) / "data" / "chunk-000" / "file-000.parquet"
    table = parquet.read_table(
        data_path,
        columns=["episode_index", "frame_index", "index"],
    )
    episode_column = np.asarray(table["episode_index"], dtype=np.int64)
    frame_column = np.asarray(table["frame_index"], dtype=np.int64)
    index_column = np.asarray(table["index"], dtype=np.int64)
    specs: list[HeldOutCaseSpec] = []
    for episode in heldout_episodes:
        selected = episode_column == episode
        frames = frame_column[selected]
        absolute = index_column[selected]
        if len(frames) < chunk_size:
            raise ValueError(f"held-out episode {episode} has only {len(frames)} frames")
        if not np.array_equal(frames, np.arange(len(frames), dtype=np.int64)):
            raise ValueError(f"held-out episode {episode} frame identities are not contiguous")
        positions = np.linspace(
            0,
            len(frames) - chunk_size,
            samples_per_episode,
            dtype=np.int64,
        )
        if len(set(int(position) for position in positions)) != samples_per_episode:
            raise ValueError(f"held-out episode {episode} cannot supply unique evaluation cases")
        specs.extend(
            HeldOutCaseSpec(
                episode=int(episode),
                frame_index=int(frames[position]),
                absolute_index=int(absolute[position]),
            )
            for position in positions
        )
    return tuple(specs)


def _to_numpy(value: Any) -> np.ndarray:
    value = value.detach().cpu()
    if str(value.dtype) == "torch.bfloat16":
        value = value.float()
    return np.ascontiguousarray(value.numpy()).copy()


@dataclass(frozen=True)
class BridgeBatch:
    """One LeRobot-prepared microbatch with no Torch object crossing the boundary."""

    episode: int
    frame_index: int
    absolute_index: int
    task: str
    physical_action_dim: int
    pixel_values: np.ndarray
    pixel_attention_mask: np.ndarray
    input_ids: np.ndarray
    text_attention_mask: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    action_is_pad: np.ndarray
    raw_state: np.ndarray
    raw_actions: np.ndarray
    observation: Mapping[str, np.ndarray | str]


class TrainingDataBridge:
    """Pinned LeRobot loader and processors feeding owned NumPy microbatches."""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        episodes: tuple[int, ...],
        sampler_seed: int = SAMPLER_SEED,
        stats: Mapping[str, Mapping[str, object]] | None = None,
        dataset_id: str = DATASET_ID,
        dataset_root: str | Path | None = None,
        dataset_revision: str | None = DATASET_REVISION,
    ) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.datasets import EpisodeAwareSampler
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.utils.collate import lerobot_collate_fn

        cache_dir = Path(cache_dir)
        dataset_root = (
            cache_dir / "datasets" / "svla_so101_pickplace"
            if dataset_root is None
            else Path(dataset_root)
        )
        metadata = LeRobotDatasetMetadata(
            dataset_id,
            root=dataset_root,
            revision=dataset_revision,
        )
        if not episodes or any(index < 0 or index >= metadata.total_episodes for index in episodes):
            raise ValueError(f"bridge episodes are outside the pinned dataset: {episodes}")
        config = SmolVLAConfig.from_pretrained(
            CHECKPOINT_ID,
            revision=CHECKPOINT_REVISION,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        config.device = "cpu"
        tokenizer_snapshot = Path(
            snapshot_download(
                BASE_VLM_ID,
                revision=BASE_VLM_REVISION,
                cache_dir=cache_dir,
                allow_patterns=list(_BASE_VLM_PROCESSOR_FILES),
                local_files_only=True,
            )
        )
        delta_timestamps = resolve_delta_timestamps(config, metadata)
        dataset = LeRobotDataset(
            dataset_id,
            root=dataset_root,
            episodes=list(episodes),
            delta_timestamps=delta_timestamps,
            revision=dataset_revision,
            video_backend="pyav",
            return_uint8=True,
        )
        active_stats = dataset.meta.stats if stats is None else stats
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=CHECKPOINT_ID,
            pretrained_revision=CHECKPOINT_REVISION,
            preprocessor_overrides={
                "device_processor": {"device": "cpu"},
                "normalizer_processor": {
                    "features": {
                        **config.input_features,
                        **config.output_features,
                    },
                    "norm_map": config.normalization_mapping,
                    "stats": active_stats,
                },
                "rename_observations_processor": {"rename_map": _CAMERA_RENAME_MAP},
                "tokenizer_processor": {"tokenizer_name": str(tokenizer_snapshot)},
            },
        )
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=list(episodes),
            shuffle=True,
            seed=sampler_seed,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        collate = lerobot_collate_fn if dataset.meta.has_language_columns else None
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate,
        )
        self.cache_dir = cache_dir
        self.episodes = tuple(episodes)
        self.sampler_seed = sampler_seed
        self.config = config
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.loader = loader
        self._iterator: Iterator[dict[str, Any]] = iter(loader)
        self._samples_consumed = 0

    def semantic_evidence(self) -> dict[str, object]:
        """Fingerprint every materialized object that controls bridge semantics."""

        reader = getattr(self.dataset, "reader", None)
        reader_hf_dataset = getattr(reader, "hf_dataset", None)
        if reader_hf_dataset is None or self.dataset.hf_dataset is not reader_hf_dataset:
            raise RuntimeError("training bridge materialized dataset binding changed")
        if self.loader.dataset is not self.dataset:
            raise RuntimeError("training bridge loader dataset binding changed")
        if self.loader.sampler is not self.sampler:
            raise RuntimeError("training bridge loader sampler binding changed")
        if getattr(self.loader.batch_sampler, "sampler", None) is not self.sampler:
            raise RuntimeError("training bridge batch sampler binding changed")
        if getattr(self._iterator, "_dataset", None) is not self.dataset:
            raise RuntimeError("training bridge iterator dataset binding changed")
        if getattr(self._iterator, "_index_sampler", None) is not getattr(
            self.loader, "_index_sampler", None
        ):
            raise RuntimeError("training bridge iterator sampler binding changed")
        if self._samples_consumed != 0 or getattr(self._iterator, "_num_yielded", None) != 0:
            raise RuntimeError("training bridge semantic evidence requires a fresh iterator")
        tokenizer_steps = [
            step
            for step in self.preprocessor.steps
            if getattr(step.__class__, "_registry_name", None)
            == "tokenizer_processor"
        ]
        if len(tokenizer_steps) != 1:
            raise RuntimeError("training bridge tokenizer topology changed")
        tokenizer = tokenizer_steps[0].input_tokenizer
        if tokenizer is None or not hasattr(tokenizer, "backend_tokenizer"):
            raise RuntimeError("training bridge has no materialized tokenizer")
        tokenizer_backend = tokenizer.backend_tokenizer.to_str().encode("utf-8")
        processor_steps = []
        for step in self.preprocessor.steps:
            item: dict[str, object] = {
                "type": f"{type(step).__module__}.{type(step).__qualname__}",
                "registry_name": getattr(step.__class__, "_registry_name", None),
                "config": _semantic_value(step.get_config()),
            }
            if hasattr(step, "_tensor_stats"):
                item["tensor_stats"] = _semantic_value(step._tensor_stats)
            processor_steps.append(item)
        components = {
            "bridge": {
                "cache_dir": str(self.cache_dir),
                "episodes": list(self.episodes),
                "sampler_seed": self.sampler_seed,
            },
            "config": _semantic_value(vars(self.config)),
            "metadata": {
                "repo_id": self.dataset.meta.repo_id,
                "revision": self.dataset.meta.revision,
                "info": _semantic_value(self.dataset.meta.info),
                "episodes": _semantic_value(self.dataset.meta.episodes),
                "tasks": _semantic_value(self.dataset.meta.tasks),
                "total_episodes": self.dataset.meta.total_episodes,
                "camera_keys": list(self.dataset.meta.camera_keys),
                "has_language_columns": self.dataset.meta.has_language_columns,
            },
            "dataset": {
                "repo_id": self.dataset.repo_id,
                "revision": self.dataset.revision,
                "episodes": list(self.dataset.episodes),
                "delta_timestamps": _semantic_value(self.dataset.delta_timestamps),
                "absolute_to_relative_idx": _semantic_value(
                    self.dataset.absolute_to_relative_idx
                ),
                "video_backend": self.dataset._video_backend,
                "return_uint8": self.dataset._return_uint8,
                "materialized_rows": _materialized_hf_dataset_evidence(
                    reader_hf_dataset
                ),
            },
            "sampler": _semantic_value(vars(self.sampler)),
            "preprocessor": {
                "name": self.preprocessor.name,
                "steps": processor_steps,
                "tokenizer_type": (
                    f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
                ),
                "tokenizer_backend_sha256": hashlib.sha256(
                    tokenizer_backend
                ).hexdigest(),
                "tokenizer_special_tokens": _semantic_value(
                    tokenizer.special_tokens_map
                ),
                "tokenizer_vocab_size": len(tokenizer),
            },
            "loader": {
                "batch_size": self.loader.batch_size,
                "num_workers": self.loader.num_workers,
                "drop_last": self.loader.drop_last,
                "pin_memory": self.loader.pin_memory,
                "sampler_is_bound": self.loader.sampler is self.sampler,
                "collate_fn": _semantic_value(self.loader.collate_fn),
            },
        }
        component_hashes = {
            name: _stable_json_sha256(value)
            for name, value in sorted(components.items())
        }
        return {
            "format_version": 1,
            "sha256": _stable_json_sha256(component_hashes),
            "components": component_hashes,
        }

    def _prepare(self, batch: dict[str, Any]) -> BridgeBatch:
        import torch
        from lerobot.policies.common.vla_utils import pad_vector, resize_with_pad

        raw_state = batch["observation.state"].detach().cpu().float().clone()
        raw_actions = batch["action"].detach().cpu().float().clone()
        raw_observation: dict[str, np.ndarray | str] = {}
        for source_key, target_key in _CAMERA_RENAME_MAP.items():
            if source_key in batch:
                camera = batch[source_key]
                raw_observation[target_key] = _to_numpy(camera[0])
        raw_observation["observation.state"] = _to_numpy(raw_state[0, -1] if raw_state.ndim > 2 else raw_state[0])

        for camera_key in self.dataset.meta.camera_keys:
            if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0
        processed = self.preprocessor(batch)

        images = []
        masks = []
        present = [key for key in self.config.image_features if key in processed]
        missing = [key for key in self.config.image_features if key not in processed]
        if not present:
            raise ValueError("processed bridge batch has no configured image feature")
        image = None
        mask = None
        for key in present:
            image = processed[key][:, -1] if processed[key].ndim == 5 else processed[key]
            if self.config.resize_imgs_with_padding is not None:
                image = resize_with_pad(
                    image,
                    self.config.resize_imgs_with_padding[1],
                    self.config.resize_imgs_with_padding[0],
                    pad_value=0,
                )
            image = image * 2.0 - 1.0
            if f"{key}_padding_mask" in processed:
                mask = processed[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
            images.append(image)
            masks.append(mask)
        assert image is not None and mask is not None
        for empty_index in range(len(missing)):
            if empty_index >= self.config.empty_cameras:
                break
            images.append(torch.ones_like(image) * -1)
            masks.append(torch.zeros_like(mask))

        state = processed["observation.state"]
        state = state[:, -1] if state.ndim > 2 else state
        state = pad_vector(state, self.config.max_state_dim)
        actions = pad_vector(processed["action"], self.config.max_action_dim)
        task_value = processed["task"]
        if not isinstance(task_value, list) or len(task_value) != 1 or not isinstance(task_value[0], str):
            raise ValueError(f"unexpected bridge task value: {task_value!r}")
        raw_observation["task"] = task_value[0]

        episode = int(batch["episode_index"].reshape(-1)[0])
        frame_index = int(batch["frame_index"].reshape(-1)[0])
        absolute_index = int(batch["index"].reshape(-1)[0])
        return BridgeBatch(
            episode=episode,
            frame_index=frame_index,
            absolute_index=absolute_index,
            task=task_value[0],
            physical_action_dim=int(self.config.action_feature.shape[0]),
            pixel_values=_to_numpy(torch.cat(images, dim=0).float()),
            pixel_attention_mask=_to_numpy(
                torch.cat([value.reshape(-1, 1) for value in masks], dim=0).bool()
            ),
            input_ids=_to_numpy(processed["observation.language.tokens"]),
            text_attention_mask=_to_numpy(
                processed["observation.language.attention_mask"].bool()
            ),
            state=_to_numpy(state.float()),
            actions=_to_numpy(actions.float()),
            action_is_pad=_to_numpy(processed["action_is_pad"].bool()),
            raw_state=_to_numpy(raw_state.float()),
            raw_actions=_to_numpy(raw_actions.float()),
            observation=raw_observation,
        )

    def frame(self, *, episode: int, frame_index: int) -> BridgeBatch:
        """Prepare one exact episode-relative frame without changing sampler state."""

        if episode not in self.episodes:
            raise ValueError(f"episode {episode} is not part of this bridge")
        starts = self.dataset.meta.episodes["dataset_from_index"]
        ends = self.dataset.meta.episodes["dataset_to_index"]
        absolute_index = int(starts[episode]) + frame_index
        if frame_index < 0 or absolute_index >= int(ends[episode]):
            raise IndexError(f"frame {frame_index} is outside episode {episode}")
        relative_index = self.dataset.absolute_to_relative_idx[absolute_index]
        from torch.utils.data._utils.collate import default_collate

        return self._prepare(default_collate([self.dataset[relative_index]]))

    def next_batch(self) -> BridgeBatch:
        """Return the next reproducibly shuffled one-sample NumPy microbatch."""

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            batch = next(self._iterator)
        if batch is None:
            raise RuntimeError("LeRobot data loader returned an empty batch")
        prepared = self._prepare(batch)
        self._samples_consumed += 1
        return prepared

    def state_dict(self) -> dict[str, object]:
        """Return enough state to reproduce the exact next sampler position."""

        num_samples = len(self.sampler)
        epoch, start_index = divmod(self._samples_consumed, num_samples)
        return {
            "format_version": 1,
            "samples_consumed": self._samples_consumed,
            "num_samples": num_samples,
            "epoch": epoch,
            "start_index": start_index,
            "sampler_seed": self.sampler_seed,
            "episodes": list(self.episodes),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a validated sampler position without decoding skipped frames."""

        required = {
            "format_version",
            "samples_consumed",
            "num_samples",
            "epoch",
            "start_index",
            "sampler_seed",
            "episodes",
        }
        if set(state) != required:
            raise ValueError(
                f"bridge state fields differ; missing={sorted(required - set(state))}, "
                f"unexpected={sorted(set(state) - required)}"
            )
        if state["format_version"] != 1:
            raise ValueError(f"unsupported bridge state version {state['format_version']!r}")
        num_samples = len(self.sampler)
        if state["num_samples"] != num_samples:
            raise ValueError(
                f"bridge sample count changed: checkpoint={state['num_samples']}, current={num_samples}"
            )
        if state["sampler_seed"] != self.sampler_seed:
            raise ValueError("bridge sampler seed differs from the checkpoint")
        if tuple(state["episodes"]) != self.episodes:
            raise ValueError("bridge episode split differs from the checkpoint")
        samples_consumed = int(state["samples_consumed"])
        if samples_consumed < 0:
            raise ValueError("bridge samples consumed must be nonnegative")
        expected_epoch, expected_start = divmod(samples_consumed, num_samples)
        if int(state["epoch"]) != expected_epoch or int(state["start_index"]) != expected_start:
            raise ValueError("bridge epoch/offset do not match samples consumed")
        self.sampler.load_state_dict(
            {"epoch": expected_epoch, "start_index": expected_start}
        )
        self._iterator = iter(self.loader)
        self._samples_consumed = samples_consumed
