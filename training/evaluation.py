"""Frozen held-out MAE, Torch round trip, and stats-active parity for T3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable, Mapping

import mlx.core as mx
import numpy as np

from reference.discovery import DATASET_ID, DATASET_REVISION
from training.data import TrainingArtifact, TrainingArtifactWriter
from training.dataset import (
    SPLIT_SEED,
    TrainingDataBridge,
    compute_train_statistics,
    make_episode_split,
    make_heldout_case_specs,
)
from training.export import resolve_base_checkpoint
from training.finetune import write_run_state
from training.preprocessing import (
    StatsAwareSmolVLAPreprocessor,
    load_stats_aware_policy,
)


MAE_IMPROVEMENT_RATIO_MAXIMUM = 0.9
TORCH_MLX_MAE_RATIO_MINIMUM = 0.95
TORCH_MLX_MAE_RATIO_MAXIMUM = 1.05
INFERENCE_MAX_ABSOLUTE_TOLERANCE = 5e-3
EVALUATION_SAMPLE_COUNT = 56
EVALUATION_NOISE_SEED = 20_260_902
_SAMPLES_PER_EPISODE = 7
_EXPECTED_TENSORS_PER_CASE = 5


@dataclass(frozen=True)
class EvaluationCase:
    """One immutable raw held-out observation, target, and flow draw."""

    ordinal: int
    episode: int
    frame_index: int
    absolute_index: int
    task: str
    camera1: np.ndarray
    camera2: np.ndarray
    state: np.ndarray
    target_action: np.ndarray
    noise: np.ndarray

    @property
    def observation(self) -> dict[str, np.ndarray | str]:
        return {
            "observation.images.camera1": self.camera1,
            "observation.images.camera2": self.camera2,
            "observation.state": self.state,
            "task": self.task,
        }


def capture_evaluation_artifact(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    split_seed: int = SPLIT_SEED,
    noise_seed: int = EVALUATION_NOISE_SEED,
) -> dict[str, object]:
    """Capture 56 unseen observations/targets/noise arrays before training."""

    cache_dir = Path(cache_dir)
    split = make_episode_split(num_episodes=50, seed=split_seed)
    dataset_root = cache_dir / "datasets" / "svla_so101_pickplace"
    stats = compute_train_statistics(dataset_root, split.train_episodes)
    specs = make_heldout_case_specs(
        dataset_root,
        split.holdout_episodes,
        samples_per_episode=_SAMPLES_PER_EPISODE,
        chunk_size=50,
    )
    if len(specs) != EVALUATION_SAMPLE_COUNT:
        raise RuntimeError(f"held-out case count changed: {len(specs)}")
    bridge = TrainingDataBridge(
        cache_dir=cache_dir,
        episodes=split.holdout_episodes,
        sampler_seed=split_seed,
        stats=stats.processor_stats,
    )
    generator = np.random.default_rng(noise_seed)
    writer = TrainingArtifactWriter(Path(output_dir))
    case_metadata = []
    for ordinal, spec in enumerate(specs):
        batch = bridge.frame(episode=spec.episode, frame_index=spec.frame_index)
        if batch.absolute_index != spec.absolute_index:
            raise RuntimeError("evaluation bridge identity differs from frozen specification")
        camera1 = np.asarray(batch.observation["observation.images.camera1"])
        camera2 = np.asarray(batch.observation["observation.images.camera2"])
        state = np.asarray(batch.observation["observation.state"], dtype=np.float32)
        target = np.asarray(batch.raw_actions[0, 0], dtype=np.float32)
        noise = generator.standard_normal((1, 50, 32), dtype=np.float32)
        prefix = f"cases/{ordinal:03d}"
        writer.add(f"{prefix}/camera1", camera1)
        writer.add(f"{prefix}/camera2", camera2)
        writer.add(f"{prefix}/state", state)
        writer.add(f"{prefix}/target_action", target)
        writer.add(f"{prefix}/noise", noise)
        case_metadata.append(
            {
                "ordinal": ordinal,
                "episode": spec.episode,
                "frame_index": spec.frame_index,
                "absolute_index": spec.absolute_index,
                "task": batch.task,
            }
        )
    return writer.finalize(
        {
            "format_version": 1,
            "artifact_type": "smolvla-lora-heldout-evaluation",
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "split_seed": split_seed,
            "noise_seed": noise_seed,
            "sample_count": len(specs),
            "samples_per_episode": _SAMPLES_PER_EPISODE,
            "train_episodes": list(split.train_episodes),
            "heldout_episodes": list(split.holdout_episodes),
            "train_statistics_sha256": stats.sha256,
            "metric": "physical first-action MAE over six action dimensions",
            "cases": case_metadata,
        }
    )


def load_evaluation_cases(root: str | Path) -> tuple[EvaluationCase, ...]:
    """Strictly verify and load every frozen held-out case."""

    artifact = TrainingArtifact(Path(root))
    names = artifact.verify_all()
    metadata = artifact.metadata
    required = {
        "artifact_type": "smolvla-lora-heldout-evaluation",
        "sample_count": EVALUATION_SAMPLE_COUNT,
        "samples_per_episode": _SAMPLES_PER_EPISODE,
        "noise_seed": EVALUATION_NOISE_SEED,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"evaluation artifact {key} changed: {metadata.get(key)!r}")
    if len(names) != EVALUATION_SAMPLE_COUNT * _EXPECTED_TENSORS_PER_CASE:
        raise ValueError("evaluation artifact tensor count changed")
    raw_cases = metadata.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != EVALUATION_SAMPLE_COUNT:
        raise ValueError("evaluation artifact case metadata is incomplete")
    cases = []
    for ordinal, raw in enumerate(raw_cases):
        if not isinstance(raw, dict) or raw.get("ordinal") != ordinal:
            raise ValueError(f"evaluation case metadata changed at ordinal {ordinal}")
        task = raw.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError(f"evaluation task is invalid at ordinal {ordinal}")
        prefix = f"cases/{ordinal:03d}"
        case = EvaluationCase(
            ordinal=ordinal,
            episode=int(raw["episode"]),
            frame_index=int(raw["frame_index"]),
            absolute_index=int(raw["absolute_index"]),
            task=task,
            camera1=artifact.load(f"{prefix}/camera1"),
            camera2=artifact.load(f"{prefix}/camera2"),
            state=artifact.load(f"{prefix}/state").astype(np.float32, copy=False),
            target_action=artifact.load(f"{prefix}/target_action").astype(np.float32, copy=False),
            noise=artifact.load(f"{prefix}/noise").astype(np.float32, copy=False),
        )
        if case.camera1.dtype != np.uint8 or case.camera2.dtype != np.uint8:
            raise ValueError(f"evaluation cameras must remain uint8 at ordinal {ordinal}")
        if case.state.shape != (6,) or case.target_action.shape != (6,):
            raise ValueError(f"evaluation physical vector shape changed at ordinal {ordinal}")
        if case.noise.shape != (1, 50, 32) or not np.isfinite(case.noise).all():
            raise ValueError(f"evaluation noise is invalid at ordinal {ordinal}")
        cases.append(case)
    if len({case.absolute_index for case in cases}) != EVALUATION_SAMPLE_COUNT:
        raise ValueError("evaluation case identities are not unique")
    return tuple(cases)


def absolute_error(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """Return float64-accumulated absolute-error sum and element count."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shapes differ: {prediction.shape} != {target.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("MAE inputs contain non-finite values")
    error = np.abs(prediction - target)
    return float(np.sum(error, dtype=np.float64)), int(error.size)


@dataclass(frozen=True)
class MAEEvaluation:
    """Complete per-case physical-action MAE evidence for one policy."""

    framework: str
    device: str
    dtype: str
    sample_count: int
    element_count: int
    mae: float
    absolute_error_sum: float
    samples: tuple[Mapping[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["samples"] = [dict(sample) for sample in self.samples]
        return value


def load_base_stats_policy(
    *,
    cache_dir: str | Path,
    native_cache: str | Path,
):
    """Load the immutable fp32 base with the frozen train-only statistics."""

    from smolvla_mlx.policy import SmolVLAMLX

    cache_dir = Path(cache_dir)
    split = make_episode_split(num_episodes=50, seed=SPLIT_SEED)
    stats = compute_train_statistics(
        cache_dir / "datasets" / "svla_so101_pickplace",
        split.train_episodes,
    )
    checkpoint = resolve_base_checkpoint(cache_dir)
    with mx.stream(mx.cpu):
        policy = SmolVLAMLX.from_pretrained(
            checkpoint,
            cache_dir=native_cache,
            dtype=mx.float32,
        )
        policy.preprocessor = StatsAwareSmolVLAPreprocessor(
            base=policy.preprocessor,
            state_mean=stats.stats["observation.state"]["mean"],
            state_std=stats.stats["observation.state"]["std"],
            action_mean=stats.stats["action"]["mean"],
            action_std=stats.stats["action"]["std"],
        )
    return policy


def evaluate_mlx_policy(
    policy,
    cases: tuple[EvaluationCase, ...],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> MAEEvaluation:
    """Score a native fp32 policy on identical physical first-action targets."""

    error_sum = 0.0
    element_count = 0
    records = []
    with mx.stream(mx.cpu):
        if mx.default_device() != mx.cpu:
            raise RuntimeError("held-out MLX evaluation must run on the fp32 CPU compatibility lane")
        for index, case in enumerate(cases):
            normalized = policy.predict_action_chunk(
                case.observation,
                noise=mx.array(case.noise).astype(mx.float32),
            )
            physical = policy.preprocessor.unnormalize_actions(normalized)
            mx.eval(physical)
            prediction = np.asarray(physical.astype(mx.float32))[0, 0]
            case_sum, case_count = absolute_error(prediction, case.target_action)
            error_sum += case_sum
            element_count += case_count
            records.append(
                {
                    "ordinal": case.ordinal,
                    "episode": case.episode,
                    "frame_index": case.frame_index,
                    "absolute_index": case.absolute_index,
                    "absolute_error_sum": case_sum,
                    "element_count": case_count,
                }
            )
            if progress is not None:
                progress(index + 1, len(cases))
    if element_count == 0:
        raise RuntimeError("held-out MLX evaluation produced no metric elements")
    return MAEEvaluation(
        framework="mlx",
        device="Device(cpu, 0)",
        dtype="float32",
        sample_count=len(cases),
        element_count=element_count,
        mae=error_sum / element_count,
        absolute_error_sum=error_sum,
        samples=tuple(records),
    )


def _torch_observation(case: EvaluationCase):
    import torch

    return {
        "observation.images.camera1": torch.from_numpy(case.camera1.copy()).float() / 255.0,
        "observation.images.camera2": torch.from_numpy(case.camera2.copy()).float() / 255.0,
        "observation.state": torch.from_numpy(case.state.copy()).float(),
        "task": case.task,
    }


def evaluate_torch_export(
    reference,
    cases: tuple[EvaluationCase, ...],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> MAEEvaluation:
    """Score the strict Torch export on the exact same cases and flow noise."""

    import torch

    error_sum = 0.0
    element_count = 0
    records = []
    for index, case in enumerate(cases):
        batch = reference.preprocessor(_torch_observation(case))
        noise = torch.from_numpy(case.noise.copy()).float()
        with torch.inference_mode():
            normalized = reference.policy.predict_action_chunk(batch, noise=noise)
            physical = reference.postprocessor(normalized)
        prediction = physical.detach().cpu().float().numpy()[0, 0]
        case_sum, case_count = absolute_error(prediction, case.target_action)
        error_sum += case_sum
        element_count += case_count
        records.append(
            {
                "ordinal": case.ordinal,
                "episode": case.episode,
                "frame_index": case.frame_index,
                "absolute_index": case.absolute_index,
                "absolute_error_sum": case_sum,
                "element_count": case_count,
            }
        )
        if progress is not None:
            progress(index + 1, len(cases))
    return MAEEvaluation(
        framework="torch",
        device=str(reference.device),
        dtype=str(reference.dtype).removeprefix("torch."),
        sample_count=len(cases),
        element_count=element_count,
        mae=error_sum / element_count,
        absolute_error_sum=error_sum,
        samples=tuple(records),
    )


@dataclass(frozen=True)
class StatsActiveParity:
    """P0 preprocessing, P1 normalized-flow, and P2 stats-active action parity."""

    sample_count: int
    preprocessing_max_abs: float
    normalized_action_max_abs: float
    physical_action_max_abs: float
    physical_action_standardized_max_abs: float
    gate_max_abs: float
    passed: bool
    samples: tuple[Mapping[str, object], ...]


def run_stats_active_parity(
    mlx_policy,
    reference,
    cases: tuple[EvaluationCase, ...],
) -> StatsActiveParity:
    """Run one fixed case per held-out episode through the three parity levels."""

    import torch
    from lerobot.policies.common.vla_utils import resize_with_pad

    selected = []
    seen = set()
    for case in cases:
        if case.episode not in seen:
            selected.append(case)
            seen.add(case.episode)
    records = []
    p0_max = 0.0
    p1_max = 0.0
    p2_raw_max = 0.0
    p2_standardized_max = 0.0
    action_std = np.asarray(mlx_policy.preprocessor.action_std, dtype=np.float32)
    with mx.stream(mx.cpu):
        for case in selected:
            observation = _torch_observation(case)
            torch_batch = reference.preprocessor(observation)
            images = []
            masks = []
            for key in reference.policy.config.image_features:
                if key not in torch_batch:
                    continue
                image = torch_batch[key]
                image = image[:, -1] if image.ndim == 5 else image
                width, height = reference.policy.config.resize_imgs_with_padding
                image = resize_with_pad(image, height, width, pad_value=0)
                images.append(image * 2.0 - 1.0)
                masks.append(torch.ones(image.shape[0], dtype=torch.bool))
            torch_pixels = torch.cat(images, dim=0).cpu().float().numpy()
            torch_masks = torch.cat([mask.reshape(-1, 1) for mask in masks], dim=0).numpy()
            torch_state = torch_batch["observation.state"].cpu().float().numpy()
            torch_ids = torch_batch["observation.language.tokens"].cpu().numpy()
            torch_text_mask = torch_batch["observation.language.attention_mask"].cpu().numpy()

            native = mlx_policy.preprocessor(case.observation)
            mx.eval(
                native.pixel_values,
                native.pixel_attention_mask,
                native.input_ids,
                native.text_attention_mask,
                native.state,
            )
            if not np.array_equal(np.asarray(native.pixel_attention_mask), torch_masks):
                raise RuntimeError("stats-active parity image masks differ")
            if not np.array_equal(np.asarray(native.input_ids), torch_ids):
                raise RuntimeError("stats-active parity token IDs differ")
            if not np.array_equal(np.asarray(native.text_attention_mask), torch_text_mask):
                raise RuntimeError("stats-active parity token masks differ")
            preprocessing_max = max(
                float(np.max(np.abs(np.asarray(native.pixel_values) - torch_pixels))),
                float(np.max(np.abs(np.asarray(native.state) - torch_state))),
            )

            noise = torch.from_numpy(case.noise.copy()).float()
            with torch.inference_mode():
                torch_normalized = reference.policy.predict_action_chunk(torch_batch, noise=noise)
                torch_physical = reference.postprocessor(torch_normalized)
            mlx_normalized = mlx_policy.predict_action_chunk(
                case.observation,
                noise=mx.array(case.noise).astype(mx.float32),
            )
            mlx_physical = mlx_policy.preprocessor.unnormalize_actions(mlx_normalized)
            mx.eval(mlx_normalized, mlx_physical)
            torch_normalized_array = torch_normalized.cpu().float().numpy()
            torch_physical_array = torch_physical.cpu().float().numpy()
            mlx_normalized_array = np.asarray(mlx_normalized.astype(mx.float32))
            mlx_physical_array = np.asarray(mlx_physical.astype(mx.float32))
            normalized_max = float(
                np.max(np.abs(mlx_normalized_array - torch_normalized_array))
            )
            physical_difference = np.abs(mlx_physical_array - torch_physical_array)
            physical_max = float(np.max(physical_difference))
            standardized_max = float(np.max(physical_difference / action_std))
            p0_max = max(p0_max, preprocessing_max)
            p1_max = max(p1_max, normalized_max)
            p2_raw_max = max(p2_raw_max, physical_max)
            p2_standardized_max = max(p2_standardized_max, standardized_max)
            records.append(
                {
                    "episode": case.episode,
                    "frame_index": case.frame_index,
                    "preprocessing_max_abs": preprocessing_max,
                    "normalized_action_max_abs": normalized_max,
                    "physical_action_max_abs": physical_max,
                    "physical_action_standardized_max_abs": standardized_max,
                }
            )
    gate_max = max(p0_max, p1_max, p2_standardized_max)
    return StatsActiveParity(
        sample_count=len(selected),
        preprocessing_max_abs=p0_max,
        normalized_action_max_abs=p1_max,
        physical_action_max_abs=p2_raw_max,
        physical_action_standardized_max_abs=p2_standardized_max,
        gate_max_abs=gate_max,
        passed=gate_max <= INFERENCE_MAX_ABSOLUTE_TOLERANCE,
        samples=tuple(records),
    )


@dataclass(frozen=True)
class OutcomeGates:
    """All three independent immutable T3 outcome decisions."""

    passed: bool
    improvement_passed: bool
    roundtrip_passed: bool
    parity_passed: bool
    fine_to_base_ratio: float
    torch_to_mlx_ratio: float
    parity_max_abs: float


def evaluate_outcome_gates(
    *,
    base_mlx_mae: float,
    fine_mlx_mae: float,
    torch_mae: float,
    parity_max_abs: float,
) -> OutcomeGates:
    """Apply the brief's immutable thresholds without rounding or adjustment."""

    values = (base_mlx_mae, fine_mlx_mae, torch_mae, parity_max_abs)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("outcome metrics must be finite and nonnegative")
    if base_mlx_mae == 0 or fine_mlx_mae == 0:
        raise ValueError("outcome MAE ratios require nonzero MLX denominators")
    fine_to_base = fine_mlx_mae / base_mlx_mae
    torch_to_mlx = torch_mae / fine_mlx_mae
    improvement = fine_to_base <= MAE_IMPROVEMENT_RATIO_MAXIMUM
    roundtrip = TORCH_MLX_MAE_RATIO_MINIMUM <= torch_to_mlx <= TORCH_MLX_MAE_RATIO_MAXIMUM
    parity = parity_max_abs <= INFERENCE_MAX_ABSOLUTE_TOLERANCE
    return OutcomeGates(
        passed=improvement and roundtrip and parity,
        improvement_passed=improvement,
        roundtrip_passed=roundtrip,
        parity_passed=parity,
        fine_to_base_ratio=fine_to_base,
        torch_to_mlx_ratio=torch_to_mlx,
        parity_max_abs=parity_max_abs,
    )


def capture_and_evaluate_base(
    *,
    cache_dir: str | Path,
    native_cache: str | Path,
    evaluation_dir: str | Path,
    output_path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[MAEEvaluation, str]:
    """Freeze held-out inputs and baseline MAE before the training run exists."""

    metadata = capture_evaluation_artifact(
        cache_dir=cache_dir,
        output_dir=evaluation_dir,
    )
    cases = load_evaluation_cases(evaluation_dir)
    policy = load_base_stats_policy(cache_dir=cache_dir, native_cache=native_cache)
    evaluation = evaluate_mlx_policy(policy, cases, progress=progress)
    report = {
        "format_version": 1,
        "artifact_type": "smolvla-lora-base-heldout-evaluation",
        "evaluation_manifest_sha256": metadata["manifest_sha256"],
        "train_statistics_sha256": metadata["train_statistics_sha256"],
        "sample_count": evaluation.sample_count,
        "element_count": evaluation.element_count,
        "mlx_mae": evaluation.mae,
        "absolute_error_sum": evaluation.absolute_error_sum,
        "device": evaluation.device,
        "dtype": evaluation.dtype,
        "samples": [dict(sample) for sample in evaluation.samples],
    }
    digest = write_run_state(output_path, report)
    return evaluation, digest
