"""Stats-active base-checkpoint parity against its independent golden set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from reference.goldens import GoldenStore
from smolvla_mlx.policy import SmolVLAMLX
from smolvla_mlx.statistical import StatisticalResult


_ROOT = Path("tests/golden-stats-active")
_CHECKPOINT = Path("reference/artifacts/stats-active-base")


@dataclass(frozen=True)
class _Case:
    name: str
    task: str

    def array(self, name: str) -> np.ndarray:
        return GoldenStore(_ROOT).load(f"{self.name}/{name}")

    def observation(self) -> dict[str, object]:
        return {
            "observation.images.camera1": self.array("raw/camera1"),
            "observation.images.camera2": self.array("raw/camera2"),
            "observation.state": self.array("raw/state"),
            "task": self.task,
        }


@pytest.fixture(scope="session")
def stats_active_cases() -> tuple[_Case, ...]:
    metadata_path = _ROOT / "metadata.json"
    if not (_ROOT / "manifest.json").is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Stats-active goldens are absent; run scripts/make_stats_active_reference.py "
            "and scripts/make_goldens.py with --checkpoint reference/artifacts/stats-active-base"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["checkpoint"] == {
        "id": "lerobot/smolvla_base+svla_so101_pickplace-stats",
        "revision": (
            "c83c3163b8ca9b7e67c509fffd9121e66cb96205+"
            "f641879e22172be7e8161d5e6c1503c2d2feb657"
        ),
    }
    return tuple(_Case(name=sample["name"], task=sample["task"]) for sample in metadata["samples"])


@pytest.fixture(scope="module", params=("float32", "bfloat16"))
def stats_active_policy(request: pytest.FixtureRequest) -> tuple[SmolVLAMLX, str]:
    with mx.stream(mx.cpu):
        policy = SmolVLAMLX.from_pretrained(
            _CHECKPOINT,
            cache_dir=Path(".cache/smolvla_mlx") / f"stats-active-{request.param}",
            dtype=request.param,
            tokenizer_dir=next(
                Path(".cache/hf").glob(
                    "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/"
                    "snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
                )
            ),
        )
    return policy, request.param


def test_stats_active_preprocessing_matches_all_reference_cases(
    stats_active_cases: tuple[_Case, ...],
    stats_active_policy: tuple[SmolVLAMLX, str],
) -> None:
    policy, _ = stats_active_policy
    assert policy.config.state_normalization == "mean_std"
    assert policy.config.action_normalization == "mean_std"
    for case in stats_active_cases:
        actual = policy.preprocessor(case.observation())
        np.testing.assert_allclose(
            np.asarray(actual.pixel_values),
            case.array("preprocessed/pixel_values"),
            atol=1e-5,
            rtol=0,
        )
        np.testing.assert_allclose(
            np.asarray(actual.state),
            case.array("preprocessed/state_normalized"),
            atol=1e-6,
            rtol=0,
        )


def test_stats_active_normalized_actions_match_all_reference_cases(
    stats_active_cases: tuple[_Case, ...],
    stats_active_policy: tuple[SmolVLAMLX, str],
) -> None:
    policy, dtype = stats_active_policy
    tolerance = 5e-3 if dtype == "float32" else 5e-2
    with mx.stream(mx.cpu):
        for case in stats_active_cases:
            actual = policy.predict_action_chunk(
                case.observation(),
                noise=mx.array(case.array("noise"), dtype=mx.float32),
            )
            mx.eval(actual)
            difference = np.max(
                np.abs(
                    np.asarray(actual.astype(mx.float32))
                    - case.array("actions/normalized").astype(np.float32, copy=False)
                )
            )
            assert difference <= tolerance, (case.name, dtype, difference)


def test_stats_active_unnormalization_matches_reference(
    stats_active_cases: tuple[_Case, ...],
    stats_active_policy: tuple[SmolVLAMLX, str],
) -> None:
    policy, dtype = stats_active_policy
    tolerance = 5e-5 if dtype == "float32" else 5e-2
    for case in stats_active_cases:
        normalized = mx.array(case.array("actions/normalized"), dtype=mx.float32)
        actual = policy.preprocessor.unnormalize_actions(normalized)
        mx.eval(actual)
        np.testing.assert_allclose(
            np.asarray(actual),
            case.array("actions/unnormalized"),
            atol=tolerance,
            rtol=0,
        )


def test_stats_active_fifty_frame_statistical_gate() -> None:
    result = StatisticalResult.from_json(Path(".cache/statistical-stats-active.json"))
    assert result.sample_count == 50
    assert result.mlx_fp32_mae <= 1.05 * result.torch_fp32_mae
    assert result.mlx_bf16_mae <= 1.05 * result.torch_fp32_mae
