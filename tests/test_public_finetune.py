"""Pinned public multitask SmolVLA fine-tune conversion and parity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
import mlx.core as mx
import numpy as np
import pytest

from reference.goldens import GoldenStore
from scripts.make_public_finetune_goldens import MODEL_ID, MODEL_REVISION
from mlx_smolvla.policy import SmolVLAMLX
from mlx_smolvla.statistical import StatisticalResult


pytestmark = pytest.mark.slow


_ROOT = Path("tests/golden-public-finetune")
_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


@dataclass(frozen=True)
class _Case:
    name: str
    task: str
    camera_keys: tuple[str, ...]

    def array(self, name: str) -> np.ndarray:
        return GoldenStore(_ROOT).load(f"{self.name}/{name}")

    def observation(self) -> dict[str, object]:
        observation: dict[str, object] = {
            "observation.state": self.array("raw/state"),
            "task": self.task,
        }
        for camera_index, camera_key in enumerate(self.camera_keys, start=1):
            observation[camera_key] = self.array(f"raw/camera{camera_index}")
        return observation


@pytest.fixture(scope="session")
def public_finetune_checkpoint() -> Path:
    return Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=Path(".cache/hf"),
            allow_patterns=list(_FILES),
        )
    )


@pytest.fixture(scope="session")
def public_finetune_cases() -> tuple[_Case, ...]:
    metadata_path = _ROOT / "metadata.json"
    if not (_ROOT / "manifest.json").is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Public fine-tune goldens are absent; run scripts/make_public_finetune_goldens.py"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["checkpoint"] == {"id": MODEL_ID, "revision": MODEL_REVISION}
    return tuple(
        _Case(
            name=sample["name"],
            task=sample["task"],
            camera_keys=tuple(sample["camera_keys"]),
        )
        for sample in metadata["samples"]
    )


@pytest.fixture(scope="module", params=("float32", "bfloat16"))
def public_finetune_policy(
    request: pytest.FixtureRequest,
    public_finetune_checkpoint: Path,
    base_vlm_dir: Path,
) -> tuple[SmolVLAMLX, str]:
    with mx.stream(mx.cpu):
        policy = SmolVLAMLX.from_pretrained(
            public_finetune_checkpoint,
            cache_dir=Path(".cache/mlx_smolvla") / f"public-finetune-{request.param}",
            dtype=request.param,
            tokenizer_dir=base_vlm_dir,
            execution_mode="strict",
        )
    return policy, request.param


def test_public_finetune_config_and_stats_come_from_checkpoint(
    public_finetune_policy: tuple[SmolVLAMLX, str],
) -> None:
    policy, _ = public_finetune_policy
    assert policy.config.image_shapes == (
        ("observation.images.wrist_camera", (3, 480, 640)),
        ("observation.images.top_camera", (3, 480, 640)),
    )
    assert policy.config.state_shape == (6,)
    assert policy.config.action_shape == (6,)
    assert policy.config.state_normalization == "mean_std"
    assert policy.config.action_normalization == "mean_std"
    assert len(policy.loaded_parameter_names) == 500


def test_public_finetune_preprocessing_matches_reference(
    public_finetune_cases: tuple[_Case, ...],
    public_finetune_policy: tuple[SmolVLAMLX, str],
) -> None:
    policy, _ = public_finetune_policy
    for case in public_finetune_cases:
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


def test_public_finetune_normalized_actions_match_reference(
    public_finetune_cases: tuple[_Case, ...],
    public_finetune_policy: tuple[SmolVLAMLX, str],
) -> None:
    policy, dtype = public_finetune_policy
    tolerance = 5e-3 if dtype == "float32" else 5e-2
    with mx.stream(mx.cpu):
        for case in public_finetune_cases:
            actual = policy.predict_action_chunk(
                case.observation(),
                noise=mx.array(case.array("noise"), dtype=mx.float32),
            )
            mx.eval(actual)
            max_abs = float(
                np.max(
                    np.abs(
                        np.asarray(actual.astype(mx.float32))
                        - case.array("actions/normalized").astype(np.float32, copy=False)
                    )
                )
            )
            assert max_abs <= tolerance, (case.name, dtype, max_abs)


def test_any_hub_repo_id_is_resolved_without_a_hardcoded_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = __import__("mlx_smolvla.policy", fromlist=["_resolve_checkpoint"])
    captured = {}

    def fake_snapshot_download(identifier, **kwargs):
        captured["identifier"] = identifier
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(module, "snapshot_download", fake_snapshot_download)
    assert module._resolve_checkpoint("owner/custom-smolvla", tmp_path / "cache") == tmp_path
    assert captured["identifier"] == "owner/custom-smolvla"
    assert captured["revision"] is None


def test_from_pretrained_accepts_matching_hub_identifier(
    monkeypatch: pytest.MonkeyPatch,
    public_finetune_checkpoint: Path,
    base_vlm_dir: Path,
) -> None:
    module = __import__("mlx_smolvla.policy", fromlist=["snapshot_download"])

    def local_snapshot(identifier, **_kwargs):
        assert identifier == "owner/custom-smolvla"
        return str(public_finetune_checkpoint)

    monkeypatch.setattr(module, "snapshot_download", local_snapshot)
    with mx.stream(mx.cpu):
        policy = SmolVLAMLX.from_pretrained(
            "owner/custom-smolvla",
            cache_dir=Path(".cache/mlx_smolvla/public-finetune-float32"),
            dtype="float32",
            tokenizer_dir=base_vlm_dir,
            execution_mode="strict",
        )
    assert len(policy.loaded_parameter_names) == 500
    assert policy.config.image_keys == (
        "observation.images.wrist_camera",
        "observation.images.top_camera",
    )


def test_public_finetune_fifty_frame_statistical_gate() -> None:
    result = StatisticalResult.from_json(Path(".cache/statistical-public-finetune.json"))
    assert result.sample_count == 50
    assert result.mlx_fp32_mae <= 1.05 * result.torch_fp32_mae
    assert result.mlx_bf16_mae <= 1.05 * result.torch_fp32_mae
