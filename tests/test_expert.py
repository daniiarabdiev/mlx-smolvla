"""Golden tests for the native SmolVLA action expert."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from smolvla_mlx.expert import ActionExpert
from tests.test_prefix import _prefix_inputs


@dataclass(frozen=True)
class _ExpertParts:
    language: object
    state_proj: object
    expert: ActionExpert
    dtype: str


def _load_expert_parts(checkpoint_dir: Path, dtype: str) -> _ExpertParts:
    from smolvla_mlx.convert import convert_checkpoint
    from smolvla_mlx.language import TruncatedLanguageModel

    converted = convert_checkpoint(
        checkpoint_dir,
        Path(".cache/smolvla_mlx") / f"expert-{dtype}",
        dtype=dtype,
    )
    weights = mx.load(str(converted.output_path))

    language = TruncatedLanguageModel()
    language.load_weights(
        [
            (name.removeprefix("language."), value)
            for name, value in weights.items()
            if name.startswith("language.")
        ],
        strict=True,
    )

    from mlx import nn

    state_proj = nn.Linear(32, 960, bias=True)
    state_proj.load_weights(
        [(name.removeprefix("state_proj."), value) for name, value in weights.items() if name.startswith("state_proj.")],
        strict=True,
    )

    expert = ActionExpert()
    expert.load_weights(
        [
            (name.removeprefix("expert."), value)
            for name, value in weights.items()
            if name.startswith("expert.")
        ]
        + [(name, value) for name, value in weights.items() if name.startswith("action_")],
        strict=True,
    )
    return _ExpertParts(language=language, state_proj=state_proj, expert=expert, dtype=dtype)


@pytest.fixture(scope="module", params=("float32", "bfloat16"))
def expert_parts(request: pytest.FixtureRequest, checkpoint_dir: Path) -> _ExpertParts:
    with mx.stream(mx.cpu):
        return _load_expert_parts(checkpoint_dir, request.param)


def _assert_error(actual: mx.array, expected: np.ndarray, *, dtype: str) -> None:
    actual_array = np.array(actual.astype(mx.float32))
    expected_array = expected.astype(np.float32, copy=False)
    difference = actual_array - expected_array
    relative_l2 = np.linalg.norm(difference.ravel()) / max(np.linalg.norm(expected_array.ravel()), 1e-12)
    assert relative_l2 <= (1e-3 if dtype == "float32" else 3e-2), relative_l2
    if dtype == "float32":
        assert np.max(np.abs(difference)) <= 1e-3


def prefix_cache(golden, parts: _ExpertParts):
    _, _, prefix = _prefix_inputs(golden, parts)
    return parts.language.encode_prefix(prefix)


@pytest.mark.parametrize("golden", range(8), indirect=True)
def test_action_and_timestep_embedding_matches_all_real_goldens(golden, expert_parts: _ExpertParts) -> None:
    """Fails on an action projection, sinusoid, MLP, or dtype-boundary mismatch."""

    with mx.stream(mx.cpu):
        suffix = expert_parts.expert.embed_suffix(
            golden.mx("flow/step_00/x_t", mx.float32),
            mx.array([1.0], dtype=mx.float32),
        )
        mx.eval(suffix)

    _assert_error(
        suffix,
        golden.array("flow/step_00/suffix_embeddings"),
        dtype=expert_parts.dtype,
    )


@pytest.mark.parametrize("golden", [0], indirect=True)
def test_action_expert_blocks_and_velocity_match_first_denoising_step(golden, expert_parts: _ExpertParts) -> None:
    """Fails on self/cross alignment, RoPE, masking, residuals, or cache use."""

    with mx.stream(mx.cpu):
        cache = prefix_cache(golden, expert_parts)
        result = expert_parts.expert.denoise(
            cache,
            golden.mx("flow/step_00/x_t", mx.float32),
            mx.array([1.0], dtype=mx.float32),
            collect_layer_outputs=True,
        )
        mx.eval(result.suffix_embeddings, result.hidden, result.velocity, *result.layer_outputs)

    assert len(result.layer_outputs) == 16
    _assert_error(
        result.suffix_embeddings,
        golden.array("flow/step_00/suffix_embeddings"),
        dtype=expert_parts.dtype,
    )
    for layer_index, output in enumerate(result.layer_outputs):
        _assert_error(
            output,
            golden.array(f"expert/step_00/layer_{layer_index:02d}/output"),
            dtype=expert_parts.dtype,
        )
    _assert_error(result.hidden, golden.array("expert/step_00/output"), dtype=expert_parts.dtype)
    _assert_error(result.velocity, golden.array("flow/step_00/velocity"), dtype=expert_parts.dtype)
