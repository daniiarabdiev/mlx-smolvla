from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from smolvla_mlx.types import ProcessedObservation


@dataclass(frozen=True)
class _PrefixParts:
    language: object
    state_proj: nn.Linear
    dtype: str


def _load_prefix_parts(checkpoint_dir: Path, dtype: str) -> _PrefixParts:
    from smolvla_mlx.convert import convert_checkpoint
    from smolvla_mlx.language import TruncatedLanguageModel

    converted = convert_checkpoint(
        checkpoint_dir,
        Path(".cache/smolvla_mlx") / f"language-prefix-{dtype}",
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
    state_proj = nn.Linear(32, 960, bias=True)
    state_proj.load_weights(
        [(name.removeprefix("state_proj."), value) for name, value in weights.items() if name.startswith("state_proj.")],
        strict=True,
    )
    return _PrefixParts(language=language, state_proj=state_proj, dtype=dtype)


@pytest.fixture(scope="module", params=("float32", "bfloat16"))
def prefix_parts(request: pytest.FixtureRequest, checkpoint_dir: Path) -> _PrefixParts:
    with mx.stream(mx.cpu):
        return _load_prefix_parts(checkpoint_dir, request.param)


def _processed(golden) -> ProcessedObservation:
    """Build the native runtime's fp32 activation inputs.

    ``dtype`` controls compact checkpoint storage, not the policy's incoming
    observations or connector activations.  The reference creates its goldens
    after upcasting the checkpoint to CPU fp32, so keeping this boundary fp32
    is both the real runtime path and the Section 6 comparison contract.
    """

    return ProcessedObservation(
        pixel_values=golden.mx("preprocessed/pixel_values", mx.float32),
        pixel_attention_mask=golden.mx("preprocessed/pixel_mask", mx.bool_),
        input_ids=golden.mx("preprocessed/input_ids", mx.int32),
        text_attention_mask=golden.mx("preprocessed/text_attention_mask", mx.bool_),
        state=golden.mx("preprocessed/state_normalized", mx.float32),
    )


def _prefix_inputs(golden, parts: _PrefixParts):
    from smolvla_mlx.language import pad_state_to_width

    processed = _processed(golden)
    state_embedding = parts.state_proj(pad_state_to_width(processed.state, width=32))[:, None, :]
    prefix = parts.language.build_prefix(
        processed,
        golden.mx("connector/output", mx.float32),
        state_embedding,
    )
    return processed, state_embedding, prefix


def _assert_module_error(actual: mx.array, expected: np.ndarray, *, dtype: str) -> None:
    """Apply BRIEF Section 6 without adding a stricter bf16-only max bound."""

    actual_array = np.array(actual.astype(mx.float32))
    expected_array = expected.astype(np.float32, copy=False)
    difference = actual_array - expected_array
    relative_l2 = np.linalg.norm(difference.ravel()) / max(np.linalg.norm(expected_array.ravel()), 1e-12)
    assert relative_l2 <= (1e-3 if dtype == "float32" else 3e-2), relative_l2
    if dtype == "float32":
        assert np.max(np.abs(difference)) <= 1e-3


@pytest.mark.parametrize("golden", range(8), indirect=True)
def test_prefix_assembly_preserves_image_language_state_order_and_mask(golden, prefix_parts: _PrefixParts) -> None:
    """Fails if scale/order/padding/causal-boundary construction changes."""

    with mx.stream(mx.cpu):
        processed, state_embedding, prefix = _prefix_inputs(golden, prefix_parts)
        language_embeddings = prefix_parts.language.embed_language_tokens(processed.input_ids)

    _assert_module_error(language_embeddings, golden.array("language/embeddings"), dtype=prefix_parts.dtype)
    _assert_module_error(state_embedding, golden.array("state/embedding"), dtype=prefix_parts.dtype)
    _assert_module_error(prefix.embeddings, golden.array("prefix/embeddings"), dtype=prefix_parts.dtype)
    np.testing.assert_array_equal(np.array(prefix.pad_mask), golden.array("prefix/pad_mask"))
    np.testing.assert_array_equal(np.array(prefix.attention_flags), golden.array("prefix/attention_flags"))
    np.testing.assert_array_equal(np.array(prefix.attention_mask), golden.array("prefix/attention_mask"))
    np.testing.assert_array_equal(np.array(prefix.position_ids), golden.array("prefix/position_ids"))


@pytest.mark.parametrize("golden", range(8), indirect=True)
def test_prefix_cache_carries_the_exact_mask(golden, prefix_parts: _PrefixParts) -> None:
    """Fails if cached decoding loses the exact bidirectional-prefix mask."""

    with mx.stream(mx.cpu):
        _, _, prefix = _prefix_inputs(golden, prefix_parts)
        cache = prefix_parts.language.encode_prefix(prefix, stop_after=1)

    np.testing.assert_array_equal(np.array(cache.mask), golden.array("prefix/attention_mask"))
    assert len(cache.keys) == len(cache.values) == 1
