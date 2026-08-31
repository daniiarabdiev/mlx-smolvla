from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tests.test_prefix import _prefix_inputs, prefix_parts


def _assert_error(actual: mx.array, expected: np.ndarray, *, dtype: str) -> None:
    actual_array = np.array(actual.astype(mx.float32))
    expected_array = expected.astype(np.float32, copy=False)
    difference = actual_array - expected_array
    relative_l2 = np.linalg.norm(difference.ravel()) / max(np.linalg.norm(expected_array.ravel()), 1e-12)
    threshold = 1e-3 if dtype == "float32" else 3e-2
    assert relative_l2 <= threshold, relative_l2
    if dtype == "float32":
        assert np.max(np.abs(difference)) <= 1e-3


@pytest.mark.parametrize("golden", range(8), indirect=True)
def test_each_used_decoder_layer_and_exported_kv_match_goldens(golden, prefix_parts) -> None:
    """Fails on a wrong RoPE pairing/base, GQA expansion, residual, or cache layout."""

    with mx.stream(mx.cpu):
        _, _, prefix = _prefix_inputs(golden, prefix_parts)
        cache = prefix_parts.language.encode_prefix(prefix, collect_layer_outputs=True)

    assert len(cache.layer_outputs) == len(cache.keys) == len(cache.values) == 16
    for layer_index, layer_output in enumerate(cache.layer_outputs):
        _assert_error(
            layer_output,
            golden.array(f"vlm/layer_{layer_index:02d}/output"),
            dtype=prefix_parts.dtype,
        )
        _assert_error(
            cache.keys[layer_index],
            golden.array(f"vlm/cache/layer_{layer_index:02d}/key"),
            dtype=prefix_parts.dtype,
        )
        _assert_error(
            cache.values[layer_index],
            golden.array(f"vlm/cache/layer_{layer_index:02d}/value"),
            dtype=prefix_parts.dtype,
        )
    _assert_error(cache.hidden, golden.array("vlm/prefix/output"), dtype=prefix_parts.dtype)


@pytest.mark.parametrize("golden", [0], indirect=True)
def test_decoder_stop_after_returns_only_the_requested_cache_prefix(golden, prefix_parts) -> None:
    """Fails if the decoder runs past its requested layer cutoff."""

    with mx.stream(mx.cpu):
        _, _, prefix = _prefix_inputs(golden, prefix_parts)
        cache = prefix_parts.language.encode_prefix(prefix, stop_after=3, collect_layer_outputs=True)

    assert len(cache.layer_outputs) == len(cache.keys) == len(cache.values) == 3
    _assert_error(cache.hidden, golden.array("vlm/layer_02/output"), dtype=prefix_parts.dtype)
