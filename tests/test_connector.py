from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


_GOLDEN_TOLERANCES = {
    "float32": (1e-3, 1e-3),
    "bfloat16": (3e-2, None),
}


def _load_connector(converted_path: Path):
    from smolvla_mlx.connector import Connector

    connector = Connector()
    weights = mx.load(str(converted_path))
    connector.load_weights(
        [
            (name.removeprefix("connector."), value)
            for name, value in weights.items()
            if name.startswith("connector.")
        ],
        strict=True,
    )
    return connector


@pytest.mark.parametrize("dtype", ("float32", "bfloat16"))
@pytest.mark.parametrize("golden", range(8), indirect=True)
def test_connector_matches_golden(
    golden,
    checkpoint_dir: Path,
    dtype: str,
) -> None:
    from smolvla_mlx.convert import convert_checkpoint

    with mx.stream(mx.cpu):
        converted = convert_checkpoint(checkpoint_dir, Path(".cache/smolvla_mlx") / f"connector-{dtype}", dtype=dtype)
        connector = _load_connector(converted.output_path)
        actual = connector(golden.mx("vision/features", getattr(mx, dtype)))

    actual_array = np.array(actual.astype(mx.float32))
    expected_array = golden.array("connector/output").astype(np.float32, copy=False)
    difference = actual_array - expected_array
    relative_l2 = np.linalg.norm(difference.ravel()) / max(np.linalg.norm(expected_array.ravel()), 1e-12)
    rel_l2, max_abs = _GOLDEN_TOLERANCES[dtype]
    assert relative_l2 <= rel_l2, relative_l2
    if max_abs is not None:
        assert np.max(np.abs(difference)) <= max_abs
