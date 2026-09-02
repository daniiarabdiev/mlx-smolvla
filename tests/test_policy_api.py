"""Public native-policy loading and action-queue contracts."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from smolvla_mlx.policy import SmolVLAMLX


@dataclass(frozen=True)
class _PolicyParts:
    policy: SmolVLAMLX
    dtype: str


@pytest.fixture(scope="module", params=("float32", "bfloat16"))
def native_policy(
    request: pytest.FixtureRequest,
    checkpoint_dir: Path,
    base_vlm_dir: Path,
) -> _PolicyParts:
    with mx.stream(mx.cpu):
        policy = SmolVLAMLX.from_pretrained(
            str(checkpoint_dir),
            cache_dir=Path(".cache/smolvla_mlx") / f"policy-{request.param}",
            dtype=request.param,
            tokenizer_dir=base_vlm_dir,
            execution_mode="strict",
        )
    return _PolicyParts(policy=policy, dtype=request.param)


def test_from_pretrained_initializes_every_converted_checkpoint_parameter(native_policy: _PolicyParts) -> None:
    """Fails if a checkpoint tensor is ignored or attached to the wrong native module."""

    converted = mx.load(str(native_policy.policy.converted_weights_path))
    assert len(converted) == 500
    assert native_policy.policy.loaded_parameter_names == tuple(sorted(converted))


def test_public_execution_mode_contract_is_explicit(native_policy: _PolicyParts) -> None:
    policy = native_policy.policy
    assert inspect.signature(SmolVLAMLX.from_pretrained).parameters["execution_mode"].default == "production"
    assert inspect.signature(SmolVLAMLX.from_pretrained).parameters["quantization"].default is None
    assert policy.execution_mode == "strict"
    assert policy.execution_device == mx.cpu
    assert policy.quantization is None
    assert policy.quantization_manifest is None


def test_unknown_execution_mode_is_rejected_before_checkpoint_resolution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        SmolVLAMLX.from_pretrained(tmp_path / "missing", execution_mode="automatic")


@pytest.mark.parametrize("preset", ("everything-4bit", "vlm-3bit", "dense-bf16"))
def test_unknown_quantization_is_rejected_before_checkpoint_resolution(
    tmp_path: Path,
    preset: str,
) -> None:
    with pytest.raises(ValueError, match="quantization"):
        SmolVLAMLX.from_pretrained(tmp_path / "missing", quantization=preset)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"dtype": "float32", "quantization": "vlm-8bit"}, "bfloat16"),
        ({"execution_mode": "strict", "quantization": "vlm-4bit"}, "production"),
    ),
)
def test_quantization_rejects_unvalidated_runtime_modes_before_checkpoint_resolution(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SmolVLAMLX.from_pretrained(tmp_path / "missing", **kwargs)


@pytest.mark.parametrize(
    ("execution_mode", "outer_device", "expected_device"),
    (("production", mx.cpu, mx.gpu), ("strict", mx.gpu, mx.cpu)),
)
def test_policy_owns_its_execution_device_context(
    monkeypatch: pytest.MonkeyPatch,
    execution_mode: str,
    outer_device,
    expected_device,
) -> None:
    policy = object.__new__(SmolVLAMLX)
    policy._execution_mode = execution_mode
    observed = []

    def probe(self, observation, noise=None):
        observed.append(mx.default_device())
        return mx.array([1.0])

    monkeypatch.setattr(SmolVLAMLX, "_predict_action_chunk", probe, raising=False)
    with mx.stream(outer_device):
        result = policy.predict_action_chunk({})
        mx.eval(result)

    assert observed == [expected_device]


@pytest.mark.parametrize("golden", [0], indirect=True)
def test_select_action_uses_real_chunk_fifo_and_reset(golden, native_policy: _PolicyParts) -> None:
    """Queue behavior must use real model output, not a stubbed prediction path."""

    policy = native_policy.policy
    policy.reset()
    with mx.stream(mx.cpu):
        first = policy.select_action(golden.observation(), noise=golden.mx("noise", mx.float32))
        second = policy.select_action(golden.observation())

    assert first.shape == second.shape == (policy.config.action_dim,)
    assert policy.last_prefix_evaluations == 1
    assert policy.queued_actions == policy.config.n_action_steps - 2
    expected = golden.array("actions/unnormalized")[0]
    tolerance = 5e-3 if native_policy.dtype == "float32" else 5e-2
    np.testing.assert_allclose(first, expected[0], atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(second, expected[1], atol=tolerance, rtol=0.0)

    policy.reset()
    assert policy.queued_actions == 0
