"""Scoped CPU autodiff contracts that protect exact inference dispatch."""

from __future__ import annotations

import mlx.core as mx
import pytest


def test_pure_cpu_primitives_have_finite_nonzero_vjps() -> None:
    module = __import__(
        "training.differentiable",
        fromlist=[
            "differentiable_rms_norm",
            "differentiable_rope",
            "differentiable_softmax",
            "differentiable_silu",
        ],
    )
    values = mx.array([[0.25, -0.5, 1.5, -2.0]], dtype=mx.float32)
    weight = mx.array([0.8, 0.9, 1.1, 1.2], dtype=mx.float32)

    def objective(inputs: mx.array, scale: mx.array) -> mx.array:
        normalized = module.differentiable_rms_norm(inputs, scale, 1e-5)
        rotated = module.differentiable_rope(
            inputs.reshape(1, 1, 1, 4),
            mx.array([[3]], dtype=mx.int32),
        )
        probabilities = module.differentiable_softmax(inputs)
        activated = module.differentiable_silu(inputs)
        return mx.sum(normalized) + mx.sum(rotated) + mx.sum(probabilities) + mx.sum(activated)

    with mx.stream(mx.cpu):
        value_and_grad = mx.value_and_grad(objective, argnums=(0, 1))
        value, (input_gradient, weight_gradient) = value_and_grad(values, weight)
        mx.eval(value, input_gradient, weight_gradient)

    assert bool(mx.all(mx.isfinite(input_gradient)))
    assert bool(mx.all(mx.isfinite(weight_gradient)))
    assert bool(mx.any(mx.abs(input_gradient) > 0))
    assert bool(mx.any(mx.abs(weight_gradient) > 0))


def _runtime_callables() -> dict[str, object]:
    rms = __import__("smolvla_mlx.rmsnorm", fromlist=["ReferenceRMSNorm"])
    language = __import__("smolvla_mlx.language", fromlist=["reference_rope"])
    expert = __import__("smolvla_mlx.expert", fromlist=["reference_softmax"])
    return {
        "norm_call": rms.ReferenceRMSNorm.__call__,
        "rms_rope": rms.reference_rope,
        "rms_softmax": rms.reference_softmax,
        "rms_silu": rms.reference_silu,
        "language_rope": language.reference_rope,
        "language_softmax": language.reference_softmax,
        "language_silu": language.reference_silu,
        "expert_softmax": expert.reference_softmax,
        "expert_silu": expert.reference_silu,
    }


def test_cpu_autodiff_scope_patches_only_while_active_and_restores_on_exit() -> None:
    module = __import__("training.differentiable", fromlist=["differentiable_cpu_primitives"])
    original = _runtime_callables()

    with mx.stream(mx.cpu):
        with module.differentiable_cpu_primitives():
            active = _runtime_callables()
            assert active != original
            assert active["rms_rope"] is module.differentiable_rope
            assert active["language_rope"] is module.differentiable_rope
            assert active["rms_softmax"] is module.differentiable_softmax
            assert active["language_softmax"] is module.differentiable_softmax
            assert active["expert_softmax"] is module.differentiable_softmax
            assert active["rms_silu"] is module.differentiable_silu
            assert active["language_silu"] is module.differentiable_silu
            assert active["expert_silu"] is module.differentiable_silu

    assert _runtime_callables() == original


def test_cpu_autodiff_scope_restores_after_exception_and_rejects_nesting() -> None:
    module = __import__("training.differentiable", fromlist=["differentiable_cpu_primitives"])
    original = _runtime_callables()

    with pytest.raises(RuntimeError, match="intentional"):
        with mx.stream(mx.cpu):
            with module.differentiable_cpu_primitives():
                with pytest.raises(RuntimeError, match="already active"):
                    with module.differentiable_cpu_primitives():
                        pass
                raise RuntimeError("intentional scope failure")

    assert _runtime_callables() == original


def test_cpu_autodiff_scope_refuses_non_cpu_execution() -> None:
    module = __import__("training.differentiable", fromlist=["differentiable_cpu_primitives"])

    with mx.stream(mx.gpu):
        with pytest.raises(RuntimeError, match="MLX CPU"):
            with module.differentiable_cpu_primitives():
                pass
