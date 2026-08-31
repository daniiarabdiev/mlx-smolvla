"""Real full-architecture gradient and resource gate for Stage T0."""

from __future__ import annotations


def test_audit_reads_the_installed_mlx_distribution_version() -> None:
    module = __import__("training.audit", fromlist=["installed_mlx_version"])

    assert module.installed_mlx_version() == "0.32.2"


def test_full_random_weight_training_step_has_finite_selected_gradients() -> None:
    module = __import__("training.audit", fromlist=["run_training_readiness_audit"])

    result = module.run_training_readiness_audit(seed=0)
    payload = result.as_dict()

    assert payload["device"].startswith("Device(gpu")
    assert payload["dtype"] == "bfloat16"
    assert payload["seed"] == 0
    assert payload["microbatch"] == 1
    assert payload["camera_count"] == 2
    assert payload["image_shape"] == [2, 3, 512, 512]
    assert payload["action_shape"] == [1, 50, 32]
    assert payload["physical_action_dim"] == 6
    assert payload["trainable_tensor_count"] > 0
    assert payload["trainable_scalar_count"] > 0
    assert payload["gradient_tensor_count"] == payload["trainable_tensor_count"]
    assert len(payload["selected_parameter_names"]) == payload["trainable_tensor_count"]
    assert len(set(payload["canonical_parameter_names"])) == payload["trainable_tensor_count"]
    assert payload["all_gradients_finite"] is True
    assert payload["zero_norm_gradient_tensors"] == []
    assert payload["loss"] > 0.0
    assert payload["forward_ms"] > 0.0
    assert payload["forward_backward_ms"] > 0.0
    assert payload["active_memory_bytes"] > 0
    assert payload["peak_memory_bytes"] >= payload["active_memory_bytes"]
    assert payload["disk_free_before_bytes"] >= 40 * 1024**3
    assert payload["disk_free_after_bytes"] >= 40 * 1024**3
