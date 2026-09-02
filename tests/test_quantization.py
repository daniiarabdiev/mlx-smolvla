"""Stage Q P2-3 VLM-only quantization experiment contracts."""

from __future__ import annotations

import copy
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest


class _Root(nn.Module):
    def __init__(self, input_dims: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dims, 64, bias=False)


class _ToyPolicy:
    def __init__(self, *, vlm_input_dims: int = 64) -> None:
        self.connector = _Root(vlm_input_dims)
        self.language = _Root(vlm_input_dims)
        self.vision = _Root()
        self.state_proj = nn.Linear(32, 64)
        self.expert = _Root()
        for root in (
            self.connector,
            self.language,
            self.vision,
            self.state_proj,
            self.expert,
        ):
            root.set_dtype(mx.bfloat16)


def _sample_errors(candidate_offset: float) -> list[dict[str, object]]:
    return [
        {
            "episode": index,
            "frame_index": 0,
            "seed": 20_260_831 + index,
            "element_count": 6,
            "torch_fp32_abs_error_sum": 60.0 + index,
            "dense_bf16_abs_error_sum": 60.5 + index,
            "candidate_abs_error_sum": 60.5 + index + candidate_offset,
        }
        for index in range(50)
    ]


def _document() -> dict[str, object]:
    from smolvla_mlx.quantization import (
        QuantizationAccuracy,
        QuantizationLatency,
        QuantizationProtocol,
        derive_quantization_decision,
        expected_topology_manifest,
    )

    protocol = QuantizationProtocol()
    variants = []
    for variant, offset in zip(protocol.variants, (0.0, 0.1, 0.2), strict=True):
        variants.append(
            {
                "variant": variant,
                "topology": expected_topology_manifest(variant),
                "accuracy": QuantizationAccuracy.from_samples(
                    variant=variant,
                    samples=_sample_errors(offset),
                ).as_dict(),
                "latency": QuantizationLatency.from_samples(
                    variant=variant,
                    samples_ms=[100.0 + index + offset for index in range(50)],
                    warmup_runs=protocol.latency_warmup_runs,
                    peak_memory_bytes=2_000_000_000,
                    device="Device(gpu, 0)",
                ).as_dict(),
            }
        )
    return {
        "artifact_type": "smolvla-mlx-vlm-quantization-experiment",
        "format_version": 1,
        "protocol": protocol.as_dict(),
        "idle": {
            "verified": True,
            "checked_at_utc": "2026-09-02T00:00:00+00:00",
            "matching_processes": [],
        },
        "environment": {
            "cpu": "Apple M5 Pro",
            "unified_memory_bytes": 48 * 1024**3,
            "macos": "26.6.2",
            "python": "3.12.13",
            "mlx": "0.32.2",
            "lerobot": "0.6.1",
        },
        "inputs": {
            "checkpoint_id": "lerobot/smolvla_base",
            "checkpoint_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
            "dataset_id": "lerobot/svla_so101_pickplace",
            "dataset_revision": "f641879e22172be7e8161d5e6c1503c2d2feb657",
            "latency_sample": "tests/golden/sample_000",
            "latency_sample_sha256": "a" * 64,
            "latency_noise_sha256": "b" * 64,
            "dense_statistical_sha256": "c506ddcfdde50297e97b9905a299d55f117680a93b257e0af335ae6c9ad5fe07",
        },
        "source": {
            "git_commit": "c" * 40,
            "tracked_worktree_clean": True,
            "sha256": {
                "scripts/benchmark_inference_comparison.py": "d" * 64,
                "scripts/experiment_quantization.py": "e" * 64,
                "smolvla_mlx/benchmark.py": "f" * 64,
                "smolvla_mlx/policy.py": "1" * 64,
                "smolvla_mlx/quantization.py": "2" * 64,
            },
        },
        "variants": variants,
        "decision": derive_quantization_decision(variants),
    }


def test_quantization_protocol_freezes_scope_gate_and_measurements() -> None:
    from smolvla_mlx.quantization import QuantizationProtocol

    protocol = QuantizationProtocol()
    assert protocol.variants == ("dense-bf16", "vlm-8bit", "vlm-4bit")
    assert protocol.quantized_roots == ("connector", "language")
    assert protocol.dense_roots == ("vision", "state_proj", "expert")
    assert protocol.group_size == 64
    assert protocol.mode == "affine"
    assert protocol.statistical_samples == 50
    assert protocol.statistical_ratio_gate == 1.05
    assert (protocol.latency_warmup_runs, protocol.latency_measured_runs) == (5, 50)
    with pytest.raises(ValueError, match="fixed"):
        QuantizationProtocol(statistical_ratio_gate=1.051)


def test_expected_real_topology_is_vlm_linear_only() -> None:
    from smolvla_mlx.quantization import (
        EXPECTED_QUANTIZED_SCALARS,
        expected_vlm_linear_paths,
    )

    paths = expected_vlm_linear_paths()
    assert len(paths) == 114
    assert EXPECTED_QUANTIZED_SCALARS == 216_391_680
    assert "connector.modality_projection.proj" in paths
    assert "language.lm_head" in paths
    assert "language.layers.0.self_attn.q_proj" in paths
    assert not any(path.startswith(("vision.", "expert.", "state_proj.")) for path in paths)


@pytest.mark.parametrize("bits", (8, 4))
def test_quantize_vlm_linears_targets_only_connector_and_language(bits: int) -> None:
    from smolvla_mlx.quantization import quantize_vlm_linears

    policy = _ToyPolicy()
    manifest = quantize_vlm_linears(policy, bits=bits)
    assert manifest.bits == bits
    assert manifest.quantized_paths == ("connector.proj", "language.proj")
    assert isinstance(policy.connector.proj, nn.QuantizedLinear)
    assert isinstance(policy.language.proj, nn.QuantizedLinear)
    assert isinstance(policy.vision.proj, nn.Linear)
    assert isinstance(policy.state_proj, nn.Linear)
    assert isinstance(policy.expert.proj, nn.Linear)
    assert policy.vision.proj.weight.dtype == mx.bfloat16
    assert policy.expert.proj.weight.dtype == mx.bfloat16


def test_quantization_rejects_ineligible_vlm_width_before_mutation() -> None:
    from smolvla_mlx.quantization import quantize_vlm_linears

    policy = _ToyPolicy(vlm_input_dims=63)
    with pytest.raises(ValueError, match="group size"):
        quantize_vlm_linears(policy, bits=4)
    assert isinstance(policy.connector.proj, nn.Linear)
    assert isinstance(policy.language.proj, nn.Linear)


def test_experiment_validator_recomputes_gates_timings_and_decision() -> None:
    from smolvla_mlx.quantization import validate_quantization_document

    document = _document()
    assert validate_quantization_document(document) == document
    assert document["decision"] == {
        "default_changed": False,
        "eligible_opt_in_variants": ["vlm-8bit", "vlm-4bit"],
        "rejected_variants": [],
    }

    changed = copy.deepcopy(document)
    changed["variants"][1]["accuracy"]["statistical_ratio"] = 0.1
    with pytest.raises(ValueError, match="accuracy"):
        validate_quantization_document(changed)

    wrong_scope = copy.deepcopy(document)
    wrong_scope["variants"][1]["topology"]["quantized_paths"][0] = "vision.bad"
    with pytest.raises(ValueError, match="topology"):
        validate_quantization_document(wrong_scope)

    wrong_decision = copy.deepcopy(document)
    wrong_decision["decision"]["default_changed"] = True
    with pytest.raises(ValueError, match="decision"):
        validate_quantization_document(wrong_decision)


def test_experiment_script_has_isolated_accuracy_and_latency_workers() -> None:
    source = Path("scripts/experiment_quantization.py").read_text(encoding="utf-8")
    assert "_tracked_worktree_is_clean()" in source
    assert "_competing_processes()" in source
    assert 'choices=("accuracy", "latency")' in source
    assert 'choices=("dense-bf16", "vlm-8bit", "vlm-4bit")' in source
    assert "subprocess.run(" in source
