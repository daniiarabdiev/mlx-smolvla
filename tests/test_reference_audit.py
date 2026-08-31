from pathlib import Path
import json
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_runtime_audit_resolves_every_architecture_hypothesis() -> None:
    from reference.audit import inspect_reference
    from reference.policy import ReferencePolicy, load_dataset_observation

    reference = ReferencePolicy.load(cache_dir=Path(".cache/hf"))
    sample = load_dataset_observation(cache_dir=Path(".cache/hf"), index=0)

    audit = inspect_reference(reference, sample)

    assert audit["parameters"] == {
        "total": 450_046_176,
        "vlm": 350_165_184,
        "expert": 98_245_840,
        "state_projection": 31_680,
        "action_input_projection": 23_760,
        "action_output_projection": 23_072,
        "time_mlp": 1_556_640,
    }
    assert audit["model"]["vlm_layers"] == 16
    assert audit["model"]["expert_layers"] == 16
    assert audit["model"]["text_hidden_size"] == 960
    assert audit["model"]["expert_hidden_size"] == 720
    assert audit["model"]["expert_intermediate_size"] == 2048
    assert audit["model"]["attention_heads"] == 15
    assert audit["model"]["key_value_heads"] == 5
    assert audit["model"]["head_dim"] == 64
    assert audit["model"]["self_attention_layers"] == list(range(0, 16, 2))
    assert audit["model"]["cross_attention_layers"] == list(range(1, 16, 2))

    assert audit["vision"]["input_shape"] == [2, 3, 512, 512]
    assert audit["vision"]["encoder_output_shape"] == [2, 1024, 768]
    assert audit["vision"]["connector_output_shape"] == [2, 64, 960]
    assert audit["vision"]["pixel_shuffle_scale"] == 4
    assert audit["vision"]["activation"] == "gelu_pytorch_tanh"

    assert audit["boundaries"]["language_tokens"] == [1, 48]
    assert audit["boundaries"]["padded_state"] == [1, 32]
    assert audit["boundaries"]["state_embedding"] == [1, 1, 960]
    assert audit["boundaries"]["prefix"] == [1, 177, 960]
    assert audit["boundaries"]["suffix"] == [1, 50, 720]
    assert audit["boundaries"]["first_cache_key"] == [1, 5, 177, 64]
    assert audit["boundaries"]["cache_layers"] == 16

    assert audit["attention"]["prefix_content_is_bidirectional"] is True
    assert audit["attention"]["state_attends_prefix"] is True
    assert audit["attention"]["prefix_cannot_attend_state"] is True
    assert audit["attention"]["suffix_is_causal"] is True

    assert audit["preprocessing"]["resize"] == {
        "height": 512,
        "width": 512,
        "interpolation": "bilinear",
        "align_corners": False,
        "padding_edges": ["left", "top"],
        "pad_before_pixel_normalization": 0.0,
        "pixel_transform": "x * 2 - 1",
    }
    assert audit["preprocessing"]["tokenizer_max_length"] == 48
    assert audit["preprocessing"]["tokenizer_padding_side"] == "right"
    assert audit["preprocessing"]["state_normalization_effective"] == "identity"
    assert audit["preprocessing"]["action_unnormalization_effective"] == "identity"
    assert audit["preprocessing"]["saved_stat_keys"] == [
        "so100-blue.buffer.action",
        "so100-red.buffer.action",
        "so100.buffer.action",
    ]

    assert audit["flow"]["steps"] == 10
    assert audit["flow"]["velocity_dim"] == 32
    assert audit["flow"]["output_action_dim"] == 6
    assert audit["flow"]["dt"] == -0.1
    assert audit["flow"]["times"] == pytest.approx(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    )
    assert audit["flow"]["update"] == "x_t = x_t + dt * v_t"
    assert audit["queue"]["n_action_steps"] == 50
    assert audit["queue"]["refill_when_empty"] is True


@pytest.mark.slow
def test_inspect_reference_script_writes_machine_readable_and_markdown_evidence(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    json_path = tmp_path / "audit.json"
    markdown_path = tmp_path / "architecture.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_reference.py",
            "--cache-dir",
            ".cache/hf",
            "--json",
            str(json_path),
            "--write",
            str(markdown_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    report = markdown_path.read_text(encoding="utf-8")
    assert payload["runtime"]["model"]["vlm_layers"] == 16
    assert len(payload["weight_inventory"]) == 500
    assert payload["weight_inventory"][0]["name"] == "model.action_in_proj.bias"
    assert "## Verified architecture" in report
    assert "`model.action_in_proj.bias`" in report
