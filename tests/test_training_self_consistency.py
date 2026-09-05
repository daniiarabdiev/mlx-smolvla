"""Prospectively fixed PyTorch self-consistency-floor contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest


pytestmark = pytest.mark.slow


def _actions(value: float = 0.0) -> np.ndarray:
    return np.full((56, 50, 6), value, dtype=np.float64)


def _case_identities() -> tuple[dict[str, int], ...]:
    return tuple(
        {
            "ordinal": ordinal,
            "episode": ordinal // 7,
            "frame_index": ordinal,
            "absolute_index": 1_000 + ordinal,
        }
        for ordinal in range(56)
    )


def _runtime_metadata(item) -> dict[str, object]:
    return {
        "actual_threads": 6 if item.requested_threads is None else item.requested_threads,
        "interop_threads": 12,
        "torch_version": "2.11.0",
        "lerobot_version": "0.6.1",
        "transformers_version": "5.5.4",
        "platform": "test-platform",
        "machine": "arm64",
        "python_version": "3.12.13",
        "mps_built": True,
        "mps_available": True,
        "mps_fallback_environment": "1" if item.mps_fallback else None,
        "mps_environment": {
            key: (
                "1"
                if item.mps_fallback and key == "PYTORCH_ENABLE_MPS_FALLBACK"
                else None
            )
            for key in __import__(
                "mlx_smolvla._lab.training.self_consistency",
                fromlist=["MPS_ENVIRONMENT_KEYS"],
            ).MPS_ENVIRONMENT_KEYS
        },
        "cpu_thread_environment": {
            key: None
            for key in __import__(
                "mlx_smolvla._lab.training.self_consistency",
                fromlist=["CPU_THREAD_ENVIRONMENT_KEYS"],
            ).CPU_THREAD_ENVIRONMENT_KEYS
        },
        "worker_seed": 20_260_901,
        "deterministic_algorithms": False,
        "float32_matmul_precision": "highest",
        "float64_compatibility_path": (
            "projection_weight_dtype" if item.dtype == "float64" else None
        ),
    }


def _variant_metadata(module, *, max_threads: int = 12):
    input_digest = _input_hashes()["combined_sha256"]
    result = {}
    for item in module.perturbation_plan(max_threads=max_threads):
        document = {
            "format_version": 1,
            "artifact_type": "smolvla-pytorch-self-consistency-variant",
            "procedure_id": module.PROCEDURE_ID,
            **item.as_dict(),
            **_runtime_metadata(item),
            "input_combined_sha256": input_digest,
            "sample_count": 56,
            "normalized_action_chunk_shape": [50, 6],
            "normalized_actions_dtype": item.dtype,
            "normalized_actions_sha256": hashlib.sha256(item.name.encode()).hexdigest(),
        }
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        document["variant_artifact_sha256"] = hashlib.sha256(payload).hexdigest()
        result[item.name] = document
    return result


def _input_hashes() -> dict[str, object]:
    def group(files: dict[str, str]) -> dict[str, object]:
        payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        return {"tree_sha256": hashlib.sha256(payload).hexdigest(), "files": files}

    groups: dict[str, object] = {
        "checkpoint_export": group({"model.safetensors": "2" * 64}),
        "evaluation_artifact": group({"manifest.json": "4" * 64}),
        "pinned_dataset": group({"meta/info.json": "b" * 64}),
        "tokenizer_snapshot": group({"tokenizer.json": "6" * 64}),
        "implementation": group({"training/self_consistency.py": "8" * 64}),
    }
    payload = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    return {**groups, "combined_sha256": hashlib.sha256(payload).hexdigest()}


def _report(module) -> dict[str, object]:
    actions = {item.name: _actions() for item in module.perturbation_plan(max_threads=12)}
    actions["cpu_fp32_threads_1"][3, 2, 1] = 0.125
    actions["cpu_fp32_threads_max"][5, 4, 2] = -0.25
    for index, name in enumerate(module.MPS_VARIANTS, start=1):
        actions[name][7, 6, 3] = 0.075 * index
    actions["cpu_float64"][11, 8, 4] = -0.5
    return module.assemble_floor_report(
        actions=actions,
        variant_metadata=_variant_metadata(module),
        case_identities=_case_identities(),
        input_sha256=_input_hashes(),
        checkpoint_path=".cache/training/t3/export",
        purpose="retrospective_diagnostic",
        created_at_utc="2026-09-01T12:00:00.000000+00:00",
        created_at_ns=1_788_264_000_000_000_000,
    )


def test_perturbation_set_is_exact_and_max_threads_is_explicit() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["perturbation_plan"],
    )

    plan = module.perturbation_plan(max_threads=12)

    assert tuple(item.name for item in plan) == (
        "cpu_fp32_baseline",
        "cpu_fp32_threads_1",
        "cpu_fp32_threads_max",
        "mps_fp32_fallback_1",
        "mps_fp32_fallback_2",
        "mps_fp32_fallback_3",
        "mps_fp32_fallback_4",
        "mps_fp32_fallback_5",
        "cpu_float64",
    )
    assert plan[0].requested_threads is None
    assert plan[1].requested_threads == 1
    assert plan[2].requested_threads == 12
    assert all(item.device == "mps" and item.mps_fallback for item in plan[3:8])
    assert plan[8].device == "cpu" and plan[8].dtype == "float64"
    with pytest.raises(ValueError, match="maximum thread count"):
        module.perturbation_plan(max_threads=0)


def test_floor_envelope_includes_every_nonbaseline_perturbation() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["assemble_floor_report"],
    )

    report = _report(module)

    assert report["artifact_type"] == "smolvla-pytorch-self-consistency-floor"
    assert report["sample_count"] == 56
    assert report["normalized_action_chunk_shape"] == [50, 6]
    assert report["F"] == 0.5
    assert report["F64"] == 0.5
    assert report["variants"]["cpu_fp32_threads_1"]["max_abs_vs_baseline"] == 0.125
    assert report["variants"]["cpu_fp32_threads_max"]["worst_case"]["ordinal"] == 5
    assert report["variants"]["mps_fp32_fallback_5"]["max_abs_vs_baseline"] == 0.375
    assert report["variants"]["cpu_float64"]["max_abs_vs_baseline"] == 0.5
    assert report["context"]["original_mlx_vs_baseline_normalized_max_abs"] == (
        0.17762404680252075
    )
    assert report["context"]["verdict"] == "informational_only"


def test_prospective_floor_uses_checkpoint_generic_identity_and_context() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["assemble_floor_report", "validate_floor_report"],
    )
    actions = {
        item.name: _actions()
        for item in module.perturbation_plan(max_threads=12)
    }

    report = module.assemble_floor_report(
        actions=actions,
        variant_metadata=_variant_metadata(module),
        case_identities=_case_identities(),
        input_sha256=_input_hashes(),
        checkpoint_path=".cache/training/t3b/export",
        purpose="prospective_gate",
        created_at_utc="2026-09-01T12:00:00.000000+00:00",
        created_at_ns=1_788_264_000_000_000_000,
    )

    assert report["source_identity"]["checkpoint_role"] == (
        "prospective-trained-merged-fp32-export"
    )
    assert report["context"] == {
        "comparison_status": "not_run",
        "verdict": "prospective_floor",
    }
    module.validate_floor_report(report)


@pytest.mark.parametrize(
    "dominant_mps_variant",
    [f"mps_fp32_fallback_{index}" for index in range(1, 6)],
)
def test_any_mps_process_slot_can_define_the_floor(
    dominant_mps_variant: str,
) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["assemble_floor_report"],
    )
    actions = {
        item.name: _actions()
        for item in module.perturbation_plan(max_threads=12)
    }
    actions[dominant_mps_variant][17, 3, 2] = -0.625

    report = module.assemble_floor_report(
        actions=actions,
        variant_metadata=_variant_metadata(module),
        case_identities=_case_identities(),
        input_sha256=_input_hashes(),
        checkpoint_path=".cache/training/t3/export",
        purpose="retrospective_diagnostic",
        created_at_utc="2026-09-01T12:00:00.000000+00:00",
        created_at_ns=1_788_264_000_000_000_000,
    )

    assert report["F"] == 0.625
    assert report["variants"][dominant_mps_variant]["worst_case"]["ordinal"] == 17


def test_mps_process_slots_cannot_be_missing_or_relabelled() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["assemble_floor_report", "validate_floor_report"],
    )
    plan = module.perturbation_plan(max_threads=12)
    actions = {item.name: _actions() for item in plan}
    metadata = _variant_metadata(module)
    del actions[module.MPS_VARIANTS[2]]
    with pytest.raises(ValueError, match="perturbation outputs"):
        module.assemble_floor_report(
            actions=actions,
            variant_metadata=metadata,
            case_identities=_case_identities(),
            input_sha256=_input_hashes(),
            checkpoint_path=".cache/training/t3/export",
            purpose="retrospective_diagnostic",
            created_at_utc="2026-09-01T12:00:00.000000+00:00",
            created_at_ns=1_788_264_000_000_000_000,
        )

    changed = json.loads(json.dumps(_report(module)))
    changed["variants"][module.MPS_VARIANTS[1]]["replicate_index"] = 1
    with pytest.raises(ValueError, match="perturbation plan"):
        module.validate_floor_report(changed)


def test_floor_rejects_missing_changed_or_nonfinite_variant_outputs() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["assemble_floor_report"],
    )
    plan = module.perturbation_plan(max_threads=12)
    metadata = _variant_metadata(module)
    actions = {item.name: _actions() for item in plan}
    common = {
        "variant_metadata": metadata,
        "case_identities": _case_identities(),
        "input_sha256": _input_hashes(),
        "checkpoint_path": ".cache/training/t3/export",
        "purpose": "retrospective_diagnostic",
        "created_at_utc": "2026-09-01T12:00:00+00:00",
        "created_at_ns": 1_788_264_000_000_000_000,
    }

    with pytest.raises(ValueError, match="perturbation outputs"):
        module.assemble_floor_report(
            actions={key: value for key, value in actions.items() if key != plan[-1].name},
            **common,
        )
    changed = dict(actions)
    changed[plan[-1].name] = np.zeros((55, 50, 6), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        module.assemble_floor_report(actions=changed, **common)
    changed = {key: value.copy() for key, value in actions.items()}
    changed[plan[1].name][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module.assemble_floor_report(actions=changed, **common)


def test_floor_validation_binds_all_inputs_and_recomputes_envelopes() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["validate_floor_report"],
    )
    report = _report(module)

    module.validate_floor_report(report)

    changed = json.loads(json.dumps(report))
    changed["input_sha256"]["checkpoint_export"]["files"]["model.safetensors"] = "x" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        module.validate_floor_report(changed)
    changed = json.loads(json.dumps(report))
    changed["input_sha256"]["checkpoint_export"]["files"]["model.safetensors"] = "c" * 64
    groups = {
        key: value
        for key, value in changed["input_sha256"].items()
        if key != "combined_sha256"
    }
    changed["input_sha256"]["combined_sha256"] = hashlib.sha256(
        json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="tree SHA-256"):
        module.validate_floor_report(changed)
    changed = json.loads(json.dumps(report))
    changed["F"] = 0.499
    with pytest.raises(ValueError, match="envelope"):
        module.validate_floor_report(changed)
    changed = json.loads(json.dumps(report))
    changed["variants"]["cpu_float64"]["case_max_abs"][11] = 0.25
    with pytest.raises(ValueError, match="variant maximum"):
        module.validate_floor_report(changed)


def test_floor_timestamp_text_and_nanoseconds_must_identify_same_instant() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["validate_floor_report"],
    )
    changed = json.loads(json.dumps(_report(module)))
    changed["created_at_ns"] += 1_000_000_000

    with pytest.raises(ValueError, match="same instant"):
        module.validate_floor_report(changed)


@pytest.mark.parametrize(
    ("variant_name", "field", "replacement", "message"),
    [
        ("cpu_fp32_baseline", "input_combined_sha256", None, "input digest"),
        ("cpu_fp32_threads_1", "actual_threads", 2, "thread count"),
        ("mps_fp32_fallback_1", "mps_fallback_environment", None, "fallback"),
        ("mps_fp32_fallback_1", "mps_available", False, "MPS"),
        ("cpu_float64", "float64_compatibility_path", None, "float64"),
        ("cpu_float64", "normalized_actions_dtype", "float32", "output dtype"),
    ],
)
def test_floor_validation_enforces_variant_runtime_evidence(
    variant_name: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["validate_floor_report"],
    )
    changed = json.loads(json.dumps(_report(module)))
    if replacement is None:
        del changed["variants"][variant_name][field]
    else:
        changed["variants"][variant_name][field] = replacement

    with pytest.raises(ValueError, match=message):
        module.validate_floor_report(changed)


def test_floor_write_is_atomic_hashable_and_round_trips(tmp_path: Path) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["write_floor_report"],
    )
    report = _report(module)
    output = tmp_path / "floor.json"

    digest = module.write_floor_report(output, report)
    payload = output.read_bytes()

    assert digest == hashlib.sha256(payload).hexdigest()
    assert json.loads(payload) == report
    assert not tuple(tmp_path.glob(".floor.json.*"))


def test_input_tree_hashes_every_file_and_rejects_escape_symlinks(
    tmp_path: Path,
) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["hash_input_tree"],
    )
    allowed = tmp_path / "allowed"
    root = allowed / "snapshot"
    blobs = allowed / "blobs"
    root.mkdir(parents=True)
    blobs.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (blobs / "tokenizer").write_bytes(b"tokens")
    (root / "tokenizer.json").symlink_to(Path("../blobs/tokenizer"))

    first = module.hash_input_tree(root, allowed_root=allowed)
    second = module.hash_input_tree(root, allowed_root=allowed)

    assert first == second
    assert first["files"] == {
        "config.json": hashlib.sha256(b"{}\n").hexdigest(),
        "tokenizer.json": hashlib.sha256(b"tokens").hexdigest(),
    }
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (root / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes the allowed root"):
        module.hash_input_tree(root, allowed_root=allowed)


def test_variant_artifact_binds_plan_inputs_and_normalized_actions(
    tmp_path: Path,
) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["read_variant_artifact", "write_variant_artifact"],
    )
    variant = module.perturbation_plan(max_threads=12)[1]
    values = _actions(0.25).astype(np.float32)
    output = tmp_path / variant.name
    metadata = {
        **_runtime_metadata(variant),
    }

    digest = module.write_variant_artifact(
        output,
        variant=variant,
        normalized_actions=values,
        input_combined_sha256="a" * 64,
        metadata=metadata,
    )
    loaded, recorded = module.read_variant_artifact(
        output,
        expected_variant=variant,
        expected_input_combined_sha256="a" * 64,
    )

    np.testing.assert_array_equal(loaded, values)
    assert recorded["variant_artifact_sha256"] == digest
    assert recorded["normalized_actions_sha256"] == hashlib.sha256(
        (output / "normalized_actions.npy").read_bytes()
    ).hexdigest()
    changed = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    changed["input_combined_sha256"] = "b" * 64
    (output / "metadata.json").write_text(
        json.dumps(changed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="input digest"):
        module.read_variant_artifact(
            output,
            expected_variant=variant,
            expected_input_combined_sha256="a" * 64,
        )


def test_cache_path_rejects_symlinked_existing_ancestor(tmp_path: Path) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["_require_cache_path"],
    )
    link = Path(".cache") / f"escape-{tmp_path.name}"
    link.symlink_to(tmp_path, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="symlinked ancestor"):
            module._require_cache_path(
                link / "floor.json",
                label="floor output",
                directory=False,
            )
    finally:
        link.unlink()


def test_floor_outputs_cannot_overlap_hashed_inputs_or_escape_worker_root(
    tmp_path: Path,
) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["_require_floor_paths_disjoint", "run_reference_variant"],
    )
    checkpoint = Path(".cache/training/t3/export").resolve()
    evaluation = Path(".cache/training/t3-evaluation").resolve()
    cache = Path(".cache/hf").resolve()
    work = Path(".cache/training/t3").resolve()

    with pytest.raises(ValueError, match="overlaps checkpoint"):
        module._require_floor_paths_disjoint(
            checkpoint_dir=checkpoint,
            evaluation_dir=evaluation,
            cache_dir=cache,
            work_dir=checkpoint / "workers",
            output_path=Path(".cache/training/t3/floor.json").resolve(),
        )
    with pytest.raises(ValueError, match="repository-local .cache"):
        module.run_reference_variant(
            variant=module.perturbation_plan(max_threads=12)[0],
            checkpoint_dir=checkpoint,
            evaluation_dir=evaluation,
            cache_dir=cache,
            work_dir=work,
            input_combined_sha256="a" * 64,
            output_dir=tmp_path / "escaped-variant",
        )


def test_float64_compatibility_step_preserves_double_projection_inputs() -> None:
    import torch

    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["_float64_denoise_step"],
    )

    class Expert:
        def forward(self, **kwargs):
            suffix = kwargs["inputs_embeds"][1]
            return (None, suffix), None

    class TinyModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"use_cache": False, "chunk_size": 3})()
            self.vlm_with_expert = Expert()
            self.action_out_proj = torch.nn.Linear(2, 1, bias=False).double()

        def embed_suffix(self, x_t, timestep):
            del timestep
            return (
                x_t,
                torch.ones((1, 3), dtype=torch.bool),
                torch.ones((1, 3), dtype=torch.bool),
            )

    model = TinyModel()
    output = module._float64_denoise_step(
        model,
        prefix_pad_masks=torch.ones((1, 2), dtype=torch.bool),
        past_key_values=None,
        x_t=torch.ones((1, 3, 2), dtype=torch.float64),
        timestep=torch.ones((1,), dtype=torch.float64),
    )

    assert output.dtype == torch.float64
    assert output.shape == (1, 3, 1)


def test_torch_export_loader_honors_nondefault_float64_policy_and_processor() -> None:
    import gc
    import torch

    evaluation = __import__(
        "mlx_smolvla._lab.training.evaluation",
        fromlist=["_torch_observation", "load_evaluation_cases"],
    )
    reference_export = __import__(
        "mlx_smolvla._lab.training.reference_export",
        fromlist=["TorchExportPolicy"],
    )
    case = evaluation.load_evaluation_cases(
        Path(".cache/training/t3-evaluation")
    )[0]
    reference = reference_export.TorchExportPolicy.load(
        Path(".cache/training/t3/export"),
        cache_dir=Path(".cache/hf"),
        device="cpu",
        dtype=torch.float64,
    )
    batch = reference.preprocessor(evaluation._torch_observation(case))
    floating = [
        value
        for value in batch.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]

    assert reference.device.type == "cpu"
    assert reference.dtype == torch.float64
    assert floating and all(value.dtype == torch.float64 for value in floating)
    del reference, batch, floating
    gc.collect()


def test_self_consistency_cli_exposes_the_frozen_plan_without_model_loading() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compute_self_consistency_floor.py",
            "--list-perturbations",
            "--max-threads",
            "12",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["procedure_id"] == "smolvla-pytorch-self-consistency-v3"
    assert [item["name"] for item in payload["perturbations"]] == [
        "cpu_fp32_baseline",
        "cpu_fp32_threads_1",
        "cpu_fp32_threads_max",
        "mps_fp32_fallback_1",
        "mps_fp32_fallback_2",
        "mps_fp32_fallback_3",
        "mps_fp32_fallback_4",
        "mps_fp32_fallback_5",
        "cpu_float64",
    ]


def test_floor_input_evidence_names_every_hashed_location() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["collect_floor_input_evidence", "collect_floor_input_hashes"],
    )
    arguments = {
        "checkpoint_dir": ".cache/training/t3/export",
        "evaluation_dir": ".cache/training/t3-evaluation",
        "cache_dir": ".cache/hf",
    }

    inputs, tokenizer_snapshot = module.collect_floor_input_hashes(**arguments)
    evidence = module.collect_floor_input_evidence(**arguments)

    assert evidence["checkpoint_export"] == {
        "mode": "exact_tree",
        "root": ".cache/training/t3/export",
    }
    assert evidence["evaluation_artifact"] == {
        "mode": "exact_tree",
        "root": ".cache/training/t3-evaluation",
    }
    assert evidence["tokenizer_snapshot"] == {
        "mode": "contained_symlink_tree",
        "root": tokenizer_snapshot.relative_to(Path.cwd()).as_posix(),
        "allowed_root": ".cache/hf",
    }
    for group_name in ("pinned_dataset", "implementation"):
        assert evidence[group_name]["mode"] == "named_files"
        assert set(evidence[group_name]["paths"]) == set(
            inputs[group_name]["files"]
        )
        assert all(
            (Path.cwd() / recorded_path).is_file()
            for recorded_path in evidence[group_name]["paths"].values()
        )


def test_self_consistency_cli_assembles_completed_workers_without_rerunning_them() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=[
            "collect_floor_input_hashes",
            "perturbation_plan",
            "write_variant_artifact",
        ],
    )
    training_cache = Path(".cache/training").resolve()
    training_cache.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(prefix="self-consistency-cli-test-", dir=training_cache)
    )
    output = work_dir.with_name(f"{work_dir.name}-floor.json")
    try:
        inputs, _ = module.collect_floor_input_hashes(
            checkpoint_dir=".cache/training/t3/export",
            evaluation_dir=".cache/training/t3-evaluation",
            cache_dir=".cache/hf",
        )
        (work_dir / "input_sha256.json").write_text(
            json.dumps(inputs, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        plan = module.perturbation_plan(max_threads=18)
        for index, item in enumerate(plan):
            dtype = np.float64 if item.dtype == "float64" else np.float32
            actions = np.zeros((56, 50, 6), dtype=dtype)
            if item.name != module.BASELINE_VARIANT:
                actions[...] = index / 8
            module.write_variant_artifact(
                work_dir / "variants" / item.name,
                variant=item,
                normalized_actions=actions,
                input_combined_sha256=inputs["combined_sha256"],
                metadata=_runtime_metadata(item),
            )
        variants = sorted((work_dir / "variants").glob("*/metadata.json"))
        assert len(variants) == 9
        before_mtimes = {path: path.stat().st_mtime_ns for path in variants}
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/compute_self_consistency_floor.py",
                "--assemble-only",
                "--checkpoint",
                ".cache/training/t3/export",
                "--evaluation-dir",
                ".cache/training/t3-evaluation",
                "--cache-dir",
                ".cache/hf",
                "--work-dir",
                str(work_dir),
                "--input-manifest",
                str(work_dir / "input_sha256.json"),
                "--output",
                str(output),
                "--purpose",
                "retrospective_diagnostic",
                "--max-threads",
                "18",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        summary = json.loads(completed.stdout)
        report = json.loads(output.read_text(encoding="utf-8"))
        assert summary["mode"] == "assemble_only"
        assert summary["workers_started"] == 0
        assert report["F"] == 1.0
        assert report["F64"] == 1.0
        assert before_mtimes == {path: path.stat().st_mtime_ns for path in variants}
        assert not tuple(output.parent.glob(f".{output.name}.*"))
    finally:
        if output.exists():
            output.unlink()
        shutil.rmtree(work_dir)


def test_worker_environment_clears_every_documented_mps_switch() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["_worker_environment"],
    )
    plan = module.perturbation_plan(max_threads=12)
    inherited = {
        "PATH": "/bin",
        **{key: "inherited" for key in module.MPS_ENVIRONMENT_KEYS},
        **{key: "inherited" for key in module.CPU_THREAD_ENVIRONMENT_KEYS},
    }

    cpu = module._worker_environment(plan[0], inherited)
    mps = module._worker_environment(plan[3], inherited)

    assert cpu["PATH"] == "/bin"
    assert all(key not in cpu for key in module.MPS_ENVIRONMENT_KEYS)
    assert all(key not in cpu for key in module.CPU_THREAD_ENVIRONMENT_KEYS)
    assert mps["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert all(
        key not in mps
        for key in module.MPS_ENVIRONMENT_KEYS
        if key != "PYTORCH_ENABLE_MPS_FALLBACK"
    )
    assert module._mps_environment_snapshot(mps) == {
        key: ("1" if key == "PYTORCH_ENABLE_MPS_FALLBACK" else None)
        for key in module.MPS_ENVIRONMENT_KEYS
    }
    assert module._cpu_thread_environment_snapshot(mps) == {
        key: None for key in module.CPU_THREAD_ENVIRONMENT_KEYS
    }


def test_variant_root_rejects_a_symlinked_parent_on_reuse(
    tmp_path: Path,
) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["_require_variant_root"],
    )
    work = Path(".cache") / f"self-consistency-symlink-{os.getpid()}"
    target = tmp_path / "variants"
    target.mkdir()
    work.mkdir()
    (work / "variants").symlink_to(target, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="variant root.*symlink"):
            module._require_variant_root(work, create=False)
    finally:
        (work / "variants").unlink()
        work.rmdir()


@pytest.mark.parametrize("mode", ["run", "assemble"])
def test_floor_orchestration_rejects_symlinked_variant_root(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["assemble_existing_floor", "run_self_consistency_floor"],
    )
    work = Path(".cache") / f"self-consistency-{mode}-symlink-{os.getpid()}"
    output = Path(".cache") / f"self-consistency-{mode}-floor-{os.getpid()}.json"
    target = tmp_path / f"{mode}-variants"
    target.mkdir()
    work.mkdir()
    (work / "variants").symlink_to(target, target_is_directory=True)
    inputs = _input_hashes()
    monkeypatch.setattr(
        module,
        "collect_floor_input_hashes",
        lambda **kwargs: (inputs, Path(".cache/hf/tokenizer")),
    )
    try:
        with pytest.raises(ValueError, match="variant root.*symlink"):
            if mode == "run":
                module.run_self_consistency_floor(
                    checkpoint_dir=".cache/training/t3/export",
                    evaluation_dir=".cache/training/t3-evaluation",
                    cache_dir=".cache/hf",
                    work_dir=work,
                    output_path=output,
                    purpose="retrospective_diagnostic",
                    max_threads=12,
                )
            else:
                (work / "input_sha256.json").write_text(
                    json.dumps(inputs),
                    encoding="utf-8",
                )
                module.assemble_existing_floor(
                    checkpoint_dir=".cache/training/t3/export",
                    evaluation_dir=".cache/training/t3-evaluation",
                    cache_dir=".cache/hf",
                    work_dir=work,
                    input_manifest_path=work / "input_sha256.json",
                    output_path=output,
                    purpose="retrospective_diagnostic",
                    max_threads=12,
                )
    finally:
        if output.exists():
            output.unlink()
        manifest = work / "input_sha256.json"
        if manifest.exists():
            manifest.unlink()
        (work / "variants").unlink()
        work.rmdir()


@pytest.mark.parametrize("nanosecond_remainder", [499, 500, 999])
def test_utc_timestamp_generation_truncates_submicroseconds(
    nanosecond_remainder: int,
) -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["_utc_from_ns"],
    )
    created_at_ns = 1_788_264_000_123_456_000 + nanosecond_remainder

    created_at_utc = module._utc_from_ns(created_at_ns)

    assert created_at_utc == "2026-09-01T12:00:00.123456+00:00"
    assert module._validate_timestamp(created_at_utc, created_at_ns) == (
        created_at_utc,
        created_at_ns,
    )


def test_utc_timestamp_generation_handles_second_rollover_exactly() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["_utc_from_ns"],
    )
    created_at_ns = 1_788_264_001_000_000_999

    assert module._utc_from_ns(created_at_ns) == (
        "2026-09-01T12:00:01.000000+00:00"
    )


def test_mps_floor_has_five_separate_process_slots() -> None:
    module = __import__(
        "mlx_smolvla._lab.training.self_consistency",
        fromlist=["MPS_REPETITIONS", "perturbation_plan"],
    )
    plan = module.perturbation_plan(max_threads=12)
    mps = [item for item in plan if item.device == "mps"]

    assert module.MPS_REPETITIONS == 5
    assert [item.name for item in mps] == [
        "mps_fp32_fallback_1",
        "mps_fp32_fallback_2",
        "mps_fp32_fallback_3",
        "mps_fp32_fallback_4",
        "mps_fp32_fallback_5",
    ]
    assert all(item.dtype == "float32" and item.mps_fallback for item in mps)
    assert [item.family for item in mps] == ["mps_fp32_fallback"] * 5
    assert [item.replicate_index for item in mps] == [1, 2, 3, 4, 5]
    assert [item.replicate_count for item in mps] == [5] * 5


@pytest.mark.parametrize(
    ("worker", "expected_fallback"),
    [
        ("cpu_fp32_baseline", None),
        ("mps_fp32_fallback_1", "1"),
    ],
)
@pytest.mark.parametrize("worker_syntax", ["split", "equals"])
def test_hidden_worker_environment_is_fixed_before_numpy_import(
    worker: str,
    expected_fallback: str | None,
    worker_syntax: str,
) -> None:
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "99",
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "inherited",
    }
    worker_arguments = (
        ["--worker", worker]
        if worker_syntax == "split"
        else [f"--worker={worker}"]
    )
    code = (
        "import os,sys; "
        f"sys.argv={['floor', *worker_arguments]!r}; "
        "import scripts.compute_self_consistency_floor; "
        "print(repr(os.environ.get('OPENBLAS_NUM_THREADS'))); "
        "print(repr(os.environ.get('PYTORCH_MPS_FAST_MATH'))); "
        "print(repr(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'))); "
        "print('numpy' in sys.modules)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "None",
        "None",
        repr(expected_fallback),
        "True",
    ]
