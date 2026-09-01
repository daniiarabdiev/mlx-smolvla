"""PyTorch self-consistency floors for trained SmolVLA checkpoints.

The pure report-building functions in this module deliberately do not import
Torch, LeRobot, or MLX. Model execution lives behind the command-line worker so
each arithmetic perturbation starts in a clean process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
import platform
import random
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence
import io

from training.floor_runtime import (
    CPU_THREAD_ENVIRONMENT_KEYS,
    MPS_ENVIRONMENT_KEYS,
    cpu_thread_environment_snapshot as _cpu_thread_environment_snapshot,
    mps_environment_snapshot as _mps_environment_snapshot,
    worker_environment as _fixed_worker_environment,
)

import numpy as np


PROCEDURE_ID = "smolvla-pytorch-self-consistency-v3"
FLOOR_ARTIFACT_TYPE = "smolvla-pytorch-self-consistency-floor"
BASELINE_VARIANT = "cpu_fp32_baseline"
FLOAT64_VARIANT = "cpu_float64"
MPS_REPETITIONS = 5
MPS_VARIANTS = tuple(
    f"mps_fp32_fallback_{index}" for index in range(1, MPS_REPETITIONS + 1)
)
EVALUATION_SAMPLE_COUNT = 56
NORMALIZED_ACTION_CHUNK_SHAPE = (50, 6)
WORKER_SEED = 20_260_901
ORIGINAL_T3_MLX_MAX_ABS = 0.17762404680252075
PRE_V2_MPS_MAX_ABS = 0.35390492528676987
PRE_V2_FLOOR_SHA256 = "463cec238365c0bc9912e3fed3c011f86ad62dd330a213ca1dd5efd18c8e7c2a"
PRE_V2_INPUT_SHA256 = "38b5d0da2039e5d495dd28b15c4fa10a0f0c7bdf0c80da1622408bebedc705c1"
PRE_V2_MPS_ACTIONS_SHA256 = (
    "e309853482af9cfe1215a3d5854ecbe2edd134c6a8a2225495a7d5ecb95061b2"
)
PRE_V2_MPS_ARTIFACT_SHA256 = (
    "edbffeba927db307dc1ed23ddf3e73cbbe0b17c17f99b5390f81db8e919a84fa"
)
CLEAN_MPS_ACTIONS_SHA256 = (
    "065573d47fbfe17ff99ed6ce3fcd4c84ece91ef388db8dc8508c22cfc8d5e8d9"
)
FAST_MATH_MPS_ACTIONS_SHA256 = (
    "9bdacae0f03ddcd0638138b820699650fc8a08b149e7bb0d808ddd498d6d9574"
)
FAST_MATH_MPS_MAX_ABS = 1.8492990136146545
V2_SINGLE_FLOOR_SHA256 = (
    "cf867a1afb7c1fb4a815678e3d7b9403a3492b781dd535d7af052d45aa119168"
)
V2_SINGLE_INPUT_SHA256 = (
    "dbc33a69f3d777dcdf9667ee8da335d52efcb2e627c72cf372e680e1b8879a3a"
)
V2_SINGLE_MPS_ACTIONS_SHA256 = (
    "3f71aa4b8132db6bff311f14024cae80ba2d1ba0218cf80f2dfd31b7518274b6"
)
V2_SINGLE_MPS_MAX_ABS = 1.7168622612953186
MINIMUM_FREE_BYTES = 40 * 1024**3
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_CACHE = _REPOSITORY_ROOT / ".cache"
_VARIANT_RUNTIME_FIELDS = {
    "actual_threads",
    "interop_threads",
    "torch_version",
    "lerobot_version",
    "transformers_version",
    "platform",
    "machine",
    "python_version",
    "mps_built",
    "mps_available",
    "mps_fallback_environment",
    "mps_environment",
    "cpu_thread_environment",
    "worker_seed",
    "deterministic_algorithms",
    "float32_matmul_precision",
    "float64_compatibility_path",
}
_VARIANT_DOCUMENT_FIELDS = {
    "format_version",
    "artifact_type",
    "procedure_id",
    "name",
    "device",
    "dtype",
    "requested_threads",
    "mps_fallback",
    "family",
    "replicate_index",
    "replicate_count",
    *_VARIANT_RUNTIME_FIELDS,
    "input_combined_sha256",
    "sample_count",
    "normalized_action_chunk_shape",
    "normalized_actions_dtype",
    "normalized_actions_sha256",
    "variant_artifact_sha256",
}
_VARIANT_DERIVED_FIELDS = {
    "max_abs_vs_baseline",
    "case_max_abs",
    "worst_case",
}


@dataclass(frozen=True)
class Perturbation:
    """One prospectively fixed reference arithmetic configuration."""

    name: str
    device: str
    dtype: str
    requested_threads: int | None
    mps_fallback: bool
    family: str
    replicate_index: int
    replicate_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def perturbation_plan(*, max_threads: int) -> tuple[Perturbation, ...]:
    """Return the exact T3B Section 1 perturbation set in execution order."""

    if isinstance(max_threads, bool) or not isinstance(max_threads, int) or max_threads <= 0:
        raise ValueError("maximum thread count must be a positive integer")
    return (
        Perturbation(
            BASELINE_VARIANT,
            "cpu",
            "float32",
            None,
            False,
            BASELINE_VARIANT,
            1,
            1,
        ),
        Perturbation(
            "cpu_fp32_threads_1",
            "cpu",
            "float32",
            1,
            False,
            "cpu_fp32_threads_1",
            1,
            1,
        ),
        Perturbation(
            "cpu_fp32_threads_max",
            "cpu",
            "float32",
            max_threads,
            False,
            "cpu_fp32_threads_max",
            1,
            1,
        ),
        *(
            Perturbation(
                name,
                "mps",
                "float32",
                None,
                True,
                "mps_fp32_fallback",
                index,
                MPS_REPETITIONS,
            )
            for index, name in enumerate(MPS_VARIANTS, start=1)
        ),
        Perturbation(
            FLOAT64_VARIANT,
            "cpu",
            "float64",
            None,
            False,
            FLOAT64_VARIANT,
            1,
            1,
        ),
    )


def _worker_environment(
    variant: Perturbation,
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a fixed child environment before Torch imports the MPS backend."""

    return _fixed_worker_environment(
        mps_fallback=variant.mps_fallback,
        inherited=inherited,
    )


def _require_sha256(label: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_input_tree(
    root: str | Path,
    *,
    allowed_root: str | Path | None = None,
) -> dict[str, object]:
    """Hash the content of every logical file below ``root``.

    Hugging Face snapshots use file symlinks into a sibling blob store, so safe
    symlinks are dereferenced while targets outside ``allowed_root`` are refused.
    """

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"input tree is not a directory: {root}")
    allowed = Path(allowed_root if allowed_root is not None else root).resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_relative_to(allowed):
        raise ValueError(f"input tree escapes the allowed root: {root}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise FileNotFoundError(f"input tree entry is missing: {relative}") from error
        if not resolved.is_relative_to(allowed):
            raise ValueError(f"input tree entry escapes the allowed root: {relative}")
        if not resolved.is_file():
            raise ValueError(f"input tree entry is not a regular file: {relative}")
        files[relative] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if not files:
        raise ValueError(f"input tree contains no files: {root}")
    return {
        "tree_sha256": hashlib.sha256(_canonical_json(files)).hexdigest(),
        "files": files,
    }


def _hash_named_files(files: Mapping[str, Path]) -> dict[str, object]:
    hashes: dict[str, str] = {}
    for name, path in sorted(files.items()):
        if Path(name).is_absolute() or not name:
            raise ValueError(f"input hash label must be relative: {name!r}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise FileNotFoundError(f"input file is missing: {path}") from error
        if not resolved.is_file():
            raise ValueError(f"input path is not a regular file: {path}")
        hashes[name] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if not hashes:
        raise ValueError("named input file set cannot be empty")
    return {
        "tree_sha256": hashlib.sha256(_canonical_json(hashes)).hexdigest(),
        "files": hashes,
    }


def _require_cache_path(path: str | Path, *, label: str, directory: bool = True) -> Path:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    if not absolute.is_relative_to(_REPOSITORY_CACHE):
        raise ValueError(f"{label} must stay under the repository-local .cache directory")
    missing_parts: list[str] = []
    ancestor = absolute
    while not os.path.lexists(ancestor):
        missing_parts.append(ancestor.name)
        if ancestor.parent == ancestor:
            raise ValueError(f"{label} has no safe existing ancestor")
        ancestor = ancestor.parent
    if ancestor.is_symlink():
        raise ValueError(f"{label} has a symlinked ancestor: {ancestor}")
    resolved_ancestor = ancestor.resolve(strict=True)
    if resolved_ancestor != ancestor:
        raise ValueError(f"{label} has a symlinked ancestor: {ancestor}")
    resolved = resolved_ancestor.joinpath(*reversed(missing_parts))
    resolved_cache = _REPOSITORY_CACHE.resolve(strict=True)
    if not resolved.is_relative_to(resolved_cache):
        raise ValueError(f"{label} resolves outside the repository-local .cache directory")
    if directory:
        if not resolved.is_dir() or resolved.is_symlink():
            raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    return resolved


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _require_variant_root(work_dir: str | Path, *, create: bool) -> Path:
    """Return a real in-cache variants directory, never a symlinked reuse path."""

    work = _require_cache_path(
        work_dir,
        label="self-consistency work directory",
    )
    variant_root = work / "variants"
    if create:
        variant_root.mkdir(exist_ok=True)
    try:
        return _require_cache_path(
            variant_root,
            label="self-consistency variant root",
        )
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            "self-consistency variant root is missing or has a symlinked ancestor"
        ) from error


def _require_floor_paths_disjoint(
    *,
    checkpoint_dir: Path,
    evaluation_dir: Path,
    cache_dir: Path,
    work_dir: Path,
    output_path: Path,
) -> None:
    inputs = {
        "checkpoint": checkpoint_dir,
        "evaluation": evaluation_dir,
        "Hugging Face cache": cache_dir,
    }
    outputs = {"work directory": work_dir, "floor output": output_path}
    for output_label, output in outputs.items():
        for input_label, input_path in inputs.items():
            if _paths_overlap(output, input_path):
                raise ValueError(
                    f"self-consistency {output_label} overlaps {input_label} input"
                )
    if _paths_overlap(work_dir, output_path):
        raise ValueError("self-consistency work directory and floor output overlap")


def collect_floor_input_hashes(
    *,
    checkpoint_dir: str | Path,
    evaluation_dir: str | Path,
    cache_dir: str | Path,
) -> tuple[dict[str, object], Path]:
    """Hash every checkpoint, case/noise, processor, and implementation input."""

    checkpoint_dir = _require_cache_path(
        checkpoint_dir, label="self-consistency checkpoint"
    )
    evaluation_dir = _require_cache_path(
        evaluation_dir, label="self-consistency evaluation artifact"
    )
    cache_dir = _require_cache_path(cache_dir, label="Hugging Face cache")

    from training.evaluation import _require_pinned_dataset_root
    from training.reference_export import resolve_tokenizer_snapshot
    from reference.discovery import DATASET_REVISION

    dataset_root = _require_pinned_dataset_root(cache_dir)
    tokenizer_snapshot = resolve_tokenizer_snapshot(cache_dir)
    dataset_revision_tree = (
        dataset_root
        / ".cache"
        / "huggingface"
        / "trees"
        / f"{DATASET_REVISION}.json"
    )
    pinned_dataset_files = {
        "data/chunk-000/file-000.parquet": (
            dataset_root / "data/chunk-000/file-000.parquet"
        ),
        "meta/info.json": dataset_root / "meta/info.json",
        "meta/stats.json": dataset_root / "meta/stats.json",
        "meta/tasks.parquet": dataset_root / "meta/tasks.parquet",
        f"revision/{dataset_revision_tree.name}": dataset_revision_tree,
    }

    import lerobot.policies.smolvla.modeling_smolvla as modeling_smolvla
    import lerobot.policies.smolvla.smolvlm_with_expert as smolvlm_with_expert

    lerobot_root = Path(modeling_smolvla.__file__).resolve().parents[2]

    def distribution_record(distribution_name: str) -> Path:
        from importlib.metadata import distribution

        installed = distribution(distribution_name)
        record = next(
            (
                item
                for item in installed.files or ()
                if item.name == "RECORD" and item.parent.name.endswith(".dist-info")
            ),
            None,
        )
        if record is None:
            raise FileNotFoundError(
                f"installed distribution has no RECORD: {distribution_name}"
            )
        return Path(installed.locate_file(record))

    implementation_files = {
        "training/self_consistency.py": Path(__file__),
        "training/floor_runtime.py": _REPOSITORY_ROOT / "training/floor_runtime.py",
        "training/evaluation.py": _REPOSITORY_ROOT / "training/evaluation.py",
        "training/data.py": _REPOSITORY_ROOT / "training/data.py",
        "training/dataset.py": _REPOSITORY_ROOT / "training/dataset.py",
        "training/reference_export.py": _REPOSITORY_ROOT
        / "training/reference_export.py",
        "training/t3_contract.py": _REPOSITORY_ROOT / "training/t3_contract.py",
        "reference/discovery.py": _REPOSITORY_ROOT / "reference/discovery.py",
        "scripts/compute_self_consistency_floor.py": _REPOSITORY_ROOT
        / "scripts/compute_self_consistency_floor.py",
        "lerobot/policies/factory.py": lerobot_root / "policies/factory.py",
        "lerobot/policies/common/flow_matching.py": lerobot_root
        / "policies/common/flow_matching.py",
        "lerobot/policies/common/vla_utils.py": lerobot_root
        / "policies/common/vla_utils.py",
        "lerobot/policies/smolvla/configuration_smolvla.py": lerobot_root
        / "policies/smolvla/configuration_smolvla.py",
        "lerobot/modeling_smolvla.py": Path(modeling_smolvla.__file__),
        "lerobot/policies/smolvla/processor_smolvla.py": lerobot_root
        / "policies/smolvla/processor_smolvla.py",
        "lerobot/smolvlm_with_expert.py": Path(smolvlm_with_expert.__file__),
        "lerobot/processor/batch_processor.py": lerobot_root
        / "processor/batch_processor.py",
        "lerobot/processor/converters.py": lerobot_root / "processor/converters.py",
        "lerobot/processor/device_processor.py": lerobot_root
        / "processor/device_processor.py",
        "lerobot/processor/factory.py": lerobot_root / "processor/factory.py",
        "lerobot/processor/newline_task_processor.py": lerobot_root
        / "processor/newline_task_processor.py",
        "lerobot/processor/normalize_processor.py": lerobot_root
        / "processor/normalize_processor.py",
        "lerobot/processor/pipeline.py": lerobot_root / "processor/pipeline.py",
        "lerobot/processor/rename_processor.py": lerobot_root
        / "processor/rename_processor.py",
        "lerobot/processor/tokenizer_processor.py": lerobot_root
        / "processor/tokenizer_processor.py",
        "distribution/lerobot/RECORD": distribution_record("lerobot"),
        "distribution/torch/RECORD": distribution_record("torch"),
        "distribution/transformers/RECORD": distribution_record("transformers"),
        "distribution/numpy/RECORD": distribution_record("numpy"),
        "distribution/safetensors/RECORD": distribution_record("safetensors"),
        "distribution/huggingface-hub/RECORD": distribution_record(
            "huggingface-hub"
        ),
        "distribution/tokenizers/RECORD": distribution_record("tokenizers"),
        "uv.lock": _REPOSITORY_ROOT / "uv.lock",
    }
    groups: dict[str, object] = {
        "checkpoint_export": hash_input_tree(
            checkpoint_dir, allowed_root=_REPOSITORY_CACHE
        ),
        "evaluation_artifact": hash_input_tree(
            evaluation_dir, allowed_root=_REPOSITORY_CACHE
        ),
        "pinned_dataset": _hash_named_files(pinned_dataset_files),
        "tokenizer_snapshot": hash_input_tree(
            tokenizer_snapshot, allowed_root=cache_dir
        ),
        "implementation": _hash_named_files(implementation_files),
    }
    combined = hashlib.sha256(_canonical_json(groups)).hexdigest()
    inputs = {**groups, "combined_sha256": combined}
    return _validated_input_hashes(inputs), tokenizer_snapshot


def _numpy_payload(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return buffer.getvalue()


def _write_bytes_synced(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_variant_document(
    value: object,
    *,
    expected_variant: Perturbation,
    expected_input_combined_sha256: str,
    include_derived: bool,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("variant metadata must be a JSON object")
    expected_fields = _VARIANT_DOCUMENT_FIELDS | (
        _VARIANT_DERIVED_FIELDS if include_derived else set()
    )
    if set(value) != expected_fields:
        missing = expected_fields - set(value)
        if "input_combined_sha256" in missing:
            raise ValueError("variant input digest is missing")
        if "mps_fallback_environment" in missing:
            raise ValueError("variant MPS fallback evidence is missing")
        if "mps_environment" in missing:
            raise ValueError("variant MPS environment evidence is missing")
        if "float64_compatibility_path" in missing:
            raise ValueError("variant float64 compatibility evidence is missing")
        raise ValueError(
            "variant metadata fields differ from the fixed schema: "
            f"missing={sorted(missing)}, unexpected={sorted(set(value) - expected_fields)}"
        )
    if (
        value.get("format_version") != 1
        or value.get("artifact_type")
        != "smolvla-pytorch-self-consistency-variant"
        or value.get("procedure_id") != PROCEDURE_ID
        or value.get("sample_count") != EVALUATION_SAMPLE_COUNT
        or value.get("normalized_action_chunk_shape")
        != list(NORMALIZED_ACTION_CHUNK_SHAPE)
    ):
        raise ValueError("variant artifact identity is invalid")
    if any(
        value.get(name) != expected
        for name, expected in expected_variant.as_dict().items()
    ):
        raise ValueError("variant identity differs from the fixed perturbation plan")
    expected_input = _require_sha256(
        "expected variant input", expected_input_combined_sha256
    )
    if value.get("input_combined_sha256") != expected_input:
        raise ValueError("variant input digest differs from the floor inputs")

    actual_threads = value.get("actual_threads")
    interop_threads = value.get("interop_threads")
    if (
        isinstance(actual_threads, bool)
        or not isinstance(actual_threads, int)
        or actual_threads <= 0
        or (
            expected_variant.requested_threads is not None
            and actual_threads != expected_variant.requested_threads
        )
    ):
        raise ValueError("variant actual thread count differs from its request")
    if (
        isinstance(interop_threads, bool)
        or not isinstance(interop_threads, int)
        or interop_threads <= 0
    ):
        raise ValueError("variant interop thread count is invalid")
    for field in (
        "torch_version",
        "lerobot_version",
        "transformers_version",
        "platform",
        "machine",
        "python_version",
    ):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"variant runtime field is invalid: {field}")
    if not isinstance(value.get("mps_built"), bool) or not isinstance(
        value.get("mps_available"), bool
    ):
        raise ValueError("variant MPS runtime evidence is invalid")
    if expected_variant.device == "mps":
        if not value["mps_built"] or not value["mps_available"]:
            raise ValueError("MPS perturbation requires built and available MPS")
        if value.get("mps_fallback_environment") != "1":
            raise ValueError("MPS fallback was not enabled before Torch import")
    elif value.get("mps_fallback_environment") is not None:
        raise ValueError("non-MPS perturbation must not inherit MPS fallback state")
    expected_mps_environment = {
        key: (
            "1"
            if expected_variant.device == "mps"
            and key == "PYTORCH_ENABLE_MPS_FALLBACK"
            else None
        )
        for key in MPS_ENVIRONMENT_KEYS
    }
    if value.get("mps_environment") != expected_mps_environment:
        raise ValueError("variant MPS environment differs from the fixed procedure")
    expected_cpu_environment = {
        key: None for key in CPU_THREAD_ENVIRONMENT_KEYS
    }
    if value.get("cpu_thread_environment") != expected_cpu_environment:
        raise ValueError(
            "variant CPU thread environment differs from the fixed procedure"
        )
    if value.get("worker_seed") != WORKER_SEED:
        raise ValueError("variant worker seed differs from the fixed procedure")
    if value.get("deterministic_algorithms") is not False:
        raise ValueError("variant deterministic-algorithm mode changed")
    if value.get("float32_matmul_precision") != "highest":
        raise ValueError("variant float32 matmul precision changed")
    expected_float64_marker = (
        "projection_weight_dtype" if expected_variant.dtype == "float64" else None
    )
    if value.get("float64_compatibility_path") != expected_float64_marker:
        raise ValueError("variant float64 compatibility path is invalid")
    if value.get("normalized_actions_dtype") != expected_variant.dtype:
        raise ValueError("variant output dtype differs from the perturbation dtype")
    _require_sha256(
        "variant normalized actions", value.get("normalized_actions_sha256")
    )
    recorded_artifact = _require_sha256(
        "variant artifact", value.get("variant_artifact_sha256")
    )
    artifact_document = {
        key: value[key]
        for key in _VARIANT_DOCUMENT_FIELDS
        if key != "variant_artifact_sha256"
    }
    if hashlib.sha256(_canonical_json(artifact_document)).hexdigest() != recorded_artifact:
        raise ValueError("variant artifact SHA-256 does not match its metadata")
    return dict(value)


def write_variant_artifact(
    root: str | Path,
    *,
    variant: Perturbation,
    normalized_actions: np.ndarray,
    input_combined_sha256: str,
    metadata: Mapping[str, object],
) -> str:
    """Atomically persist one isolated perturbation's complete output."""

    root = Path(root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"variant artifact already exists: {root}")
    input_digest = _require_sha256("variant input", input_combined_sha256)
    actions = np.asarray(normalized_actions)
    expected_shape = (EVALUATION_SAMPLE_COUNT, *NORMALIZED_ACTION_CHUNK_SHAPE)
    if actions.shape != expected_shape:
        raise ValueError(f"variant normalized action shape {actions.shape} != {expected_shape}")
    if not np.issubdtype(actions.dtype, np.floating) or not np.isfinite(actions).all():
        raise ValueError("variant normalized actions must be finite floating point")
    if not isinstance(metadata, Mapping) or set(metadata) != _VARIANT_RUNTIME_FIELDS:
        raise ValueError("variant runtime metadata differs from the fixed schema")

    actions_payload = _numpy_payload(actions)
    actions_digest = hashlib.sha256(actions_payload).hexdigest()
    document: dict[str, object] = {
        "format_version": 1,
        "artifact_type": "smolvla-pytorch-self-consistency-variant",
        "procedure_id": PROCEDURE_ID,
        **variant.as_dict(),
        **dict(metadata),
        "input_combined_sha256": input_digest,
        "sample_count": EVALUATION_SAMPLE_COUNT,
        "normalized_action_chunk_shape": list(NORMALIZED_ACTION_CHUNK_SHAPE),
        "normalized_actions_dtype": actions.dtype.name,
        "normalized_actions_sha256": actions_digest,
    }
    artifact_digest = hashlib.sha256(_canonical_json(document)).hexdigest()
    document["variant_artifact_sha256"] = artifact_digest
    _validate_variant_document(
        document,
        expected_variant=variant,
        expected_input_combined_sha256=input_digest,
        include_derived=False,
    )
    metadata_payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        _write_bytes_synced(temporary / "normalized_actions.npy", actions_payload)
        _write_bytes_synced(temporary / "metadata.json", metadata_payload)
        temporary.replace(root)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return artifact_digest


def read_variant_artifact(
    root: str | Path,
    *,
    expected_variant: Perturbation,
    expected_input_combined_sha256: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load and verify one perturbation output before floor aggregation."""

    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"variant artifact directory is missing or unsafe: {root}")
    action_path = root / "normalized_actions.npy"
    metadata_path = root / "metadata.json"
    if (
        not action_path.is_file()
        or action_path.is_symlink()
        or not metadata_path.is_file()
        or metadata_path.is_symlink()
    ):
        raise FileNotFoundError(f"variant artifact is incomplete or unsafe: {root}")
    try:
        document = json.loads(metadata_path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("variant metadata is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("variant metadata must be a JSON object")
    document = _validate_variant_document(
        document,
        expected_variant=expected_variant,
        expected_input_combined_sha256=expected_input_combined_sha256,
        include_derived=False,
    )
    actions_payload = action_path.read_bytes()
    if hashlib.sha256(actions_payload).hexdigest() != _require_sha256(
        "variant normalized actions", document.get("normalized_actions_sha256")
    ):
        raise ValueError("variant normalized-action SHA-256 mismatch")
    try:
        actions = np.load(io.BytesIO(actions_payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("variant normalized actions are not a valid NumPy array") from error
    expected_shape = (EVALUATION_SAMPLE_COUNT, *NORMALIZED_ACTION_CHUNK_SHAPE)
    if (
        actions.shape != expected_shape
        or actions.dtype.name != document.get("normalized_actions_dtype")
        or not np.issubdtype(actions.dtype, np.floating)
        or not np.isfinite(actions).all()
    ):
        raise ValueError("variant normalized actions differ from their metadata")
    return np.ascontiguousarray(actions), document


def _float64_denoise_step(
    model,
    prefix_pad_masks,
    past_key_values,
    x_t,
    timestep,
):
    """LeRobot's denoise step with its fp32 output cast generalized to weight dtype.

    LeRobot 0.6.1 intentionally upcasts the expert output to fp32 before the
    action projection. Once the complete model is cast to float64, that literal
    fp32 cast is incompatible with the float64 projection. This copy is otherwise
    line-for-line equivalent and casts to the projection's actual weight dtype.
    """

    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, timestep)
    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
        batch_size, suffix_len, prefix_len
    )
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat(
        [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
    )
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    outputs_embeds, _ = model.vlm_with_expert.forward(
        attention_mask=full_att_2d_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=model.config.use_cache,
    )
    if past_key_values is not None:
        past_key_values.crop(prefix_len)
    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -model.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=model.action_out_proj.weight.dtype)
    return model.action_out_proj(suffix_out)


def run_reference_variant(
    *,
    variant: Perturbation,
    checkpoint_dir: str | Path,
    evaluation_dir: str | Path,
    cache_dir: str | Path,
    work_dir: str | Path,
    input_combined_sha256: str,
    output_dir: str | Path,
) -> str:
    """Run all frozen cases for one perturbation and persist its chunks."""

    if variant.mps_fallback and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        raise RuntimeError(
            "MPS perturbation requires PYTORCH_ENABLE_MPS_FALLBACK=1 before Torch import"
        )
    checkpoint_dir = _require_cache_path(
        checkpoint_dir, label="self-consistency checkpoint"
    )
    evaluation_dir = _require_cache_path(
        evaluation_dir, label="self-consistency evaluation artifact"
    )
    cache_dir = _require_cache_path(cache_dir, label="Hugging Face cache")
    work_dir = _require_cache_path(
        work_dir, label="self-consistency work directory"
    )
    output_dir = _require_cache_path(
        output_dir,
        label="self-consistency variant output",
        directory=False,
    )
    expected_output = work_dir / "variants" / variant.name
    if output_dir != expected_output:
        raise ValueError(
            "self-consistency variant output must match its verified work directory"
        )

    import torch
    from types import MethodType

    random.seed(WORKER_SEED)
    np.random.seed(WORKER_SEED)
    torch.manual_seed(WORKER_SEED)
    if variant.device == "mps":
        torch.mps.manual_seed(WORKER_SEED)
    torch.set_float32_matmul_precision("highest")
    if variant.requested_threads is not None:
        torch.set_num_threads(variant.requested_threads)
    if variant.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("fixed MPS perturbation requested but MPS is unavailable")
    dtype = {"float32": torch.float32, "float64": torch.float64}[variant.dtype]

    from training.evaluation import _torch_observation, load_evaluation_cases
    from training.reference_export import TorchExportPolicy

    cases = load_evaluation_cases(evaluation_dir)
    reference = TorchExportPolicy.load(
        checkpoint_dir,
        cache_dir=cache_dir,
        device=variant.device,
        dtype=dtype,
    )
    if reference.device.type != variant.device or reference.dtype != dtype:
        raise RuntimeError("loaded reference policy differs from perturbation device/dtype")
    if variant.dtype == "float64":
        reference.policy.model.denoise_step = MethodType(
            _float64_denoise_step,
            reference.policy.model,
        )

    outputs: list[np.ndarray] = []
    for index, case in enumerate(cases):
        reference.policy.reset()
        batch = reference.preprocessor(_torch_observation(case))
        floating_inputs = [
            value
            for value in batch.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ]
        if not floating_inputs or any(
            value.device.type != variant.device or value.dtype != dtype
            for value in floating_inputs
        ):
            raise RuntimeError(
                "reference preprocessor did not honor perturbation device/dtype"
            )
        noise = torch.from_numpy(case.noise.copy()).to(
            device=variant.device,
            dtype=dtype,
        )
        with torch.inference_mode():
            normalized = reference.policy.predict_action_chunk(batch, noise=noise)
        if normalized.shape != (1, *NORMALIZED_ACTION_CHUNK_SHAPE):
            raise RuntimeError(
                f"reference normalized action shape changed: {tuple(normalized.shape)}"
            )
        if normalized.device.type != variant.device or normalized.dtype != dtype:
            raise RuntimeError(
                "reference normalized output differs from perturbation device/dtype"
            )
        output = normalized.detach().cpu().numpy()[0]
        if not np.isfinite(output).all():
            raise RuntimeError(f"non-finite normalized actions at case {case.ordinal}")
        outputs.append(np.ascontiguousarray(output))
        if index == 0 or (index + 1) % 7 == 0 or index + 1 == len(cases):
            print(
                f"{variant.name} evaluation {index + 1}/{len(cases)}",
                flush=True,
            )
    actions = np.stack(outputs, axis=0)
    runtime_metadata = {
        "actual_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "torch_version": torch.__version__,
        "lerobot_version": package_version("lerobot"),
        "transformers_version": package_version("transformers"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_environment": os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK"
        ),
        "mps_environment": _mps_environment_snapshot(),
        "cpu_thread_environment": _cpu_thread_environment_snapshot(),
        "worker_seed": WORKER_SEED,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "float64_compatibility_path": (
            "projection_weight_dtype" if variant.dtype == "float64" else None
        ),
    }
    return write_variant_artifact(
        output_dir,
        variant=variant,
        normalized_actions=actions,
        input_combined_sha256=input_combined_sha256,
        metadata=runtime_metadata,
    )


def _atomic_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def _case_identities_from_hashed_metadata(
    evaluation_dir: Path,
    inputs: Mapping[str, object],
) -> tuple[dict[str, int], ...]:
    metadata_path = evaluation_dir / "metadata.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise FileNotFoundError("held-out evaluation metadata is missing or unsafe")
    expected_digest = inputs["evaluation_artifact"]["files"].get("metadata.json")
    actual_digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("held-out evaluation metadata differs from the input manifest")
    try:
        metadata = json.loads(metadata_path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("held-out evaluation metadata is not valid JSON") from error
    source_identity = _fixed_source_identity()
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("artifact_type") != "smolvla-lora-heldout-evaluation"
        or metadata.get("sample_count") != EVALUATION_SAMPLE_COUNT
        or metadata.get("noise_seed") != source_identity["evaluation"]["noise_seed"]
        or metadata.get("dataset") != source_identity["dataset"]
    ):
        raise ValueError("held-out evaluation metadata identity changed")
    raw_cases = metadata.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("held-out evaluation case identities are missing")
    try:
        identities = tuple(
            {
                "ordinal": int(case["ordinal"]),
                "episode": int(case["episode"]),
                "frame_index": int(case["frame_index"]),
                "absolute_index": int(case["absolute_index"]),
            }
            for case in raw_cases
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("held-out evaluation case identities are invalid") from error
    return _validated_cases(identities)


def assemble_existing_floor(
    *,
    checkpoint_dir: str | Path,
    evaluation_dir: str | Path,
    cache_dir: str | Path,
    work_dir: str | Path,
    input_manifest_path: str | Path,
    output_path: str | Path,
    purpose: str,
    max_threads: int,
) -> tuple[dict[str, object], str]:
    """Aggregate completed workers without importing or loading a policy model."""

    if purpose not in {"retrospective_diagnostic", "prospective_gate"}:
        raise ValueError("self-consistency floor purpose is invalid")
    checkpoint_dir = _require_cache_path(
        checkpoint_dir, label="self-consistency checkpoint"
    )
    evaluation_dir = _require_cache_path(
        evaluation_dir, label="self-consistency evaluation artifact"
    )
    cache_dir = _require_cache_path(cache_dir, label="Hugging Face cache")
    work_dir = _require_cache_path(
        work_dir, label="self-consistency work directory"
    )
    input_manifest_path = _require_cache_path(
        input_manifest_path,
        label="self-consistency input manifest",
        directory=False,
    )
    output_path = _require_cache_path(
        output_path, label="self-consistency floor output", directory=False
    )
    if input_manifest_path != work_dir / "input_sha256.json":
        raise ValueError("input manifest must be the verified work-directory manifest")
    if not input_manifest_path.is_file() or input_manifest_path.is_symlink():
        raise FileNotFoundError("self-consistency input manifest is missing or unsafe")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"self-consistency floor already exists: {output_path}")
    _require_floor_paths_disjoint(
        checkpoint_dir=checkpoint_dir,
        evaluation_dir=evaluation_dir,
        cache_dir=cache_dir,
        work_dir=work_dir,
        output_path=output_path,
    )
    try:
        recorded_inputs = json.loads(input_manifest_path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("self-consistency input manifest is not valid JSON") from error
    inputs = _validated_input_hashes(recorded_inputs)
    current_inputs, _ = collect_floor_input_hashes(
        checkpoint_dir=checkpoint_dir,
        evaluation_dir=evaluation_dir,
        cache_dir=cache_dir,
    )
    if current_inputs != inputs:
        raise ValueError("self-consistency input manifest differs from current inputs")

    variant_root = _require_variant_root(work_dir, create=False)
    actions: dict[str, np.ndarray] = {}
    metadata: dict[str, Mapping[str, object]] = {}
    for variant in perturbation_plan(max_threads=max_threads):
        variant_actions, variant_metadata = read_variant_artifact(
            variant_root / variant.name,
            expected_variant=variant,
            expected_input_combined_sha256=str(inputs["combined_sha256"]),
        )
        actions[variant.name] = variant_actions
        metadata[variant.name] = variant_metadata
    identities = _case_identities_from_hashed_metadata(evaluation_dir, inputs)
    final_inputs, _ = collect_floor_input_hashes(
        checkpoint_dir=checkpoint_dir,
        evaluation_dir=evaluation_dir,
        cache_dir=cache_dir,
    )
    if final_inputs != inputs:
        raise RuntimeError("self-consistency inputs changed during floor assembly")
    created_at_ns = time.time_ns()
    created_at_utc = _utc_from_ns(created_at_ns)
    report = assemble_floor_report(
        actions=actions,
        variant_metadata=metadata,
        case_identities=identities,
        input_sha256=inputs,
        checkpoint_path=checkpoint_dir.relative_to(_REPOSITORY_ROOT).as_posix(),
        purpose=purpose,
        created_at_utc=created_at_utc,
        created_at_ns=created_at_ns,
    )
    return report, write_floor_report(output_path, report)


def run_self_consistency_floor(
    *,
    checkpoint_dir: str | Path,
    evaluation_dir: str | Path,
    cache_dir: str | Path,
    work_dir: str | Path,
    output_path: str | Path,
    purpose: str,
    max_threads: int,
    script_path: str | Path | None = None,
) -> tuple[dict[str, object], str]:
    """Run/reuse isolated workers and atomically write a complete floor."""

    if purpose not in {"retrospective_diagnostic", "prospective_gate"}:
        raise ValueError("self-consistency floor purpose is invalid")
    checkpoint_dir = _require_cache_path(
        checkpoint_dir, label="self-consistency checkpoint"
    )
    evaluation_dir = _require_cache_path(
        evaluation_dir, label="self-consistency evaluation artifact"
    )
    cache_dir = _require_cache_path(cache_dir, label="Hugging Face cache")
    work_dir = _require_cache_path(
        work_dir, label="self-consistency work directory", directory=False
    )
    output_path = _require_cache_path(
        output_path, label="self-consistency floor output", directory=False
    )
    _require_floor_paths_disjoint(
        checkpoint_dir=checkpoint_dir,
        evaluation_dir=evaluation_dir,
        cache_dir=cache_dir,
        work_dir=work_dir,
        output_path=output_path,
    )
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"self-consistency floor already exists: {output_path}")
    if shutil.disk_usage(_REPOSITORY_CACHE).free < MINIMUM_FREE_BYTES:
        raise RuntimeError("self-consistency floor requires at least 40 GiB free")
    work_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.is_symlink():
        raise ValueError("self-consistency work directory cannot be a symlink")

    inputs, _ = collect_floor_input_hashes(
        checkpoint_dir=checkpoint_dir,
        evaluation_dir=evaluation_dir,
        cache_dir=cache_dir,
    )
    _atomic_json(work_dir / "input_sha256.json", inputs)
    input_digest = str(inputs["combined_sha256"])
    plan = perturbation_plan(max_threads=max_threads)
    canonical_script = (
        _REPOSITORY_ROOT / "scripts/compute_self_consistency_floor.py"
    ).resolve(strict=True)
    script = Path(script_path if script_path is not None else canonical_script).resolve(
        strict=True
    )
    if script != canonical_script:
        raise ValueError("self-consistency workers must use the hashed canonical script")
    variant_root = _require_variant_root(work_dir, create=True)
    actions: dict[str, np.ndarray] = {}
    metadata: dict[str, Mapping[str, object]] = {}
    for variant in plan:
        destination = variant_root / variant.name
        if not destination.exists():
            command = [
                sys.executable,
                str(script),
                "--worker",
                variant.name,
                "--max-threads",
                str(max_threads),
                "--checkpoint",
                str(checkpoint_dir),
                "--evaluation-dir",
                str(evaluation_dir),
                "--cache-dir",
                str(cache_dir),
                "--work-dir",
                str(work_dir),
                "--input-sha256",
                input_digest,
                "--variant-output",
                str(destination),
            ]
            environment = _worker_environment(variant)
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"self-consistency worker failed for {variant.name}: "
                    f"exit {completed.returncode}"
                )
        variant_actions, variant_metadata = read_variant_artifact(
            destination,
            expected_variant=variant,
            expected_input_combined_sha256=input_digest,
        )
        actions[variant.name] = variant_actions
        metadata[variant.name] = variant_metadata

    final_inputs, _ = collect_floor_input_hashes(
        checkpoint_dir=checkpoint_dir,
        evaluation_dir=evaluation_dir,
        cache_dir=cache_dir,
    )
    if final_inputs != inputs:
        raise RuntimeError("self-consistency inputs changed while workers were running")
    from training.evaluation import load_evaluation_cases

    cases = load_evaluation_cases(evaluation_dir)
    identities = tuple(
        {
            "ordinal": case.ordinal,
            "episode": case.episode,
            "frame_index": case.frame_index,
            "absolute_index": case.absolute_index,
        }
        for case in cases
    )
    created_at_ns = time.time_ns()
    created_at_utc = _utc_from_ns(created_at_ns)
    checkpoint_relative = checkpoint_dir.relative_to(_REPOSITORY_ROOT).as_posix()
    report = assemble_floor_report(
        actions=actions,
        variant_metadata=metadata,
        case_identities=identities,
        input_sha256=inputs,
        checkpoint_path=checkpoint_relative,
        purpose=purpose,
        created_at_utc=created_at_utc,
        created_at_ns=created_at_ns,
    )
    digest = write_floor_report(output_path, report)
    return report, digest


def _validated_input_hashes(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("input_sha256 must be a mapping")
    expected_groups = {
        "checkpoint_export",
        "evaluation_artifact",
        "pinned_dataset",
        "tokenizer_snapshot",
        "implementation",
    }
    if set(value) != expected_groups | {"combined_sha256"}:
        raise ValueError("input_sha256 groups differ from the fixed floor schema")
    normalized: dict[str, object] = {}
    for group_name in sorted(expected_groups):
        group = value[group_name]
        if not isinstance(group, Mapping) or set(group) != {"tree_sha256", "files"}:
            raise ValueError(f"{group_name} hash group is incomplete")
        files = group["files"]
        if not isinstance(files, Mapping) or not files:
            raise ValueError(f"{group_name} file hashes are incomplete")
        normalized_files: dict[str, str] = {}
        for path, digest in sorted(files.items()):
            candidate = Path(path) if isinstance(path, str) else Path()
            if (
                not isinstance(path, str)
                or not path
                or candidate.is_absolute()
                or any(part in ("", ".", "..") for part in candidate.parts)
            ):
                raise ValueError(f"{group_name} contains an invalid relative input path")
            normalized_files[path] = _require_sha256(
                f"{group_name} input {path}", digest
            )
        tree_sha256 = _require_sha256(
            f"{group_name} tree", group["tree_sha256"]
        )
        computed_tree_sha256 = hashlib.sha256(
            _canonical_json(normalized_files)
        ).hexdigest()
        if tree_sha256 != computed_tree_sha256:
            raise ValueError(
                f"{group_name} tree SHA-256 does not bind its file hashes"
            )
        normalized[group_name] = {
            "tree_sha256": tree_sha256,
            "files": normalized_files,
        }
    recorded_combined = _require_sha256(
        "combined input", value["combined_sha256"]
    )
    computed_combined = hashlib.sha256(_canonical_json(normalized)).hexdigest()
    if recorded_combined != computed_combined:
        raise ValueError("combined input SHA-256 does not bind the input hash groups")
    return {**normalized, "combined_sha256": recorded_combined}


def _validated_cases(
    case_identities: Sequence[Mapping[str, object]],
) -> tuple[dict[str, int], ...]:
    if len(case_identities) != EVALUATION_SAMPLE_COUNT:
        raise ValueError(
            f"self-consistency floor requires {EVALUATION_SAMPLE_COUNT} case identities"
        )
    normalized: list[dict[str, int]] = []
    for ordinal, value in enumerate(case_identities):
        required = {"ordinal", "episode", "frame_index", "absolute_index"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"case identity {ordinal} differs from the fixed schema")
        try:
            record = {key: int(value[key]) for key in required}
        except (TypeError, ValueError) as error:
            raise ValueError(f"case identity {ordinal} is invalid") from error
        if record["ordinal"] != ordinal:
            raise ValueError(f"case identity ordinal changed at {ordinal}")
        normalized.append(
            {
                "ordinal": record["ordinal"],
                "episode": record["episode"],
                "frame_index": record["frame_index"],
                "absolute_index": record["absolute_index"],
            }
        )
    if len({item["absolute_index"] for item in normalized}) != len(normalized):
        raise ValueError("self-consistency case absolute indices are not unique")
    return tuple(normalized)


def _utc_from_ns(created_at_ns: int) -> str:
    """Format epoch nanoseconds as UTC, truncating to JSON's microsecond precision."""

    if (
        isinstance(created_at_ns, bool)
        or not isinstance(created_at_ns, int)
        or created_at_ns <= 0
    ):
        raise ValueError("floor nanosecond timestamp must be a positive integer")
    seconds, nanoseconds = divmod(created_at_ns, 1_000_000_000)
    created = datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(
        microseconds=nanoseconds // 1_000
    )
    return created.isoformat(timespec="microseconds")


def _validate_timestamp(created_at_utc: object, created_at_ns: object) -> tuple[str, int]:
    if not isinstance(created_at_utc, str):
        raise ValueError("floor UTC timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(created_at_utc)
    except ValueError as error:
        raise ValueError("floor UTC timestamp is invalid") from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise ValueError("floor UTC timestamp must use UTC")
    if isinstance(created_at_ns, bool) or not isinstance(created_at_ns, int) or created_at_ns <= 0:
        raise ValueError("floor nanosecond timestamp must be a positive integer")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed.astimezone(timezone.utc) - epoch
    parsed_microseconds = (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )
    if created_at_ns // 1_000 != parsed_microseconds:
        raise ValueError("floor UTC and nanosecond timestamps must identify the same instant")
    return created_at_utc, created_at_ns


def _fixed_source_identity() -> dict[str, object]:
    from reference.discovery import (
        BASE_VLM_ID,
        BASE_VLM_REVISION,
        DATASET_ID,
        DATASET_REVISION,
    )
    from training.t3_contract import (
        FROZEN_EVALUATION_MANIFEST_SHA256,
        FROZEN_EVALUATION_METADATA_SHA256,
    )

    return {
        "checkpoint_role": "retained-t3-merged-fp32-export",
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "tokenizer": {"id": BASE_VLM_ID, "revision": BASE_VLM_REVISION},
        "evaluation": {
            "sample_count": EVALUATION_SAMPLE_COUNT,
            "noise_seed": 20_260_902,
            "manifest_sha256": FROZEN_EVALUATION_MANIFEST_SHA256,
            "metadata_sha256": FROZEN_EVALUATION_METADATA_SHA256,
        },
    }


def _floor_context(purpose: str) -> dict[str, object]:
    historical = None
    if purpose == "retrospective_diagnostic":
        historical = {
            "status": "preserved_pre_v2_uncontrolled_environment_observation",
            "max_abs_vs_baseline": PRE_V2_MPS_MAX_ABS,
            "floor_sha256": PRE_V2_FLOOR_SHA256,
            "input_combined_sha256": PRE_V2_INPUT_SHA256,
            "normalized_actions_sha256": PRE_V2_MPS_ACTIONS_SHA256,
            "variant_artifact_sha256": PRE_V2_MPS_ARTIFACT_SHA256,
            "recorded_mps_environment": {
                "PYTORCH_ENABLE_MPS_FALLBACK": "1"
            },
            "unrecorded_environment_limit": (
                "the remaining documented PYTORCH_MPS_* variables were not captured"
            ),
            "included_in_v3_F": False,
            "clean_default_followup": {
                "fresh_process_count": 6,
                "all_byte_identical": True,
                "normalized_actions_sha256": CLEAN_MPS_ACTIONS_SHA256,
                "included_in_v3_F": False,
            },
            "fast_math_diagnostic": {
                "PYTORCH_MPS_FAST_MATH": "1",
                "max_abs_vs_baseline": FAST_MATH_MPS_MAX_ABS,
                "normalized_actions_sha256": FAST_MATH_MPS_ACTIONS_SHA256,
                "included_in_v3_F": False,
            },
            "sanitized_v2_single_process": {
                "floor_sha256": V2_SINGLE_FLOOR_SHA256,
                "input_combined_sha256": V2_SINGLE_INPUT_SHA256,
                "max_abs_vs_baseline": V2_SINGLE_MPS_MAX_ABS,
                "normalized_actions_sha256": V2_SINGLE_MPS_ACTIONS_SHA256,
                "included_in_v3_F": False,
            },
        }
    return {
        "original_mlx_vs_baseline_normalized_max_abs": ORIGINAL_T3_MLX_MAX_ABS,
        "verdict": (
            "informational_only"
            if purpose == "retrospective_diagnostic"
            else "prospective_floor"
        ),
        "historical_mps_evidence": historical,
    }


def assemble_floor_report(
    *,
    actions: Mapping[str, np.ndarray],
    variant_metadata: Mapping[str, Mapping[str, object]],
    case_identities: Sequence[Mapping[str, object]],
    input_sha256: Mapping[str, object],
    checkpoint_path: str,
    purpose: str,
    created_at_utc: str,
    created_at_ns: int,
) -> dict[str, object]:
    """Aggregate complete normalized chunks into a validated floor report."""

    if purpose not in {"retrospective_diagnostic", "prospective_gate"}:
        raise ValueError("self-consistency floor purpose is invalid")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.startswith(
        ".cache/training/"
    ):
        raise ValueError("checkpoint path must be repository-relative training cache")
    identities = _validated_cases(case_identities)
    inputs = _validated_input_hashes(input_sha256)
    created_at_utc, created_at_ns = _validate_timestamp(created_at_utc, created_at_ns)

    metadata_names = set(variant_metadata)
    action_names = set(actions)
    if metadata_names != action_names:
        raise ValueError("self-consistency perturbation outputs are incomplete")
    max_threads_values = {
        value.get("requested_threads")
        for name, value in variant_metadata.items()
        if name == "cpu_fp32_threads_max"
    }
    if len(max_threads_values) != 1:
        raise ValueError("maximum-thread perturbation metadata is incomplete")
    max_threads = next(iter(max_threads_values))
    if isinstance(max_threads, bool) or not isinstance(max_threads, int):
        raise ValueError("maximum-thread perturbation metadata is invalid")
    plan = perturbation_plan(max_threads=max_threads)
    order = tuple(item.name for item in plan)
    if action_names != set(order):
        raise ValueError("self-consistency perturbation outputs differ from the fixed set")

    expected_shape = (EVALUATION_SAMPLE_COUNT, *NORMALIZED_ACTION_CHUNK_SHAPE)
    normalized_actions: dict[str, np.ndarray] = {}
    for item in plan:
        array = np.asarray(actions[item.name])
        if array.shape != expected_shape:
            raise ValueError(
                f"{item.name} normalized action shape {array.shape} != {expected_shape}"
            )
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f"{item.name} normalized actions must be finite floating point")
        normalized_actions[item.name] = np.asarray(array, dtype=np.float64)

    baseline = normalized_actions[BASELINE_VARIANT]
    variants: dict[str, dict[str, object]] = {}
    for item in plan:
        metadata = _validate_variant_document(
            variant_metadata[item.name],
            expected_variant=item,
            expected_input_combined_sha256=str(inputs["combined_sha256"]),
            include_derived=False,
        )
        differences = np.max(
            np.abs(normalized_actions[item.name] - baseline),
            axis=(1, 2),
        )
        case_max_abs = [float(value) for value in differences]
        maximum = float(np.max(differences))
        worst_ordinal = int(np.argmax(differences))
        variants[item.name] = {
            **dict(metadata),
            "max_abs_vs_baseline": maximum,
            "case_max_abs": case_max_abs,
            "worst_case": {
                **identities[worst_ordinal],
                "max_abs_vs_baseline": maximum,
            },
        }

    nonbaseline_maxima = [
        float(variants[item.name]["max_abs_vs_baseline"])
        for item in plan
        if item.name != BASELINE_VARIANT
    ]
    report: dict[str, object] = {
        "format_version": 1,
        "artifact_type": FLOOR_ARTIFACT_TYPE,
        "procedure_id": PROCEDURE_ID,
        "purpose": purpose,
        "created_at_utc": created_at_utc,
        "created_at_ns": created_at_ns,
        "checkpoint_path": checkpoint_path,
        "sample_count": EVALUATION_SAMPLE_COUNT,
        "normalized_action_chunk_shape": list(NORMALIZED_ACTION_CHUNK_SHAPE),
        "metric": "max(abs(normalized_action_chunk_p - normalized_action_chunk_baseline))",
        "source_identity": _fixed_source_identity(),
        "input_sha256": inputs,
        "case_identities": [dict(item) for item in identities],
        "perturbation_order": list(order),
        "variants": variants,
        "F": max(nonbaseline_maxima),
        "F64": float(variants[FLOAT64_VARIANT]["max_abs_vs_baseline"]),
        "context": _floor_context(purpose),
    }
    validate_floor_report(report)
    return report


def validate_floor_report(report: object) -> dict[str, object]:
    """Validate a persisted floor without trusting its recorded maxima."""

    if not isinstance(report, Mapping):
        raise ValueError("self-consistency floor must be a JSON object")
    required = {
        "format_version",
        "artifact_type",
        "procedure_id",
        "purpose",
        "created_at_utc",
        "created_at_ns",
        "checkpoint_path",
        "sample_count",
        "normalized_action_chunk_shape",
        "metric",
        "source_identity",
        "input_sha256",
        "case_identities",
        "perturbation_order",
        "variants",
        "F",
        "F64",
        "context",
    }
    if set(report) != required:
        raise ValueError("self-consistency floor fields differ from the fixed schema")
    if (
        report["format_version"] != 1
        or report["artifact_type"] != FLOOR_ARTIFACT_TYPE
        or report["procedure_id"] != PROCEDURE_ID
        or report["sample_count"] != EVALUATION_SAMPLE_COUNT
        or report["normalized_action_chunk_shape"]
        != list(NORMALIZED_ACTION_CHUNK_SHAPE)
    ):
        raise ValueError("self-consistency floor identity is invalid")
    if report["purpose"] not in {"retrospective_diagnostic", "prospective_gate"}:
        raise ValueError("self-consistency floor purpose is invalid")
    if not isinstance(report["checkpoint_path"], str) or not report[
        "checkpoint_path"
    ].startswith(".cache/training/"):
        raise ValueError("self-consistency floor checkpoint path is invalid")
    if report["metric"] != (
        "max(abs(normalized_action_chunk_p - normalized_action_chunk_baseline))"
    ):
        raise ValueError("self-consistency floor metric changed")
    if report["source_identity"] != _fixed_source_identity():
        raise ValueError("self-consistency source identity changed")
    _validate_timestamp(report["created_at_utc"], report["created_at_ns"])
    _validated_input_hashes(report["input_sha256"])
    identities = _validated_cases(report["case_identities"])

    order = report["perturbation_order"]
    variants = report["variants"]
    if not isinstance(order, list) or not isinstance(variants, Mapping):
        raise ValueError("self-consistency floor variants are invalid")
    max_variant = variants.get("cpu_fp32_threads_max")
    if not isinstance(max_variant, Mapping):
        raise ValueError("maximum-thread floor variant is missing")
    max_threads = max_variant.get("requested_threads")
    if isinstance(max_threads, bool) or not isinstance(max_threads, int):
        raise ValueError("maximum-thread floor variant is invalid")
    plan = perturbation_plan(max_threads=max_threads)
    expected_order = [item.name for item in plan]
    if order != expected_order or set(variants) != set(expected_order):
        raise ValueError("self-consistency perturbation order changed")

    maxima: dict[str, float] = {}
    for item in plan:
        value = _validate_variant_document(
            variants[item.name],
            expected_variant=item,
            expected_input_combined_sha256=str(
                report["input_sha256"]["combined_sha256"]
            ),
            include_derived=True,
        )
        case_values = value.get("case_max_abs")
        if not isinstance(case_values, list) or len(case_values) != EVALUATION_SAMPLE_COUNT:
            raise ValueError(f"{item.name} floor case maxima are incomplete")
        try:
            numeric = [float(case_value) for case_value in case_values]
            recorded = float(value["max_abs_vs_baseline"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{item.name} floor variant maximum is invalid") from error
        if not all(math.isfinite(case_value) and case_value >= 0 for case_value in numeric):
            raise ValueError(f"{item.name} floor case maxima must be finite and nonnegative")
        computed = max(numeric)
        if recorded != computed:
            raise ValueError(f"{item.name} variant maximum does not match its cases")
        worst = value.get("worst_case")
        worst_ordinal = numeric.index(computed)
        expected_worst = {
            **identities[worst_ordinal],
            "max_abs_vs_baseline": computed,
        }
        if worst != expected_worst:
            raise ValueError(f"{item.name} worst-case identity is inconsistent")
        maxima[item.name] = recorded
    default_thread_counts = {
        int(variants[name]["actual_threads"])
        for name in (BASELINE_VARIANT, *MPS_VARIANTS, FLOAT64_VARIANT)
    }
    if len(default_thread_counts) != 1:
        raise ValueError("default-thread perturbations used different thread counts")
    invariant_runtime_fields = (
        "interop_threads",
        "torch_version",
        "lerobot_version",
        "transformers_version",
        "platform",
        "machine",
        "python_version",
        "mps_built",
        "mps_available",
    )
    for field in invariant_runtime_fields:
        if len({variants[name][field] for name in expected_order}) != 1:
            raise ValueError(f"variant runtime field changed across workers: {field}")
    if maxima[BASELINE_VARIANT] != 0.0:
        raise ValueError("self-consistency baseline must compare exactly with itself")
    try:
        recorded_f = float(report["F"])
        recorded_f64 = float(report["F64"])
    except (TypeError, ValueError) as error:
        raise ValueError("self-consistency floor envelopes are invalid") from error
    computed_f = max(
        value for name, value in maxima.items() if name != BASELINE_VARIANT
    )
    if recorded_f != computed_f or recorded_f64 != maxima[FLOAT64_VARIANT]:
        raise ValueError("self-consistency floor envelope does not match variant maxima")
    context = report["context"]
    if context != _floor_context(str(report["purpose"])):
        raise ValueError("self-consistency floor context changed")
    return dict(report)


def write_floor_report(path: str | Path, report: Mapping[str, object]) -> str:
    """Validate and atomically write one complete floor JSON document."""

    validated = validate_floor_report(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(validated, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()
