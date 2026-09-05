"""Prospective, evidence-bound parity gates for trained SmolVLA checkpoints.

The file evaluator is the authoritative entry point. It reconstructs the
PyTorch floor from every raw worker array, verifies a real-clock start marker,
recomputes every aggregate from 56 per-case records, verifies named source
files, revalidates all file snapshots, and only then installs a decision with
no-overwrite semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Mapping, Sequence

from mlx_smolvla._lab.training.t3_contract import (
    FROZEN_BASE_REPORT_SHA256,
    FROZEN_TRAIN_STATISTICS_SHA256,
)


PROCEDURE_ID = "smolvla-trained-checkpoint-parity-v1"
START_MARKER_ARTIFACT_TYPE = "smolvla-trained-comparison-start-marker"
COMPARISON_ARTIFACT_TYPE = "smolvla-trained-checkpoint-mlx-comparison"
EVALUATION_ARTIFACT_TYPE = "smolvla-trained-checkpoint-parity-evaluation"

IMAGE_PREPROCESSING_MAX_ABS = 1e-5
STATE_PREPROCESSING_MAX_ABS = 1e-6
FINE_TO_BASE_MAE_RATIO_MAXIMUM = 0.9
TORCH_TO_MLX_MAE_RATIO_MINIMUM = 0.95
TORCH_TO_MLX_MAE_RATIO_MAXIMUM = 1.05
DETERMINISTIC_FALLBACK_MAX_ABS = 0.005
REFERENCE_FLOOR_MULTIPLIER = 3.0

_SAMPLE_COUNT = 56
_ELEMENT_COUNT = _SAMPLE_COUNT * 6
_ACTION_CHUNK_SHAPE = [50, 6]
_IDENTITY_FIELDS = ("ordinal", "episode", "frame_index", "absolute_index")
_EVIDENCE_FILE_NAMES = {
    "base_report",
    "native_conversion_model",
    "native_conversion_name_map",
    "comparison_implementation",
}
_FLOOR_INPUT_GROUP_MODES = {
    "checkpoint_export": "exact_tree",
    "evaluation_artifact": "exact_tree",
    "pinned_dataset": "named_files",
    "tokenizer_snapshot": "contained_symlink_tree",
    "implementation": "named_files",
}
_PARITY_METRIC_FIELDS = (
    "image_preprocessing_max_abs",
    "state_preprocessing_max_abs",
    "preprocessing_max_abs",
    "normalized_action_max_abs",
    "physical_action_max_abs",
    "physical_action_standardized_max_abs",
)
_METRIC_FIELDS = {
    "base_mlx_mae",
    "fine_mlx_mae",
    "torch_mae",
    "image_preprocessing_max_abs",
    "state_preprocessing_max_abs",
    "normalized_action_max_abs",
}
_RESULT_METRIC_FIELDS = _METRIC_FIELDS | {
    "fine_to_base_mae_ratio",
    "torch_to_mlx_mae_ratio",
}
_GATE_FIELDS = {
    "image_preprocessing_passed",
    "state_preprocessing_passed",
    "heldout_improvement_passed",
    "torch_mlx_roundtrip_passed",
    "fixed_gates_passed",
    "deterministic_parity_passed",
    "passed",
}


@dataclass(frozen=True)
class FileSnapshot:
    """One descriptor-backed immutable view of a regular file."""

    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str
    payload: bytes


@dataclass(frozen=True)
class FloorBundle:
    """A floor reconstructed from raw worker outputs."""

    report: dict[str, object]
    bundle_sha256: str
    snapshots: tuple[FileSnapshot, ...]


@dataclass(frozen=True)
class FloorInputBundle:
    """Concrete, descriptor-read files supporting every floor hash group."""

    evidence: dict[str, object]
    files: dict[str, dict[str, FileSnapshot]]
    tree_roots: dict[str, Path]
    links: tuple["SymlinkSnapshot", ...]

    @property
    def snapshots(self) -> tuple[FileSnapshot, ...]:
        return tuple(
            snapshot
            for group_name in sorted(self.files)
            for _, snapshot in sorted(self.files[group_name].items())
        )


@dataclass(frozen=True)
class SymlinkSnapshot:
    """Stable identity and target of one explicitly permitted file symlink."""

    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    target: str
    resolved_path: Path


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _pretty_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _require_mapping(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields differ from the fixed schema")
    return value


def _require_sha256(label: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_positive_ns(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive nanosecond timestamp")
    return value


def _require_nonnegative_float(label: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and nonnegative")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and nonnegative") from error
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return numeric


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _reject_path_overlaps(
    outputs: Mapping[str, Path],
    protected: Mapping[str, Path],
) -> None:
    for output_label, output_path in outputs.items():
        for protected_label, protected_path in protected.items():
            if _paths_overlap(output_path, protected_path):
                raise ValueError(
                    f"{output_label} overlaps protected {protected_label}"
                )


def _utc_from_ns(value_ns: int) -> str:
    value_ns = _require_positive_ns("timestamp", value_ns)
    seconds, nanoseconds = divmod(value_ns, 1_000_000_000)
    value = datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(
        microseconds=nanoseconds // 1_000
    )
    return value.isoformat(timespec="microseconds")


def _validate_timestamp(
    label: str,
    value_utc: object,
    value_ns: object,
) -> tuple[str, int]:
    if not isinstance(value_utc, str):
        raise ValueError(f"{label} UTC timestamp must be a string")
    value_ns = _require_positive_ns(label, value_ns)
    try:
        parsed = datetime.fromisoformat(value_utc)
    except ValueError as error:
        raise ValueError(f"{label} UTC timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} UTC timestamp must use UTC")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed - epoch
    parsed_microseconds = (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )
    if value_ns // 1_000 != parsed_microseconds:
        raise ValueError(f"{label} text and nanoseconds identify different instants")
    return value_utc, value_ns


def _snapshot_file(path: str | Path, *, label: str) -> FileSnapshot:
    """Read a regular file through one no-follow descriptor."""

    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
            raise FileNotFoundError(f"{label} is missing or unsafe: {path}") from error
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FileNotFoundError(f"{label} is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise RuntimeError(f"{label} changed while it was read")
    return FileSnapshot(
        path=path,
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )


def _snapshot_json(path: str | Path, *, label: str) -> tuple[object, FileSnapshot]:
    snapshot = _snapshot_file(path, label=label)
    try:
        value = json.loads(snapshot.payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return value, snapshot


def _revalidate_snapshots(snapshots: Sequence[FileSnapshot]) -> None:
    """Prove every path still names the exact bytes originally validated."""

    for expected in snapshots:
        current = _snapshot_file(expected.path, label="parity input")
        if (
            current.device != expected.device
            or current.inode != expected.inode
            or current.size != expected.size
            or current.mtime_ns != expected.mtime_ns
            or current.sha256 != expected.sha256
        ):
            raise RuntimeError(f"parity input changed before report install: {expected.path}")


def _atomic_json_no_clobber(path: str | Path, value: object) -> str:
    """Install canonical JSON atomically and fail if any writer won the path."""

    path = Path(path)
    payload = _pretty_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"parity artifact already exists: {path}") from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def _write_private_snapshot(path: Path, snapshot: FileSnapshot) -> None:
    """Materialize captured bytes into a new owner-only regular file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(snapshot.payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_snapshot_conversion(
    *,
    source_model: FileSnapshot,
    converted_model: FileSnapshot,
    name_map: FileSnapshot,
):
    """Run semantic conversion checks on immutable copies of captured bytes."""

    from mlx_smolvla.convert import validate_converted_checkpoint

    with tempfile.TemporaryDirectory(prefix="smolvla-parity-conversion-") as raw:
        private_root = Path(raw)
        source_path = private_root / "source" / "model.safetensors"
        converted_path = private_root / "converted" / "model.float32.safetensors"
        name_map_path = private_root / "converted" / "name_map.json"
        _write_private_snapshot(source_path, source_model)
        _write_private_snapshot(converted_path, converted_model)
        _write_private_snapshot(name_map_path, name_map)
        return validate_converted_checkpoint(
            source_path.parent,
            converted_path,
            name_map_path,
            dtype="float32",
        )


def _load_floor_bundle(floor: object, *, variant_root: str | Path) -> FloorBundle:
    """Rebuild a floor from the exact raw arrays and worker metadata."""

    from mlx_smolvla._lab.training.self_consistency import (
        assemble_floor_report,
        perturbation_plan,
        validate_floor_report,
    )

    try:
        validated = validate_floor_report(floor)
        variants = validated["variants"]
        max_threads = variants["cpu_fp32_threads_max"]["requested_threads"]
        plan = perturbation_plan(max_threads=max_threads)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"raw variant floor metadata is invalid: {error}") from error

    root = Path(variant_root)
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"raw variant root is missing or unsafe: {root}")
    actions: dict[str, object] = {}
    metadata: dict[str, Mapping[str, object]] = {}
    snapshots: list[FileSnapshot] = []
    bundle_files: dict[str, dict[str, str]] = {}
    import numpy as np

    for item in plan:
        artifact = root / item.name
        if not artifact.is_dir() or artifact.is_symlink():
            raise FileNotFoundError(
                f"raw variant {item.name} directory is missing or unsafe: {artifact}"
            )
        metadata_value, metadata_snapshot = _snapshot_json(
            artifact / "metadata.json", label=f"raw variant {item.name} metadata"
        )
        actions_snapshot = _snapshot_file(
            artifact / "normalized_actions.npy", label=f"raw variant {item.name} actions"
        )
        if not isinstance(metadata_value, Mapping):
            raise ValueError(f"raw variant {item.name} metadata is not an object")
        if metadata_value.get("normalized_actions_sha256") != actions_snapshot.sha256:
            raise ValueError(f"raw variant {item.name} action digest differs from metadata")
        try:
            array = np.load(io.BytesIO(actions_snapshot.payload), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"raw variant {item.name} actions are invalid") from error
        if array.dtype.name != metadata_value.get("normalized_actions_dtype"):
            raise ValueError(f"raw variant {item.name} dtype differs from metadata")
        actions[item.name] = np.ascontiguousarray(array)
        metadata[item.name] = dict(metadata_value)
        snapshots.extend((metadata_snapshot, actions_snapshot))
        bundle_files[item.name] = {
            "metadata_sha256": metadata_snapshot.sha256,
            "normalized_actions_sha256": actions_snapshot.sha256,
        }

    try:
        rebuilt = assemble_floor_report(
            actions=actions,
            variant_metadata=metadata,
            case_identities=validated["case_identities"],
            input_sha256=validated["input_sha256"],
            checkpoint_path=str(validated["checkpoint_path"]),
            purpose=str(validated["purpose"]),
            created_at_utc=str(validated["created_at_utc"]),
            created_at_ns=int(validated["created_at_ns"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"raw variant reconstruction failed: {error}") from error
    if rebuilt != validated:
        raise ValueError("raw variant reconstruction differs from the floor report")
    bundle_sha256 = hashlib.sha256(_canonical_json(bundle_files)).hexdigest()
    return FloorBundle(rebuilt, bundle_sha256, tuple(snapshots))


def validate_floor_bundle(
    floor: object,
    *,
    variant_root: str | Path,
) -> dict[str, object]:
    """Validate the JSON floor against all nine raw perturbation artifacts."""

    bundle = _load_floor_bundle(floor, variant_root=variant_root)
    return {"report": bundle.report, "bundle_sha256": bundle.bundle_sha256}


_MARKER_FIELDS = {
    "format_version",
    "artifact_type",
    "procedure_id",
    "created_at_utc",
    "created_at_ns",
    "comparison_path",
    "floor_sha256",
    "floor_procedure_id",
    "floor_created_at_ns",
    "floor_file_mtime_ns",
    "floor_bundle_sha256",
    "checkpoint_path",
    "input_combined_sha256",
}


def validate_comparison_start_marker(marker: object) -> dict[str, object]:
    value = _require_mapping(marker, fields=_MARKER_FIELDS, label="comparison marker")
    if (
        value["format_version"] != 1
        or value["artifact_type"] != START_MARKER_ARTIFACT_TYPE
        or value["procedure_id"] != PROCEDURE_ID
    ):
        raise ValueError("comparison marker identity is invalid")
    _validate_timestamp("comparison marker", value["created_at_utc"], value["created_at_ns"])
    for field in ("floor_sha256", "floor_bundle_sha256", "input_combined_sha256"):
        _require_sha256(field.replace("_", " "), value[field])
    for field in ("floor_created_at_ns", "floor_file_mtime_ns"):
        _require_positive_ns(field.replace("_", " "), value[field])
    if not isinstance(value["comparison_path"], str) or not value["comparison_path"]:
        raise ValueError("comparison marker target path is invalid")
    if not isinstance(value["floor_procedure_id"], str) or not value["floor_procedure_id"]:
        raise ValueError("comparison marker floor procedure is invalid")
    if not isinstance(value["checkpoint_path"], str) or not value["checkpoint_path"].startswith(
        ".cache/training/"
    ):
        raise ValueError("comparison marker checkpoint path is invalid")
    if not (
        value["floor_created_at_ns"]
        <= value["floor_file_mtime_ns"]
        < value["created_at_ns"]
    ):
        raise ValueError("comparison marker does not strictly follow the floor")
    return dict(value)


def create_comparison_start_marker(
    *,
    floor_path: str | Path,
    variant_root: str | Path,
    output_path: str | Path,
    comparison_path: str | Path | None = None,
) -> tuple[dict[str, object], str]:
    """Create the one-shot marker immediately before the first MLX inference."""

    floor_path = Path(floor_path)
    output_path = Path(output_path)
    target = (
        output_path.with_name("comparison.json")
        if comparison_path is None
        else Path(comparison_path)
    )
    if len(
        {
            str(floor_path.resolve()),
            str(output_path.resolve()),
            str(target.resolve()),
        }
    ) != 3:
        raise ValueError("floor, start marker, and comparison paths must be distinct")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"comparison target already exists: {target}")
    floor, floor_snapshot = _snapshot_json(
        floor_path, label="prospective self-consistency floor"
    )
    bundle = _load_floor_bundle(floor, variant_root=variant_root)
    if bundle.report.get("purpose") != "prospective_gate":
        raise ValueError("comparison marker requires a prospective floor")
    checkpoint_root = Path(bundle.report["checkpoint_path"])
    _reject_path_overlaps(
        {
            "comparison marker output": output_path,
            "comparison target": target,
        },
        {
            "raw floor variant tree": Path(variant_root),
            "checkpoint export tree": checkpoint_root,
        },
    )
    created_at_ns = time.time_ns()
    if bundle.report["created_at_ns"] >= created_at_ns:
        raise ValueError("floor creation timestamp must precede comparison marker")
    if floor_snapshot.mtime_ns >= created_at_ns:
        raise ValueError("floor file timestamp must precede comparison marker")
    marker = {
        "format_version": 1,
        "artifact_type": START_MARKER_ARTIFACT_TYPE,
        "procedure_id": PROCEDURE_ID,
        "created_at_utc": _utc_from_ns(created_at_ns),
        "created_at_ns": created_at_ns,
        "comparison_path": str(target.resolve()),
        "floor_sha256": floor_snapshot.sha256,
        "floor_procedure_id": bundle.report["procedure_id"],
        "floor_created_at_ns": bundle.report["created_at_ns"],
        "floor_file_mtime_ns": floor_snapshot.mtime_ns,
        "floor_bundle_sha256": bundle.bundle_sha256,
        "checkpoint_path": bundle.report["checkpoint_path"],
        "input_combined_sha256": bundle.report["input_sha256"]["combined_sha256"],
    }
    marker = validate_comparison_start_marker(marker)
    _revalidate_snapshots((floor_snapshot, *bundle.snapshots))
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"comparison target already exists: {target}")
    return marker, _atomic_json_no_clobber(output_path, marker)


_FLOOR_BINDING_FIELDS = {
    "floor_sha256",
    "floor_procedure_id",
    "floor_created_at_ns",
    "floor_file_mtime_ns",
    "input_combined_sha256",
    "floor_bundle_sha256",
}
_MARKER_BINDING_FIELDS = {
    "marker_sha256",
    "marker_created_at_ns",
    "marker_file_mtime_ns",
    "floor_bundle_sha256",
}
_COMPARISON_FIELDS = {
    "format_version",
    "artifact_type",
    "procedure_id",
    "created_at_utc",
    "created_at_ns",
    "checkpoint_path",
    "sample_count",
    "normalized_action_chunk_shape",
    "floor_binding",
    "start_marker_binding",
    "source_identity",
    "input_sha256",
    "floor_input_evidence",
    "case_identities",
    "evidence_files",
    "conversion_validation",
    "base_mlx_evaluation",
    "fine_mlx_evaluation",
    "torch_evaluation",
    "stats_active_parity",
    "metrics",
}
_CONVERSION_VALIDATION_FIELDS = {
    "source_model_sha256",
    "converted_model_sha256",
    "name_map_sha256",
    "dtype",
    "tensor_count",
    "parameter_count",
}


def _validated_evidence_files(value: object) -> dict[str, dict[str, str]]:
    files = _require_mapping(
        value, fields=_EVIDENCE_FILE_NAMES, label="comparison evidence files"
    )
    result: dict[str, dict[str, str]] = {}
    for name, item in files.items():
        entry = _require_mapping(item, fields={"path", "sha256"}, label=f"{name} evidence file")
        path = entry["path"]
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ValueError(f"{name} evidence path is invalid")
        result[name] = {
            "path": path,
            "sha256": _require_sha256(f"{name} evidence", entry["sha256"]),
        }
    if result["base_report"]["sha256"] != FROZEN_BASE_REPORT_SHA256:
        raise ValueError("base report differs from the pre-training frozen base report")
    return result


def _require_recorded_path(label: str, value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} path is invalid")
    candidate = Path(value)
    if any(part in ("", ".", "..") for part in candidate.parts):
        raise ValueError(f"{label} path is invalid")
    return value


def _validated_floor_input_evidence(
    value: object,
    *,
    expected_inputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    evidence = _require_mapping(
        value,
        fields=set(_FLOOR_INPUT_GROUP_MODES),
        label="floor input evidence",
    )
    normalized: dict[str, object] = {}
    for group_name, expected_mode in _FLOOR_INPUT_GROUP_MODES.items():
        entry = evidence[group_name]
        if expected_mode in {"exact_tree", "contained_symlink_tree"}:
            fields = (
                {"mode", "root"}
                if expected_mode == "exact_tree"
                else {"mode", "root", "allowed_root"}
            )
            record = _require_mapping(
                entry,
                fields=fields,
                label=f"{group_name} floor input evidence",
            )
            if record["mode"] != expected_mode:
                raise ValueError(f"{group_name} floor input evidence mode changed")
            normalized_entry = {
                "mode": expected_mode,
                "root": _require_recorded_path(
                    f"{group_name} floor input root", record["root"]
                ),
            }
            if expected_mode == "contained_symlink_tree":
                normalized_entry["allowed_root"] = _require_recorded_path(
                    f"{group_name} allowed root", record["allowed_root"]
                )
            normalized[group_name] = normalized_entry
            continue

        record = _require_mapping(
            entry,
            fields={"mode", "paths"},
            label=f"{group_name} floor input evidence",
        )
        if record["mode"] != "named_files":
            raise ValueError(f"{group_name} floor input evidence mode changed")
        paths = record["paths"]
        if not isinstance(paths, Mapping) or not paths:
            raise ValueError(f"{group_name} floor input paths are incomplete")
        expected_files: Mapping[str, object] | None = None
        if expected_inputs is not None:
            expected_group = expected_inputs.get(group_name)
            if not isinstance(expected_group, Mapping) or not isinstance(
                expected_group.get("files"), Mapping
            ):
                raise ValueError(f"{group_name} floor input hash group is invalid")
            expected_files = expected_group["files"]
            if set(paths) != set(expected_files):
                raise ValueError(
                    f"{group_name} floor input paths differ from the hashed inventory"
                )
        normalized_paths: dict[str, str] = {}
        for logical_name, path in sorted(paths.items()):
            if (
                not isinstance(logical_name, str)
                or not logical_name
                or Path(logical_name).is_absolute()
                or any(part in ("", ".", "..") for part in Path(logical_name).parts)
            ):
                raise ValueError(f"{group_name} floor input label is invalid")
            normalized_paths[logical_name] = _require_recorded_path(
                f"{group_name} floor input {logical_name}", path
            )
        normalized[group_name] = {
            "mode": "named_files",
            "paths": normalized_paths,
        }
    return normalized


def _validated_conversion(
    value: object,
    *,
    evidence_files: Mapping[str, Mapping[str, str]],
    expected_source_model_sha256: object | None = None,
) -> dict[str, object]:
    conversion = _require_mapping(
        value,
        fields=_CONVERSION_VALIDATION_FIELDS,
        label="native conversion validation",
    )
    source_sha256 = _require_sha256(
        "native conversion source model", conversion["source_model_sha256"]
    )
    converted_sha256 = _require_sha256(
        "native converted model", conversion["converted_model_sha256"]
    )
    name_map_sha256 = _require_sha256(
        "native conversion name map", conversion["name_map_sha256"]
    )
    if conversion["dtype"] != "float32":
        raise ValueError("trained checkpoint conversion must use canonical float32")
    for field in ("tensor_count", "parameter_count"):
        count = conversion[field]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"native conversion {field.replace('_', ' ')} is invalid")
    if converted_sha256 != evidence_files["native_conversion_model"]["sha256"]:
        raise ValueError("native conversion model digest differs from its evidence file")
    if name_map_sha256 != evidence_files["native_conversion_name_map"]["sha256"]:
        raise ValueError("native conversion name-map digest differs from its evidence file")
    if (
        expected_source_model_sha256 is not None
        and source_sha256
        != _require_sha256(
            "floor checkpoint model", expected_source_model_sha256
        )
    ):
        raise ValueError("native conversion source model differs from the floor-bound export")
    return {
        "source_model_sha256": source_sha256,
        "converted_model_sha256": converted_sha256,
        "name_map_sha256": name_map_sha256,
        "dtype": "float32",
        "tensor_count": int(conversion["tensor_count"]),
        "parameter_count": int(conversion["parameter_count"]),
    }


def _validated_identities(value: object) -> list[dict[str, int]]:
    if not isinstance(value, list) or len(value) != _SAMPLE_COUNT:
        raise ValueError("trained comparison case identities are invalid")
    identities: list[dict[str, int]] = []
    for ordinal, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != set(_IDENTITY_FIELDS):
            raise ValueError("trained comparison case identities are invalid")
        if any(
            isinstance(item[field], bool) or not isinstance(item[field], int)
            for field in _IDENTITY_FIELDS
        ):
            raise ValueError("trained comparison case identities are invalid")
        if item["ordinal"] != ordinal:
            raise ValueError("trained comparison case identity order changed")
        identities.append({field: int(item[field]) for field in _IDENTITY_FIELDS})
    if len({item["absolute_index"] for item in identities}) != _SAMPLE_COUNT:
        raise ValueError("trained comparison case identities are not unique")
    return identities


def _validate_samples(
    value: object,
    identities: Sequence[Mapping[str, int]],
    *,
    label: str,
) -> tuple[list[dict[str, object]], float]:
    if not isinstance(value, list) or len(value) != _SAMPLE_COUNT:
        raise ValueError(f"{label} sample evidence is incomplete")
    records: list[dict[str, object]] = []
    sums: list[float] = []
    fields = set(_IDENTITY_FIELDS) | {"absolute_error_sum", "element_count"}
    for identity, sample in zip(identities, value, strict=True):
        if not isinstance(sample, Mapping) or set(sample) != fields:
            raise ValueError(f"{label} sample evidence is inconsistent")
        if any(sample[field] != identity[field] for field in _IDENTITY_FIELDS):
            raise ValueError(f"{label} sample identities differ from the frozen cases")
        error_sum = _require_nonnegative_float(
            f"{label} sample absolute error", sample["absolute_error_sum"]
        )
        if sample["element_count"] != 6:
            raise ValueError(f"{label} sample element count is invalid")
        records.append(dict(sample))
        sums.append(error_sum)
    return records, math.fsum(sums)


def _validate_base_evaluation(
    value: object,
    identities: Sequence[Mapping[str, int]],
    *,
    evaluation_manifest_sha256: str,
) -> tuple[dict[str, object], float]:
    fields = {
        "format_version",
        "artifact_type",
        "evaluation_manifest_sha256",
        "train_statistics_sha256",
        "sample_count",
        "element_count",
        "device",
        "dtype",
        "mlx_mae",
        "absolute_error_sum",
        "samples",
    }
    report = _require_mapping(value, fields=fields, label="base MLX evaluation")
    if (
        report["format_version"] != 1
        or report["artifact_type"] != "smolvla-lora-base-heldout-evaluation"
        or report["evaluation_manifest_sha256"] != evaluation_manifest_sha256
        or report["train_statistics_sha256"] != FROZEN_TRAIN_STATISTICS_SHA256
        or report["sample_count"] != _SAMPLE_COUNT
        or report["element_count"] != _ELEMENT_COUNT
    ):
        raise ValueError("base MLX evaluation identity is invalid")
    if not isinstance(report["device"], str) or not report["device"]:
        raise ValueError("base MLX evaluation device is invalid")
    if not isinstance(report["dtype"], str) or not report["dtype"]:
        raise ValueError("base MLX evaluation dtype is invalid")
    _, total = _validate_samples(report["samples"], identities, label="base MLX")
    recorded_total = _require_nonnegative_float(
        "base MLX absolute error", report["absolute_error_sum"]
    )
    mae = _require_nonnegative_float("base MLX MAE", report["mlx_mae"])
    if total != recorded_total or mae != total / _ELEMENT_COUNT or mae == 0:
        raise ValueError("base MLX evaluation aggregate differs from sample evidence")
    return dict(report), mae


def _validate_mae_evaluation(
    value: object,
    identities: Sequence[Mapping[str, int]],
    *,
    framework: str,
) -> tuple[dict[str, object], float]:
    fields = {
        "framework",
        "device",
        "dtype",
        "sample_count",
        "element_count",
        "mae",
        "absolute_error_sum",
        "samples",
    }
    report = _require_mapping(value, fields=fields, label=f"{framework} evaluation")
    if (
        report["framework"] != framework
        or report["sample_count"] != _SAMPLE_COUNT
        or report["element_count"] != _ELEMENT_COUNT
    ):
        raise ValueError(f"{framework} evaluation identity is invalid")
    if not isinstance(report["device"], str) or not report["device"]:
        raise ValueError(f"{framework} evaluation device is invalid")
    if not isinstance(report["dtype"], str) or not report["dtype"]:
        raise ValueError(f"{framework} evaluation dtype is invalid")
    _, total = _validate_samples(report["samples"], identities, label=framework)
    recorded_total = _require_nonnegative_float(
        f"{framework} absolute error", report["absolute_error_sum"]
    )
    mae = _require_nonnegative_float(f"{framework} MAE", report["mae"])
    if total != recorded_total or mae != total / _ELEMENT_COUNT:
        raise ValueError(f"{framework} evaluation aggregate differs from sample evidence")
    if framework == "mlx" and mae == 0:
        raise ValueError("trained parity MAE ratios require nonzero MLX denominators")
    return dict(report), mae


def _validate_parity_evidence(
    value: object,
    identities: Sequence[Mapping[str, int]],
) -> tuple[dict[str, object], dict[str, float]]:
    fields = {"sample_count", "gate_max_abs", "samples", *_PARITY_METRIC_FIELDS}
    report = _require_mapping(value, fields=fields, label="stats-active parity")
    if report["sample_count"] != _SAMPLE_COUNT:
        raise ValueError("stats-active parity sample count is invalid")
    samples = report["samples"]
    if not isinstance(samples, list) or len(samples) != _SAMPLE_COUNT:
        raise ValueError("stats-active parity sample evidence is incomplete")
    maxima = {field: 0.0 for field in _PARITY_METRIC_FIELDS}
    sample_fields = set(_IDENTITY_FIELDS) | set(_PARITY_METRIC_FIELDS)
    records: list[dict[str, object]] = []
    for identity, sample in zip(identities, samples, strict=True):
        if not isinstance(sample, Mapping) or set(sample) != sample_fields:
            raise ValueError("stats-active parity sample evidence is inconsistent")
        if any(sample[field] != identity[field] for field in _IDENTITY_FIELDS):
            raise ValueError("stats-active parity differs from the frozen case identities")
        numeric = {
            field: _require_nonnegative_float(f"stats-active {field}", sample[field])
            for field in _PARITY_METRIC_FIELDS
        }
        if numeric["preprocessing_max_abs"] != max(
            numeric["image_preprocessing_max_abs"], numeric["state_preprocessing_max_abs"]
        ):
            raise ValueError("stats-active per-case preprocessing aggregate is inconsistent")
        for field in _PARITY_METRIC_FIELDS:
            maxima[field] = max(maxima[field], numeric[field])
        records.append(dict(sample))
    summary = {
        field: _require_nonnegative_float(f"stats-active {field}", report[field])
        for field in _PARITY_METRIC_FIELDS
    }
    if summary != maxima:
        raise ValueError("stats-active parity summary differs from sample evidence")
    if summary["preprocessing_max_abs"] != max(
        summary["image_preprocessing_max_abs"], summary["state_preprocessing_max_abs"]
    ):
        raise ValueError("stats-active preprocessing aggregate is inconsistent")
    gate_max = _require_nonnegative_float("stats-active gate maximum", report["gate_max_abs"])
    if gate_max != max(summary.values()):
        raise ValueError("stats-active parity gate maximum excludes a parity boundary")
    result = dict(report)
    result["samples"] = records
    return result, summary


def _validated_evidence(
    comparison: Mapping[str, object],
    identities: Sequence[Mapping[str, int]],
    *,
    evaluation_manifest_sha256: str,
) -> tuple[dict[str, object], dict[str, float]]:
    base, base_mae = _validate_base_evaluation(
        comparison["base_mlx_evaluation"],
        identities,
        evaluation_manifest_sha256=evaluation_manifest_sha256,
    )
    fine, fine_mae = _validate_mae_evaluation(
        comparison["fine_mlx_evaluation"], identities, framework="mlx"
    )
    torch, torch_mae = _validate_mae_evaluation(
        comparison["torch_evaluation"], identities, framework="torch"
    )
    parity, parity_summary = _validate_parity_evidence(
        comparison["stats_active_parity"], identities
    )
    metrics = {
        "base_mlx_mae": base_mae,
        "fine_mlx_mae": fine_mae,
        "torch_mae": torch_mae,
        "image_preprocessing_max_abs": parity_summary["image_preprocessing_max_abs"],
        "state_preprocessing_max_abs": parity_summary["state_preprocessing_max_abs"],
        "normalized_action_max_abs": parity_summary["normalized_action_max_abs"],
    }
    recorded = _require_mapping(
        comparison["metrics"], fields=_METRIC_FIELDS, label="comparison metrics"
    )
    recorded_numeric = {
        name: _require_nonnegative_float(name.replace("_", " "), recorded[name])
        for name in _METRIC_FIELDS
    }
    if recorded_numeric != metrics:
        raise ValueError("comparison metric summary differs from per-case evidence")
    return {
        "base_mlx_evaluation": base,
        "fine_mlx_evaluation": fine,
        "torch_evaluation": torch,
        "stats_active_parity": parity,
    }, metrics


def _validate_comparison(
    comparison: object,
) -> tuple[dict[str, object], int, dict[str, dict[str, str]]]:
    value = _require_mapping(comparison, fields=_COMPARISON_FIELDS, label="trained comparison")
    if (
        value["format_version"] != 1
        or value["artifact_type"] != COMPARISON_ARTIFACT_TYPE
        or value["procedure_id"] != PROCEDURE_ID
        or value["sample_count"] != _SAMPLE_COUNT
        or value["normalized_action_chunk_shape"] != _ACTION_CHUNK_SHAPE
    ):
        raise ValueError("trained comparison identity is invalid")
    _, created_at_ns = _validate_timestamp(
        "comparison creation", value["created_at_utc"], value["created_at_ns"]
    )
    if not isinstance(value["checkpoint_path"], str) or not value["checkpoint_path"].startswith(
        ".cache/training/"
    ):
        raise ValueError("trained comparison checkpoint path is invalid")
    if not isinstance(value["source_identity"], Mapping):
        raise ValueError("trained comparison source identity is invalid")
    if not isinstance(value["input_sha256"], Mapping):
        raise ValueError("trained comparison input hashes are invalid")
    _validated_floor_input_evidence(value["floor_input_evidence"])
    _validated_identities(value["case_identities"])
    floor_binding = _require_mapping(
        value["floor_binding"], fields=_FLOOR_BINDING_FIELDS, label="floor binding"
    )
    for field in ("floor_sha256", "input_combined_sha256", "floor_bundle_sha256"):
        _require_sha256(field.replace("_", " "), floor_binding[field])
    for field in ("floor_created_at_ns", "floor_file_mtime_ns"):
        _require_positive_ns(field.replace("_", " "), floor_binding[field])
    if not isinstance(floor_binding["floor_procedure_id"], str) or not floor_binding[
        "floor_procedure_id"
    ]:
        raise ValueError("floor binding procedure is invalid")
    marker_binding = _require_mapping(
        value["start_marker_binding"], fields=_MARKER_BINDING_FIELDS, label="start marker binding"
    )
    for field in ("marker_sha256", "floor_bundle_sha256"):
        _require_sha256(field.replace("_", " "), marker_binding[field])
    for field in ("marker_created_at_ns", "marker_file_mtime_ns"):
        _require_positive_ns(field.replace("_", " "), marker_binding[field])
    evidence_files = _validated_evidence_files(value["evidence_files"])
    _validated_conversion(
        value["conversion_validation"], evidence_files=evidence_files
    )
    return dict(value), created_at_ns, evidence_files


def _thresholds(*, floor: float, floor64: float) -> dict[str, float]:
    normalized_action_max_abs = max(
        DETERMINISTIC_FALLBACK_MAX_ABS,
        REFERENCE_FLOOR_MULTIPLIER * floor,
    )
    if not math.isfinite(normalized_action_max_abs):
        raise ValueError("derived deterministic threshold must be finite")
    return {
        "image_preprocessing_max_abs": IMAGE_PREPROCESSING_MAX_ABS,
        "state_preprocessing_max_abs": STATE_PREPROCESSING_MAX_ABS,
        "fine_to_base_mae_ratio_maximum": FINE_TO_BASE_MAE_RATIO_MAXIMUM,
        "torch_to_mlx_mae_ratio_minimum": TORCH_TO_MLX_MAE_RATIO_MINIMUM,
        "torch_to_mlx_mae_ratio_maximum": TORCH_TO_MLX_MAE_RATIO_MAXIMUM,
        "deterministic_fallback_max_abs": DETERMINISTIC_FALLBACK_MAX_ABS,
        "reference_floor_multiplier": REFERENCE_FLOOR_MULTIPLIER,
        "reference_floor": floor,
        "reference_floor_float64": floor64,
        "normalized_action_max_abs": normalized_action_max_abs,
    }


def _decision(
    metrics: Mapping[str, float], thresholds: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, bool]]:
    if metrics["base_mlx_mae"] == 0 or metrics["fine_mlx_mae"] == 0:
        raise ValueError("trained parity MAE ratios require nonzero MLX denominators")
    fine_to_base = metrics["fine_mlx_mae"] / metrics["base_mlx_mae"]
    torch_to_mlx = metrics["torch_mae"] / metrics["fine_mlx_mae"]
    if not math.isfinite(fine_to_base) or not math.isfinite(torch_to_mlx):
        raise ValueError("trained parity derived MAE ratios must be finite")
    recorded_metrics = {
        **dict(metrics),
        "fine_to_base_mae_ratio": fine_to_base,
        "torch_to_mlx_mae_ratio": torch_to_mlx,
    }
    image = metrics["image_preprocessing_max_abs"] <= thresholds[
        "image_preprocessing_max_abs"
    ]
    state = metrics["state_preprocessing_max_abs"] <= thresholds[
        "state_preprocessing_max_abs"
    ]
    improvement = fine_to_base <= thresholds["fine_to_base_mae_ratio_maximum"]
    roundtrip = (
        thresholds["torch_to_mlx_mae_ratio_minimum"]
        <= torch_to_mlx
        <= thresholds["torch_to_mlx_mae_ratio_maximum"]
    )
    fixed = image and state and improvement and roundtrip
    deterministic = metrics["normalized_action_max_abs"] <= thresholds[
        "normalized_action_max_abs"
    ]
    gates = {
        "image_preprocessing_passed": image,
        "state_preprocessing_passed": state,
        "heldout_improvement_passed": improvement,
        "torch_mlx_roundtrip_passed": roundtrip,
        "fixed_gates_passed": fixed,
        "deterministic_parity_passed": deterministic,
        "passed": fixed and deterministic,
    }
    return recorded_metrics, gates


_RESULT_SOURCE_FIELDS = {
    "floor_sha256",
    "floor_bundle_sha256",
    "comparison_sha256",
    "start_marker_sha256",
    "floor_procedure_id",
    "checkpoint_path",
    "input_combined_sha256",
    "floor_created_at_ns",
    "floor_file_mtime_ns",
    "start_marker_created_at_ns",
    "start_marker_file_mtime_ns",
    "comparison_created_at_ns",
    "comparison_file_mtime_ns",
    "evidence_files",
    "floor_input_evidence",
    "conversion_validation",
}
_THRESHOLD_FIELDS = {
    "image_preprocessing_max_abs",
    "state_preprocessing_max_abs",
    "fine_to_base_mae_ratio_maximum",
    "torch_to_mlx_mae_ratio_minimum",
    "torch_to_mlx_mae_ratio_maximum",
    "deterministic_fallback_max_abs",
    "reference_floor_multiplier",
    "reference_floor",
    "reference_floor_float64",
    "normalized_action_max_abs",
}
_RESULT_FIELDS = {
    "format_version",
    "artifact_type",
    "procedure_id",
    "evaluated_at_utc",
    "evaluated_at_ns",
    "source",
    "thresholds",
    "evidence",
    "metrics",
    "gates",
}


def evaluate_trained_parity_documents(
    *,
    floor: object,
    floor_sha256: str,
    floor_file_mtime_ns: int,
    floor_bundle_sha256: str,
    start_marker: object,
    start_marker_sha256: str,
    start_marker_file_mtime_ns: int,
    comparison: object,
    comparison_sha256: str,
    comparison_file_mtime_ns: int,
    evaluated_at_ns: int,
) -> dict[str, object]:
    """Recompute evidence after chronology and exact source bindings pass."""

    from mlx_smolvla._lab.training.self_consistency import validate_floor_report

    if not isinstance(floor, Mapping) or floor.get("purpose") != "prospective_gate":
        raise ValueError("trained parity requires a prospective self-consistency floor")
    validated_floor = validate_floor_report(floor)
    marker = validate_comparison_start_marker(start_marker)
    comparison_value, comparison_created_ns, evidence_files = _validate_comparison(comparison)
    floor_sha256 = _require_sha256("floor SHA-256", floor_sha256)
    floor_bundle_sha256 = _require_sha256("floor bundle SHA-256", floor_bundle_sha256)
    start_marker_sha256 = _require_sha256("start marker SHA-256", start_marker_sha256)
    comparison_sha256 = _require_sha256("comparison SHA-256", comparison_sha256)
    floor_file_mtime_ns = _require_positive_ns("floor file timestamp", floor_file_mtime_ns)
    start_marker_file_mtime_ns = _require_positive_ns(
        "start marker file timestamp", start_marker_file_mtime_ns
    )
    comparison_file_mtime_ns = _require_positive_ns(
        "comparison file timestamp", comparison_file_mtime_ns
    )
    evaluated_at_ns = _require_positive_ns("evaluation timestamp", evaluated_at_ns)

    if marker["floor_sha256"] != floor_sha256:
        raise ValueError("comparison marker floor SHA-256 differs from the floor file")
    if marker["floor_bundle_sha256"] != floor_bundle_sha256:
        raise ValueError("comparison marker raw floor bundle differs from the floor")
    if marker["floor_procedure_id"] != validated_floor["procedure_id"]:
        raise ValueError("comparison marker floor procedure differs from the floor")
    if marker["floor_created_at_ns"] != validated_floor["created_at_ns"]:
        raise ValueError("comparison marker floor creation differs from the floor")
    if marker["floor_file_mtime_ns"] != floor_file_mtime_ns:
        raise ValueError("comparison marker floor timestamp differs from the floor")
    if marker["checkpoint_path"] != validated_floor["checkpoint_path"]:
        raise ValueError("comparison marker checkpoint differs from the floor")
    if marker["input_combined_sha256"] != validated_floor["input_sha256"]["combined_sha256"]:
        raise ValueError("comparison marker input digest differs from the floor")
    if marker["created_at_ns"] > start_marker_file_mtime_ns:
        raise ValueError("comparison marker file timestamp precedes marker creation")

    expected_floor_binding = {
        "floor_sha256": floor_sha256,
        "floor_procedure_id": validated_floor["procedure_id"],
        "floor_created_at_ns": validated_floor["created_at_ns"],
        "floor_file_mtime_ns": floor_file_mtime_ns,
        "input_combined_sha256": validated_floor["input_sha256"]["combined_sha256"],
        "floor_bundle_sha256": floor_bundle_sha256,
    }
    if comparison_value["floor_binding"] != expected_floor_binding:
        raise ValueError("comparison floor binding differs from the validated floor")
    expected_marker_binding = {
        "marker_sha256": start_marker_sha256,
        "marker_created_at_ns": marker["created_at_ns"],
        "marker_file_mtime_ns": start_marker_file_mtime_ns,
        "floor_bundle_sha256": floor_bundle_sha256,
    }
    if comparison_value["start_marker_binding"] != expected_marker_binding:
        raise ValueError("comparison start marker binding is inconsistent")
    if not (
        validated_floor["created_at_ns"]
        <= floor_file_mtime_ns
        < marker["created_at_ns"]
        <= start_marker_file_mtime_ns
        <= comparison_created_ns
        <= comparison_file_mtime_ns
        <= evaluated_at_ns
    ):
        raise ValueError("trained comparison chronology is invalid")
    if comparison_value["checkpoint_path"] != validated_floor["checkpoint_path"]:
        raise ValueError("comparison checkpoint differs from the floor checkpoint")
    if comparison_value["source_identity"] != validated_floor["source_identity"]:
        raise ValueError("comparison source identity differs from the floor")
    if comparison_value["input_sha256"] != validated_floor["input_sha256"]:
        raise ValueError("comparison input hashes differ from the floor")
    floor_input_evidence = _validated_floor_input_evidence(
        comparison_value["floor_input_evidence"],
        expected_inputs=validated_floor["input_sha256"],
    )
    identities = _validated_identities(comparison_value["case_identities"])
    if identities != validated_floor["case_identities"]:
        raise ValueError("comparison case identities differ from the floor")

    evaluation_identity = validated_floor["source_identity"].get("evaluation")
    if not isinstance(evaluation_identity, Mapping):
        raise ValueError("floor evaluation source identity is invalid")
    manifest_sha256 = _require_sha256(
        "floor evaluation manifest", evaluation_identity.get("manifest_sha256")
    )
    embedded_base_sha256 = hashlib.sha256(
        _pretty_json(comparison_value["base_mlx_evaluation"])
    ).hexdigest()
    if embedded_base_sha256 != evidence_files["base_report"]["sha256"]:
        raise ValueError("comparison base report body digest differs from its binding")
    checkpoint_files = validated_floor["input_sha256"]["checkpoint_export"]["files"]
    if not isinstance(checkpoint_files, Mapping):
        raise ValueError("floor checkpoint export file binding is invalid")
    conversion_validation = _validated_conversion(
        comparison_value["conversion_validation"],
        evidence_files=evidence_files,
        expected_source_model_sha256=checkpoint_files.get("model.safetensors"),
    )
    evidence, metrics = _validated_evidence(
        comparison_value, identities, evaluation_manifest_sha256=manifest_sha256
    )
    floor_value = _require_nonnegative_float("reference floor", validated_floor["F"])
    floor64 = _require_nonnegative_float("reference float64 floor", validated_floor["F64"])
    thresholds = _thresholds(floor=floor_value, floor64=floor64)
    result_metrics, gates = _decision(metrics, thresholds)
    result = {
        "format_version": 1,
        "artifact_type": EVALUATION_ARTIFACT_TYPE,
        "procedure_id": PROCEDURE_ID,
        "evaluated_at_utc": _utc_from_ns(evaluated_at_ns),
        "evaluated_at_ns": evaluated_at_ns,
        "source": {
            "floor_sha256": floor_sha256,
            "floor_bundle_sha256": floor_bundle_sha256,
            "comparison_sha256": comparison_sha256,
            "start_marker_sha256": start_marker_sha256,
            "floor_procedure_id": validated_floor["procedure_id"],
            "checkpoint_path": validated_floor["checkpoint_path"],
            "input_combined_sha256": validated_floor["input_sha256"]["combined_sha256"],
            "floor_created_at_ns": validated_floor["created_at_ns"],
            "floor_file_mtime_ns": floor_file_mtime_ns,
            "start_marker_created_at_ns": marker["created_at_ns"],
            "start_marker_file_mtime_ns": start_marker_file_mtime_ns,
            "comparison_created_at_ns": comparison_created_ns,
            "comparison_file_mtime_ns": comparison_file_mtime_ns,
            "evidence_files": evidence_files,
            "floor_input_evidence": floor_input_evidence,
            "conversion_validation": conversion_validation,
        },
        "thresholds": thresholds,
        "evidence": evidence,
        "metrics": result_metrics,
        "gates": gates,
    }
    return validate_trained_parity_report(result)


def validate_trained_parity_report(report: object) -> dict[str, object]:
    """Recompute all aggregates, thresholds, ratios, and decisions in a result."""

    value = _require_mapping(report, fields=_RESULT_FIELDS, label="parity result")
    if (
        value["format_version"] != 1
        or value["artifact_type"] != EVALUATION_ARTIFACT_TYPE
        or value["procedure_id"] != PROCEDURE_ID
    ):
        raise ValueError("trained parity result identity is invalid")
    _, evaluated_at_ns = _validate_timestamp(
        "evaluation", value["evaluated_at_utc"], value["evaluated_at_ns"]
    )
    source = _require_mapping(
        value["source"], fields=_RESULT_SOURCE_FIELDS, label="parity result source"
    )
    for field in (
        "floor_sha256",
        "floor_bundle_sha256",
        "comparison_sha256",
        "start_marker_sha256",
        "input_combined_sha256",
    ):
        _require_sha256(field.replace("_", " "), source[field])
    evidence_files = _validated_evidence_files(source["evidence_files"])
    _validated_floor_input_evidence(source["floor_input_evidence"])
    _validated_conversion(
        source["conversion_validation"], evidence_files=evidence_files
    )
    if not isinstance(source["floor_procedure_id"], str) or not source[
        "floor_procedure_id"
    ]:
        raise ValueError("parity result floor procedure is invalid")
    if not isinstance(source["checkpoint_path"], str) or not source["checkpoint_path"].startswith(
        ".cache/training/"
    ):
        raise ValueError("parity result checkpoint is invalid")
    source_times = {
        field: _require_positive_ns(field.replace("_", " "), source[field])
        for field in (
            "floor_created_at_ns",
            "floor_file_mtime_ns",
            "start_marker_created_at_ns",
            "start_marker_file_mtime_ns",
            "comparison_created_at_ns",
            "comparison_file_mtime_ns",
        )
    }
    if not (
        source_times["floor_created_at_ns"]
        <= source_times["floor_file_mtime_ns"]
        < source_times["start_marker_created_at_ns"]
        <= source_times["start_marker_file_mtime_ns"]
        <= source_times["comparison_created_at_ns"]
        <= source_times["comparison_file_mtime_ns"]
        <= evaluated_at_ns
    ):
        raise ValueError("parity result chronology is invalid")

    evidence = _require_mapping(
        value["evidence"],
        fields={
            "base_mlx_evaluation",
            "fine_mlx_evaluation",
            "torch_evaluation",
            "stats_active_parity",
        },
        label="parity result evidence",
    )
    base = evidence["base_mlx_evaluation"]
    if not isinstance(base, Mapping) or not isinstance(base.get("samples"), list):
        raise ValueError("parity result base evidence is invalid")
    if hashlib.sha256(_pretty_json(base)).hexdigest() != evidence_files[
        "base_report"
    ]["sha256"]:
        raise ValueError("parity result base report body digest differs from its binding")
    identities = [
        {field: sample[field] for field in _IDENTITY_FIELDS}
        for sample in base["samples"]
        if isinstance(sample, Mapping) and all(field in sample for field in _IDENTITY_FIELDS)
    ]
    identities = _validated_identities(identities)
    manifest_sha256 = _require_sha256(
        "base evaluation manifest", base.get("evaluation_manifest_sha256")
    )
    synthetic_comparison = {
        **dict(evidence),
        "metrics": {field: value["metrics"][field] for field in _METRIC_FIELDS},
    }
    _, base_metrics = _validated_evidence(
        synthetic_comparison, identities, evaluation_manifest_sha256=manifest_sha256
    )
    thresholds = _require_mapping(
        value["thresholds"], fields=_THRESHOLD_FIELDS, label="parity threshold"
    )
    floor_value = _require_nonnegative_float("reference floor", thresholds["reference_floor"])
    floor64 = _require_nonnegative_float(
        "reference float64 floor", thresholds["reference_floor_float64"]
    )
    if dict(thresholds) != _thresholds(floor=floor_value, floor64=floor64):
        raise ValueError("trained parity thresholds differ from the fixed procedure")
    expected_metrics, expected_gates = _decision(base_metrics, thresholds)
    recorded_metrics = _require_mapping(
        value["metrics"], fields=_RESULT_METRIC_FIELDS, label="parity result metrics"
    )
    if dict(recorded_metrics) != expected_metrics:
        raise ValueError("trained parity derived metrics are inconsistent")
    gates = _require_mapping(value["gates"], fields=_GATE_FIELDS, label="parity gates")
    if dict(gates) != expected_gates:
        raise ValueError("trained parity gate decisions are inconsistent")
    return dict(value)


def _resolve_evidence_path(root: Path, recorded: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"comparison evidence root is missing or symlinked: {root}")
    root = Path(os.path.abspath(root))
    candidate = Path(recorded)
    candidate = candidate if candidate.is_absolute() else root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"comparison evidence escapes its root: {recorded}") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"comparison evidence path is symlinked: {recorded}")
    return candidate


def _tree_inventory(
    root: Path,
    *,
    label: str,
    allowed_symlink_root: Path | None = None,
) -> tuple[dict[str, Path], tuple[SymlinkSnapshot, ...]]:
    """Enumerate one exact tree, optionally binding contained file symlinks."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root is missing, unsafe, or not a directory")
    if allowed_symlink_root is not None:
        if allowed_symlink_root.is_symlink() or not allowed_symlink_root.is_dir():
            raise ValueError(f"{label} allowed root is missing or unsafe")
        allowed_symlink_root = allowed_symlink_root.resolve(strict=True)
    files: dict[str, Path] = {}
    links: list[SymlinkSnapshot] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise ValueError(f"{label} tree cannot be enumerated") from error
        for entry in entries:
            path = directory / entry.name
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                if allowed_symlink_root is None:
                    raise ValueError(f"{label} tree contains a symlink: {relative}")
                before = os.lstat(path)
                target = os.readlink(path)
                try:
                    resolved = path.resolve(strict=True)
                except OSError as error:
                    raise ValueError(
                        f"{label} tree contains a broken symlink: {relative}"
                    ) from error
                after = os.lstat(path)
                identity_before = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                )
                identity_after = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if identity_before != identity_after or os.readlink(path) != target:
                    raise RuntimeError(f"{label} symlink changed while it was read")
                if not resolved.is_relative_to(allowed_symlink_root):
                    raise ValueError(
                        f"{label} tree symlink escapes its allowed root: {relative}"
                    )
                if not resolved.is_file() or resolved.is_symlink():
                    raise ValueError(
                        f"{label} tree symlink does not resolve to a regular file: {relative}"
                    )
                files[relative] = resolved
                links.append(
                    SymlinkSnapshot(
                        path=path,
                        device=after.st_dev,
                        inode=after.st_ino,
                        size=after.st_size,
                        mtime_ns=after.st_mtime_ns,
                        target=target,
                        resolved_path=resolved,
                    )
                )
                continue
            if entry.is_dir(follow_symlinks=False):
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                files[relative] = path
            else:
                raise ValueError(f"{label} tree contains a non-regular entry: {relative}")

    visit(root)
    if not files:
        raise ValueError(f"{label} tree contains no files")
    return files, tuple(links)


def _snapshot_floor_inputs(
    *,
    evidence_root: Path,
    recorded_evidence: object,
    floor_inputs: Mapping[str, object],
) -> FloorInputBundle:
    evidence = _validated_floor_input_evidence(
        recorded_evidence,
        expected_inputs=floor_inputs,
    )
    snapshots: dict[str, dict[str, FileSnapshot]] = {}
    tree_roots: dict[str, Path] = {}
    links: list[SymlinkSnapshot] = []
    for group_name, mode in _FLOOR_INPUT_GROUP_MODES.items():
        expected_group = floor_inputs[group_name]
        if not isinstance(expected_group, Mapping) or not isinstance(
            expected_group.get("files"), Mapping
        ):
            raise ValueError(f"{group_name} floor input hash group is invalid")
        expected_files = expected_group["files"]
        entry = evidence[group_name]
        paths: Mapping[str, Path]
        if mode in {"exact_tree", "contained_symlink_tree"}:
            root = _resolve_evidence_path(evidence_root, entry["root"])
            allowed_root = None
            if mode == "contained_symlink_tree":
                allowed_root = _resolve_evidence_path(
                    evidence_root, entry["allowed_root"]
                )
            paths, group_links = _tree_inventory(
                root,
                label=group_name.replace("_", " "),
                allowed_symlink_root=allowed_root,
            )
            links.extend(group_links)
            if set(paths) != set(expected_files):
                raise ValueError(
                    f"{group_name.replace('_', ' ')} inventory differs from the floor binding"
                )
            tree_roots[group_name] = root
        else:
            paths = {
                logical_name: _resolve_evidence_path(evidence_root, recorded_path)
                for logical_name, recorded_path in entry["paths"].items()
            }
        group_snapshots: dict[str, FileSnapshot] = {}
        for logical_name, path in sorted(paths.items()):
            snapshot = _snapshot_file(
                path,
                label=f"{group_name} floor input {logical_name}",
            )
            if snapshot.sha256 != expected_files[logical_name]:
                raise ValueError(
                    f"{group_name} input differs from the floor binding: {logical_name}"
                )
            group_snapshots[logical_name] = snapshot
        snapshots[group_name] = group_snapshots
    return FloorInputBundle(
        evidence=evidence,
        files=snapshots,
        tree_roots=tree_roots,
        links=tuple(links),
    )


def _revalidate_floor_input_locations(
    bundle: FloorInputBundle,
    *,
    evidence_root: Path,
) -> None:
    for group_name, mode in _FLOOR_INPUT_GROUP_MODES.items():
        expected = {
            logical_name: snapshot.path
            for logical_name, snapshot in bundle.files[group_name].items()
        }
        entry = bundle.evidence[group_name]
        if mode in {"exact_tree", "contained_symlink_tree"}:
            root = _resolve_evidence_path(evidence_root, entry["root"])
            if root != bundle.tree_roots[group_name]:
                raise RuntimeError(
                    f"{group_name} floor input root changed before report install"
                )
            allowed_root = None
            if mode == "contained_symlink_tree":
                allowed_root = _resolve_evidence_path(
                    evidence_root, entry["allowed_root"]
                )
            current, current_links = _tree_inventory(
                root,
                label=group_name.replace("_", " "),
                allowed_symlink_root=allowed_root,
            )
            if current != expected:
                raise RuntimeError(
                    f"{group_name} floor input inventory changed before report install"
                )
            expected_links = tuple(
                link for link in bundle.links if link.path.is_relative_to(root)
            )
            if current_links != expected_links:
                raise RuntimeError(
                    f"{group_name} floor input symlinks changed before report install"
                )
        else:
            current = {
                logical_name: _resolve_evidence_path(evidence_root, recorded_path)
                for logical_name, recorded_path in entry["paths"].items()
            }
            if current != expected:
                raise RuntimeError(
                    f"{group_name} floor input paths changed before report install"
                )


def _validate_training_manifest(
    snapshot: FileSnapshot,
    *,
    checkpoint_files: Mapping[str, object],
) -> tuple[int, int]:
    try:
        manifest = json.loads(snapshot.payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("checkpoint training manifest is not valid JSON") from error
    required = {
        "format_version",
        "artifact_type",
        "dtype",
        "tensor_count",
        "parameter_count",
        "source_checkpoint",
        "metadata",
        "file_sha256",
    }
    manifest = _require_mapping(
        manifest,
        fields=required,
        label="checkpoint training manifest",
    )
    if (
        manifest["format_version"] != 1
        or manifest["artifact_type"]
        != "smolvla-mlx-merged-training-checkpoint"
        or manifest["dtype"] != "float32"
    ):
        raise ValueError("checkpoint training manifest identity is invalid")
    source = _require_mapping(
        manifest["source_checkpoint"],
        fields={"repo_id", "revision"},
        label="checkpoint training manifest source",
    )
    from mlx_smolvla._lab.reference.discovery import CHECKPOINT_ID, CHECKPOINT_REVISION

    if dict(source) != {
        "repo_id": CHECKPOINT_ID,
        "revision": CHECKPOINT_REVISION,
    }:
        raise ValueError("checkpoint training manifest source is invalid")
    if not isinstance(manifest["metadata"], Mapping):
        raise ValueError("checkpoint training manifest metadata is invalid")
    tensor_count = manifest["tensor_count"]
    parameter_count = manifest["parameter_count"]
    if (
        isinstance(tensor_count, bool)
        or not isinstance(tensor_count, int)
        or tensor_count <= 0
        or isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
    ):
        raise ValueError("checkpoint training manifest inventory is invalid")
    expected_file_hashes = {
        name: digest
        for name, digest in checkpoint_files.items()
        if name != "training_manifest.json"
    }
    file_hashes = manifest["file_sha256"]
    if not isinstance(file_hashes, Mapping) or dict(file_hashes) != expected_file_hashes:
        raise ValueError("checkpoint training manifest file hashes differ from the export")
    return tensor_count, parameter_count


def evaluate_trained_parity_files(
    *,
    floor_path: str | Path,
    variant_root: str | Path,
    start_marker_path: str | Path,
    comparison_path: str | Path,
    output_path: str | Path,
    evidence_root: str | Path = ".",
    evaluated_at_ns: int | None = None,
) -> tuple[dict[str, object], str]:
    """Validate every concrete input, then install one no-clobber verdict."""

    floor_path = Path(floor_path)
    marker_path = Path(start_marker_path)
    comparison_path = Path(comparison_path)
    output_path = Path(output_path)
    named_paths = [floor_path, marker_path, comparison_path, output_path]
    if len({str(path.resolve()) for path in named_paths}) != len(named_paths):
        raise ValueError("parity inputs and output paths must be distinct")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"parity output already exists: {output_path}")

    floor, floor_snapshot = _snapshot_json(
        floor_path, label="prospective self-consistency floor"
    )
    bundle = _load_floor_bundle(floor, variant_root=variant_root)
    marker, marker_snapshot = _snapshot_json(marker_path, label="comparison start marker")
    comparison, comparison_snapshot = _snapshot_json(
        comparison_path, label="trained comparison"
    )
    if not isinstance(marker, Mapping):
        raise ValueError("comparison start marker must be an object")
    if Path(str(marker.get("comparison_path", ""))).resolve() != comparison_path.resolve():
        raise ValueError("comparison start marker was issued for a different comparison path")
    comparison_value, _, evidence_file_records = _validate_comparison(comparison)

    evidence_snapshots: list[FileSnapshot] = []
    evidence_snapshot_by_name: dict[str, FileSnapshot] = {}
    evidence_paths: dict[str, Path] = {}
    base_document: object | None = None
    for name, record in evidence_file_records.items():
        evidence_path = _resolve_evidence_path(Path(evidence_root), record["path"])
        evidence_paths[name] = evidence_path
        snapshot = _snapshot_file(evidence_path, label=f"{name} evidence")
        if snapshot.sha256 != record["sha256"]:
            raise ValueError(f"{name} evidence SHA-256 differs from the actual file")
        evidence_snapshots.append(snapshot)
        evidence_snapshot_by_name[name] = snapshot
        if name == "base_report":
            try:
                base_document = json.loads(snapshot.payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError("base report evidence is not valid JSON") from error
    if base_document != comparison_value["base_mlx_evaluation"]:
        raise ValueError("comparison base evidence differs from the frozen base report file")

    if not isinstance(floor, Mapping):
        raise ValueError("prospective floor must be a JSON object")
    checkpoint_path = floor.get("checkpoint_path")
    floor_inputs = floor.get("input_sha256")
    checkpoint_group = (
        floor_inputs.get("checkpoint_export")
        if isinstance(floor_inputs, Mapping)
        else None
    )
    if (
        not isinstance(checkpoint_path, str)
        or not isinstance(floor_inputs, Mapping)
        or not isinstance(checkpoint_group, Mapping)
        or not isinstance(checkpoint_group.get("files"), Mapping)
    ):
        raise ValueError("floor checkpoint export binding is invalid")
    floor_input_bundle = _snapshot_floor_inputs(
        evidence_root=Path(evidence_root),
        recorded_evidence=comparison_value["floor_input_evidence"],
        floor_inputs=floor_inputs,
    )
    _reject_path_overlaps(
        {"parity output": output_path},
        {
            "raw floor variant tree": Path(variant_root),
            **{
                f"{group_name} floor input tree": root
                for group_name, root in floor_input_bundle.tree_roots.items()
            },
        },
    )
    checkpoint_root = floor_input_bundle.tree_roots["checkpoint_export"]
    expected_checkpoint_root = _resolve_evidence_path(
        Path(evidence_root), checkpoint_path
    )
    if checkpoint_root != expected_checkpoint_root:
        raise ValueError("checkpoint input evidence root differs from the floor path")
    checkpoint_snapshots = floor_input_bundle.files["checkpoint_export"]
    source_model_snapshot = checkpoint_snapshots.get("model.safetensors")
    manifest_snapshot = checkpoint_snapshots.get("training_manifest.json")
    if source_model_snapshot is None:
        raise ValueError("floor checkpoint export is missing model.safetensors")
    if manifest_snapshot is None:
        raise ValueError("floor checkpoint export is missing training_manifest.json")
    manifest_tensor_count, manifest_parameter_count = _validate_training_manifest(
        manifest_snapshot,
        checkpoint_files=checkpoint_group["files"],
    )
    conversion_result = _validate_snapshot_conversion(
        source_model=source_model_snapshot,
        converted_model=evidence_snapshot_by_name["native_conversion_model"],
        name_map=evidence_snapshot_by_name["native_conversion_name_map"],
    )
    actual_conversion = {
        "source_model_sha256": conversion_result.source_model_sha256,
        "converted_model_sha256": conversion_result.converted_model_sha256,
        "name_map_sha256": conversion_result.name_map_sha256,
        "dtype": conversion_result.dtype,
        "tensor_count": conversion_result.tensor_count,
        "parameter_count": conversion_result.parameter_count,
    }
    declared_conversion = _validated_conversion(
        comparison_value["conversion_validation"],
        evidence_files=evidence_file_records,
        expected_source_model_sha256=checkpoint_group["files"].get(
            "model.safetensors"
        ),
    )
    if actual_conversion != declared_conversion:
        raise ValueError(
            "native conversion validation differs from the floor-bound source files"
        )
    if (
        actual_conversion["tensor_count"] != manifest_tensor_count
        or actual_conversion["parameter_count"] != manifest_parameter_count
    ):
        raise ValueError(
            "training manifest inventory differs from the native conversion inventory"
        )

    evaluated_at_ns = time.time_ns() if evaluated_at_ns is None else evaluated_at_ns
    report = evaluate_trained_parity_documents(
        floor=floor,
        floor_sha256=floor_snapshot.sha256,
        floor_file_mtime_ns=floor_snapshot.mtime_ns,
        floor_bundle_sha256=bundle.bundle_sha256,
        start_marker=marker,
        start_marker_sha256=marker_snapshot.sha256,
        start_marker_file_mtime_ns=marker_snapshot.mtime_ns,
        comparison=comparison,
        comparison_sha256=comparison_snapshot.sha256,
        comparison_file_mtime_ns=comparison_snapshot.mtime_ns,
        evaluated_at_ns=evaluated_at_ns,
    )
    snapshots = (
        floor_snapshot,
        *bundle.snapshots,
        marker_snapshot,
        comparison_snapshot,
        *evidence_snapshots,
        *floor_input_bundle.snapshots,
    )
    for name, record in evidence_file_records.items():
        if _resolve_evidence_path(Path(evidence_root), record["path"]) != evidence_paths[
            name
        ]:
            raise RuntimeError(f"comparison evidence path changed before install: {name}")
    _revalidate_floor_input_locations(
        floor_input_bundle,
        evidence_root=Path(evidence_root),
    )
    _revalidate_snapshots(snapshots)
    digest = _atomic_json_no_clobber(output_path, report)
    return report, digest
