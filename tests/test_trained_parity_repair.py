"""A repaired comparison cannot rewrite history or soften a fixed gate."""

from copy import deepcopy
import importlib
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def _module():
    name = "scripts.check_trained_parity_repair"
    assert importlib.util.find_spec(name) is not None, "repair guard is not implemented"
    return importlib.import_module(name)


def test_repair_requires_floor_write_before_start_before_outcome() -> None:
    module = _module()
    module.check_chronology(10, 11, 12)
    module.check_chronology(10, 11, 12, 13)
    module.check_chronology(10, 11, 12, 13, 14, 15)
    # Reject a floor written after comparison, a backdated marker, or an old outcome.
    for values in ((10, 13, 12, 13, 14, 15), (10, 11, 12, 11, 14, 15),
                   (10, 11, 12, 13, 12, 15), (10, 11, 12, 13, 16, 15),
                   (10, 9, 12, 13, 14, 15), (True, 11, 12, 13, 14, 15)):
        with pytest.raises(ValueError, match="chronology"):
            module.check_chronology(*values)


def test_repair_output_is_new_and_cannot_clobber_history(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "repair"
    module.reserve_output(output)
    protected = output / "outcome.json"
    protected.write_text("original evidence")
    with pytest.raises(FileExistsError):
        module.reserve_output(output)
    assert protected.read_text() == "original evidence"
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)
    with pytest.raises((FileExistsError, ValueError)):
        module.reserve_output(alias)
    assert protected.read_text() == "original evidence"


def test_repair_rejects_changed_inputs_before_install(tmp_path: Path) -> None:
    module = _module()
    protected = tmp_path / "input.json"
    protected.write_text("before")
    snapshot = module.capture_file(protected)
    module.revalidate_files([snapshot])
    protected.write_text("after!")
    with pytest.raises(RuntimeError, match="changed"):
        module.revalidate_files([snapshot])


def test_repair_output_cannot_change_a_protected_input_tree(tmp_path: Path) -> None:
    module = _module()
    protected = tmp_path / "export"
    protected.mkdir()
    for destination in (protected, protected / "repair", tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            module.reject_output_overlap(destination, [protected])
    assert list(protected.iterdir()) == []
    module.reject_output_overlap(tmp_path / "separate-repair", [protected])


@pytest.mark.slow
def test_repair_recomputes_old_failure_and_rejects_softened_limits() -> None:
    module = _module()
    # Genuine saved failing comparison: no model work or changed checkpoint here.
    outcome = json.loads(Path(".cache/training/t3b/outcome.json").read_text())
    decision = module.validate_fixed_outcome(outcome)
    assert decision["passed"] is False
    assert decision["parity_passed"] is False
    assert decision["improvement_passed"] is True
    assert decision["roundtrip_passed"] is True
    softened = deepcopy(outcome)
    softened["thresholds"]["stats_active_parity_max_abs"] = 0.5
    with pytest.raises(ValueError, match="fixed limits"):
        module.validate_fixed_outcome(softened)
    forged = deepcopy(outcome)
    forged["gates"]["passed"] = True
    with pytest.raises(ValueError, match="decision"):
        module.validate_fixed_outcome(forged)


@pytest.mark.parametrize(("physical_max", "passes"), [(0.005, True), (0.00500001, False)])
@pytest.mark.slow
def test_repair_keeps_physical_action_limit_even_when_normalized_is_exact(
    physical_max: float, passes: bool
) -> None:
    module = _module()
    outcome = json.loads(Path(".cache/training/t3b/outcome.json").read_text())
    parity = outcome["stats_active_parity"]
    for sample in parity["samples"]:
        for metric in ("image_preprocessing_max_abs", "state_preprocessing_max_abs",
                       "preprocessing_max_abs", "normalized_action_max_abs",
                       "physical_action_max_abs", "physical_action_standardized_max_abs"):
            sample[metric] = physical_max if metric == "physical_action_max_abs" else 0.0
            parity[metric] = sample[metric]
    parity["gate_max_abs"] = physical_max
    parity["passed"] = passes
    outcome["gates"].update(passed=passes, parity_passed=passes, parity_max_abs=physical_max)
    assert module.validate_fixed_outcome(outcome)["passed"] is passes


@pytest.mark.slow
def test_repair_rejects_incomplete_or_forged_sample_evidence() -> None:
    module = _module()
    outcome = json.loads(Path(".cache/training/t3b/outcome.json").read_text())
    incomplete = deepcopy(outcome)
    incomplete["stats_active_parity"]["samples"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        module.validate_fixed_outcome(incomplete)
    forged = deepcopy(outcome)
    forged["stats_active_parity"]["physical_action_max_abs"] = 0.0
    with pytest.raises(ValueError, match="sample evidence"):
        module.validate_fixed_outcome(forged)


def test_repair_does_not_adopt_changed_validated_bytes(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "worker.json"
    source.write_text("validated bytes")
    snapshot = module._snapshot_file(source, label="worker")
    source.write_text("replacement bytes")
    assert hasattr(module, "binding_from_snapshot"), "validated snapshot binding is missing"
    binding = module.binding_from_snapshot(snapshot)
    with pytest.raises(RuntimeError, match="changed"):
        module.revalidate_files([binding])


@pytest.mark.parametrize("replaced", ["outcome.json", "converted.safetensors"])
def test_repair_rechecks_promoted_outcome_and_inputs_before_verdict(
    tmp_path: Path, replaced: str
) -> None:
    module = _module()
    outcome = tmp_path / "outcome.json"
    converted = tmp_path / "converted.safetensors"
    outcome.write_text("original outcome")
    converted.write_text("original converted weights")
    bindings = [module.capture_file(outcome), module.capture_file(converted)]
    (tmp_path / replaced).write_text("replaced after promotion")
    destination = tmp_path / "verdict.json"
    assert hasattr(module, "install_verdict"), "final publication guard is missing"
    with pytest.raises(RuntimeError, match="changed"):
        module.install_verdict(destination, {"passed": True}, bindings)
    assert not destination.exists()


@pytest.mark.slow
def test_repair_canonical_outcome_install_is_no_clobber(tmp_path: Path) -> None:
    module = _module()
    staging = tmp_path / "staging.json"
    payload = Path(".cache/training/t3b/outcome.json").read_bytes()
    staging.write_bytes(payload)
    output = tmp_path / "outcome.json"
    output.write_text("concurrent writer")
    assert hasattr(module, "install_outcome"), "no-clobber outcome installation is missing"
    with pytest.raises(FileExistsError):
        module.install_outcome(staging, output, hashlib.sha256(payload).hexdigest())
    assert output.read_text() == "concurrent writer"
    with pytest.raises(RuntimeError, match="changed"):
        module.install_outcome(staging, tmp_path / "fresh.json", "0" * 64)
    assert not (tmp_path / "fresh.json").exists()


@pytest.mark.slow
def test_repair_does_not_accept_coherently_inflated_base_mae() -> None:
    module = _module()
    outcome = json.loads(Path(".cache/training/t3b/outcome.json").read_text())
    base = outcome["base_mlx_evaluation"]
    base["absolute_error_sum"] *= 2
    base["mlx_mae"] *= 2
    for sample in base["samples"]:
        sample["absolute_error_sum"] *= 2
    outcome["gates"]["fine_to_base_ratio"] /= 2
    with pytest.raises(ValueError, match="frozen base"):
        module.validate_fixed_outcome(outcome)
