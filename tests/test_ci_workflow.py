"""Stage Q P2-4 disabled macOS CI and activation-contract tests."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_macos_15_workflow_is_valid_but_unconditionally_disabled() -> None:
    path = Path(".github/workflows/macos-15.yml")
    source = path.read_text(encoding="utf-8")
    workflow = yaml.load(source, Loader=yaml.BaseLoader)

    assert isinstance(workflow, dict)
    assert workflow["on"] == {"workflow_dispatch": ""}
    assert workflow["permissions"] == {"contents": "read"}
    assert "push" not in workflow["on"]
    assert "pull_request" not in workflow["on"]

    job = workflow["jobs"]["full-suite"]
    assert job["if"] == "${{ false }}"
    assert job["runs-on"] == "macos-15"
    assert job["timeout-minutes"] == "360"
    assert "secrets." not in source


def test_disabled_workflow_preserves_the_intended_hermetic_full_suite() -> None:
    workflow = yaml.load(
        Path(".github/workflows/macos-15.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["full-suite"]
    environment = job["env"]
    for name in ("HF_HOME", "UV_CACHE_DIR", "MLX_SMOLVLA_CACHE"):
        assert environment[name].startswith("${{ github.workspace }}/.cache/")

    commands = "\n".join(
        step.get("run", "") for step in job["steps"] if isinstance(step, dict)
    )
    assert "uv==0.11.25" in commands
    assert "uv sync --extra reference" in commands
    assert "make goldens" in commands
    assert "make training-goldens" in commands
    assert "make optimizer-goldens" in commands
    assert "make test" in commands
    assert "48 * 1024**3" in commands
    assert "80 * 1024**3" in commands


def test_ci_document_records_hosted_limits_and_exact_activation_requirements() -> None:
    document = Path("CI.md").read_text(encoding="utf-8")
    assert "7 GB" in document and "14 GB" in document
    assert "48 GiB" in document and "80 GiB" in document
    assert "self-hosted" in document
    assert "Apple Silicon" in document
    assert "T3/T3B/T4" in document
    assert "No secrets are required" in document
    assert "https://docs.github.com/en/actions/reference/runners/github-hosted-runners" in document
    assert "https://docs.github.com/en/actions/reference/runners/larger-runners" in document
