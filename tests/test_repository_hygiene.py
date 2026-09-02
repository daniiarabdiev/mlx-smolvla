"""Public tracked-tree and historical-correction contracts."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = {
    ".github",
    ".gitignore",
    "AGENTS.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CLAUDE.md",  # Explicit coding-agent-guide companion required by the brief.
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "NOTICE",
    "README.md",
    "docs",
    "examples",
    "hardware",
    "mlx_smolvla",
    "pyproject.toml",
    "reference",
    "scripts",
    "tests",
    "training",
    "uv.lock",
}


def _public_files() -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(
        ROOT / raw.decode("utf-8")
        for raw in completed.stdout.split(b"\0")
        if raw
    )


def test_public_root_matches_the_release_allowlist_exactly() -> None:
    actual = {path.relative_to(ROOT).parts[0] for path in _public_files()}

    assert actual == PUBLIC_ROOT


def test_operator_configuration_is_ignored_and_not_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".codex/config.toml"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".codex/config.toml"],
        cwd=ROOT,
        check=False,
    )

    assert tracked.returncode != 0
    assert ignored.returncode == 0


def test_history_index_records_the_first_port_claim_correction() -> None:
    history = (ROOT / "docs/history/README.md").read_text(encoding="utf-8")

    assert "made in error" in history
    assert "corrected on 2026-09-02" in history
    assert "tokimoa/smolvla-mlx" in history


def test_erroneous_first_port_claim_exists_only_in_history() -> None:
    erroneous = "Nobody has shipped native Apple Silicon inference for this model."
    offenders = []
    for path in _public_files():
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.suffix.lower() not in {".md", ".toml", ".cff"}:
            continue
        if relative.parts[:2] == ("docs", "history"):
            continue
        if erroneous in path.read_text(encoding="utf-8", errors="replace"):
            offenders.append(str(relative))

    assert offenders == []


def test_public_markdown_links_resolve_locally() -> None:
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    broken: list[str] = []
    for path in _public_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() != ".md" or relative.parts[:2] == ("docs", "history"):
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        for raw_target in link_pattern.findall(source):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{relative}: {raw_target}")

    assert broken == []


def test_public_tree_contains_no_absolute_operator_home_path() -> None:
    absolute_home_prefix = b"/" + b"Users" + b"/"
    offenders = []
    for path in _public_files():
        relative = path.relative_to(ROOT)
        if not path.is_file() or relative.parts[:2] == ("docs", "history"):
            continue
        if absolute_home_prefix in path.read_bytes():
            offenders.append(str(relative))

    assert offenders == []


def test_active_spec_references_and_generated_output_defaults_use_moved_paths() -> None:
    expected_references = {
        "docs/ARCHITECTURE.md": ("history/BRIEF.md", "history/BRIEF_T3B.md"),
        ".github/workflows/macos-15.yml": ("docs/history/BRIEF_FULL.md",),
        "reference/discovery.py": ("docs/history/BRIEF.md",),
        "scripts/inspect_reference.py": ("docs/history/BRIEF.md",),
    }
    expected_defaults = {
        "scripts/bench.py": 'default=Path("docs/BENCHMARK.md")',
        "scripts/profile_inference_dtypes.py": (
            'default=Path("docs/evidence/BF16_PROFILE.json")'
        ),
        "scripts/benchmark_inference_comparison.py": (
            'default=Path("docs/evidence/INFERENCE_COMPARISON.json")'
        ),
        "scripts/experiment_quantization.py": (
            'default=Path("docs/evidence/QUANTIZATION_EXPERIMENT.json")'
        ),
    }

    for relative_path, references in expected_references.items():
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        for reference in references:
            assert reference in source, (relative_path, reference)

    for relative_path, expected in expected_defaults.items():
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert expected in source, relative_path
