"""Contracts for the public 0.1.2 repository surface."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/daniiarabdiev/mlx-smolvla"


def _project() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def test_public_distribution_metadata_is_complete_and_canonical() -> None:
    project = _project()

    assert project["name"] == "mlx-smolvla"
    assert project["version"] == "0.1.2"
    assert project["readme"] == "README.md"
    assert project["license"] == "Apache-2.0"
    assert project["keywords"] == [
        "apple-silicon",
        "lerobot",
        "mlx",
        "robotics",
        "smolvla",
        "vision-language-action",
    ]
    assert set(project["classifiers"]) >= {
        "Development Status :: 3 - Alpha",
        "Environment :: MacOS X",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    }
    assert project["urls"] == {
        "Changelog": f"{REPOSITORY}/blob/main/CHANGELOG.md",
        "Documentation": f"{REPOSITORY}/tree/main/docs",
        "Homepage": REPOSITORY,
        "Issues": f"{REPOSITORY}/issues",
        "Repository": REPOSITORY,
    }


def test_source_and_package_versions_match_public_release() -> None:
    from mlx_smolvla import __version__

    assert __version__ == "0.1.2"


def test_public_community_files_are_present_and_actionable() -> None:
    expected = (
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "CITATION.cff",
        "CLAUDE.md",
        ".github/CODE_OF_CONDUCT.md",
        ".github/SECURITY.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
    )
    for relative_path in expected:
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert path.read_text(encoding="utf-8").strip(), relative_path

    bug_form = (ROOT / ".github/ISSUE_TEMPLATE/bug_report.yml").read_text(
        encoding="utf-8"
    )
    assert "mlx-smolvla doctor" in bug_form


def test_public_agent_guide_preserves_engineering_contracts() -> None:
    guide = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8").strip()

    assert "make test-fast" in guide
    assert "make test" in guide
    assert "mlx-smolvla serve" in guide
    assert "mlx-smolvla train" in guide
    assert "torch" in guide.lower()
    assert "transformers" in guide.lower()
    assert "lerobot" in guide.lower()
    assert "never loosen" in guide.lower()
    assert 'execution_mode="production"' in guide
    assert 'execution_mode="strict"' in guide
    assert "/" + "Users" + "/" not in guide
    assert "HUMAN_TASKS.md" not in guide
    assert claude == "See [AGENTS.md](AGENTS.md)."


def test_readme_acknowledges_the_independent_prior_mlx_port_factually() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Related projects" in readme
    assert "https://huggingface.co/tokimoa/smolvla-mlx" in readme
    assert "2026-07-29" in readme
    assert "verified parity gates" in readme.lower()
    assert "without torch or transformers at runtime" in readme.lower()
    assert "lerobot-protocol serving" in readme.lower()
    assert "training" in readme.lower()


def test_readme_states_the_hardware_extra_python_floor() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "reference, serve, train, and hardware extras" in readme


def test_makefile_exposes_fast_lane_without_changing_full_lane() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "test-fast:" in makefile
    assert "uv run --all-extras pytest -m 'not slow' $(TESTS)" in makefile
    assert "test:\n\tuv run --all-extras pytest $(TESTS)" in makefile
