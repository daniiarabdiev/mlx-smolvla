"""Canonical-name and one-release cache compatibility contracts."""

from __future__ import annotations

from pathlib import Path
import tomllib
import warnings


def test_distribution_import_cli_and_native_extension_use_canonical_name() -> None:
    from reference._build_backend import setup_kwargs

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["name"] == "mlx-smolvla"
    assert project["project"]["scripts"] == {
        "mlx-smolvla": "mlx_smolvla.cli:main"
    }

    configuration = setup_kwargs()
    assert "mlx_smolvla" in configuration["packages"]
    assert "smolvla_mlx" not in configuration["packages"]
    assert configuration["ext_modules"][0].name == "mlx_smolvla._rmsnorm_native"


def test_cli_identifies_itself_with_canonical_name() -> None:
    from mlx_smolvla.cli import _parser

    assert _parser().prog == "mlx-smolvla"


def test_new_cache_environment_variable_and_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from mlx_smolvla.cache import resolve_cache_dir

    configured = tmp_path / "configured"
    monkeypatch.setenv("MLX_SMOLVLA_CACHE", str(configured))
    monkeypatch.delenv("SMOLVLA_MLX_CACHE", raising=False)
    assert resolve_cache_dir() == configured.resolve()

    monkeypatch.delenv("MLX_SMOLVLA_CACHE")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert resolve_cache_dir() == tmp_path / ".cache" / "mlx_smolvla"


def test_legacy_cache_environment_variable_warns_for_one_release(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from mlx_smolvla.cache import resolve_cache_dir

    legacy = tmp_path / "legacy"
    monkeypatch.delenv("MLX_SMOLVLA_CACHE", raising=False)
    monkeypatch.setenv("SMOLVLA_MLX_CACHE", str(legacy))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_cache_dir()

    assert resolved == legacy.resolve()
    assert len(caught) == 1
    assert issubclass(caught[0].category, FutureWarning)
    assert "MLX_SMOLVLA_CACHE" in str(caught[0].message)
    assert "SMOLVLA_MLX_CACHE" in str(caught[0].message)
    assert "one release" in str(caught[0].message)


def test_new_cache_variable_wins_and_legacy_variable_still_warns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from mlx_smolvla.cache import resolve_cache_dir

    current = tmp_path / "current"
    monkeypatch.setenv("MLX_SMOLVLA_CACHE", str(current))
    monkeypatch.setenv("SMOLVLA_MLX_CACHE", str(tmp_path / "ignored"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_cache_dir()

    assert resolved == current.resolve()
    assert len(caught) == 1
    assert "ignored" in str(caught[0].message)


def test_makefile_and_workflow_export_only_the_canonical_cache_variable() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/macos-15.yml").read_text(encoding="utf-8")

    assert "MLX_SMOLVLA_CACHE" in makefile
    assert ".cache/mlx_smolvla" in makefile
    assert "SMOLVLA_MLX_CACHE" not in makefile
    assert "MLX_SMOLVLA_CACHE" in workflow
    assert ".cache/mlx_smolvla" in workflow
    assert "SMOLVLA_MLX_CACHE" not in workflow


def test_native_build_toggle_uses_canonical_environment_prefix(monkeypatch) -> None:
    from reference._build_backend import setup_kwargs

    monkeypatch.setenv("MLX_SMOLVLA_BUILD_NATIVE", "0")
    monkeypatch.delenv("SMOLVLA_MLX_BUILD_NATIVE", raising=False)

    configuration = setup_kwargs()

    assert configuration["ext_modules"] == []
