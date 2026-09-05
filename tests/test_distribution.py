"""Distribution metadata contracts for the dependency-isolated runtime."""

from __future__ import annotations

from importlib.metadata import distribution
from importlib.util import find_spec
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def test_installed_runtime_does_not_require_the_vendored_mlx_vlm_stack() -> None:
    """Catch an accidental dependency on the unused MLX-VLM server/audio stack."""

    requirements = [
        Requirement(requirement)
        for requirement in distribution("mlx-smolvla").requires or ()
    ]
    names = {requirement.name.lower() for requirement in requirements if requirement.marker is None}

    assert "mlx-vlm" not in names
    assert "torch" not in names
    assert "lerobot" not in names
    assert "transformers" not in names


def test_installed_runtime_requires_the_audited_dependency_versions() -> None:
    """Keep source installs on the exact dependency set validated by parity tests."""

    requirements = [
        Requirement(requirement)
        for requirement in distribution("mlx-smolvla").requires or ()
    ]
    runtime = {
        requirement.name.lower(): str(requirement.specifier)
        for requirement in requirements
        if requirement.marker is None
    }

    mlx_requirement = SpecifierSet(runtime.pop("mlx"))
    assert "0.32.0" in mlx_requirement
    assert "0.32.1" in mlx_requirement
    assert "0.32.2" in mlx_requirement
    assert "0.31.2" not in mlx_requirement
    assert "0.32.3" not in mlx_requirement
    assert "0.33.0" not in mlx_requirement
    assert runtime == {
        "huggingface-hub": "<2,>=1.29.0",
        "numpy": "<3,>=2.2.6",
        "pillow": "<13,>=12.3.0",
        "safetensors": "<0.9,>=0.8.0",
        "tokenizers": "<0.23,>=0.22.2",
    }


def test_base_distribution_supports_cpython_311_through_313() -> None:
    requires_python = distribution("mlx-smolvla").metadata["Requires-Python"]
    supported = SpecifierSet(requires_python)

    assert "3.11" in supported
    assert "3.12" in supported
    assert "3.13" in supported
    assert "3.10" not in supported
    assert "3.14" not in supported


def test_reference_extra_is_guarded_by_lerobots_python_312_floor() -> None:
    requirements = [
        Requirement(requirement)
        for requirement in distribution("mlx-smolvla").requires or ()
    ]
    guarded = [
        requirement
        for requirement in requirements
        if requirement.name.lower() in {"lerobot", "torch"}
        and requirement.marker is not None
        and requirement.marker.evaluate(
            {"extra": "reference", "python_version": "3.12"}
        )
    ]

    assert {requirement.name.lower() for requirement in guarded} == {"lerobot", "torch"}
    for requirement in guarded:
        assert requirement.marker is not None
        assert requirement.marker.evaluate(
            {"extra": "reference", "python_version": "3.11"}
        ) is False
        assert requirement.marker.evaluate(
            {"extra": "reference", "python_version": "3.12"}
        ) is True


def test_serve_extra_is_optional_and_guarded_by_lerobots_python_312_floor() -> None:
    requirements = [
        Requirement(requirement)
        for requirement in distribution("mlx-smolvla").requires or ()
    ]
    serve_requirements = [
        requirement
        for requirement in requirements
        if requirement.name.lower() == "lerobot"
        and requirement.marker is not None
        and requirement.marker.evaluate({"extra": "serve", "python_version": "3.12"})
    ]

    assert len(serve_requirements) == 1
    requirement = serve_requirements[0]
    assert requirement.specifier == SpecifierSet("==0.6.1")
    assert requirement.extras == {"async"}
    assert requirement.marker.evaluate({"extra": "serve", "python_version": "3.11"}) is False
    assert "serve" in (distribution("mlx-smolvla").metadata.get_all("Provides-Extra") or [])


def test_native_extension_build_is_optional_and_targets_macos_14(
    monkeypatch,
) -> None:
    from _build_backend import setup_kwargs

    monkeypatch.setenv("MLX_SMOLVLA_BUILD_NATIVE", "0")
    monkeypatch.delenv("MACOSX_DEPLOYMENT_TARGET", raising=False)
    configuration = setup_kwargs()

    assert os.environ["MACOSX_DEPLOYMENT_TARGET"] == "14.0"
    assert configuration["ext_modules"] == []
    assert configuration["cmdclass"] == {}


def test_native_extension_is_built_by_default(monkeypatch) -> None:
    from _build_backend import setup_kwargs

    monkeypatch.delenv("MLX_SMOLVLA_BUILD_NATIVE", raising=False)
    configuration = setup_kwargs()

    assert len(configuration["ext_modules"]) == 1
    assert configuration["ext_modules"][0].name == "mlx_smolvla._rmsnorm_native"
    assert Path(configuration["ext_modules"][0].sourcedir).name == "native"
    assert "build_ext" in configuration["cmdclass"]


def test_training_package_is_shipped_as_an_optional_surface() -> None:
    metadata = distribution("mlx-smolvla").metadata

    assert find_spec("mlx_smolvla._lab.training") is not None
    assert "train" in (metadata.get_all("Provides-Extra") or [])


def test_trained_parity_subpackage_is_in_the_wheel_package_list() -> None:
    from _build_backend import setup_kwargs

    configuration = setup_kwargs()

    assert "mlx_smolvla.training" in configuration["packages"]
    assert "mlx_smolvla._lab.reference" in configuration["packages"]
    assert find_spec("mlx_smolvla.training.trained_parity") is not None


def test_built_wheel_contains_and_imports_the_trained_parity_surface(
    tmp_path: Path,
) -> None:
    wheel_dir = tmp_path / "wheel"
    source = tmp_path / "source"
    shutil.copytree(
        Path.cwd(),
        source,
        ignore=shutil.ignore_patterns(".cache", ".git", ".venv", "build", "dist"),
    )
    environment = dict(os.environ)
    environment["UV_CACHE_DIR"] = str((Path.cwd() / ".cache" / "uv").resolve())
    built = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir), str(source)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    wheels = tuple(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    assert {name.split("/")[0] for name in names if ".dist-info/" not in name} == {"mlx_smolvla"}
    assert not any("_build_backend" in name for name in names)
    assert "mlx_smolvla/server.py" in names
    assert "mlx_smolvla/training/trained_parity.py" in names
    assert "mlx_smolvla/_lab/training/t3_contract.py" in names
    assert "mlx_smolvla/_lab/training/ux.py" in names
    assert "mlx_smolvla/_lab/training/benchmark.py" in names
    assert "mlx_smolvla/_lab/reference/discovery.py" in names

    target = tmp_path / "installed"
    installed = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(target),
            "--no-deps",
            str(wheels[0]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr
    environment["PYTHONPATH"] = str(target)
    for decoy in ("training", "reference"):
        (tmp_path / decoy).mkdir()
        (tmp_path / decoy / "__init__.py").write_text("")
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib; "
                "import mlx_smolvla.training.trained_parity as p; "
                "print(pathlib.Path(p.__file__).resolve())"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    assert str(target.resolve()) in imported.stdout


def test_extension_free_wheel_imports_with_pure_mlx_fallback(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheel"
    source = tmp_path / "source"
    shutil.copytree(
        Path.cwd(),
        source,
        ignore=shutil.ignore_patterns(".cache", ".git", ".venv", "build", "dist"),
    )
    environment = dict(os.environ)
    environment["UV_CACHE_DIR"] = str((Path.cwd() / ".cache" / "uv").resolve())
    environment["MLX_SMOLVLA_BUILD_NATIVE"] = "0"
    built = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir), str(source)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    wheel = next(wheel_dir.glob("*.whl"))
    assert wheel.name.endswith("py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.endswith((".so", ".dylib")) for name in archive.namelist())

    target = tmp_path / "installed"
    installed = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(target),
            "--no-deps",
            str(wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr
    site_packages = next(
        Path(entry)
        for entry in sys.path
        if Path(entry).name == "site-packages" and Path(entry).is_dir()
    )
    checked = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import pathlib, sys; "
                f"sys.path[:0] = [{str(target)!r}, {str(site_packages)!r}]; "
                "import mlx_smolvla.rmsnorm as r; "
                "assert r.cpu_compatibility_backend() == 'pure-mlx-fallback'; "
                "print(pathlib.Path(r.__file__).resolve())"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
    assert str(target.resolve()) in checked.stdout
