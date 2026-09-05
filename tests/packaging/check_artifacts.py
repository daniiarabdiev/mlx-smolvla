"""Check built distributions without importing the source tree or pytest conftest.

Usage: python tests/packaging/check_artifacts.py --wheel dist/*.whl \
    --expected-version 0.1.2 [--sdist dist/*.tar.gz] [--python /venv/bin/python]

The optional interpreter must already have this wheel and base dependencies
installed. No model, data, hardware, reference extras, or local evidence is used.
"""
from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile


def check_wheel(wheel: Path, expected_version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}
        info = f"mlx_smolvla-{expected_version}.dist-info"
        assert {name.split("/")[0] for name in names} == {"mlx_smolvla", info}, names
        assert not any("_build_backend" in name or "build_support" in name for name in names)
        metadata = BytesParser().parsebytes(archive.read(f"{info}/METADATA"))
        assert metadata["Name"] == "mlx-smolvla"
        assert metadata["Version"] == expected_version
        assert metadata["Requires-Python"] == ">=3.11,<3.14" or metadata["Requires-Python"] == "<3.14,>=3.11"
        mlx_requirement = [item for item in metadata.get_all("Requires-Dist", [])
                           if re.match(r"mlx\s*[<>=!( ]", item)]
        assert len(mlx_requirement) == 1, mlx_requirement
        bounds = set(mlx_requirement[0].replace("mlx", "", 1).replace(" ", "").strip("()").split(","))
        assert bounds == {">=0.32.0", "<0.32.3"}, mlx_requirement
        wheel_metadata = BytesParser().parsebytes(archive.read(f"{info}/WHEEL"))
        assert wheel_metadata["Root-Is-Purelib"] == "false"
        tags = wheel_metadata.get_all("Tag", [])
        assert len(tags) == 1 and re.fullmatch(r"cp31[123]-cp31[123]-macosx_14_0_arm64", tags[0]), tags
        python_tag, abi_tag, _ = tags[0].split("-")
        assert python_tag == abi_tag
        assert wheel.name == f"mlx_smolvla-{expected_version}-{tags[0]}.whl", wheel.name
        native = [name for name in names if name.startswith("mlx_smolvla/_rmsnorm_native") and name.endswith(".so")]
        assert len(native) == 1 and f"cpython-{python_tag[2:]}-darwin.so" in native[0], native
        tree = ast.parse(archive.read("mlx_smolvla/__init__.py"))
        versions = [ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
        assert versions == [expected_version], versions
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            for node in ast.walk(ast.parse(archive.read(name), filename=name)):
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    roots = {(node.module or "").split(".")[0]}
                else:
                    continue
                assert not roots & {"training", "reference"}, (name, node.lineno, roots)
    print(f"PASS wheel layout, metadata, native extension and namespace: {wheel.name}")


def check_sdist(sdist: Path, expected_version: str) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        files = {member.name: member for member in archive.getmembers() if member.isfile()}
        roots = {name.split("/")[0] for name in files}
        assert len(roots) == 1, roots
        root = next(iter(roots))
        assert root == f"mlx_smolvla-{expected_version}", root
        def read(relative: str) -> bytes:
            stream = archive.extractfile(files[f"{root}/{relative}"])
            assert stream is not None
            return stream.read()
        project = tomllib.loads(read("pyproject.toml").decode())
        assert project["project"]["version"] == expected_version
        assert BytesParser().parsebytes(read("PKG-INFO"))["Version"] == expected_version
        backend = project["build-system"]["build-backend"].split(":")[0]
        assert not backend.startswith(("mlx_smolvla.", "reference.", "training.")), backend
        backend_relative = backend.replace(".", "/") + ".py"
        assert any(f"{root}/{str(PurePosixPath(path) / backend_relative)}" in files
                   for path in project["build-system"]["backend-path"]), backend_relative
        assert not any(name.endswith(".so") for name in files), "sdist contains a prebuilt extension"
    print(f"PASS source archive version and standalone backend: {sdist.name}")


def check_installed(python: Path, expected_version: str) -> None:
    # Keep cwd on sys.path: -I would conceal the original shadowing regression.
    code = r'''
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import mlx.core as mx
mx.set_default_device(mx.cpu)
import mlx_smolvla
import mlx_smolvla.cli
import mlx_smolvla.rmsnorm as rmsnorm
import mlx_smolvla.training.trained_parity
import mlx_smolvla._lab.training.ux
import mlx_smolvla._lab.reference.discovery
assert metadata.version("mlx-smolvla") == sys.argv[1] == mlx_smolvla.__version__
installed = Path(mlx_smolvla.__file__).resolve()
assert installed.is_relative_to(Path(sys.prefix).resolve()), installed
assert not {"training", "reference", "torch", "transformers", "lerobot"} & sys.modules.keys()
if metadata.version("mlx") == "0.32.2":
    assert rmsnorm._rmsnorm_native is not None, "native extension failed to load on its build MLX version"
print("PASS installed package and training imports under decoy cwd:", installed)
if mx.metal.is_available():
    result = subprocess.run([str(Path(sys.executable).parent / "mlx-smolvla"), "doctor"],
                            check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["package_version"] == sys.argv[1], report
    assert report["metal_default"] is True, report
    print("PASS installed doctor on Metal-capable runner:", result.stdout.strip())
else:
    print("NOT RUN: doctor Metal check; runner reports no Metal device. No inference was tested.")
'''
    with tempfile.TemporaryDirectory(prefix="mlx-smolvla-wheel-smoke-") as directory:
        cwd = Path(directory)
        for name in ("training", "reference"):
            (cwd / name).mkdir()
            (cwd / name / "__init__.py").write_text(f"raise AssertionError('decoy {name} imported')\n")
        env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONNOUSERSITE="1",
                   MLX_SMOLVLA_CACHE=str(cwd / "empty-cache"), HF_HOME=str(cwd / "empty-hf"))
        subprocess.run([str(python.absolute()), "-c", code, expected_version], cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, default=os.environ.get("MLX_SMOLVLA_WHEEL"))
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.wheel is None:
        parser.error("--wheel or MLX_SMOLVLA_WHEEL is required")
    check_wheel(args.wheel, args.expected_version)
    if args.sdist:
        check_sdist(args.sdist, args.expected_version)
    if args.python:
        check_installed(args.python, args.expected_version)


if __name__ == "__main__":
    main()
