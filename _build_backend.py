"""PEP 517 backend for the optional MLX CMake extension.

This standalone backend is included only in the source distribution; it does
not import the runtime package or ship in the wheel.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mlx import extension
from setuptools import find_packages, setup
from setuptools.build_meta import _BuildMetaBackend


def setup_kwargs() -> dict[str, Any]:
    """Return setup arguments for the current, explicitly selected build mode."""

    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "14.0")
    build_native = os.environ.get("MLX_SMOLVLA_BUILD_NATIVE", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    packages = find_packages(include=("mlx_smolvla", "mlx_smolvla.*"))
    lab_packages = find_packages(include=("training", "training.*", "reference", "reference.*"))
    packages += [f"mlx_smolvla._lab.{name}" for name in lab_packages]
    ext_modules = (
        [
            extension.CMakeExtension(
                "mlx_smolvla._rmsnorm_native",
                sourcedir=str(Path("mlx_smolvla/native")),
            )
        ]
        if build_native
        else []
    )
    return {
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": extension.CMakeBuild} if build_native else {},
        "packages": packages,
        "package_dir": {
            "mlx_smolvla._lab.training": "training",
            "mlx_smolvla._lab.reference": "reference",
        },
        "include_package_data": False,
        "package_data": {
            "mlx_smolvla": [
                "*.so",
                "native/CMakeLists.txt",
                "native/*.cpp",
                "native/*.h",
            ]
        }
        if build_native
        else {"mlx_smolvla": ["native/CMakeLists.txt", "native/*.cpp", "native/*.h"]},
        "exclude_package_data": (
            {} if build_native else {"mlx_smolvla": ["*.so", "*.dylib"]}
        ),
        "zip_safe": False,
    }


class _ProjectBackend(_BuildMetaBackend):
    def run_setup(self, setup_script: str = "setup.py") -> None:
        del setup_script
        setup(**setup_kwargs())


_backend = _ProjectBackend()

build_editable = _backend.build_editable
build_sdist = _backend.build_sdist
build_wheel = _backend.build_wheel
get_requires_for_build_editable = _backend.get_requires_for_build_editable
get_requires_for_build_sdist = _backend.get_requires_for_build_sdist
get_requires_for_build_wheel = _backend.get_requires_for_build_wheel
prepare_metadata_for_build_editable = _backend.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _backend.prepare_metadata_for_build_wheel
