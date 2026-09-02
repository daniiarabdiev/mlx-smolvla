import os

from mlx import extension
from setuptools import find_packages, setup


MACOSX_DEPLOYMENT_TARGET = os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "14.0")
BUILD_NATIVE = os.environ.get("MLX_SMOLVLA_BUILD_NATIVE", "1").lower() not in {
    "0",
    "false",
    "no",
}

PACKAGES = find_packages(
    include=(
        "mlx_smolvla",
        "mlx_smolvla.*",
        "training",
        "training.*",
        "reference",
        "reference.*",
    )
)

EXT_MODULES = (
    [extension.CMakeExtension("mlx_smolvla._rmsnorm_native")]
    if BUILD_NATIVE
    else []
)
CMDCLASS = {"build_ext": extension.CMakeBuild} if BUILD_NATIVE else {}


if __name__ == "__main__":
    setup(
        ext_modules=EXT_MODULES,
        cmdclass=CMDCLASS,
        packages=PACKAGES,
        include_package_data=False,
        package_data={"mlx_smolvla": ["*.so"]} if BUILD_NATIVE else {},
        exclude_package_data=(
            {} if BUILD_NATIVE else {"mlx_smolvla": ["*.so", "*.dylib"]}
        ),
        zip_safe=False,
    )
