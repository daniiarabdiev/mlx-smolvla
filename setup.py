from mlx import extension
from setuptools import find_packages, setup


PACKAGES = find_packages(
    include=(
        "smolvla_mlx",
        "smolvla_mlx.*",
        "training",
        "training.*",
        "reference",
        "reference.*",
    )
)


if __name__ == "__main__":
    setup(
        ext_modules=[extension.CMakeExtension("smolvla_mlx._rmsnorm_native")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=PACKAGES,
        include_package_data=False,
        package_data={"smolvla_mlx": ["*.so"]},
        zip_safe=False,
    )
