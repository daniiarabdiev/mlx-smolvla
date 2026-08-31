from mlx import extension
from setuptools import setup


if __name__ == "__main__":
    setup(
        ext_modules=[extension.CMakeExtension("smolvla_mlx._rmsnorm_native")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["smolvla_mlx"],
        include_package_data=False,
        package_data={"smolvla_mlx": ["*.so"]},
        zip_safe=False,
    )
