"""Private lab namespace; repository sources retain their provenance paths.

Setuptools maps training/ and reference/ into this namespace in distributions.
In a source checkout, resolve the same packages relative to this file, never
relative to the caller's working directory.
"""
from pathlib import Path

_source_root = Path(__file__).resolve().parents[2]
if (_source_root / "_build_backend.py").is_file() and (_source_root / "pyproject.toml").is_file():
    __path__.append(str(_source_root))
del _source_root
