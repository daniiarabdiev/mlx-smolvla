"""Distribution metadata contracts for the dependency-isolated runtime."""

from __future__ import annotations

from importlib.metadata import distribution

from packaging.requirements import Requirement


def test_installed_runtime_does_not_require_the_vendored_mlx_vlm_stack() -> None:
    """Catch an accidental dependency on the unused MLX-VLM server/audio stack."""

    requirements = [Requirement(requirement) for requirement in distribution("smolvla-mlx").requires or ()]
    names = {requirement.name.lower() for requirement in requirements if requirement.marker is None}

    assert "mlx-vlm" not in names
    assert "torch" not in names
    assert "lerobot" not in names
    assert "transformers" not in names


def test_installed_runtime_requires_the_audited_dependency_versions() -> None:
    """Keep source installs on the exact dependency set validated by parity tests."""

    requirements = [Requirement(requirement) for requirement in distribution("smolvla-mlx").requires or ()]
    runtime = {
        requirement.name.lower(): str(requirement.specifier)
        for requirement in requirements
        if requirement.marker is None
    }

    assert runtime == {
        "huggingface-hub": "==1.29.0",
        "mlx": "==0.32.2",
        "numpy": "==2.2.6",
        "pillow": "==12.3.0",
        "safetensors": "==0.8.0",
        "tokenizers": "==0.22.2",
    }
