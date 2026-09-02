"""Public macOS/MLX compatibility and diagnostics contracts."""

from __future__ import annotations

from pathlib import Path

from mlx_smolvla.compatibility import assess_compatibility
from mlx_smolvla.doctor import collect_doctor_report


def test_every_published_mlx_032_release_has_the_verified_runtime_contract() -> None:
    for version in ("0.32.0", "0.32.1", "0.32.2"):
        verdict = assess_compatibility(
            macos_version="14.0",
            machine="arm64",
            mlx_version=version,
        )

        assert verdict.supported is True
        assert verdict.status == "verified"
        assert verdict.minimum_macos == "14.0"
        assert verdict.mlx_specifier == ">=0.32.0,<0.32.3"
        assert "macOS 14" in verdict.message
        assert f"MLX {version}" in verdict.message


def test_compatibility_rejection_is_actionable_for_each_unsupported_axis() -> None:
    old_macos = assess_compatibility(
        macos_version="13.7.8",
        machine="arm64",
        mlx_version="0.32.2",
    )
    intel = assess_compatibility(
        macos_version="15.7",
        machine="x86_64",
        mlx_version="0.32.2",
    )
    unverified_mlx = assess_compatibility(
        macos_version="15.7",
        machine="arm64",
        mlx_version="0.33.0",
    )

    assert old_macos.supported is False
    assert old_macos.status == "unsupported"
    assert "upgrade to macOS 14 or newer" in old_macos.message
    assert intel.supported is False
    assert "Apple Silicon" in intel.message
    assert unverified_mlx.supported is False
    assert "install MLX >=0.32.0,<0.32.3" in unverified_mlx.message


def test_doctor_reports_real_environment_cache_and_release_compatibility(
    tmp_path: Path,
) -> None:
    (tmp_path / "first.bin").write_bytes(b"abc")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "second.bin").write_bytes(b"12345")

    report = collect_doctor_report(cache_dir=tmp_path).as_dict()

    assert set(report) == {
        "cache_path",
        "cache_size_bytes",
        "chip",
        "compatibility",
        "macos_version",
        "metal_default",
        "mlx_version",
        "mlx_wheel_tag",
        "package_version",
        "python_version",
        "serve_extra_installed",
        "train_extra_installed",
    }
    assert report["cache_path"] == str(tmp_path.resolve())
    assert report["cache_size_bytes"] == 8
    assert report["package_version"]
    assert report["python_version"]
    assert report["mlx_version"]
    assert report["macos_version"]
    assert report["chip"]
    assert isinstance(report["metal_default"], bool)
    assert isinstance(report["serve_extra_installed"], bool)
    assert isinstance(report["train_extra_installed"], bool)
    assert report["compatibility"]["status"] in {"verified", "unsupported"}
