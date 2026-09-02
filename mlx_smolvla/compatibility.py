"""Audited Apple Silicon, macOS, and MLX runtime compatibility policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Literal


MINIMUM_MACOS = "14.0"
MLX_SPECIFIER = ">=0.32.0,<0.32.3"
VERIFIED_MLX_VERSIONS = ("0.32.0", "0.32.1", "0.32.2")

CompatibilityStatus = Literal["verified", "unsupported"]


@dataclass(frozen=True)
class CompatibilityVerdict:
    """One actionable result from the public compatibility policy."""

    supported: bool
    status: CompatibilityStatus
    message: str
    minimum_macos: str = MINIMUM_MACOS
    mlx_specifier: str = MLX_SPECIFIER

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _numeric_version(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def assess_compatibility(
    *,
    macos_version: str,
    machine: str,
    mlx_version: str,
) -> CompatibilityVerdict:
    """Return the release verdict without importing optional dependencies."""

    if machine.lower() not in {"arm64", "aarch64"}:
        return CompatibilityVerdict(
            supported=False,
            status="unsupported",
            message="mlx-smolvla requires an Apple Silicon Mac (arm64).",
        )

    parsed_macos = _numeric_version(macos_version)
    if parsed_macos is None or parsed_macos < (14, 0, 0):
        return CompatibilityVerdict(
            supported=False,
            status="unsupported",
            message=(
                f"macOS {macos_version or 'unknown'} is unsupported; "
                "upgrade to macOS 14 or newer."
            ),
        )

    if mlx_version not in VERIFIED_MLX_VERSIONS:
        return CompatibilityVerdict(
            supported=False,
            status="unsupported",
            message=(
                f"MLX {mlx_version or 'unknown'} is outside the verified range; "
                f"install MLX {MLX_SPECIFIER}."
            ),
        )

    return CompatibilityVerdict(
        supported=True,
        status="verified",
        message=(
            f"verified: MLX {mlx_version} on Apple Silicon with macOS 14 or newer"
        ),
    )
