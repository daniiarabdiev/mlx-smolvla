"""Environment diagnostics for support requests and release preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata, util
import os
from pathlib import Path
import platform
import subprocess

import mlx.core as mx

from mlx_smolvla.cache import resolve_cache_dir
from mlx_smolvla.compatibility import CompatibilityVerdict, assess_compatibility
from mlx_smolvla.rmsnorm import (
    cpu_compatibility_backend,
    native_extension_unavailable_reason,
)


@dataclass(frozen=True)
class DoctorReport:
    package_version: str
    python_version: str
    macos_version: str
    chip: str
    mlx_version: str
    mlx_wheel_tag: str
    metal_default: bool
    cache_path: str
    cache_size_bytes: int
    serve_extra_installed: bool
    train_extra_installed: bool
    compatibility: CompatibilityVerdict
    cpu_compatibility_backend: str
    native_extension_unavailable_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _mlx_wheel_tag() -> str:
    try:
        wheel = metadata.distribution("mlx").read_text("WHEEL") or ""
    except metadata.PackageNotFoundError:
        return "not installed"
    tags = [line.removeprefix("Tag:").strip() for line in wheel.splitlines() if line.startswith("Tag:")]
    return ",".join(tags) if tags else "source/unknown"


def _chip_name() -> str:
    if platform.system() == "Darwin":
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    return platform.machine() or "unknown"


def _cache_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        directories[:] = [
            name for name in directories if not (Path(root) / name).is_symlink()
        ]
        for name in files:
            candidate = Path(root) / name
            try:
                total += candidate.lstat().st_size
            except FileNotFoundError:
                continue
    return total


def _all_modules_available(*names: str) -> bool:
    return all(util.find_spec(name) is not None for name in names)


def collect_doctor_report(cache_dir: Path | str | None = None) -> DoctorReport:
    """Inspect the real environment without creating or modifying the cache."""

    resolved_cache = resolve_cache_dir(cache_dir)
    macos_version = platform.mac_ver()[0]
    machine = platform.machine()
    mlx_version = _distribution_version("mlx")
    compatibility = assess_compatibility(
        macos_version=macos_version,
        machine=machine,
        mlx_version=mlx_version,
    )
    return DoctorReport(
        package_version=_distribution_version("mlx-smolvla"),
        python_version=platform.python_version(),
        macos_version=macos_version or "unknown",
        chip=_chip_name(),
        mlx_version=mlx_version,
        mlx_wheel_tag=_mlx_wheel_tag(),
        metal_default=mx.default_device() == mx.gpu,
        cache_path=str(resolved_cache),
        cache_size_bytes=_cache_size(resolved_cache),
        serve_extra_installed=_all_modules_available("grpc", "lerobot", "torch"),
        train_extra_installed=_all_modules_available("av", "lerobot", "torch"),
        compatibility=compatibility,
        cpu_compatibility_backend=cpu_compatibility_backend(),
        native_extension_unavailable_reason=native_extension_unavailable_reason(),
    )
