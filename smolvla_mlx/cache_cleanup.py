"""Narrow, repository-scoped cleanup for explicitly disposable debug caches."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import stat
import sys


@dataclass(frozen=True)
class CacheCleanupReport:
    """Auditable selection and deletion result for one cleanup invocation."""

    cache_dir: str
    dry_run: bool
    candidates: tuple[str, ...]
    candidate_bytes: int
    removed: tuple[str, ...]
    removed_bytes: int


@dataclass(frozen=True)
class CacheInventoryEntry:
    """One top-level native-cache entry and its frozen cleanup disposition."""

    name: str
    kind: str
    bytes: int
    retention: str
    regenerable: bool
    reason: str


@dataclass(frozen=True)
class CacheInventoryReport:
    """Complete top-level inventory for the repository-native model cache."""

    cache_dir: str
    total_bytes: int
    entries: tuple[CacheInventoryEntry, ...]


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _validate_cache_path(repository_root: Path, cache_dir: Path) -> None:
    expected = repository_root / ".cache" / "smolvla_mlx"
    if cache_dir != expected:
        raise ValueError(f"clean-cache requires the exact repository cache {expected}, got {cache_dir}")
    if repository_root.is_symlink() or not repository_root.is_dir():
        raise ValueError(f"repository root is missing or a symlink: {repository_root}")
    for path in (repository_root / ".cache", cache_dir):
        if path.is_symlink():
            raise ValueError(f"clean-cache refuses a symlink path: {path}")
        if not path.is_dir():
            raise ValueError(f"clean-cache path is not a directory: {path}")


def _is_cleanup_name(name: str) -> bool:
    return name == "benchmark-debug" or name.startswith("debug-")


def _tree_size(path: Path) -> int:
    total = path.lstat().st_size
    for root, directories, files in os.walk(path, topdown=True, followlinks=False):
        root_path = Path(root)
        for name in directories:
            total += (root_path / name).lstat().st_size
        for name in files:
            total += (root_path / name).lstat().st_size
    return total


def inventory_cache(
    *,
    repository_root: str | Path,
    cache_dir: str | Path,
) -> CacheInventoryReport:
    """Describe every top-level native-cache entry without mutating it."""

    repository = _absolute(repository_root)
    cache = _absolute(cache_dir)
    _validate_cache_path(repository, cache)

    entries: list[CacheInventoryEntry] = []
    for entry in sorted(os.scandir(cache), key=lambda item: item.name):
        path = cache / entry.name
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            kind = "symlink"
            size = path.lstat().st_size
        elif stat.S_ISDIR(mode):
            kind = "directory"
            size = _tree_size(path)
        elif stat.S_ISREG(mode):
            kind = "file"
            size = path.lstat().st_size
        else:
            kind = "other"
            size = path.lstat().st_size

        disposable = _is_cleanup_name(entry.name)
        entries.append(
            CacheInventoryEntry(
                name=entry.name,
                kind=kind,
                bytes=size,
                retention="delete" if disposable else "keep",
                regenerable=True,
                reason=(
                    "explicitly disposable debug experiment"
                    if disposable
                    else "retained by the frozen boundary; costly or unnecessary to regenerate"
                ),
            )
        )

    frozen_entries = tuple(entries)
    return CacheInventoryReport(
        cache_dir=str(cache),
        total_bytes=sum(entry.bytes for entry in frozen_entries),
        entries=frozen_entries,
    )


def clean_cache(
    *,
    repository_root: str | Path,
    cache_dir: str | Path,
    dry_run: bool = False,
) -> CacheCleanupReport:
    """Delete only top-level `debug-*` and exact `benchmark-debug` directories."""

    repository = _absolute(repository_root)
    cache = _absolute(cache_dir)
    _validate_cache_path(repository, cache)

    candidates: list[tuple[str, Path, int]] = []
    for entry in sorted(os.scandir(cache), key=lambda item: item.name):
        if not _is_cleanup_name(entry.name):
            continue
        path = cache / entry.name
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"clean-cache refuses an allowlisted symlink: {path}")
        if not stat.S_ISDIR(mode):
            raise ValueError(f"clean-cache candidate is not a real directory: {path}")
        candidates.append((entry.name, path, _tree_size(path)))

    names = tuple(name for name, _, _ in candidates)
    candidate_bytes = sum(size for _, _, size in candidates)
    if dry_run:
        return CacheCleanupReport(
            cache_dir=str(cache),
            dry_run=True,
            candidates=names,
            candidate_bytes=candidate_bytes,
            removed=(),
            removed_bytes=0,
        )

    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("clean-cache requires a symlink-attack-resistant shutil.rmtree")
    removed: list[str] = []
    removed_bytes = 0
    for name, path, size in candidates:
        current_mode = path.lstat().st_mode
        if stat.S_ISLNK(current_mode) or not stat.S_ISDIR(current_mode):
            raise RuntimeError(f"clean-cache candidate changed after validation: {path}")
        shutil.rmtree(path)
        removed.append(name)
        removed_bytes += size
    return CacheCleanupReport(
        cache_dir=str(cache),
        dry_run=False,
        candidates=names,
        candidate_bytes=candidate_bytes,
        removed=tuple(removed),
        removed_bytes=removed_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/smolvla_mlx"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--inventory", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.inventory:
            report = inventory_cache(
                repository_root=args.repository_root,
                cache_dir=args.cache_dir,
            )
        else:
            report = clean_cache(
                repository_root=args.repository_root,
                cache_dir=args.cache_dir,
                dry_run=args.dry_run,
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"smolvla-mlx clean-cache: {error}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
