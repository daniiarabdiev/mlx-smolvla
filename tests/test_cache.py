from pathlib import Path

import pytest


def test_explicit_cache_path_wins(tmp_path: Path) -> None:
    from smolvla_mlx.cache import resolve_cache_dir

    assert resolve_cache_dir(tmp_path) == tmp_path.resolve()


def _cache_tree(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    cache = repository / ".cache" / "smolvla_mlx"
    cache.mkdir(parents=True)
    for name in ("debug-attention", "debug-rms", "benchmark-debug"):
        directory = cache / name
        directory.mkdir()
        (directory / "payload.bin").write_bytes(b"debug payload")
    for name in ("converted", "policy-float32", "statistical-fp32"):
        directory = cache / name
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"expensive")
    (cache / "sum_probe").write_bytes(b"probe")
    return repository, cache


def test_cache_cleanup_dry_run_selects_only_frozen_allowlist(tmp_path: Path) -> None:
    from smolvla_mlx.cache_cleanup import clean_cache

    repository, cache = _cache_tree(tmp_path)
    report = clean_cache(repository_root=repository, cache_dir=cache, dry_run=True)

    assert report.dry_run is True
    assert report.removed == ()
    assert report.candidates == ("benchmark-debug", "debug-attention", "debug-rms")
    assert {path.name for path in cache.iterdir()} == {
        "benchmark-debug",
        "converted",
        "debug-attention",
        "debug-rms",
        "policy-float32",
        "statistical-fp32",
        "sum_probe",
    }


def test_cache_inventory_reports_size_retention_and_regenerability(tmp_path: Path) -> None:
    from smolvla_mlx.cache_cleanup import inventory_cache

    repository, cache = _cache_tree(tmp_path)
    report = inventory_cache(repository_root=repository, cache_dir=cache)

    assert report.cache_dir == str(cache)
    assert report.total_bytes == sum(item.bytes for item in report.entries)
    assert tuple(item.name for item in report.entries) == (
        "benchmark-debug",
        "converted",
        "debug-attention",
        "debug-rms",
        "policy-float32",
        "statistical-fp32",
        "sum_probe",
    )
    by_name = {item.name: item for item in report.entries}
    assert by_name["debug-attention"].retention == "delete"
    assert by_name["debug-attention"].regenerable is True
    assert by_name["converted"].retention == "keep"
    assert by_name["converted"].regenerable is True
    assert by_name["sum_probe"].kind == "file"


def test_cache_cleanup_removes_only_safe_debug_directories(tmp_path: Path) -> None:
    from smolvla_mlx.cache_cleanup import clean_cache

    repository, cache = _cache_tree(tmp_path)
    report = clean_cache(repository_root=repository, cache_dir=cache)

    assert report.dry_run is False
    assert report.removed == ("benchmark-debug", "debug-attention", "debug-rms")
    assert report.removed_bytes > 0
    assert {path.name for path in cache.iterdir()} == {
        "converted",
        "policy-float32",
        "statistical-fp32",
        "sum_probe",
    }


@pytest.mark.parametrize(
    "target",
    ("repository", "training", "outside", "traversal"),
)
def test_cache_cleanup_rejects_root_training_outside_and_traversal(
    tmp_path: Path,
    target: str,
) -> None:
    from smolvla_mlx.cache_cleanup import clean_cache

    repository, cache = _cache_tree(tmp_path)
    targets = {
        "repository": repository,
        "training": repository / ".cache" / "training",
        "outside": tmp_path / "outside",
        "traversal": cache / ".." / "training",
    }
    targets["training"].mkdir(exist_ok=True)
    targets["outside"].mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="exact repository cache"):
        clean_cache(repository_root=repository, cache_dir=targets[target])
    assert (cache / "debug-rms").is_dir()


def test_cache_cleanup_rejects_symlink_cache_and_allowed_candidate(tmp_path: Path) -> None:
    from smolvla_mlx.cache_cleanup import clean_cache

    repository, cache = _cache_tree(tmp_path)
    real_cache = tmp_path / "real-cache"
    real_cache.mkdir()
    cache.rename(cache.with_name("original"))
    cache.symlink_to(real_cache, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        clean_cache(repository_root=repository, cache_dir=cache)

    cache.unlink()
    cache.with_name("original").rename(cache)
    target = cache / "debug-link"
    target.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        clean_cache(repository_root=repository, cache_dir=cache)
    assert (cache / "debug-rms").is_dir()


def test_cache_cleanup_rejects_allowlisted_name_when_not_a_directory(tmp_path: Path) -> None:
    from smolvla_mlx.cache_cleanup import clean_cache

    repository, cache = _cache_tree(tmp_path)
    (cache / "debug-file").write_bytes(b"not a directory")

    with pytest.raises(ValueError, match="not a real directory"):
        clean_cache(repository_root=repository, cache_dir=cache)
    assert (cache / "debug-rms").is_dir()
