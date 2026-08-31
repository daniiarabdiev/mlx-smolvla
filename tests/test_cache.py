from pathlib import Path


def test_explicit_cache_path_wins(tmp_path: Path) -> None:
    from smolvla_mlx.cache import resolve_cache_dir

    assert resolve_cache_dir(tmp_path) == tmp_path.resolve()
