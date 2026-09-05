"""Native MLX inference for SmolVLA."""

__all__ = ["ExecutionMode", "QuantizationPreset", "SmolVLAMLX", "resolve_cache_dir"]
__version__ = "0.1.2"


def __getattr__(name: str) -> object:
    """Load public APIs on demand so worker bootstrap imports stay lightweight."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(
        ".cache" if name == "resolve_cache_dir" else ".policy",
        __name__,
    )
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
