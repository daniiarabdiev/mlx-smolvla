"""Import-safe environment contract for self-consistency worker processes."""

from __future__ import annotations

import os
from typing import Mapping, Sequence


MPS_ENVIRONMENT_KEYS = (
    "PYTORCH_DEBUG_MPS_ALLOCATOR",
    "PYTORCH_MPS_LOG_PROFILE_INFO",
    "PYTORCH_MPS_TRACE_SIGNPOSTS",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_PREFER_METAL",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)
CPU_THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def worker_environment(
    *,
    mps_fallback: bool,
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment with every controlled runtime key sanitized."""

    environment = dict(os.environ if inherited is None else inherited)
    for key in (*MPS_ENVIRONMENT_KEYS, *CPU_THREAD_ENVIRONMENT_KEYS):
        environment.pop(key, None)
    if mps_fallback:
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return environment


def apply_worker_environment(*, mps_fallback: bool) -> None:
    """Sanitize the current process before importing NumPy or Torch."""

    fixed = worker_environment(mps_fallback=mps_fallback)
    os.environ.clear()
    os.environ.update(fixed)


def mps_environment_snapshot(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    source = os.environ if environment is None else environment
    return {key: source.get(key) for key in MPS_ENVIRONMENT_KEYS}


def cpu_thread_environment_snapshot(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    source = os.environ if environment is None else environment
    return {key: source.get(key) for key in CPU_THREAD_ENVIRONMENT_KEYS}


def bootstrap_hidden_worker(argv: Sequence[str]) -> None:
    """Fix hidden-worker state early; ordinary/list/assembly modes are untouched."""

    worker_name: str | None = None
    for index, argument in enumerate(argv):
        if argument == "--worker":
            if index + 1 < len(argv):
                worker_name = argv[index + 1]
            break
        if argument.startswith("--worker="):
            worker_name = argument.partition("=")[2]
            break
    if not worker_name:
        return
    apply_worker_environment(
        mps_fallback=worker_name.startswith("mps_fp32_fallback_")
    )
