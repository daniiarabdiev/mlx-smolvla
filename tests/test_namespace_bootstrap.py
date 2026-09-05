"""Private namespace imports retain the pre-numerical-runtime bootstrap boundary."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


def test_floor_bootstrap_precedes_numpy_and_public_exports_remain_available() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import os
                import sys

                os.environ['OPENBLAS_NUM_THREADS'] = '99'
                from mlx_smolvla._lab.training.floor_runtime import bootstrap_hidden_worker
                assert 'numpy' not in sys.modules
                assert 'mlx.core' not in sys.modules
                bootstrap_hidden_worker(['--worker', 'cpu_fp32_baseline'])
                assert 'OPENBLAS_NUM_THREADS' not in os.environ

                from mlx_smolvla import (
                    ExecutionMode, QuantizationPreset, SmolVLAMLX, resolve_cache_dir,
                )
                from mlx_smolvla import policy, cache
                assert SmolVLAMLX is policy.SmolVLAMLX
                assert ExecutionMode is policy.ExecutionMode
                assert QuantizationPreset is policy.QuantizationPreset
                assert resolve_cache_dir is cache.resolve_cache_dir
                assert 'numpy' in sys.modules

                import mlx_smolvla
                try:
                    mlx_smolvla.missing_public_export
                except AttributeError:
                    pass
                else:
                    raise AssertionError('unknown export should raise AttributeError')
                """
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_t3b_provenance_accepts_canonical_bootstrap_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_smolvla._lab.training import finetune, runtime_provenance

    evidence = {
        "format_version": 1,
        "native_dependency_scope": (
            "direct-extension-origin-bound; transitive-dyld-images-inventory-hashed-only"
        ),
        "frozen": True,
        "modules": {
            name: [{
                "origin": "/source/module.py",
                "kind": "source",
                "file_sha256": "a" * 64,
                "code_sha256": "b" * 64,
            }]
            for name in (
                "__main__",
                "mlx_smolvla._lab.training",
                "mlx_smolvla._lab.training.runtime_provenance",
                "mlx_smolvla._lab.training.finetune",
            )
        },
    }
    monkeypatch.setattr(finetune.sys, "flags", SimpleNamespace(isolated=1, no_site=1))
    monkeypatch.setattr(runtime_provenance, "runtime_provenance_evidence", lambda: evidence)

    assert finetune._require_t3b_runtime_provenance() is evidence

    del evidence["modules"]["mlx_smolvla._lab.training"]
    with pytest.raises(RuntimeError, match="lacks required modules"):
        finetune._require_t3b_runtime_provenance()
