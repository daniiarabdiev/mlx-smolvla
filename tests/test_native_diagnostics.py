"""Optional native loader failures must leave the supported MLX runtime usable."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("error_type", ["ImportError", "OSError", "RuntimeError"])
def test_native_loader_failure_keeps_doctor_and_cpu_fallback_usable(
    error_type: str, tmp_path: Path
) -> None:
    # Fail the extension loader in a fresh interpreter, before the package's
    # eager policy imports can cache rmsnorm. All other imports stay real.
    script = textwrap.dedent(
        """
        import builtins
        import importlib.abc
        import json
        import sys

        class BrokenNativeLoader(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mlx_smolvla._rmsnorm_native":
                    raise getattr(builtins, sys.argv[1])("native loader probe failed")
                return None

        sys.meta_path.insert(0, BrokenNativeLoader())
        from mlx_smolvla.doctor import collect_doctor_report
        from mlx_smolvla.rmsnorm import ReferenceRMSNorm
        import mlx.core as mx

        with mx.stream(mx.cpu):
            result = ReferenceRMSNorm(720, 1e-5)(mx.ones((1, 720)))
            mx.eval(result)
            assert result.shape == (1, 720)
            assert mx.all(mx.isfinite(result)).item()

        print(json.dumps(collect_doctor_report(sys.argv[2]).as_dict()))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, error_type, str(tmp_path / "unused-cache")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["cpu_compatibility_backend"] == "pure-mlx-fallback"
    reason = report["native_extension_unavailable_reason"]
    assert error_type in reason
    assert "native loader probe failed" in reason
    assert not (tmp_path / "unused-cache").exists()


@pytest.mark.parametrize("mlx_version", ["0.32.0", "0.32.1", "0.32.2"])
def test_doctor_distinguishes_native_backend_from_abi_fallback(
    mlx_version: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mlx_smolvla import rmsnorm
    from mlx_smolvla.doctor import collect_doctor_report

    monkeypatch.setattr(rmsnorm, "_rmsnorm_native", object())
    monkeypatch.setattr(rmsnorm, "_runtime_mlx_version", lambda: mlx_version)
    report = collect_doctor_report(tmp_path).as_dict()
    if mlx_version == "0.32.2":
        assert report["cpu_compatibility_backend"] == "native-reference"
        assert report["native_extension_unavailable_reason"] is None
    else:
        assert report["cpu_compatibility_backend"] == "pure-mlx-fallback"
        reason = report["native_extension_unavailable_reason"]
        assert mlx_version in reason
        assert "0.32.2" in reason
