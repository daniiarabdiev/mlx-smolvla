"""The reference must evaluate stored fp32 weights, not bf16-rounded copies."""

from __future__ import annotations

import gc
from pathlib import Path
import shutil

import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from mlx_smolvla._lab.training.export import resolve_base_checkpoint
from mlx_smolvla._lab.training.reference_export import TorchExportPolicy


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def precision_probe_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real complete checkpoint with fp32-only values, without training/inference."""

    source = resolve_base_checkpoint(Path(".cache/hf"))
    output = tmp_path_factory.mktemp("reference-precision-checkpoint")
    for path in source.iterdir():
        if path.name.startswith("policy_") or path.name == "config.json":
            shutil.copyfile(path, output / path.name)
    values = {
        name: value.float()
        for name, value in load_file(source / "model.safetensors").items()
    }
    name = "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight"
    # Each literal is representable in fp32 but loses information through bf16.
    values[name].view(-1)[:4] = torch.tensor(
        [1.0001, -0.1234567, 0.000000001234567, 0.33333334]
    )
    save_file(values, output / "model.safetensors")
    return output


@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", torch.float32), ("cpu", torch.float64), ("mps", torch.float32)],
)
def test_reference_export_preserves_every_stored_parameter(
    precision_probe_export: Path, device: str, dtype: torch.dtype
) -> None:
    """Converting to fp32 only after loading into bf16 must fail this check."""

    loaded = TorchExportPolicy.load(
        precision_probe_export, cache_dir=Path(".cache/hf"), device=device, dtype=dtype
    )
    try:
        parameters = dict(loaded.policy.named_parameters())
        with safe_open(precision_probe_export / "model.safetensors", framework="pt") as source:
            assert set(parameters) == set(source.keys())
            for name, parameter in parameters.items():
                assert parameter.dtype == dtype, name
                assert parameter.device.type == device, name
                torch.testing.assert_close(
                    parameter.detach().cpu(),
                    source.get_tensor(name).to(dtype),
                    rtol=0,
                    atol=0,
                    msg=lambda message, name=name: f"Stored weight changed: {name}\n{message}",
                )
    finally:
        del loaded
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


def test_reference_export_still_rejects_incomplete_state(
    precision_probe_export: Path, tmp_path: Path
) -> None:
    """The precision repair must not weaken strict name/shape loading."""

    for path in precision_probe_export.iterdir():
        if path.name != "model.safetensors":
            shutil.copyfile(path, tmp_path / path.name)
    save_file({"unexpected.weight": torch.ones(1)}, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        TorchExportPolicy.load(tmp_path, cache_dir=Path(".cache/hf"))
