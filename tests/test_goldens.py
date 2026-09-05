from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_golden_writer_hashes_exact_bytes_and_store_loads_them(tmp_path: Path) -> None:
    from mlx_smolvla._lab.reference.goldens import GoldenStore, GoldenWriter

    writer = GoldenWriter(tmp_path)
    writer.add("sample_000/noise", np.arange(6, dtype=np.float32).reshape(2, 3))
    manifest = writer.finalize()

    assert manifest["sample_000/noise"]["shape"] == [2, 3]
    assert manifest["sample_000/noise"]["dtype"] == "float32"
    assert len(manifest["sample_000/noise"]["sha256"]) == 64
    np.testing.assert_array_equal(
        GoldenStore(tmp_path).load("sample_000/noise"),
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )


def test_manifest_is_stable_when_the_same_arrays_are_written_twice(tmp_path: Path) -> None:
    from mlx_smolvla._lab.reference.goldens import GoldenWriter

    value = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    first = GoldenWriter(tmp_path)
    first.add("sample_000/value", value)
    first.finalize()
    first_bytes = (tmp_path / "manifest.json").read_bytes()

    second = GoldenWriter(tmp_path)
    second.add("sample_000/value", value)
    second.finalize()

    assert (tmp_path / "manifest.json").read_bytes() == first_bytes


def test_sample_plan_spans_eight_real_episodes() -> None:
    from mlx_smolvla._lab.reference.goldens import GOLDEN_SAMPLE_SPECS

    assert len(GOLDEN_SAMPLE_SPECS) >= 8
    assert len({spec.episode for spec in GOLDEN_SAMPLE_SPECS}) >= 8
    assert all(spec.frame_index >= 0 for spec in GOLDEN_SAMPLE_SPECS)


@pytest.mark.slow
def test_real_reference_capture_contains_all_audited_boundaries(tmp_path: Path) -> None:
    from mlx_smolvla._lab.reference.goldens import GoldenWriter, capture_sample
    from mlx_smolvla._lab.reference.policy import ReferencePolicy, load_dataset_observation

    reference = ReferencePolicy.load(cache_dir=Path(".cache/hf"))
    sample = load_dataset_observation(cache_dir=Path(".cache/hf"), index=0, episode=0)
    writer = GoldenWriter(tmp_path)
    metadata = capture_sample(
        writer,
        reference,
        sample,
        sample_name="sample_000",
        episode=0,
        frame_index=0,
        seed=20260831,
    )
    manifest = writer.finalize()
    writer.write_metadata({"samples": [metadata]})

    assert metadata["episode"] == 0
    assert manifest["sample_000/preprocessed/pixel_values"]["shape"] == [2, 3, 512, 512]
    assert manifest["sample_000/vlm/cache/layer_00/key"]["shape"] == [1, 5, 177, 64]
    assert manifest["sample_000/vlm/layer_15/output"]["shape"] == [1, 177, 960]
    assert manifest["sample_000/expert/step_09/layer_15/output"]["shape"] == [1, 50, 720]
    assert manifest["sample_000/flow/step_09/velocity"]["shape"] == [1, 50, 32]
    assert manifest["sample_000/actions/normalized"]["shape"] == [1, 50, 6]
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["samples"][0]["task"]


@pytest.mark.slow
def test_make_goldens_script_captures_a_selected_real_sample(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/make_goldens.py",
            "--cache-dir",
            ".cache/hf",
            "--output",
            str(tmp_path),
            "--sample-index",
            "0",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "sample_000/actions/unnormalized" in manifest
