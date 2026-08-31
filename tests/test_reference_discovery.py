from pathlib import Path
import subprocess
import sys


def test_discovery_finds_installed_smolvla_sources(tmp_path: Path) -> None:
    from reference.discovery import discover_reference

    result = discover_reference(cache_dir=tmp_path)

    assert result.lerobot_version == "0.6.1"
    assert result.policy_source.is_file()
    assert result.config_source.is_file()
    assert result.expert_source.is_file()
    assert "SmolVLMWithExpertModel" in result.expert_source.read_text(encoding="utf-8")
    assert result.checkpoint_id == "lerobot/smolvla_base"


def test_discovery_pins_checkpoint_and_real_dataset(tmp_path: Path) -> None:
    from reference.discovery import discover_reference

    result = discover_reference(cache_dir=tmp_path)

    assert result.checkpoint_revision == "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
    assert result.checkpoint_config.is_file()
    assert result.tensor_count == 500
    assert result.parameter_count == 450_046_176
    assert result.dataset_id == "lerobot/svla_so101_pickplace"
    assert result.dataset_revision == "f641879e22172be7e8161d5e6c1503c2d2feb657"
    assert result.camera_keys == (
        "observation.images.side",
        "observation.images.up",
    )
    assert result.state_shape == (6,)
    assert result.action_shape == (6,)
    assert result.has_language_tasks


def test_discovery_cli_writes_architecture_evidence(tmp_path: Path) -> None:
    output = tmp_path / "ARCHITECTURE.md"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "reference.discovery",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--write",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = output.read_text(encoding="utf-8")
    assert "LeRobot 0.6.1" in report
    assert "PyTorch 2.11.0" in report
    assert "500 tensors" in report
    assert "450,046,176 parameters" in report
    assert "lerobot/svla_so101_pickplace" in report
