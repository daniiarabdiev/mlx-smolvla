from pathlib import Path
import json

import pytest


def test_config_matches_audited_checkpoint(checkpoint_dir: Path) -> None:
    from smolvla_mlx.config import SmolVLAConfig

    config = SmolVLAConfig.from_pretrained_files(checkpoint_dir)

    assert config.action_dim == 6
    assert config.state_dim == 6
    assert config.chunk_size == 50
    assert config.n_action_steps == 50
    assert config.max_action_dim == 32
    assert config.max_state_dim == 32
    assert config.num_steps == 10
    assert config.image_size == (512, 512)
    assert config.tokenizer_max_length == 48
    assert config.vlm_layers == 16
    assert config.expert_hidden_size == 720
    assert config.image_keys == (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    assert config.empty_cameras == 0
    assert config.image_shapes == (
        ("observation.images.camera1", (3, 256, 256)),
        ("observation.images.camera2", (3, 256, 256)),
        ("observation.images.camera3", (3, 256, 256)),
    )
    assert config.state_shape == (6,)
    assert config.action_shape == (6,)
    assert config.state_normalization == "identity"
    assert config.action_normalization == "identity"


def test_architecture_mismatch_reports_checkpoint_input_contract(
    tmp_path: Path,
    checkpoint_dir: Path,
) -> None:
    from smolvla_mlx.config import SmolVLAConfig

    raw = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    raw["max_action_dim"] = 64
    altered = tmp_path / "altered"
    altered.mkdir()
    (altered / "config.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        SmolVLAConfig.from_pretrained_files(altered)
    message = str(caught.value)
    assert "max_action_dim=64" in message
    assert "observation.images.camera1" in message
    assert "(3, 256, 256)" in message
    assert "observation.state" in message
    assert "(6,)" in message
