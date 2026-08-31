from pathlib import Path


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
    assert config.state_normalization == "identity"
    assert config.action_normalization == "identity"
