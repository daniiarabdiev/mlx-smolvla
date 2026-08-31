# Architecture

## Reference pins

- LeRobot 0.6.1
- PyTorch 2.11.0 (CPU fp32 golden reference)
- Transformers 5.5.4
- Policy source: `/Users/dan/Desktop/workshop/robotics-mlx-contrib/.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`
- Configuration source: `/Users/dan/Desktop/workshop/robotics-mlx-contrib/.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/configuration_smolvla.py`

## Checkpoint inventory

- Repository: `lerobot/smolvla_base`
- Revision: `c83c3163b8ca9b7e67c509fffd9121e66cb96205`
- Inventory: 500 tensors; 450,046,176 parameters
- Configuration cache path: `/Users/dan/Desktop/workshop/robotics-mlx-contrib/.cache/hf/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/config.json`

## Golden dataset

- Repository: `lerobot/svla_so101_pickplace`
- Revision: `f641879e22172be7e8161d5e6c1503c2d2feb657`
- Cameras (2): `observation.images.side`, `observation.images.up`
- State shape: `[6]`
- Action shape: `[6]`
- Language task table present: `True`

The remaining architecture hypotheses from `BRIEF.md` Section 3 are resolved
by the source and runtime audit in Phase 1.
