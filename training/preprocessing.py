"""Training-only stats-aware composition around the native MLX preprocessor."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_smolvla.preprocessing import SmolVLAPreprocessor


_PREPROCESSOR_STATE = "policy_preprocessor_step_5_normalizer_processor.safetensors"
_POSTPROCESSOR_STATE = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"


def _validated_vector(value: object, name: str) -> mx.array:
    array = mx.array(value).astype(mx.float32)
    mx.eval(array)
    if array.ndim != 1 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty vector, got {array.shape}")
    if not bool(mx.all(mx.isfinite(array))):
        raise ValueError(f"{name} contains non-finite values")
    return array


class StatsAwareSmolVLAPreprocessor:
    """Apply standard LeRobot state/action stats around native preprocessing."""

    def __init__(
        self,
        *,
        base: SmolVLAPreprocessor | object,
        state_mean: object,
        state_std: object,
        action_mean: object,
        action_std: object,
        eps: float = 1e-8,
    ) -> None:
        if eps < 0 or not np.isfinite(eps):
            raise ValueError(f"normalization epsilon must be finite and nonnegative, got {eps}")
        self.base = base
        self.config = base.config
        self.state_mean = _validated_vector(state_mean, "state mean")
        self.state_std = _validated_vector(state_std, "state std")
        self.action_mean = _validated_vector(action_mean, "action mean")
        self.action_std = _validated_vector(action_std, "action std")
        self.eps = float(eps)
        if self.state_mean.shape != self.state_std.shape:
            raise ValueError("state mean/std shapes differ")
        if self.action_mean.shape != self.action_std.shape:
            raise ValueError("action mean/std shapes differ")
        if bool(mx.any(self.state_std < 0)) or bool(mx.any(self.action_std < 0)):
            raise ValueError("normalization standard deviations cannot be negative")

    @classmethod
    def from_pretrained_files(
        cls,
        base: SmolVLAPreprocessor | object,
        checkpoint_dir: str | Path,
    ) -> "StatsAwareSmolVLAPreprocessor":
        """Read exact standard processor state tensors from one checkpoint."""

        checkpoint_dir = Path(checkpoint_dir)
        pre_path = checkpoint_dir / _PREPROCESSOR_STATE
        post_path = checkpoint_dir / _POSTPROCESSOR_STATE
        if not pre_path.is_file() or not post_path.is_file():
            raise FileNotFoundError(
                f"stats-aware loading requires {pre_path.name} and {post_path.name}"
            )
        pre = mx.load(str(pre_path))
        post = mx.load(str(post_path))
        required_pre = {
            "observation.state.mean",
            "observation.state.std",
            "action.mean",
            "action.std",
        }
        required_post = {"action.mean", "action.std"}
        if not required_pre <= set(pre):
            raise ValueError(f"preprocessor stats are missing {sorted(required_pre - set(pre))}")
        if not required_post <= set(post):
            raise ValueError(f"postprocessor stats are missing {sorted(required_post - set(post))}")
        for key in required_post:
            if not bool(mx.array_equal(pre[key].astype(mx.float32), post[key].astype(mx.float32))):
                raise ValueError(f"pre/postprocessor disagree on {key}")
        return cls(
            base=base,
            state_mean=pre["observation.state.mean"],
            state_std=pre["observation.state.std"],
            action_mean=post["action.mean"],
            action_std=post["action.std"],
        )

    def __call__(self, observation):
        processed = self.base(observation)
        state = (
            processed.state.astype(mx.float32) - self.state_mean[None, :]
        ) / (self.state_std[None, :] + self.eps)
        return replace(processed, state=state)

    def normalize_actions(self, actions: mx.array) -> mx.array:
        """Normalize physical actions exactly like LeRobot's MEAN_STD step."""

        return (
            actions.astype(mx.float32) - self.action_mean
        ) / (self.action_std + self.eps)

    def unnormalize_actions(self, actions: mx.array) -> mx.array:
        """Convert normalized actions back to physical dataset units."""

        return actions.astype(mx.float32) * self.action_std + self.action_mean


def load_stats_aware_policy(
    model_id: str | Path,
    *,
    cache_dir: str | Path,
    dtype: object = mx.bfloat16,
    tokenizer_dir: str | Path | None = None,
):
    """Strictly load a native policy and activate its standard saved stats."""

    from mlx_smolvla.policy import SmolVLAMLX

    checkpoint_dir = Path(model_id).expanduser()
    if not checkpoint_dir.is_dir():
        raise ValueError("stats-aware T3 policy loading currently requires a local checkpoint")
    policy = SmolVLAMLX.from_pretrained(
        checkpoint_dir,
        cache_dir=cache_dir,
        dtype=dtype,
        tokenizer_dir=tokenizer_dir,
        execution_mode="strict",
    )
    if (
        policy.config.state_normalization == "mean_std"
        and policy.config.action_normalization == "mean_std"
        and policy.preprocessor.state_mean is not None
        and policy.preprocessor.state_std is not None
        and policy.preprocessor.action_mean is not None
        and policy.preprocessor.action_std is not None
    ):
        return policy
    policy.preprocessor = StatsAwareSmolVLAPreprocessor.from_pretrained_files(
        policy.preprocessor,
        checkpoint_dir,
    )
    return policy
