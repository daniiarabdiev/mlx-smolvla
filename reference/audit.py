"""Runtime architecture audit for the pinned PyTorch SmolVLA reference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import make_att_2d_masks

from reference.policy import ReferencePolicy, ReferenceSample


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _shape(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def _normalizer_stat_keys(pipeline: object) -> list[str]:
    """Return exactly the persisted tensor-statistic feature names."""

    for step in getattr(pipeline, "steps"):
        stats = getattr(step, "_tensor_stats", None)
        if stats is not None:
            return sorted(stats)
    raise RuntimeError("The saved policy pipeline did not contain a normalization step")


def _effective_normalization(
    reference: ReferencePolicy,
    sample: ReferenceSample,
    batch: Mapping[str, torch.Tensor],
) -> tuple[str, str]:
    """Characterize the checkpoint pipeline instead of trusting its declared modes."""

    raw_state = sample.observation["observation.state"]
    if not isinstance(raw_state, torch.Tensor):  # pragma: no cover - type guard for mapping input
        raise TypeError("The reference sample state must be a tensor")
    state_is_identity = torch.equal(batch["observation.state"], raw_state.unsqueeze(0))

    action_probe = torch.linspace(
        -1.0,
        1.0,
        steps=reference.config.chunk_size * reference.config.action_feature.shape[0],
        dtype=torch.float32,
    ).reshape(1, reference.config.chunk_size, reference.config.action_feature.shape[0])
    action_is_identity = torch.equal(reference.postprocessor(action_probe), action_probe)
    return (
        "identity" if state_is_identity else "mean_std",
        "identity" if action_is_identity else "mean_std",
    )


def _flow_schedule(num_steps: int) -> tuple[float, list[float], float]:
    """Run the reference integrator with a unit velocity to expose its schedule."""

    observed_times: list[float] = []

    def unit_velocity(x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        observed_times.append(float(time[0]))
        return torch.ones_like(x_t)

    initial = torch.zeros((1, 1, 1), dtype=torch.float32)
    result = euler_integrate(unit_velocity, initial, num_steps)
    return -1.0 / num_steps, observed_times, float(result.item())


def inspect_reference(reference: ReferencePolicy, sample: ReferenceSample) -> dict[str, Any]:
    """Collect execution-derived architecture facts for one real observation.

    The returned tree is deliberately JSON-safe so the audit script can persist it
    unchanged alongside its human-readable report.
    """

    policy = reference.policy
    flow_model = policy.model
    vlm_with_expert = flow_model.vlm_with_expert
    vlm = vlm_with_expert.get_vlm_model()
    batch = reference.prepare(sample.observation)

    with torch.inference_mode():
        images, image_masks = policy.prepare_images(batch)
        prepared_images = torch.cat(images, dim=0)
        vision_output = vlm.vision_model(
            pixel_values=prepared_images.to(dtype=vlm.vision_model.dtype),
            patch_attention_mask=None,
        ).last_hidden_state
        connector_output = vlm.connector(vision_output)

        padded_state = policy.prepare_state(batch)
        state_embedding = flow_model.state_proj(padded_state)[:, None, :]
        prefix, prefix_pad_mask, prefix_att_flags = flow_model.embed_prefix(
            images,
            image_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            state=padded_state,
        )
        prefix_mask = make_att_2d_masks(prefix_pad_mask, prefix_att_flags)
        prefix_positions = torch.cumsum(prefix_pad_mask, dim=1) - 1
        _, cache = vlm_with_expert.forward(
            attention_mask=prefix_mask,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )

        zero_noise = torch.zeros(
            (1, reference.config.chunk_size, reference.config.max_action_dim),
            dtype=reference.dtype,
        )
        suffix, suffix_pad_mask, suffix_att_flags = flow_model.embed_suffix(
            zero_noise, torch.ones((1,), dtype=torch.float32)
        )
        suffix_mask = make_att_2d_masks(suffix_pad_mask, suffix_att_flags)

    valid_prefix = prefix_pad_mask[0].bool()
    content_prefix = valid_prefix & ~prefix_att_flags[0].bool()
    state_prefix = valid_prefix & prefix_att_flags[0].bool()
    state_positions = torch.nonzero(state_prefix, as_tuple=False).flatten()
    if len(state_positions) != 1:
        raise RuntimeError(f"Expected one state token in the prefix, found {len(state_positions)}")
    state_index = int(state_positions.item())
    suffix_causal = torch.tril(torch.ones_like(suffix_mask[0], dtype=torch.bool))
    dt, times, unit_velocity_result = _flow_schedule(reference.config.num_steps)
    state_normalization, action_unnormalization = _effective_normalization(reference, sample, batch)

    text_config = vlm_with_expert.config.text_config
    vision_config = vlm_with_expert.config.vision_config
    self_attn_every = reference.config.self_attn_every_n_layers
    num_layers = vlm_with_expert.num_vlm_layers

    return {
        "parameters": {
            "total": _parameter_count(policy),
            "vlm": _parameter_count(vlm_with_expert.vlm),
            "expert": _parameter_count(vlm_with_expert.lm_expert),
            "state_projection": _parameter_count(flow_model.state_proj),
            "action_input_projection": _parameter_count(flow_model.action_in_proj),
            "action_output_projection": _parameter_count(flow_model.action_out_proj),
            "time_mlp": _parameter_count(flow_model.action_time_mlp_in)
            + _parameter_count(flow_model.action_time_mlp_out),
        },
        "model": {
            "base_vlm": reference.config.vlm_model_name,
            "vlm_layers": num_layers,
            "expert_layers": vlm_with_expert.num_expert_layers,
            "text_hidden_size": text_config.hidden_size,
            "expert_hidden_size": vlm_with_expert.expert_hidden_size,
            "expert_intermediate_size": vlm_with_expert.lm_expert.config.intermediate_size,
            "attention_heads": text_config.num_attention_heads,
            "key_value_heads": text_config.num_key_value_heads,
            "head_dim": text_config.head_dim,
            "rope_base": 10_000,
            "self_attention_layers": [
                index for index in range(num_layers) if index % self_attn_every == 0
            ],
            "cross_attention_layers": [
                index for index in range(num_layers) if index % self_attn_every != 0
            ],
            "state_placement": "VLM prefix after image and language tokens",
        },
        "vision": {
            "input_shape": _shape(prepared_images),
            "encoder_output_shape": _shape(vision_output),
            "connector_output_shape": _shape(connector_output),
            "image_size": vision_config.image_size,
            "patch_size": vision_config.patch_size,
            "encoder_layers": vision_config.num_hidden_layers,
            "hidden_size": vision_config.hidden_size,
            "attention_heads": vision_config.num_attention_heads,
            "intermediate_size": vision_config.intermediate_size,
            "activation": vision_config.hidden_act,
            "pixel_shuffle_scale": vlm_with_expert.config.scale_factor,
        },
        "boundaries": {
            "language_tokens": _shape(batch["observation.language.tokens"]),
            "padded_state": _shape(padded_state),
            "state_embedding": _shape(state_embedding),
            "prefix": _shape(prefix),
            "prefix_valid_tokens": int(valid_prefix.sum()),
            "suffix": _shape(suffix),
            "first_cache_key": _shape(cache.layers[0].keys),
            "first_cache_value": _shape(cache.layers[0].values),
            "cache_layers": len(cache.layers),
        },
        "attention": {
            "prefix_attention_flags": prefix_att_flags[0].to(dtype=torch.int64).tolist(),
            "prefix_content_is_bidirectional": bool(prefix_mask[0][content_prefix][:, content_prefix].all()),
            "state_attends_prefix": bool(prefix_mask[0, state_index, valid_prefix].all()),
            "prefix_cannot_attend_state": bool((~prefix_mask[0, content_prefix, state_index]).all()),
            "suffix_is_causal": bool(torch.equal(suffix_mask[0], suffix_causal)),
            "suffix_attention_flags": suffix_att_flags[0].to(dtype=torch.int64).tolist(),
        },
        "preprocessing": {
            "resize": {
                "height": reference.config.resize_imgs_with_padding[1],
                "width": reference.config.resize_imgs_with_padding[0],
                "interpolation": "bilinear",
                "align_corners": False,
                "padding_edges": ["left", "top"],
                "pad_before_pixel_normalization": 0.0,
                "pixel_transform": "x * 2 - 1",
            },
            "tokenizer_max_length": reference.config.tokenizer_max_length,
            "tokenizer_padding_side": "right",
            "tokenizer_padding": reference.config.pad_language_to,
            "task_newline_appended": True,
            "configured_state_normalization": "MEAN_STD",
            "configured_action_normalization": "MEAN_STD",
            "state_normalization_effective": state_normalization,
            "action_unnormalization_effective": action_unnormalization,
            "saved_stat_keys": _normalizer_stat_keys(reference.preprocessor),
            "configured_cameras": len(reference.config.image_features),
            "empty_cameras": reference.config.empty_cameras,
        },
        "flow": {
            "steps": reference.config.num_steps,
            "dt": dt,
            "times": times,
            "update": "x_t = x_t + dt * v_t",
            "unit_velocity_terminal_value": unit_velocity_result,
            "noise_at": "t=1",
            "actions_at": "t=0",
        },
        "queue": {
            "n_action_steps": reference.config.n_action_steps,
            "refill_when_empty": True,
            "chunk_size": reference.config.chunk_size,
        },
    }
