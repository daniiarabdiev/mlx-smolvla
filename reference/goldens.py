"""Deterministic, reference-derived golden tensor capture for SmolVLA."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.policies.common.vla_utils import make_att_2d_masks

from mlx_smolvla._lab.reference.policy import ReferencePolicy, ReferenceSample


GOLDEN_SEED = 20_260_831


@dataclass(frozen=True)
class GoldenSampleSpec:
    """One deterministic frame selected from a distinct public dataset episode."""

    name: str
    episode: int
    frame_index: int
    seed: int


GOLDEN_SAMPLE_SPECS = tuple(
    GoldenSampleSpec(
        name=f"sample_{sample_index:03d}",
        episode=episode,
        frame_index=0,
        seed=GOLDEN_SEED + sample_index,
    )
    for sample_index, episode in enumerate((0, 7, 14, 21, 28, 35, 42, 49))
)


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return np.asarray(value.numpy())
    return np.asarray(value)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write a complete replacement file without exposing a partial artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class GoldenWriter:
    """Write contiguous `.npy` tensors and a sorted manifest with byte hashes."""

    def __init__(self, root: Path):
        self.root = root
        self.entries: dict[str, dict[str, Any]] = {}

    def add(self, name: str, value: torch.Tensor | np.ndarray) -> None:
        if not name or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"Golden tensor name must be a relative path: {name!r}")
        array = np.ascontiguousarray(_as_numpy(value))
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        payload = buffer.getvalue()
        relative_path = Path(f"{name}.npy")
        _atomic_write(self.root / relative_path, payload)
        self.entries[name] = {
            "path": relative_path.as_posix(),
            "shape": list(array.shape),
            "dtype": array.dtype.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def finalize(self) -> dict[str, dict[str, Any]]:
        manifest = {name: self.entries[name] for name in sorted(self.entries)}
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _atomic_write(self.root / "manifest.json", payload)
        return manifest

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        payload = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _atomic_write(self.root / "metadata.json", payload)


class GoldenStore:
    """Load and integrity-check generated golden arrays."""

    def __init__(self, root: Path):
        self.root = root
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Golden manifest not found at {manifest_path}; run `make goldens`")
        self.manifest: dict[str, dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))

    def load(self, name: str) -> np.ndarray:
        try:
            record = self.manifest[name]
        except KeyError as error:
            raise KeyError(f"Golden tensor {name!r} is absent from {self.root / 'manifest.json'}") from error
        path = self.root / record["path"]
        payload = path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != record["sha256"]:
            raise ValueError(f"Golden tensor hash mismatch for {name}: {actual_hash} != {record['sha256']}")
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        if list(array.shape) != record["shape"] or array.dtype.name != record["dtype"]:
            raise ValueError(f"Golden tensor metadata mismatch for {name}")
        return array


class _ForwardTraceCollector:
    """Capture true residual-block outputs using transparent module hooks."""

    def __init__(self, writer: GoldenWriter, sample_name: str, reference: ReferencePolicy):
        self.writer = writer
        self.sample_name = sample_name
        self.reference = reference
        self.capture_prefix = False
        self._residuals: dict[tuple[str, int], torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current_step: int | None = None
        self._next_step = 0

    def _path(self, suffix: str) -> str:
        return f"{self.sample_name}/{suffix}"

    def _captures(self, kind: str) -> bool:
        return (kind == "vlm" and self.capture_prefix) or (kind == "expert" and self._current_step is not None)

    def _block_pre_hook(self, kind: str, layer_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if self._captures(kind):
                self._residuals[(kind, layer_index)] = inputs[0].detach()

        return hook

    def _block_mlp_hook(self, kind: str, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            key = (kind, layer_index)
            if self._captures(kind) and key in self._residuals:
                final_output = self._residuals.pop(key) + output.detach()
                if kind == "vlm":
                    name = f"vlm/layer_{layer_index:02d}/output"
                else:
                    name = f"expert/step_{self._current_step:02d}/layer_{layer_index:02d}/output"
                self.writer.add(self._path(name), final_output)

        return hook

    def _prefix_norm_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if self.capture_prefix:
            self.writer.add(self._path("vlm/prefix/output"), output)

    def _expert_norm_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if self._current_step is not None:
            self.writer.add(self._path(f"expert/step_{self._current_step:02d}/output"), output)

    def _action_input_pre_hook(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self._current_step = self._next_step
        self.writer.add(self._path(f"flow/step_{self._current_step:02d}/x_t"), inputs[0])

    def _time_mlp_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if self._current_step is None:
            raise RuntimeError("Timestep MLP ran before an action input was captured")
        self.writer.add(self._path(f"flow/step_{self._current_step:02d}/suffix_embeddings"), output)

    def _action_output_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if self._current_step is None:
            raise RuntimeError("Action output projection ran before an action input was captured")
        self.writer.add(self._path(f"flow/step_{self._current_step:02d}/velocity"), output)
        self._next_step += 1

    def install(self) -> None:
        flow_model = self.reference.policy.model
        vlm_with_expert = flow_model.vlm_with_expert
        for kind, layers in (
            ("vlm", vlm_with_expert.get_vlm_model().text_model.layers),
            ("expert", vlm_with_expert.lm_expert.layers),
        ):
            for layer_index, layer in enumerate(layers):
                self._handles.append(
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        self._block_pre_hook(kind, layer_index)
                    )
                )
                self._handles.append(layer.mlp.register_forward_hook(self._block_mlp_hook(kind, layer_index)))
        self._handles.append(
            vlm_with_expert.get_vlm_model().text_model.norm.register_forward_hook(self._prefix_norm_hook)
        )
        self._handles.append(vlm_with_expert.lm_expert.norm.register_forward_hook(self._expert_norm_hook))
        self._handles.append(flow_model.action_in_proj.register_forward_pre_hook(self._action_input_pre_hook))
        self._handles.append(flow_model.action_time_mlp_out.register_forward_hook(self._time_mlp_hook))
        self._handles.append(flow_model.action_out_proj.register_forward_hook(self._action_output_hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _tensor_observation(sample: ReferenceSample, key: str) -> torch.Tensor:
    value = sample.observation[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor observation for {key!r}, got {type(value).__name__}")
    return value


def capture_sample(
    writer: GoldenWriter,
    reference: ReferencePolicy,
    sample: ReferenceSample,
    *,
    sample_name: str,
    episode: int,
    frame_index: int,
    seed: int,
) -> dict[str, Any]:
    """Capture every audited reference boundary for one fixed frame and noise tensor."""

    policy = reference.policy
    flow_model = policy.model
    vlm_with_expert = flow_model.vlm_with_expert
    batch = reference.prepare(sample.observation)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (1, reference.config.chunk_size, reference.config.max_action_dim),
        generator=generator,
        dtype=torch.float32,
    )
    tracer = _ForwardTraceCollector(writer, sample_name, reference)

    camera_keys = tuple(key for key in reference.config.image_features if key in sample.observation)
    if not camera_keys:
        raise ValueError("Golden observation contains none of the checkpoint camera keys")
    for camera_index, camera_key in enumerate(camera_keys, start=1):
        writer.add(
            f"{sample_name}/raw/camera{camera_index}",
            _tensor_observation(sample, camera_key),
        )
    writer.add(f"{sample_name}/raw/state", _tensor_observation(sample, "observation.state"))
    writer.add(f"{sample_name}/raw/action", sample.action)
    writer.add(f"{sample_name}/noise", noise)

    tracer.install()
    try:
        with torch.inference_mode():
            images, image_masks = policy.prepare_images(batch)
            padded_state = policy.prepare_state(batch)
            pixel_values = torch.cat(images, dim=0)
            pixel_mask = torch.stack(image_masks)
            vision_output = vlm_with_expert.get_vlm_model().vision_model(
                pixel_values=pixel_values.to(dtype=vlm_with_expert.get_vlm_model().vision_model.dtype),
                patch_attention_mask=None,
            ).last_hidden_state
            connector_output = vlm_with_expert.get_vlm_model().connector(vision_output)
            language_embeddings = vlm_with_expert.embed_language_tokens(batch["observation.language.tokens"])
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

            writer.add(f"{sample_name}/preprocessed/pixel_values", pixel_values)
            writer.add(f"{sample_name}/preprocessed/pixel_mask", pixel_mask)
            writer.add(f"{sample_name}/preprocessed/input_ids", batch["observation.language.tokens"])
            writer.add(f"{sample_name}/preprocessed/text_attention_mask", batch["observation.language.attention_mask"])
            writer.add(f"{sample_name}/preprocessed/state_normalized", batch["observation.state"])
            writer.add(f"{sample_name}/vision/features", vision_output)
            writer.add(f"{sample_name}/connector/output", connector_output)
            writer.add(f"{sample_name}/language/embeddings", language_embeddings)
            writer.add(f"{sample_name}/state/embedding", state_embedding)
            writer.add(f"{sample_name}/prefix/embeddings", prefix)
            writer.add(f"{sample_name}/prefix/pad_mask", prefix_pad_mask)
            writer.add(f"{sample_name}/prefix/attention_flags", prefix_att_flags)
            writer.add(f"{sample_name}/prefix/attention_mask", prefix_mask)
            writer.add(f"{sample_name}/prefix/position_ids", prefix_positions)

            tracer.capture_prefix = True
            _, cache = vlm_with_expert.forward(
                attention_mask=prefix_mask,
                position_ids=prefix_positions,
                past_key_values=None,
                inputs_embeds=[prefix, None],
                use_cache=True,
            )
            tracer.capture_prefix = False
            for layer_index, layer_cache in enumerate(cache.layers):
                writer.add(f"{sample_name}/vlm/cache/layer_{layer_index:02d}/key", layer_cache.keys)
                writer.add(f"{sample_name}/vlm/cache/layer_{layer_index:02d}/value", layer_cache.values)

            policy.reset()
            normalized_actions = policy.predict_action_chunk(batch, noise=noise)
            actions = reference.postprocessor(normalized_actions)
            writer.add(f"{sample_name}/actions/normalized", normalized_actions)
            writer.add(f"{sample_name}/actions/unnormalized", actions)
    finally:
        tracer.close()

    task = sample.observation["task"]
    if not isinstance(task, str):
        raise TypeError(f"Expected string task, got {type(task).__name__}")
    return {
        "name": sample_name,
        "episode": episode,
        "frame_index": frame_index,
        "seed": seed,
        "task": task,
        "camera_keys": list(camera_keys),
    }
