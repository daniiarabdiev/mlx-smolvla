"""LeRobot 0.6.1-compatible asynchronous gRPC serving for native MLX.

This module intentionally lives outside the base import graph.  Import it only
through the ``serve`` extra: LeRobot's public wire contract uses gRPC, protobuf,
pickle, and CPU Torch tensors even though policy inference itself stays native
MLX.
"""

from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import pickle  # nosec B403: required by the pinned LeRobot 0.6.1 wire protocol
from queue import Empty, Queue
import threading
import time
from typing import Callable, Mapping

import grpc
import mlx.core as mx
import numpy as np
import torch

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    observations_similar,
    raw_observation_to_observation,
)
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import MAX_MESSAGE_SIZE

from mlx_smolvla.policy import ExecutionMode, QuantizationPreset, SmolVLAMLX


_LOG = logging.getLogger("mlx_smolvla.server")
_MAX_OBSERVATION_BYTES = 64 * 1024 * 1024
_CANCELLATION_POLL_SECONDS = 0.05

PolicyLoader = Callable[..., SmolVLAMLX]


@dataclass(frozen=True)
class _ObservationReceipt:
    wall_time_ns: int
    monotonic_ns: int
    received_at_utc: str


class _LatencyJSONLRecorder:
    """Exclusive, append-only telemetry sink for one supervised server session."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError as error:
            raise ValueError(
                f"latency log already exists; choose a new session path: {self.path}"
            ) from error
        except OSError as error:
            raise RuntimeError(f"could not create latency log {self.path}: {error}") from error
        self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self._sequence = 0

    def write(self, record: Mapping[str, object]) -> None:
        with self._lock:
            payload = {
                "format_version": 1,
                "event": "observation_to_action_chunk",
                "sequence": self._sequence,
                **record,
            }
            self._stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._sequence += 1

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.close()


def _is_loopback(host: str) -> bool:
    normalized = host.strip().strip("[]")
    if normalized.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _bind_target(host: str, port: int) -> str:
    normalized = host.strip()
    if ":" in normalized and not normalized.startswith("["):
        normalized = f"[{normalized}]"
    return f"{normalized}:{port}"


@dataclass(frozen=True)
class ServeConfig:
    """Validated lifecycle, execution, and network settings for the server."""

    host: str = "127.0.0.1"
    port: int = 8080
    cache_dir: Path | None = None
    tokenizer_dir: Path | None = None
    dtype: str = "bfloat16"
    execution_mode: ExecutionMode = "production"
    quantization: QuantizationPreset | None = None
    latency_log: Path | None = None
    fps: int = 30
    inference_latency: float = 0.0
    obs_queue_timeout: float = 1.0
    max_workers: int = 4
    seed: int | None = None
    allow_remote: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a non-empty string")
        if not 0 <= self.port <= 65535:
            raise ValueError(f"port must be between 0 and 65535, got {self.port}")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be 'float32' or 'bfloat16'")
        if self.execution_mode not in {"production", "strict"}:
            raise ValueError("execution_mode must be 'production' or 'strict'")
        if self.quantization not in {None, "vlm-8bit", "vlm-4bit"}:
            raise ValueError("quantization must be None, 'vlm-8bit', or 'vlm-4bit'")
        if self.quantization is not None and self.dtype != "bfloat16":
            raise ValueError("VLM quantization requires the validated bfloat16 base dtype")
        if self.quantization is not None and self.execution_mode != "production":
            raise ValueError("VLM quantization is validated only for production Metal execution")
        if self.latency_log is not None:
            if isinstance(self.latency_log, bool) or not isinstance(
                self.latency_log, (str, Path)
            ):
                raise ValueError("latency_log must be a non-empty path or None")
            if isinstance(self.latency_log, str) and not self.latency_log.strip():
                raise ValueError("latency_log must be a non-empty path or None")
            object.__setattr__(self, "latency_log", Path(self.latency_log))
        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or self.fps <= 0:
            raise ValueError(f"fps must be a positive integer, got {self.fps!r}")
        if not math.isfinite(self.inference_latency) or self.inference_latency < 0:
            raise ValueError(
                f"inference_latency must be finite and non-negative, got {self.inference_latency}"
            )
        if not math.isfinite(self.obs_queue_timeout) or self.obs_queue_timeout < 0:
            raise ValueError(
                f"obs_queue_timeout must be finite and non-negative, got {self.obs_queue_timeout}"
            )
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or self.max_workers <= 0
        ):
            raise ValueError(f"max_workers must be a positive integer, got {self.max_workers!r}")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0
        ):
            raise ValueError(f"seed must be a non-negative integer or None, got {self.seed!r}")
        if not _is_loopback(self.host) and not self.allow_remote:
            raise ValueError(
                "non-loopback binding requires allow_remote=True because LeRobot 0.6.1 "
                "uses unauthenticated gRPC and trusted-peer pickle payloads"
            )

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps


def _default_policy_loader(**kwargs) -> SmolVLAMLX:
    return SmolVLAMLX.from_pretrained(**kwargs)


class NativeMLXPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    """Pinned LeRobot RPC surface backed by one serialized native MLX policy."""

    def __init__(
        self,
        config: ServeConfig,
        *,
        policy_loader: PolicyLoader = _default_policy_loader,
    ) -> None:
        self.config = config
        self._policy_loader = policy_loader
        self._stopped = threading.Event()
        self._state_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._generation = 0
        self._prediction_index = 0
        self.observation_queue: Queue[TimedObservation] = Queue(maxsize=1)
        self._observation_receipts: dict[int, _ObservationReceipt] = {}
        self._predicted_timesteps: set[int] = set()
        self.last_processed_obs: TimedObservation | None = None
        self._latency_recorder = (
            _LatencyJSONLRecorder(config.latency_log)
            if config.latency_log is not None
            else None
        )

        self.policy: SmolVLAMLX | None = None
        self.policy_type: str | None = None
        self.pretrained_name_or_path: str | None = None
        self.lerobot_features: dict[str, dict[str, object]] | None = None
        self.actions_per_chunk: int | None = None
        self.rename_map: dict[str, str] = {}

    @property
    def running(self) -> bool:
        return not self._stopped.is_set()

    def _abort(self, context, code: grpc.StatusCode, details: str):
        context.abort(code, details)
        raise RuntimeError(details)  # pragma: no cover - gRPC abort never returns

    def _check_active(self, context) -> None:
        if not context.is_active():
            self._abort(context, grpc.StatusCode.CANCELLED, "request cancelled")
        if not self.running:
            self._abort(context, grpc.StatusCode.UNAVAILABLE, "server is stopping")

    def _clear_episode_state(self) -> None:
        with self._state_lock:
            self._generation += 1
            self.observation_queue = Queue(maxsize=1)
            self._observation_receipts = {}
            self._predicted_timesteps = set()
            self.last_processed_obs = None
            policy = self.policy
        if policy is not None:
            policy.reset()

    def Ready(self, request, context):  # noqa: N802
        self._check_active(context)
        self._clear_episode_state()
        _LOG.info("LeRobot client ready: %s", context.peer())
        return services_pb2.Empty()

    @staticmethod
    def _validate_features(features: object) -> dict[str, dict[str, object]]:
        if not isinstance(features, dict) or not features:
            raise ValueError("lerobot_features must be a non-empty dictionary")
        validated: dict[str, dict[str, object]] = {}
        for name, feature in features.items():
            if not isinstance(name, str) or not isinstance(feature, dict):
                raise ValueError("lerobot_features must map string names to dictionaries")
            dtype = feature.get("dtype")
            shape = feature.get("shape")
            if dtype not in {"float32", "image", "video"}:
                raise ValueError(f"unsupported LeRobot feature dtype for {name!r}: {dtype!r}")
            if not isinstance(shape, (tuple, list)) or not shape or not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in shape
            ):
                raise ValueError(f"invalid LeRobot feature shape for {name!r}: {shape!r}")
            if dtype == "float32" and len(shape) == 1:
                names = feature.get("names")
                if not isinstance(names, (tuple, list)) or len(names) != shape[0] or not all(
                    isinstance(value, str) and value for value in names
                ):
                    raise ValueError(f"state feature {name!r} requires one source name per value")
            validated[name] = dict(feature)
        state = validated.get("observation.state")
        if state is None or state.get("dtype") != "float32":
            raise ValueError("lerobot_features must define float32 observation.state")
        if not any(
            name.startswith("observation.images.")
            and feature.get("dtype") in {"image", "video"}
            for name, feature in validated.items()
        ):
            raise ValueError("lerobot_features must define at least one observation image")
        return validated

    @staticmethod
    def _validate_rename_map(value: object) -> dict[str, str]:
        if not isinstance(value, dict) or not all(
            isinstance(old, str)
            and old
            and isinstance(new, str)
            and new
            for old, new in value.items()
        ):
            raise ValueError("rename_map must map non-empty strings to non-empty strings")
        if len(set(value.values())) != len(value):
            raise ValueError("rename_map target names must be unique")
        return dict(value)

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        self._check_active(context)
        try:
            policy_specs = pickle.loads(request.data)  # nosec B301: exact reference protocol
        except Exception as error:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"invalid policy setup pickle: {error}")
        if not isinstance(policy_specs, RemotePolicyConfig):
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"policy setup must be RemotePolicyConfig, got {type(policy_specs).__name__}",
            )
        if policy_specs.policy_type != "smolvla":
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"policy_type must be 'smolvla', got {policy_specs.policy_type!r}",
            )
        if not isinstance(policy_specs.pretrained_name_or_path, str) or not (
            policy_specs.pretrained_name_or_path.strip()
        ):
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "pretrained_name_or_path must be a non-empty string",
            )
        if (
            isinstance(policy_specs.actions_per_chunk, bool)
            or not isinstance(policy_specs.actions_per_chunk, int)
            or policy_specs.actions_per_chunk <= 0
        ):
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "actions_per_chunk must be a positive integer",
            )
        try:
            features = self._validate_features(policy_specs.lerobot_features)
            rename_map = self._validate_rename_map(policy_specs.rename_map)
        except ValueError as error:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(error))

        acquired = self._acquire_inference_lock(context)
        try:
            self._check_active(context)
            try:
                policy = self._policy_loader(
                    model_id=policy_specs.pretrained_name_or_path,
                    cache_dir=self.config.cache_dir,
                    dtype=self.config.dtype,
                    tokenizer_dir=self.config.tokenizer_dir,
                    execution_mode=self.config.execution_mode,
                    quantization=self.config.quantization,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"could not load requested SmolVLA policy: {error}",
                )
            if policy_specs.actions_per_chunk > policy.config.chunk_size:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "actions_per_chunk exceeds the checkpoint chunk_size "
                    f"({policy_specs.actions_per_chunk} > {policy.config.chunk_size})",
                )

            image_keys = {
                rename_map.get(name, name)
                for name, feature in features.items()
                if name.startswith("observation.images.")
                and feature.get("dtype") in {"image", "video"}
            }
            unknown_images = image_keys - set(policy.config.image_keys)
            if unknown_images:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"client image features do not match checkpoint inputs: {sorted(unknown_images)}",
                )

            with self._state_lock:
                self.policy = policy
                self.policy_type = policy_specs.policy_type
                self.pretrained_name_or_path = policy_specs.pretrained_name_or_path
                self.lerobot_features = features
                self.actions_per_chunk = policy_specs.actions_per_chunk
                self.rename_map = rename_map
                self._prediction_index = 0
            self._clear_episode_state()
        finally:
            if acquired:
                self._inference_lock.release()
        _LOG.info(
            "Loaded %s for %s inference; client device field %r is transport metadata only",
            policy_specs.pretrained_name_or_path,
            self.config.execution_mode,
            policy_specs.device,
        )
        return services_pb2.Empty()

    def _receive_observation_bytes(self, request_iterator, context) -> bytes:
        payload = bytearray()
        started = False
        ended = False
        for message in request_iterator:
            self._check_active(context)
            if ended:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "observation stream contains data after TRANSFER_END",
                )
            state = message.transfer_state
            if state == services_pb2.TRANSFER_BEGIN:
                if started or payload:
                    self._abort(
                        context,
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "TRANSFER_BEGIN must be the first observation chunk",
                    )
                started = True
            elif state == services_pb2.TRANSFER_MIDDLE:
                if not started:
                    self._abort(
                        context,
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "TRANSFER_MIDDLE requires TRANSFER_BEGIN",
                    )
            elif state == services_pb2.TRANSFER_END:
                ended = True
            else:
                self._abort(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"unknown observation transfer state {state}",
                )
            payload.extend(message.data)
            if len(payload) > _MAX_OBSERVATION_BYTES:
                self._abort(
                    context,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    f"observation exceeds {_MAX_OBSERVATION_BYTES} bytes",
                )
        if not ended or not payload:
            self._abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "observation stream must end with a non-empty TRANSFER_END chunk",
            )
        return bytes(payload)

    @staticmethod
    def _validate_timed_observation(value: object) -> TimedObservation:
        if not isinstance(value, TimedObservation):
            raise ValueError(f"observation must be TimedObservation, got {type(value).__name__}")
        if not math.isfinite(value.timestamp):
            raise ValueError("observation timestamp must be finite")
        if isinstance(value.timestep, bool) or not isinstance(value.timestep, int) or value.timestep < 0:
            raise ValueError("observation timestep must be a non-negative integer")
        if not isinstance(value.observation, dict):
            raise ValueError("TimedObservation.observation must be a dictionary")
        if not isinstance(value.must_go, bool):
            raise ValueError("TimedObservation.must_go must be boolean")
        return value

    def SendObservations(self, request_iterator, context):  # noqa: N802
        self._check_active(context)
        with self._state_lock:
            if self.policy is None or self.lerobot_features is None:
                self._abort(
                    context,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "send policy instructions before observations",
                )
        payload = self._receive_observation_bytes(request_iterator, context)
        try:
            observation = self._validate_timed_observation(
                pickle.loads(payload)  # nosec B301: exact reference protocol
            )
        except Exception as error:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"invalid observation pickle: {error}")
        self._enqueue_observation(observation)
        return services_pb2.Empty()

    def _enqueue_observation(self, observation: TimedObservation) -> bool:
        with self._state_lock:
            should_enqueue = (
                observation.must_go
                or self.last_processed_obs is None
                or (
                    observation.timestep not in self._predicted_timesteps
                    and not observations_similar(
                        observation,
                        self.last_processed_obs,
                        lerobot_features=self.lerobot_features,
                    )
                )
            )
            if not should_enqueue:
                return False
            if self.observation_queue.full():
                dropped = self.observation_queue.get_nowait()
                self._observation_receipts.pop(id(dropped), None)
            received_monotonic_ns = time.perf_counter_ns()
            received_wall_ns = time.time_ns()
            self._observation_receipts[id(observation)] = _ObservationReceipt(
                wall_time_ns=received_wall_ns,
                monotonic_ns=received_monotonic_ns,
                received_at_utc=datetime.fromtimestamp(
                    received_wall_ns / 1_000_000_000,
                    tz=timezone.utc,
                ).isoformat(),
            )
            self.observation_queue.put_nowait(observation)
            return True

    def _record_chunk_latency(
        self,
        *,
        observation: TimedObservation,
        actions: list[TimedAction],
        inference_started_ns: int,
        chunk_ready_monotonic_ns: int,
        chunk_ready_wall_ns: int,
        generation: int,
    ) -> None:
        with self._state_lock:
            receipt = self._observation_receipts.pop(id(observation), None)
            policy_type = self.policy_type
            pretrained_name_or_path = self.pretrained_name_or_path
        if self._latency_recorder is None:
            return
        if receipt is None or policy_type is None or pretrained_name_or_path is None:
            raise RuntimeError("latency telemetry lost its observation or policy identity")
        self._latency_recorder.write(
            {
                "session_generation": generation,
                "observation": {
                    "client_timestamp": observation.timestamp,
                    "must_go": observation.must_go,
                    "timestep": observation.timestep,
                },
                "server": {
                    "received_at_utc": receipt.received_at_utc,
                    "chunk_ready_at_utc": datetime.fromtimestamp(
                        chunk_ready_wall_ns / 1_000_000_000,
                        tz=timezone.utc,
                    ).isoformat(),
                },
                "latency_ms": {
                    "client_observation_to_chunk": (
                        chunk_ready_wall_ns / 1_000_000 - observation.timestamp * 1_000
                    ),
                    "server_receive_to_chunk": (
                        chunk_ready_monotonic_ns - receipt.monotonic_ns
                    )
                    / 1_000_000,
                    "inference": (chunk_ready_monotonic_ns - inference_started_ns)
                    / 1_000_000,
                },
                "chunk": {
                    "action_count": len(actions),
                    "first_timestep": actions[0].timestep,
                    "last_timestep": actions[-1].timestep,
                },
                "policy": {
                    "type": policy_type,
                    "pretrained_name_or_path": pretrained_name_or_path,
                    "dtype": self.config.dtype,
                    "execution_mode": self.config.execution_mode,
                    "quantization": self.config.quantization,
                },
            }
        )

    def _policy_image_features(self) -> dict[str, PolicyFeature]:
        if self.policy is None or self.lerobot_features is None:
            raise RuntimeError("policy is not configured")
        native_shapes = dict(self.policy.config.image_shapes)
        result: dict[str, PolicyFeature] = {}
        for name, feature in self.lerobot_features.items():
            if not name.startswith("observation.images.") or feature.get("dtype") not in {
                "image",
                "video",
            }:
                continue
            target = self.rename_map.get(name, name)
            result[name] = PolicyFeature(type=FeatureType.VISUAL, shape=native_shapes[target])
        return result

    def prepare_observation(self, timed_observation: TimedObservation) -> dict[str, object]:
        """Apply the pinned client-to-policy conversion and native batch boundary."""

        if self.policy is None or self.lerobot_features is None:
            raise RuntimeError("policy is not configured")
        prepared = raw_observation_to_observation(
            timed_observation.get_observation(),
            self.lerobot_features,
            self._policy_image_features(),
        )
        native: dict[str, object] = {}
        for old_name, value in prepared.items():
            name = self.rename_map.get(old_name, old_name)
            if name in native:
                raise ValueError(f"rename_map creates duplicate observation key {name!r}")
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().numpy()
                if array.ndim in {2, 4} and array.shape[0] == 1:
                    array = array[0]
                native[name] = np.ascontiguousarray(array)
            else:
                native[name] = value
        return native

    def _acquire_inference_lock(self, context) -> bool:
        while not self._inference_lock.acquire(timeout=_CANCELLATION_POLL_SECONDS):
            self._check_active(context)
        return True

    def _next_observation(self, context) -> TimedObservation | None:
        deadline = time.monotonic() + self.config.obs_queue_timeout
        while True:
            self._check_active(context)
            remaining = deadline - time.monotonic()
            observation_queue = self.observation_queue
            if remaining <= 0:
                try:
                    observation = observation_queue.get_nowait()
                except Empty:
                    return None
            else:
                try:
                    observation = observation_queue.get(
                        timeout=min(_CANCELLATION_POLL_SECONDS, remaining)
                    )
                except Empty:
                    continue
            if not context.is_active():
                with self._state_lock:
                    if (
                        self.running
                        and observation_queue is self.observation_queue
                        and not observation_queue.full()
                    ):
                        observation_queue.put_nowait(observation)
                self._abort(context, grpc.StatusCode.CANCELLED, "request cancelled")
            self._check_active(context)
            return observation

    def _predict_timed_actions(self, observation: TimedObservation) -> list[TimedAction]:
        if self.policy is None or self.actions_per_chunk is None:
            raise RuntimeError("policy is not configured")
        native = self.prepare_observation(observation)
        self.last_processed_obs = observation
        if self.config.seed is not None:
            with self._state_lock:
                prediction_seed = self.config.seed + self._prediction_index
                self._prediction_index += 1
            mx.random.seed(prediction_seed)
        normalized = self.policy.predict_action_chunk(native)
        physical = self.policy.preprocessor.unnormalize_actions(normalized)
        mx.eval(physical)
        actions = np.asarray(physical.astype(mx.float32))
        expected = (
            1,
            self.policy.config.chunk_size,
            self.policy.config.action_dim,
        )
        if actions.shape != expected:
            raise ValueError(f"action chunk must have shape {expected}, got {actions.shape}")
        return [
            TimedAction(
                timestamp=observation.timestamp + index * self.config.environment_dt,
                timestep=observation.timestep + index,
                action=torch.from_numpy(actions[0, index].copy()),
            )
            for index in range(self.actions_per_chunk)
        ]

    def GetActions(self, request, context):  # noqa: N802
        self._check_active(context)
        acquired = self._acquire_inference_lock(context)
        started = time.perf_counter()
        try:
            with self._state_lock:
                if self.policy is None:
                    self._abort(
                        context,
                        grpc.StatusCode.FAILED_PRECONDITION,
                        "send policy instructions before requesting actions",
                    )
                generation = self._generation
            observation = self._next_observation(context)
            if observation is None:
                return services_pb2.Actions()
            with self._state_lock:
                self._predicted_timesteps.add(observation.timestep)
            inference_started_ns = time.perf_counter_ns()
            try:
                actions = self._predict_timed_actions(observation)
            except Exception as error:
                with self._state_lock:
                    self._observation_receipts.pop(id(observation), None)
                self._abort(context, grpc.StatusCode.INTERNAL, f"inference failed: {error}")
            chunk_ready_monotonic_ns = time.perf_counter_ns()
            chunk_ready_wall_ns = time.time_ns()
            self._check_active(context)
            with self._state_lock:
                if generation != self._generation:
                    self._abort(
                        context,
                        grpc.StatusCode.CANCELLED,
                        "client session changed while inference was running",
                    )
            try:
                self._record_chunk_latency(
                    observation=observation,
                    actions=actions,
                    inference_started_ns=inference_started_ns,
                    chunk_ready_monotonic_ns=chunk_ready_monotonic_ns,
                    chunk_ready_wall_ns=chunk_ready_wall_ns,
                    generation=generation,
                )
            except (OSError, RuntimeError, ValueError) as error:
                self._abort(
                    context,
                    grpc.StatusCode.INTERNAL,
                    f"latency logging failed: {error}",
                )
            remaining = self.config.inference_latency - (time.perf_counter() - started)
            while remaining > 0:
                self._check_active(context)
                interval = min(_CANCELLATION_POLL_SECONDS, remaining)
                time.sleep(interval)
                remaining -= interval
            return services_pb2.Actions(
                data=pickle.dumps(actions)  # nosec B301: exact reference protocol
            )
        finally:
            if acquired:
                self._inference_lock.release()

    def stop(self) -> None:
        self._stopped.set()
        self._clear_episode_state()
        if self._latency_recorder is not None:
            self._latency_recorder.close()


def create_server(
    config: ServeConfig,
    *,
    policy_loader: PolicyLoader = _default_policy_loader,
):
    """Build an unstarted gRPC server and return it with its bound port."""

    servicer = NativeMLXPolicyServer(config, policy_loader=policy_loader)
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_workers),
        options=(
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ),
    )
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(servicer, grpc_server)
    port = grpc_server.add_insecure_port(_bind_target(config.host, config.port))
    if port == 0:
        raise RuntimeError(f"gRPC could not bind {_bind_target(config.host, config.port)}")
    return grpc_server, servicer, port


def serve_forever(config: ServeConfig) -> None:
    """Start the server, block until interrupted, and close episode state."""

    grpc_server, servicer, port = create_server(config)
    grpc_server.start()
    _LOG.warning(
        "Serving trusted LeRobot 0.6.1 clients on %s (pickle protocol; no authentication)",
        _bind_target(config.host, port),
    )
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        _LOG.info("Stopping MLX policy server")
    finally:
        servicer.stop()
        grpc_server.stop(grace=1).wait()


__all__ = ["NativeMLXPolicyServer", "ServeConfig", "create_server", "serve_forever"]
