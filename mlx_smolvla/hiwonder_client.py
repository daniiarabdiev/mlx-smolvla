"""Repository-owned Hiwonder SO-101 I/O and four-RPC client adapters.

Vendor, gRPC, and Torch dependencies are imported only by the executable
factory so importing :mod:`mlx_smolvla` remains dependency-light.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import pickle  # nosec B403: required by the pinned LeRobot protocol
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np

from mlx_smolvla.hardware_safety import (
    HardwareSafetyProfile,
    JointRange,
    REQUIRED_SAFETY_REGISTERS,
    SafetyEnvelope,
    SafetySession,
    SafetySessionConfig,
    ranges_from_vendor_calibration,
)


SO101_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def so101_public_action_ranges() -> dict[str, JointRange]:
    """Return the full public-unit domain of the configured vendor driver."""

    return {
        **{name: JointRange(-180.0, 180.0) for name in SO101_JOINTS[:-1]},
        "gripper": JointRange(0.0, 100.0),
    }


def so101_lerobot_contract(
    *,
    height: int = 480,
    width: int = 640,
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Return the audited two-camera feature and checkpoint-key mapping."""

    if (
        isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
    ):
        raise ValueError("camera dimensions must be positive integers")
    features: dict[str, dict[str, object]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(SO101_JOINTS),),
            "names": [f"{name}.pos" for name in SO101_JOINTS],
        },
        "observation.images.wrist_camera": {
            "dtype": "video",
            "shape": (3, height, width),
        },
        "observation.images.top_camera": {
            "dtype": "video",
            "shape": (3, height, width),
        },
    }
    rename_map = {
        "observation.images.wrist_camera": "observation.images.camera1",
        "observation.images.top_camera": "observation.images.camera2",
    }
    return features, rename_map


class HardwareTelemetryRecorder:
    """No-clobber JSONL recorder that refuses raw robot/policy payload fields."""

    _FORBIDDEN_PAYLOAD_KEYS = {
        "action",
        "actions",
        "frame",
        "frames",
        "image",
        "images",
        "observation",
        "positions",
        "state",
        "target",
        "task",
    }

    def __init__(
        self,
        path: str | Path,
        *,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.parent.is_dir():
            raise FileNotFoundError(
                f"hardware telemetry parent directory does not exist: {self.path.parent}"
            )
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise
        self._wall_clock = wall_clock
        self._closed = False

    @classmethod
    def _validate_fields(cls, fields: Mapping[str, object]) -> None:
        if not isinstance(fields, Mapping) or not all(
            isinstance(key, str) and key for key in fields
        ):
            raise ValueError("telemetry fields must be a mapping with non-empty string keys")
        forbidden = cls._FORBIDDEN_PAYLOAD_KEYS.intersection(fields)
        if forbidden:
            raise ValueError(
                "hardware telemetry cannot contain raw payload fields: "
                + ", ".join(sorted(forbidden))
            )

    def write_event(self, event: str, fields: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("hardware telemetry recorder is closed")
        if not isinstance(event, str) or not event.strip():
            raise ValueError("telemetry event must be a non-empty string")
        self._validate_fields(fields)
        record = {
            "schema_version": 1,
            "event": event,
            "recorded_at": self._wall_clock(),
            **dict(fields),
        }
        encoded = json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))
        self._stream.write(encoded + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stream.close()

    def __enter__(self) -> "HardwareTelemetryRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


@dataclass
class _CompatTimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class _CompatTimedAction(_CompatTimedData):
    action: object

    def get_action(self):
        return self.action


@dataclass
class _CompatTimedObservation(_CompatTimedData):
    observation: dict[str, object]
    must_go: bool = False

    def get_observation(self):
        return self.observation


@dataclass
class _CompatRemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, object]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)


for _compat_class, _wire_name in (
    (_CompatTimedData, "TimedData"),
    (_CompatTimedAction, "TimedAction"),
    (_CompatTimedObservation, "TimedObservation"),
    (_CompatRemotePolicyConfig, "RemotePolicyConfig"),
):
    _compat_class.__module__ = "lerobot.async_inference.helpers"
    _compat_class.__name__ = _wire_name
    _compat_class.__qualname__ = _wire_name


def install_lerobot_protocol_shim():
    """Install only the three pickle globals required by the audited wire protocol."""

    module_name = "lerobot.async_inference.helpers"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return (
            existing.RemotePolicyConfig,
            existing.TimedObservation,
            existing.TimedAction,
        )

    root = sys.modules.get("lerobot")
    if root is None:
        raise RuntimeError("import the selected lerobot package before installing its protocol shim")
    parent_name = "lerobot.async_inference"
    parent = sys.modules.get(parent_name)
    if parent is None:
        parent = types.ModuleType(parent_name)
        parent.__path__ = []
        sys.modules[parent_name] = parent
        setattr(root, "async_inference", parent)
    helper = types.ModuleType(module_name)
    helper.RemotePolicyConfig = _CompatRemotePolicyConfig
    helper.TimedObservation = _CompatTimedObservation
    helper.TimedAction = _CompatTimedAction
    helper.TimedData = _CompatTimedData
    sys.modules[module_name] = helper
    setattr(parent, "helpers", helper)
    return _CompatRemotePolicyConfig, _CompatTimedObservation, _CompatTimedAction


def _send_observation_chunks(payload: bytes, message_class):
    chunk_size = 2 * 1024 * 1024
    offset = 0
    while offset < len(payload):
        end = min(offset + chunk_size, len(payload))
        if end == len(payload):
            transfer_state = 3  # TRANSFER_END
        elif offset == 0:
            transfer_state = 1  # TRANSFER_BEGIN
        else:
            transfer_state = 2  # TRANSFER_MIDDLE
        yield message_class(transfer_state=transfer_state, data=payload[offset:end])
        offset = end


def _load_vendor_api(source_root: Path):
    loaded = sys.modules.get("lerobot")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
        if not loaded_file.is_relative_to(source_root):
            raise RuntimeError(
                "a different lerobot package is already imported; start the hardware client "
                "in a fresh process"
            )
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    return OpenCVCameraConfig, SOFollowerRobotConfig, SOFollower


def open_hiwonder_follower(
    *,
    vendor_root: Path,
    follower_port: str,
    calibration_id: str,
    wrist_camera: int,
    fixed_camera: int,
    safety_profile: HardwareSafetyProfile | None = None,
    robot_serial: str | None = None,
    _vendor_loader: Callable = _load_vendor_api,
) -> "HiwonderSO101IO":
    """Open the follower and two cameras without calling vendor configure/calibrate."""

    root = Path(vendor_root).expanduser().resolve()
    source_root = root / "src"
    required = (
        source_root / "lerobot/robots/so_follower/so_follower.py",
        source_root / "lerobot/motors/hiwonder/hiwonder.py",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"vendor checkout is missing required Hiwonder sources: {root}")
    if not isinstance(follower_port, str) or not follower_port.strip():
        raise ValueError("follower port must be a non-empty string")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError("calibration id must be a non-empty string")
    if wrist_camera == fixed_camera:
        raise ValueError("wrist and fixed cameras must use different indices")

    CameraConfig, RobotConfig, Robot = _vendor_loader(source_root)
    cameras = {
        "wrist_camera": CameraConfig(
            index_or_path=wrist_camera,
            fps=30,
            width=640,
            height=480,
            warmup_s=1,
        ),
        "top_camera": CameraConfig(
            index_or_path=fixed_camera,
            fps=30,
            width=640,
            height=480,
            warmup_s=1,
        ),
    }
    robot = Robot(
        RobotConfig(
            id=calibration_id,
            port=follower_port,
            cameras=cameras,
            use_degrees=True,
            max_relative_target=1.0,
            disable_torque_on_disconnect=True,
            gripper_min_target=15.0,
            motor_model="hx30hm",
        )
    )
    adapter = HiwonderSO101IO(
        robot,
        joint_names=SO101_JOINTS,
        safety_profile=safety_profile,
        robot_serial=robot_serial,
    )
    try:
        robot.bus.connect(handshake=True)
        torque = robot.bus.sync_read("Torque_Enable", normalize=False)
        enabled = {name: torque.get(name) for name in SO101_JOINTS if torque.get(name) != 0}
        if enabled:
            raise RuntimeError(
                f"follower torque is already enabled; cut power and resolve before retry: {enabled}"
            )
        if set(robot.calibration) != set(SO101_JOINTS) or not robot.is_calibrated:
            raise RuntimeError(
                "existing follower calibration does not match hardware; do not recalibrate in first contact"
            )
        for camera in robot.cameras.values():
            camera.connect()
        adapter.read_observation()
        return adapter
    except BaseException as error:
        try:
            adapter.close()
        except BaseException as cleanup_error:
            error.add_note(f"hardware cleanup also failed: {cleanup_error}")
        raise


@dataclass(frozen=True)
class ControlLoopResult:
    stop_reason: str
    duration_seconds: float
    observations: int
    chunks: int
    camera_sample_fps: float
    observation_to_chunk_median_ms: float | None
    observation_to_chunk_p95_ms: float | None
    observation_to_motion_median_ms: float | None
    observation_to_motion_p95_ms: float | None
    clipped_joints: int
    rate_limited_joints: int
    rejected_chunks: int
    holds: int
    timeouts: int
    rejection_reasons: dict[str, int]


def _latency_stats(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)), float(np.percentile(array, 95))


def run_control_loop(
    *,
    adapter: "HiwonderSO101IO",
    transport,
    envelope: SafetyEnvelope,
    config: SafetySessionConfig,
    task: str,
    fps: int = 5,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> ControlLoopResult:
    """Run one bounded synchronous observation/action session."""

    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")

    observation_to_chunk: list[float] = []
    observation_to_motion: list[float] = []
    observations = 0
    session = SafetySession(adapter, envelope, config, clock=clock)
    try:
        transport.start()
    except BaseException as error:
        for cleanup in (adapter.close, transport.close):
            try:
                cleanup()
            except BaseException as cleanup_error:
                error.add_note(f"hardware cleanup also failed: {cleanup_error}")
        raise

    active_started_at = clock()
    active_finished_at = active_started_at
    try:
        with session:
            if session.started_at is None:  # pragma: no cover - context contract
                raise RuntimeError("safety session did not record a start time")
            active_started_at = session.started_at
            while not session.check_limits():
                cycle_started = clock()
                observation = adapter.read_observation()
                observation["task"] = task
                sent_at = clock()
                transport.send_observation(
                    observation,
                    timestep=observations,
                    timestamp=wall_clock(),
                )
                observations += 1
                chunk = transport.receive_action_chunk()
                chunk_received = clock()
                if chunk is None:
                    if session.record_timeout():
                        break
                else:
                    observation_to_chunk.append((chunk_received - sent_at) * 1000.0)
                    session.process_chunk(chunk)
                    if not config.no_motion:
                        first_write = adapter.last_write_monotonic
                        if first_write is None or first_write < sent_at:
                            raise RuntimeError("motor write timestamp was not recorded")
                        observation_to_motion.append((first_write - sent_at) * 1000.0)
                    if session.stop_reason is not None:
                        break
                elapsed = clock() - cycle_started
                sleep(max(0.0, (1.0 / fps) - elapsed))
            active_finished_at = clock()
    finally:
        transport.close()

    duration = active_finished_at - active_started_at
    chunk_median, chunk_p95 = _latency_stats(observation_to_chunk)
    motion_median, motion_p95 = _latency_stats(observation_to_motion)
    telemetry = session.telemetry
    return ControlLoopResult(
        stop_reason=session.stop_reason or "external_stop",
        duration_seconds=duration,
        observations=observations,
        chunks=telemetry.chunks,
        camera_sample_fps=observations / duration if duration > 0 else 0.0,
        observation_to_chunk_median_ms=chunk_median,
        observation_to_chunk_p95_ms=chunk_p95,
        observation_to_motion_median_ms=motion_median,
        observation_to_motion_p95_ms=motion_p95,
        clipped_joints=telemetry.clipped_joints,
        rate_limited_joints=telemetry.rate_limited_joints,
        rejected_chunks=telemetry.rejected_chunks,
        holds=telemetry.holds,
        timeouts=telemetry.timeouts,
        rejection_reasons=dict(telemetry.rejection_reasons),
    )


class LeRobotFourRPCClient:
    """Small synchronous client for the audited LeRobot 0.6.1 wire contract."""

    def __init__(
        self,
        *,
        server_address: str,
        checkpoint: str,
        lerobot_features: Mapping[str, Mapping[str, object]],
        rename_map: Mapping[str, str],
        rpc_timeout_seconds: float = 0.5,
        setup_timeout_seconds: float = 600.0,
    ) -> None:
        parsed = urlsplit(f"//{server_address}")
        host = parsed.hostname
        try:
            loopback = host == "localhost" or (
                host is not None and ipaddress.ip_address(host).is_loopback
            )
        except ValueError:
            loopback = False
        if not loopback or parsed.port is None:
            raise ValueError("hardware client server_address must be loopback host:port")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("checkpoint must be a non-empty local path")
        if (
            isinstance(rpc_timeout_seconds, bool)
            or not isinstance(rpc_timeout_seconds, (int, float))
            or not math.isfinite(rpc_timeout_seconds)
            or not 0.0 < rpc_timeout_seconds <= 0.5
        ):
            raise ValueError("RPC timeout must be finite, positive, and at most 500 ms")
        if (
            isinstance(setup_timeout_seconds, bool)
            or not isinstance(setup_timeout_seconds, (int, float))
            or not math.isfinite(setup_timeout_seconds)
            or not 0.0 < setup_timeout_seconds <= 600.0
        ):
            raise ValueError("setup timeout must be finite, positive, and at most 600 seconds")

        import grpc
        from lerobot.transport import services_pb2, services_pb2_grpc

        RemotePolicyConfig, TimedObservation, _ = install_lerobot_protocol_shim()

        self.server_address = server_address
        self.checkpoint = checkpoint
        self.lerobot_features = {
            name: dict(feature) for name, feature in lerobot_features.items()
        }
        self.rename_map = dict(rename_map)
        self.rpc_timeout_seconds = rpc_timeout_seconds
        self.setup_timeout_seconds = float(setup_timeout_seconds)
        self._grpc = grpc
        self._TimedObservation = TimedObservation
        self._services_pb2 = services_pb2
        self._send_bytes_in_chunks = _send_observation_chunks
        self._policy_config = RemotePolicyConfig(
            policy_type="smolvla",
            pretrained_name_or_path=checkpoint,
            lerobot_features=self.lerobot_features,
            actions_per_chunk=1,
            device="cpu",
            rename_map=self.rename_map,
        )
        self.channel = grpc.insecure_channel(
            server_address,
            options=(
                ("grpc.max_receive_message_length", 4 * 1024 * 1024),
                ("grpc.max_send_message_length", 4 * 1024 * 1024),
                ("grpc.enable_retries", 0),
            ),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.started = False

    def start(self) -> None:
        self._grpc.channel_ready_future(self.channel).result(timeout=self.setup_timeout_seconds)
        self.stub.Ready(
            self._services_pb2.Empty(),
            timeout=self.setup_timeout_seconds,
        )
        setup = self._services_pb2.PolicySetup(data=pickle.dumps(self._policy_config))
        self.stub.SendPolicyInstructions(setup, timeout=self.setup_timeout_seconds)
        self.started = True

    def send_observation(
        self,
        observation: Mapping[str, object],
        *,
        timestep: int,
        timestamp: float,
    ) -> None:
        if not self.started:
            raise RuntimeError("four-RPC client has not started")
        timed = self._TimedObservation(
            timestamp=timestamp,
            timestep=timestep,
            observation=dict(observation),
            must_go=True,
        )
        payload = pickle.dumps(timed)
        messages = self._send_bytes_in_chunks(
            payload,
            self._services_pb2.Observation,
        )
        self.stub.SendObservations(messages, timeout=self.rpc_timeout_seconds)

    def receive_action_chunk(self) -> np.ndarray | None:
        if not self.started:
            raise RuntimeError("four-RPC client has not started")
        try:
            response = self.stub.GetActions(
                self._services_pb2.Empty(),
                timeout=self.rpc_timeout_seconds,
            )
        except self._grpc.RpcError as error:
            if error.code() == self._grpc.StatusCode.DEADLINE_EXCEEDED:
                return None
            raise
        if not response.data:
            return None
        actions = pickle.loads(response.data)  # nosec B301: pinned trusted-peer protocol
        if not isinstance(actions, list) or not actions:
            raise ValueError("server action payload must be a non-empty list")
        arrays = []
        for action in actions:
            tensor = action.get_action()
            arrays.append(np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64))
        return np.stack(arrays, axis=0)

    def close(self) -> None:
        self.started = False
        self.channel.close()


class HiwonderSO101IO:
    """Narrow adapter over the authorized vendor follower object."""

    def __init__(
        self,
        robot,
        *,
        joint_names: Sequence[str],
        camera_names: Sequence[str] = ("wrist_camera", "top_camera"),
        safety_profile: HardwareSafetyProfile | None = None,
        robot_serial: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.robot = robot
        self.joint_names = tuple(joint_names)
        self.camera_names = tuple(camera_names)
        self.safety_profile = safety_profile
        self.robot_serial = robot_serial
        self._sleep = sleep
        self._clock = clock
        self.last_write_monotonic: float | None = None
        self.armed = False

    def read_positions(self) -> np.ndarray:
        """Read normalized public-unit joint positions in checkpoint order."""

        values = self.robot.bus.sync_read("Present_Position")
        return np.asarray([values[name] for name in self.joint_names], dtype=np.float64)

    def calibration_ranges(self):
        """Return the loaded vendor calibration in the driver's public units."""

        raw = {
            name: {
                "range_min": self.robot.calibration[name].range_min,
                "range_max": self.robot.calibration[name].range_max,
            }
            for name in self.joint_names
        }
        return ranges_from_vendor_calibration(self.joint_names, raw)

    def read_observation(self) -> dict[str, object]:
        """Capture one observation and reject missing, malformed, or black frames."""

        observation = dict(self.robot.get_observation())
        for name in self.camera_names:
            if name not in observation:
                raise RuntimeError(f"camera {name} is missing from the robot observation")
            frame = np.asarray(observation[name])
            if frame.ndim != 3 or frame.shape[2] != 3 or not np.issubdtype(frame.dtype, np.number):
                raise RuntimeError(f"camera {name} did not return an HxWx3 numeric frame")
            if not np.isfinite(frame).all():
                raise RuntimeError(f"camera {name} returned non-finite pixels")
            if not np.any(frame != 0):
                raise RuntimeError(f"camera {name} frame is all black")
            observation[name] = np.ascontiguousarray(frame)
        return observation

    def prepare_motion(self, *, move_time_ms: int) -> None:
        """Verify operator-established limits before any torque enable."""

        if self.safety_profile is None:
            raise RuntimeError("motion requires an operator-verified hardware safety profile")
        if self.robot_serial != self.safety_profile.robot_serial:
            raise RuntimeError(
                "hardware safety profile robot serial does not match the connected follower"
            )
        if isinstance(move_time_ms, bool) or not isinstance(move_time_ms, int) or move_time_ms < 200:
            raise ValueError("move time must be at least 200 ms")

        readback = {
            register: {
                name: int(value)
                for name, value in self.robot.bus.sync_read(
                    register,
                    normalize=False,
                ).items()
            }
            for register in REQUIRED_SAFETY_REGISTERS
        }
        readback["Torque_Enable"] = {
            name: int(value)
            for name, value in self.robot.bus.sync_read(
                "Torque_Enable",
                normalize=False,
            ).items()
        }
        verdict = self.safety_profile.verify_readback(self.joint_names, readback)
        if not verdict.ok:
            raise RuntimeError("hardware safety readback mismatch: " + "; ".join(verdict.mismatches))

        self.robot.bus.enable_torque()
        enabled = self.robot.bus.sync_read("Torque_Enable", normalize=False)
        if any(enabled.get(name) != 1 for name in self.joint_names):
            self.robot.bus.disable_torque(num_retry=5)
            raise RuntimeError("torque enable readback failed; torque was disabled")
        self.armed = True

    def write_positions(self, target: np.ndarray, *, move_time_ms: int) -> None:
        """Write one bounded target, then enforce the command/dwell floor."""

        if not self.armed:
            raise RuntimeError("robot is not armed")
        positions = np.asarray(target, dtype=np.float64)
        if positions.shape != (len(self.joint_names),) or not np.isfinite(positions).all():
            raise ValueError("target must be one finite value per joint")
        if isinstance(move_time_ms, bool) or not isinstance(move_time_ms, int) or move_time_ms < 200:
            raise ValueError("move time must be at least 200 ms")
        self.last_write_monotonic = self._clock()
        self.robot.bus.sync_write(
            "Goal_Position",
            {
                name: float(value)
                for name, value in zip(self.joint_names, positions, strict=True)
            },
        )
        self._sleep(move_time_ms / 1000.0)

    def disable_torque(self) -> None:
        """Disable every follower motor and require an all-zero readback."""

        self.robot.bus.disable_torque(num_retry=5)
        readback = self.robot.bus.sync_read("Torque_Enable", normalize=False)
        self.armed = False
        failures = {name: readback.get(name) for name in self.joint_names if readback.get(name) != 0}
        if failures:
            raise RuntimeError(f"torque disable readback failed: {failures}")

    def close(self) -> None:
        """Close camera threads and serial I/O without issuing a motor write."""

        errors: list[BaseException] = []
        if self.armed:
            try:
                self.disable_torque()
            except BaseException as error:
                errors.append(error)
        for camera in self.robot.cameras.values():
            if getattr(camera, "is_connected", False):
                try:
                    camera.disconnect()
                except BaseException as error:
                    errors.append(error)
        try:
            self.robot.bus.disconnect(disable_torque=False)
        except BaseException as error:
            errors.append(error)
        if errors:
            raise BaseExceptionGroup("hardware close failed", errors)
