"""Dependency-light, fail-closed safety primitives for physical robot clients."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np


REQUIRED_SAFETY_REGISTERS = (
    "Max_Torque_Limit",
    "Torque_Limit",
    "Protection_Current",
    "Overload_Torque",
    "Acceleration",
    "Goal_Velocity",
    "Moving_Velocity",
    "Maximum_Velocity_Limit",
    "Maximum_Acceleration",
)


def ranges_from_vendor_calibration(
    joint_names: Sequence[str],
    calibration: Mapping[str, Mapping[str, float]],
    *,
    encoder_max: int = 4095,
) -> dict[str, "JointRange"]:
    """Translate Hiwonder calibration bytes to SOFollower public action units."""

    names = tuple(joint_names)
    if set(calibration) != set(names):
        raise ValueError("calibration must exactly match joint names")
    if encoder_max <= 0:
        raise ValueError("encoder_max must be positive")
    ranges: dict[str, JointRange] = {}
    for name in names:
        entry = calibration[name]
        raw_min = float(entry["range_min"])
        raw_max = float(entry["range_max"])
        if name == "gripper":
            ranges[name] = JointRange(0.0, 100.0)
        else:
            half_span_degrees = (raw_max - raw_min) * 180.0 / encoder_max
            ranges[name] = JointRange(-half_span_degrees, half_span_degrees)
    return ranges


@dataclass(frozen=True)
class JointRange:
    """One joint's calibrated command interval in the driver's public units."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.lower, self.upper)).all() or self.lower >= self.upper:
            raise ValueError("joint range must contain two finite, increasing values")


@dataclass(frozen=True)
class HardwareLimitVerdict:
    ok: bool
    mismatches: tuple[str, ...]


@dataclass(frozen=True)
class HardwareSafetyProfile:
    """Operator-owned exact allowlist for already-established servo limits."""

    robot_serial: str
    verified_at: str
    procedure: str
    expected_registers: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        if not isinstance(self.robot_serial, str) or not self.robot_serial.strip():
            raise ValueError("robot serial must be a non-empty string")
        if not isinstance(self.verified_at, str) or not self.verified_at.strip():
            raise ValueError("verified_at must be a non-empty string")
        if not isinstance(self.procedure, str) or not self.procedure.strip():
            raise ValueError("operator verification procedure must be a non-empty string")
        if set(self.expected_registers) != set(REQUIRED_SAFETY_REGISTERS):
            raise ValueError("profile must define exactly the required safety registers")

        copied: dict[str, dict[str, int]] = {}
        expected_joints: set[str] | None = None
        for register in REQUIRED_SAFETY_REGISTERS:
            values = self.expected_registers[register]
            if not isinstance(values, Mapping) or not values:
                raise ValueError("safety register values must be non-empty mappings")
            joints = set(values)
            if expected_joints is None:
                expected_joints = joints
            elif joints != expected_joints:
                raise ValueError("every safety register must cover the same joints")
            if not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values.values()
            ):
                raise ValueError("safety register values must be non-negative integers")
            copied[register] = dict(values)
        object.__setattr__(self, "expected_registers", copied)

    def verify_readback(
        self,
        joint_names: Sequence[str],
        actual_registers: Mapping[str, Mapping[str, int]],
    ) -> HardwareLimitVerdict:
        """Require torque off and exact equality with every operator-approved value."""

        names = tuple(joint_names)
        mismatches: list[str] = []
        for register in REQUIRED_SAFETY_REGISTERS:
            actual = actual_registers.get(register, {})
            for name in names:
                expected = self.expected_registers[register].get(name)
                read = actual.get(name)
                if read != expected:
                    mismatches.append(
                        f"{register}.{name}: expected {expected}, read {read}"
                    )
        torque = actual_registers.get("Torque_Enable", {})
        for name in names:
            read = torque.get(name)
            if read != 0:
                mismatches.append(f"Torque_Enable.{name}: expected 0, read {read}")
        return HardwareLimitVerdict(ok=not mismatches, mismatches=tuple(mismatches))


def load_hardware_safety_profile(path: str | Path) -> HardwareSafetyProfile:
    """Load one operator-attested profile without accepting silent extensions."""

    profile_path = Path(path).expanduser().resolve(strict=True)
    try:
        document = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"hardware safety profile is not valid JSON: {error}") from error
    required = {
        "robot_serial",
        "verified_at",
        "procedure",
        "expected_registers",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError(
            "hardware safety profile must contain exactly robot_serial, verified_at, "
            "procedure, and expected_registers"
        )
    verified_at = document["verified_at"]
    if not isinstance(verified_at, str) or not verified_at.endswith("Z"):
        raise ValueError("hardware safety profile verified_at must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(verified_at.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError(
            "hardware safety profile verified_at must be an ISO-8601 UTC timestamp"
        ) from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("hardware safety profile verified_at must be an ISO-8601 UTC timestamp")
    return HardwareSafetyProfile(
        robot_serial=document["robot_serial"],
        verified_at=verified_at,
        procedure=document["procedure"],
        expected_registers=document["expected_registers"],
    )


@dataclass(frozen=True)
class ActionDecision:
    """Observable result of applying the client-side action envelope."""

    target: np.ndarray | None
    hold: bool
    reason: str | None
    clipped_joints: tuple[str, ...] = ()
    rate_limited_joints: tuple[str, ...] = ()


@dataclass(frozen=True)
class SafetySessionConfig:
    """Hard upper bounds for one supervised hardware session."""

    no_motion: bool = True
    move_time_ms: int = 200
    return_move_time_ms: int = 1000
    watchdog_seconds: float = 0.5
    timeout_limit: int = 3
    duration_seconds: float = 60.0
    chunk_limit: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.no_motion, bool):
            raise TypeError("no_motion must be a boolean")
        if (
            isinstance(self.move_time_ms, bool)
            or not isinstance(self.move_time_ms, int)
            or self.move_time_ms < 200
        ):
            raise ValueError("move time must be at least 200 ms")
        if (
            isinstance(self.return_move_time_ms, bool)
            or not isinstance(self.return_move_time_ms, int)
            or self.return_move_time_ms < self.move_time_ms
        ):
            raise ValueError("return move time must be an integer no shorter than move time")
        if (
            isinstance(self.watchdog_seconds, bool)
            or not np.isfinite(self.watchdog_seconds)
            or not 0.0 < self.watchdog_seconds <= 0.5
        ):
            raise ValueError("watchdog must fire within 500 ms")
        if (
            isinstance(self.timeout_limit, bool)
            or not isinstance(self.timeout_limit, int)
            or not 0 < self.timeout_limit <= 3
        ):
            raise ValueError("timeout limit cannot exceed three")
        if (
            isinstance(self.duration_seconds, bool)
            or not np.isfinite(self.duration_seconds)
            or not 0.0 < self.duration_seconds <= 90.0
        ):
            raise ValueError("session duration cannot exceed 90 seconds")
        if (
            isinstance(self.chunk_limit, bool)
            or not isinstance(self.chunk_limit, int)
            or not 0 < self.chunk_limit <= 20
        ):
            raise ValueError("session chunk limit cannot exceed 20")


@dataclass
class SessionTelemetry:
    chunks: int = 0
    clipped_joints: int = 0
    rate_limited_joints: int = 0
    rejected_chunks: int = 0
    holds: int = 0
    timeouts: int = 0


class SafetyEnvelope:
    """Validate and bound one action before any physical I/O is attempted."""

    def __init__(
        self,
        joint_names: Sequence[str],
        ranges: Mapping[str, JointRange],
        *,
        calibration_margin: float = 0.10,
        max_step_fraction: float = 0.02,
        max_relative_step: float = 1.0,
    ) -> None:
        self.joint_names = tuple(joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint names must be non-empty and unique")
        if set(ranges) != set(self.joint_names):
            raise ValueError("joint ranges must exactly match joint names")
        if not 0.0 <= calibration_margin < 0.5:
            raise ValueError("calibration margin must be in [0, 0.5)")
        if not np.isfinite(max_step_fraction) or max_step_fraction <= 0.0:
            raise ValueError("max step fraction must be finite and positive")
        if not np.isfinite(max_relative_step) or max_relative_step <= 0.0:
            raise ValueError("max relative step must be finite and positive")
        self.ranges = {name: ranges[name] for name in self.joint_names}
        self.calibration_margin = calibration_margin
        self.max_step_fraction = max_step_fraction
        self.max_relative_step = max_relative_step

    def evaluate(self, chunk: np.ndarray, present_positions: np.ndarray) -> ActionDecision:
        """Return a safe target for a one-action chunk."""

        actions = np.asarray(chunk, dtype=np.float64)
        present = np.asarray(present_positions, dtype=np.float64)
        expected_width = len(self.joint_names)
        if actions.shape != (1, expected_width) or present.shape != (expected_width,):
            return ActionDecision(None, True, "wrong_shape")

        action = actions[0]
        if not np.isfinite(action).all():
            return ActionDecision(None, True, "non_finite")
        lower = np.array([self.ranges[name].lower for name in self.joint_names])
        upper = np.array([self.ranges[name].upper for name in self.joint_names])
        if np.any(action < lower) or np.any(action > upper):
            return ActionDecision(None, True, "outside_calibration")
        if (
            not np.isfinite(present).all()
            or np.any(present < lower)
            or np.any(present > upper)
        ):
            return ActionDecision(None, True, "invalid_present")
        span = upper - lower
        tightened_lower = lower + self.calibration_margin * span
        tightened_upper = upper - self.calibration_margin * span
        tightened = np.clip(action, tightened_lower, tightened_upper)
        clipped_mask = tightened != action
        max_delta = np.minimum(self.max_step_fraction * span, self.max_relative_step)
        rate_limited = np.clip(tightened, present - max_delta, present + max_delta)
        rate_mask = rate_limited != tightened

        return ActionDecision(
            target=rate_limited,
            hold=False,
            reason=None,
            clipped_joints=tuple(
                name for name, clipped in zip(self.joint_names, clipped_mask, strict=True) if clipped
            ),
            rate_limited_joints=tuple(
                name for name, limited in zip(self.joint_names, rate_mask, strict=True) if limited
            ),
        )


class RobotIO(Protocol):
    """Small physical-I/O boundary shared by the fake and vendor adapter."""

    def read_positions(self) -> np.ndarray: ...

    def prepare_motion(self, *, move_time_ms: int) -> None: ...

    def write_positions(self, target: np.ndarray, *, move_time_ms: int) -> None: ...

    def disable_torque(self) -> None: ...

    def close(self) -> None: ...


class SafetySession:
    """Apply one safety envelope and own deterministic session cleanup."""

    def __init__(
        self,
        robot: RobotIO,
        envelope: SafetyEnvelope,
        config: SafetySessionConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.robot = robot
        self.envelope = envelope
        self.config = config
        self._clock = clock
        self.start_position: np.ndarray | None = None
        self.started_at: float | None = None
        self.consecutive_timeouts = 0
        self.chunk_count = 0
        self.stop_reason: str | None = None
        self.telemetry = SessionTelemetry()

    def __enter__(self) -> "SafetySession":
        self.start_position = np.asarray(self.robot.read_positions(), dtype=np.float64)
        start_decision = self.envelope.evaluate(
            self.start_position.reshape(1, -1),
            self.start_position,
        )
        if start_decision.hold or (
            not self.config.no_motion and start_decision.clipped_joints
        ):
            self.robot.close()
            self.start_position = None
            raise ValueError("start pose is outside the safe calibrated envelope")
        if not self.config.no_motion:
            try:
                self.robot.prepare_motion(move_time_ms=self.config.move_time_ms)
            except BaseException as error:
                for cleanup in (self.robot.disable_torque, self.robot.close):
                    try:
                        cleanup()
                    except BaseException as cleanup_error:
                        error.add_note(f"hardware cleanup also failed: {cleanup_error}")
                self.start_position = None
                raise
        self.started_at = self._clock()
        return self

    def process_chunk(self, chunk: np.ndarray) -> ActionDecision:
        if self.start_position is None:
            raise RuntimeError("safety session has not started")
        if self.stop_reason is not None:
            raise RuntimeError(f"safety session already stopped: {self.stop_reason}")
        if self.check_limits():
            raise RuntimeError(f"safety session already stopped: {self.stop_reason}")
        present = np.asarray(self.robot.read_positions(), dtype=np.float64)
        decision = self.envelope.evaluate(chunk, present)
        self.telemetry.chunks += 1
        self.telemetry.clipped_joints += len(decision.clipped_joints)
        self.telemetry.rate_limited_joints += len(decision.rate_limited_joints)
        if decision.hold:
            self.telemetry.rejected_chunks += 1
            self.telemetry.holds += 1
        if not self.config.no_motion:
            target = present if decision.hold else decision.target
            if target is not None:
                self.robot.write_positions(target, move_time_ms=self.config.move_time_ms)
        self.consecutive_timeouts = 0
        self.chunk_count += 1
        if not self.config.no_motion and self.chunk_count >= self.config.chunk_limit:
            self.stop_reason = "chunk_limit"
        return decision

    def check_limits(self) -> bool:
        """Update and return the terminal state from hard session caps."""

        if self.started_at is None:
            raise RuntimeError("safety session has not started")
        if self.stop_reason is None and self._clock() - self.started_at >= self.config.duration_seconds:
            self.stop_reason = "duration_limit"
        return self.stop_reason is not None

    def record_timeout(self) -> bool:
        """Hold on one watchdog expiry and report whether the session must stop."""

        if self.start_position is None:
            raise RuntimeError("safety session has not started")
        present = np.asarray(self.robot.read_positions(), dtype=np.float64)
        if not self.config.no_motion:
            self.robot.write_positions(present, move_time_ms=self.config.move_time_ms)
        self.consecutive_timeouts += 1
        self.telemetry.timeouts += 1
        self.telemetry.holds += 1
        if self.consecutive_timeouts >= self.config.timeout_limit:
            self.stop_reason = "watchdog_timeout"
        return self.stop_reason is not None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        cleanup_errors: list[BaseException] = []

        def attempt(operation) -> None:
            try:
                operation()
            except BaseException as error:
                cleanup_errors.append(error)

        if not self.config.no_motion:
            current: np.ndarray | None = None
            try:
                current = np.asarray(self.robot.read_positions(), dtype=np.float64)
            except BaseException as error:
                cleanup_errors.append(error)
            if current is not None:
                attempt(
                    lambda: self.robot.write_positions(
                        current,
                        move_time_ms=self.config.move_time_ms,
                    )
                )
            if self.start_position is not None:
                attempt(
                    lambda: self.robot.write_positions(
                        self.start_position,
                        move_time_ms=self.config.return_move_time_ms,
                    )
                )
            attempt(self.robot.disable_torque)
        attempt(self.robot.close)
        if cleanup_errors:
            if exc_value is not None:
                for error in cleanup_errors:
                    exc_value.add_note(f"hardware cleanup also failed: {error}")
            else:
                raise BaseExceptionGroup("hardware cleanup failed", cleanup_errors)
        return False
