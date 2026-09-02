"""Safety contracts for the repository-owned Hiwonder SO-101 client."""

from __future__ import annotations

import importlib
import importlib.util
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class _FakeRobot:
    """Test-only implementation of the exact I/O used by SafetySession."""

    def __init__(self) -> None:
        self.positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
        self.writes: list[tuple[np.ndarray, int]] = []
        self.disable_calls = 0
        self.close_calls = 0
        self.prepare_calls = 0
        self.fail_prepare = False
        self.events: list[str] = []
        self.fail_write_numbers: set[int] = set()
        self.write_attempts = 0

    def read_positions(self) -> np.ndarray:
        return self.positions.copy()

    def prepare_motion(self, *, move_time_ms: int) -> None:
        self.prepare_calls += 1
        self.events.append(f"prepare_motion:{move_time_ms}")
        if self.fail_prepare:
            raise RuntimeError("prepare failed after partial arming")

    def write_positions(self, target: np.ndarray, *, move_time_ms: int) -> None:
        copied = np.asarray(target, dtype=np.float64).copy()
        self.write_attempts += 1
        self.writes.append((copied, move_time_ms))
        self.events.append(f"write:{move_time_ms}")
        if self.write_attempts in self.fail_write_numbers:
            raise RuntimeError(f"write {self.write_attempts} failed")
        self.positions = copied

    def disable_torque(self) -> None:
        self.disable_calls += 1
        self.events.append("disable_torque")

    def close(self) -> None:
        self.close_calls += 1
        self.events.append("close")


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeBus:
    def __init__(self) -> None:
        self.registers: dict[str, dict[str, int | float]] = {
            "Present_Position": {
                name: value
                for name, value in zip(JOINTS, [0, 0, 0, 0, 0, 50], strict=True)
            },
            "Torque_Enable": {name: 0 for name in JOINTS},
        }
        self.sync_writes: list[tuple[str, dict[str, float]]] = []
        self.disconnect_calls: list[bool] = []
        self.enable_calls = 0
        self.disable_calls = 0
        self.fail_disable = False

    def sync_read(self, register: str, *, normalize: bool = True):
        return dict(self.registers[register])

    def sync_write(self, register: str, values: dict[str, float]) -> None:
        self.sync_writes.append((register, dict(values)))

    def enable_torque(self) -> None:
        self.enable_calls += 1
        self.registers["Torque_Enable"] = {name: 1 for name in JOINTS}

    def disable_torque(self, *, num_retry: int = 0) -> None:
        self.disable_calls += 1
        if self.fail_disable:
            raise RuntimeError("disable failed")
        self.registers["Torque_Enable"] = {name: 0 for name in JOINTS}

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls.append(disable_torque)


class _FakeCamera:
    def __init__(self) -> None:
        self.is_connected = True
        self.disconnect_calls = 0

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


class _FakeVendorRobot:
    def __init__(self) -> None:
        self.bus = _FakeBus()
        self.cameras = {
            "wrist_camera": _FakeCamera(),
            "top_camera": _FakeCamera(),
        }
        self.frames = {
            "wrist_camera": np.ones((4, 5, 3), dtype=np.uint8),
            "top_camera": np.ones((4, 5, 3), dtype=np.uint8),
        }

    def get_observation(self):
        return {
            **{
                f"{name}.pos": value
                for name, value in self.bus.registers["Present_Position"].items()
            },
            **self.frames,
        }


class _FakeTransport:
    def __init__(self, clock: _FakeClock, chunks: list[np.ndarray | None]) -> None:
        self.clock = clock
        self.chunks = list(chunks)
        self.started = False
        self.closed = False
        self.sent: list[tuple[dict[str, object], int, float]] = []

    def start(self) -> None:
        self.started = True

    def send_observation(
        self,
        observation: dict[str, object],
        *,
        timestep: int,
        timestamp: float,
    ) -> None:
        self.sent.append((dict(observation), timestep, timestamp))

    def receive_action_chunk(self) -> np.ndarray | None:
        self.clock.now += 0.05
        return self.chunks.pop(0)

    def close(self) -> None:
        self.closed = True


def _hardware_safety():
    try:
        return importlib.import_module("mlx_smolvla.hardware_safety")
    except ModuleNotFoundError:
        pytest.fail("the hardware safety module is not implemented")


def _hiwonder_client():
    try:
        return importlib.import_module("mlx_smolvla.hiwonder_client")
    except ModuleNotFoundError:
        pytest.fail("the Hiwonder client adapter is not implemented")


def _standalone_client():
    path = Path("examples/bring_your_own_robot/hiwonder_so101_client.py")
    spec = importlib.util.spec_from_file_location("hiwonder_so101_client", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _envelope(safety):
    ranges = {
        name: safety.JointRange(-100.0, 100.0) for name in JOINTS[:-1]
    } | {"gripper": safety.JointRange(0.0, 100.0)}
    return safety.SafetyEnvelope(JOINTS, ranges)


def _matching_profile(safety):
    expected = {
        register: {name: 10 for name in JOINTS}
        for register in safety.REQUIRED_SAFETY_REGISTERS
    }
    profile = safety.HardwareSafetyProfile(
        robot_serial="5C82107541",
        verified_at="2026-09-02T00:00:00Z",
        procedure="Operator-verified with Hiwonder ServoStudio and physical power cut tested.",
        expected_registers=expected,
    )
    registers = {register: dict(values) for register, values in expected.items()}
    registers["Torque_Enable"] = {name: 0 for name in JOINTS}
    return profile, registers


def test_action_is_clipped_to_tightened_calibration_before_rate_limit() -> None:
    safety = _hardware_safety()
    ranges = {
        name: safety.JointRange(-100.0, 100.0) for name in JOINTS[:-1]
    } | {"gripper": safety.JointRange(0.0, 100.0)}
    envelope = safety.SafetyEnvelope(JOINTS, ranges)

    decision = envelope.evaluate(
        np.array([[90.0, 0.0, 0.0, 0.0, 0.0, 50.0]]),
        np.array([79.5, 0.0, 0.0, 0.0, 0.0, 50.0]),
    )

    np.testing.assert_array_equal(
        decision.target,
        np.array([80.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
    )
    assert decision.hold is False
    assert decision.clipped_joints == ("shoulder_pan",)
    assert decision.rate_limited_joints == ()


def test_rate_limit_uses_present_position_and_stricter_absolute_cap() -> None:
    safety = _hardware_safety()
    ranges = {
        name: safety.JointRange(-100.0, 100.0) for name in JOINTS[:-1]
    } | {"gripper": safety.JointRange(0.0, 100.0)}
    envelope = safety.SafetyEnvelope(JOINTS, ranges)

    decision = envelope.evaluate(
        np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 55.0]]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
    )

    np.testing.assert_array_equal(
        decision.target,
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 51.0]),
    )
    assert decision.clipped_joints == ()
    assert decision.rate_limited_joints == ("shoulder_pan", "gripper")


@pytest.mark.parametrize(
    ("chunk", "reason"),
    [
        (np.zeros(6), "wrong_shape"),
        (np.zeros((2, 6)), "wrong_shape"),
        (np.array([[np.nan, 0.0, 0.0, 0.0, 0.0, 50.0]]), "non_finite"),
        (np.array([[np.inf, 0.0, 0.0, 0.0, 0.0, 50.0]]), "non_finite"),
        (np.array([[101.0, 0.0, 0.0, 0.0, 0.0, 50.0]]), "outside_normalized_range"),
        (np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -0.1]]), "outside_normalized_range"),
    ],
)
def test_invalid_chunk_is_rejected_with_a_hold_decision(
    chunk: np.ndarray, reason: str
) -> None:
    safety = _hardware_safety()
    ranges = {
        name: safety.JointRange(-100.0, 100.0) for name in JOINTS[:-1]
    } | {"gripper": safety.JointRange(0.0, 100.0)}
    envelope = safety.SafetyEnvelope(JOINTS, ranges)

    decision = envelope.evaluate(chunk, np.zeros(6))

    assert decision.target is None
    assert decision.hold is True
    assert decision.reason == reason
    assert decision.clipped_joints == ()
    assert decision.rate_limited_joints == ()


@pytest.mark.parametrize(
    "present",
    [
        np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 50.0]),
        np.array([101.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
    ],
)
def test_invalid_present_position_prevents_a_target(present: np.ndarray) -> None:
    safety = _hardware_safety()
    ranges = {
        name: safety.JointRange(-100.0, 100.0) for name in JOINTS[:-1]
    } | {"gripper": safety.JointRange(0.0, 100.0)}
    envelope = safety.SafetyEnvelope(JOINTS, ranges)

    decision = envelope.evaluate(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]), present)

    assert decision.target is None
    assert decision.hold is True
    assert decision.reason == "invalid_present"


def test_action_inside_public_domain_is_clipped_to_calibration_not_rejected() -> None:
    safety = _hardware_safety()
    calibrated = {
        name: safety.JointRange(-30.0, 30.0) for name in JOINTS[:-1]
    } | {"gripper": safety.JointRange(10.0, 90.0)}
    public = {
        name: safety.JointRange(-180.0, 180.0) for name in JOINTS[:-1]
    } | {"gripper": safety.JointRange(0.0, 100.0)}
    envelope = safety.SafetyEnvelope(JOINTS, calibrated, normalized_ranges=public)

    decision = envelope.evaluate(
        np.array([[90.0, 0.0, 0.0, 0.0, 0.0, 95.0]]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
    )

    assert decision.hold is False
    assert decision.clipped_joints == ("shoulder_pan", "gripper")
    assert decision.rate_limited_joints == ("shoulder_pan", "gripper")
    np.testing.assert_array_equal(decision.target, np.array([1, 0, 0, 0, 0, 51]))


def test_action_outside_public_driver_domain_is_rejected() -> None:
    safety = _hardware_safety()
    calibrated = {
        name: safety.JointRange(-30.0, 30.0) for name in JOINTS[:-1]
    } | {"gripper": safety.JointRange(10.0, 90.0)}
    public = {
        name: safety.JointRange(-180.0, 180.0) for name in JOINTS[:-1]
    } | {"gripper": safety.JointRange(0.0, 100.0)}
    envelope = safety.SafetyEnvelope(JOINTS, calibrated, normalized_ranges=public)

    decision = envelope.evaluate(
        np.array([[180.001, 0.0, 0.0, 0.0, 0.0, 50.0]]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
    )

    assert decision.hold is True
    assert decision.reason == "outside_normalized_range"


def test_vendor_calibration_is_converted_to_driver_public_units() -> None:
    safety = _hardware_safety()
    calibration = {
        name: {"range_min": 1000, "range_max": 3000} for name in JOINTS[:-1]
    } | {"gripper": {"range_min": 1600, "range_max": 3400}}

    ranges = safety.ranges_from_vendor_calibration(JOINTS, calibration)

    expected_body_limit = 1000.0 * 360.0 / 4095.0
    assert ranges["shoulder_pan"].lower == pytest.approx(-expected_body_limit)
    assert ranges["shoulder_pan"].upper == pytest.approx(expected_body_limit)
    assert ranges["gripper"] == safety.JointRange(0.0, 100.0)


@pytest.mark.parametrize(
    "override",
    [
        {"move_time_ms": 199},
        {"watchdog_seconds": 0.501},
        {"timeout_limit": 4},
        {"duration_seconds": 90.001},
        {"chunk_limit": 21},
    ],
)
def test_session_config_rejects_values_looser_than_safety_caps(
    override: dict[str, float | int],
) -> None:
    safety = _hardware_safety()

    with pytest.raises(ValueError):
        safety.SafetySessionConfig(**override)


@pytest.mark.parametrize(
    "override",
    [
        {"move_time_ms": True},
        {"return_move_time_ms": 199},
        {"watchdog_seconds": 0.0},
        {"watchdog_seconds": np.nan},
        {"timeout_limit": 0},
        {"timeout_limit": True},
        {"duration_seconds": 0.0},
        {"duration_seconds": np.inf},
        {"chunk_limit": 0},
        {"chunk_limit": 1.5},
        {"no_motion": "yes"},
    ],
)
def test_session_config_rejects_invalid_or_nonpositive_values(
    override: dict[str, object],
) -> None:
    safety = _hardware_safety()

    with pytest.raises((TypeError, ValueError)):
        safety.SafetySessionConfig(**override)


def test_no_motion_session_makes_no_actuator_or_torque_writes() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    session = safety.SafetySession(robot, _envelope(safety), safety.SafetySessionConfig())

    with session:
        decision = session.process_chunk(
            np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 55.0]])
        )

    np.testing.assert_array_equal(
        decision.target,
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 51.0]),
    )
    assert robot.writes == []
    assert robot.disable_calls == 0
    assert robot.close_calls == 1


def test_motion_session_writes_only_the_enveloped_target_at_move_floor() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    config = safety.SafetySessionConfig(no_motion=False, move_time_ms=200)
    session = safety.SafetySession(robot, _envelope(safety), config)

    with session:
        decision = session.process_chunk(
            np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 55.0]])
        )
        first_write = robot.writes[0]

    np.testing.assert_array_equal(first_write[0], decision.target)
    assert first_write[1] == 200


def test_rejected_motion_chunk_holds_the_read_back_position() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    session = safety.SafetySession(
        robot,
        _envelope(safety),
        safety.SafetySessionConfig(no_motion=False),
    )

    with session:
        decision = session.process_chunk(
            np.array([[np.nan, 0.0, 0.0, 0.0, 0.0, 50.0]])
        )
        hold_write = robot.writes[0]

    assert decision.hold is True
    np.testing.assert_array_equal(hold_write[0], np.array([0, 0, 0, 0, 0, 50]))
    assert hold_write[1] == 200


def test_third_consecutive_watchdog_timeout_holds_and_stops_session() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    session = safety.SafetySession(
        robot,
        _envelope(safety),
        safety.SafetySessionConfig(no_motion=False),
    )

    with session:
        assert session.record_timeout() is False
        assert session.record_timeout() is False
        assert session.record_timeout() is True
        writes_at_stop = list(robot.writes)

    assert session.stop_reason == "watchdog_timeout"
    assert len(writes_at_stop) == 3
    for target, move_time_ms in writes_at_stop:
        np.testing.assert_array_equal(target, np.array([0, 0, 0, 0, 0, 50]))
        assert move_time_ms == 200


def test_fresh_chunk_resets_consecutive_timeout_count() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    session = safety.SafetySession(robot, _envelope(safety), safety.SafetySessionConfig())

    with session:
        assert session.record_timeout() is False
        assert session.record_timeout() is False
        session.process_chunk(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]))
        assert session.record_timeout() is False
        assert session.record_timeout() is False
        assert session.stop_reason is None
        assert session.record_timeout() is True

    assert session.stop_reason == "watchdog_timeout"


def test_chunk_cap_stops_after_the_last_allowed_chunk() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    session = safety.SafetySession(
        robot,
        _envelope(safety),
        safety.SafetySessionConfig(no_motion=False, chunk_limit=2),
    )
    chunk = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]])

    with session:
        session.process_chunk(chunk)
        assert session.stop_reason is None
        session.process_chunk(chunk)
        assert session.stop_reason == "chunk_limit"
        with pytest.raises(RuntimeError, match="already stopped"):
            session.process_chunk(chunk)

    assert session.chunk_count == 2


def test_no_motion_run_is_duration_capped_not_motion_chunk_capped() -> None:
    safety = _hardware_safety()
    session = safety.SafetySession(
        _FakeRobot(),
        _envelope(safety),
        safety.SafetySessionConfig(no_motion=True, chunk_limit=2),
    )
    chunk = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]])

    with session:
        session.process_chunk(chunk)
        session.process_chunk(chunk)
        session.process_chunk(chunk)
        assert session.stop_reason is None

    assert session.chunk_count == 3


def test_duration_cap_uses_monotonic_session_time() -> None:
    safety = _hardware_safety()
    clock = _FakeClock()
    session = safety.SafetySession(
        _FakeRobot(),
        _envelope(safety),
        safety.SafetySessionConfig(duration_seconds=1.0),
        clock=clock,
    )

    with session:
        clock.now = 0.999
        assert session.check_limits() is False
        clock.now = 1.0
        assert session.check_limits() is True

    assert session.stop_reason == "duration_limit"


def test_chunk_arriving_at_duration_boundary_stops_without_error_or_write() -> None:
    safety = _hardware_safety()
    clock = _FakeClock()
    robot = _FakeRobot()
    session = safety.SafetySession(
        robot,
        _envelope(safety),
        safety.SafetySessionConfig(duration_seconds=1.0),
        clock=clock,
    )

    with session:
        clock.now = 1.0
        decision = session.process_chunk(
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]])
        )

    assert decision.hold is True
    assert decision.reason == "duration_limit"
    assert session.stop_reason == "duration_limit"
    assert session.telemetry.chunks == 0
    assert robot.writes == []


def test_keyboard_interrupt_holds_returns_to_start_and_disables_torque() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    session = safety.SafetySession(
        robot,
        _envelope(safety),
        safety.SafetySessionConfig(no_motion=False),
    )

    with pytest.raises(KeyboardInterrupt):
        with session:
            session.process_chunk(
                np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 55.0]])
            )
            raise KeyboardInterrupt

    assert robot.events == [
        "prepare_motion:200",
        "write:200",
        "write:200",
        "write:1000",
        "disable_torque",
        "close",
    ]
    np.testing.assert_array_equal(robot.writes[1][0], np.array([1, 0, 0, 0, 0, 51]))
    np.testing.assert_array_equal(robot.writes[2][0], np.array([0, 0, 0, 0, 0, 50]))


def test_cleanup_still_returns_disables_and_closes_when_hold_write_fails() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    robot.fail_write_numbers.add(2)
    session = safety.SafetySession(
        robot,
        _envelope(safety),
        safety.SafetySessionConfig(no_motion=False),
    )

    with pytest.raises(BaseExceptionGroup, match="hardware cleanup failed"):
        with session:
            session.process_chunk(
                np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 55.0]])
            )

    assert robot.events == [
        "prepare_motion:200",
        "write:200",
        "write:200",
        "write:1000",
        "disable_torque",
        "close",
    ]
    assert robot.disable_calls == 1
    assert robot.close_calls == 1


def test_invalid_start_pose_fails_before_motion_and_closes_io() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    robot.positions[0] = 101.0
    session = safety.SafetySession(
        robot,
        _envelope(safety),
        safety.SafetySessionConfig(no_motion=False),
    )

    with pytest.raises(ValueError, match="start pose"):
        with session:
            pytest.fail("invalid start pose entered session body")

    assert robot.writes == []
    assert robot.disable_calls == 0
    assert robot.close_calls == 1


def test_motion_start_pose_must_already_be_inside_tightened_envelope() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    robot.positions[0] = 90.0
    session = safety.SafetySession(
        robot,
        _envelope(safety),
        safety.SafetySessionConfig(no_motion=False),
    )

    with pytest.raises(ValueError, match="start pose"):
        with session:
            pytest.fail("endpoint-adjacent start pose entered motion session")

    assert robot.writes == []
    assert robot.disable_calls == 0
    assert robot.close_calls == 1


def test_motion_is_armed_only_after_safe_start_pose_validation() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    session = safety.SafetySession(
        robot,
        _envelope(safety),
        safety.SafetySessionConfig(no_motion=False),
    )

    with session:
        assert robot.events == ["prepare_motion:200"]

    assert robot.prepare_calls == 1


def test_partial_arming_failure_disables_torque_and_closes_io() -> None:
    safety = _hardware_safety()
    robot = _FakeRobot()
    robot.fail_prepare = True
    session = safety.SafetySession(
        robot,
        _envelope(safety),
        safety.SafetySessionConfig(no_motion=False),
    )

    with pytest.raises(RuntimeError, match="partial arming"):
        with session:
            pytest.fail("failed arming entered session body")

    assert robot.events == ["prepare_motion:200", "disable_torque", "close"]


def test_standalone_client_help_exposes_fail_closed_modes_and_caps() -> None:
    script = Path("examples/bring_your_own_robot/hiwonder_so101_client.py")

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--no-motion" in completed.stdout
    assert "--single-action" in completed.stdout
    assert "--continuous" in completed.stdout
    assert "--duration-seconds" in completed.stdout
    assert "--chunk-limit" in completed.stdout
    assert "--hardware-safety-profile" in completed.stdout


def test_motion_profile_requires_exact_safe_register_readback_and_torque_off() -> None:
    safety = _hardware_safety()
    expected = {
        register: {name: 10 for name in JOINTS}
        for register in safety.REQUIRED_SAFETY_REGISTERS
    }
    profile = safety.HardwareSafetyProfile(
        robot_serial="5C82107541",
        verified_at="2026-09-02T00:00:00Z",
        procedure="Operator-verified with Hiwonder ServoStudio and physical power cut tested.",
        expected_registers=expected,
    )
    matching = {**expected, "Torque_Enable": {name: 0 for name in JOINTS}}

    assert profile.verify_readback(JOINTS, matching).ok is True

    wrong = {register: dict(values) for register, values in matching.items()}
    wrong["Maximum_Acceleration"]["elbow_flex"] = 11
    mismatch = profile.verify_readback(JOINTS, wrong)
    assert mismatch.ok is False
    assert mismatch.mismatches == (
        "Maximum_Acceleration.elbow_flex: expected 10, read 11",
    )

    enabled = {register: dict(values) for register, values in matching.items()}
    enabled["Torque_Enable"]["shoulder_pan"] = 1
    torque = profile.verify_readback(JOINTS, enabled)
    assert torque.ok is False
    assert torque.mismatches == ("Torque_Enable.shoulder_pan: expected 0, read 1",)


def test_motion_profile_rejects_incomplete_or_unattested_data() -> None:
    safety = _hardware_safety()
    complete = {
        register: {name: 10 for name in JOINTS}
        for register in safety.REQUIRED_SAFETY_REGISTERS
    }

    missing = dict(complete)
    missing.pop("Maximum_Velocity_Limit")
    with pytest.raises(ValueError, match="required safety registers"):
        safety.HardwareSafetyProfile("serial", "date", "procedure", missing)

    with pytest.raises(ValueError, match="procedure"):
        safety.HardwareSafetyProfile("serial", "date", "", complete)

    non_integer = {register: dict(values) for register, values in complete.items()}
    non_integer["Torque_Limit"]["gripper"] = True
    with pytest.raises(ValueError, match="non-negative integers"):
        safety.HardwareSafetyProfile("serial", "date", "procedure", non_integer)


def test_hardware_profile_loader_requires_exact_json_schema(tmp_path: Path) -> None:
    safety = _hardware_safety()
    expected = {
        register: {name: 10 for name in JOINTS}
        for register in safety.REQUIRED_SAFETY_REGISTERS
    }
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "robot_serial": "5C82107541",
                "verified_at": "2026-09-02T12:00:00Z",
                "procedure": "Operator verified the low limits and physical power cut.",
                "expected_registers": expected,
            }
        ),
        encoding="utf-8",
    )

    profile = safety.load_hardware_safety_profile(path)

    assert profile.robot_serial == "5C82107541"
    assert profile.expected_registers == expected

    document = json.loads(path.read_text(encoding="utf-8"))
    document["unreviewed_override"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        safety.load_hardware_safety_profile(path)


def test_hardware_profile_loader_rejects_non_utc_attestation(tmp_path: Path) -> None:
    safety = _hardware_safety()
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "robot_serial": "5C82107541",
                "verified_at": "2026-09-02 12:00:00",
                "procedure": "Operator verified the low limits and physical power cut.",
                "expected_registers": {
                    register: {name: 10 for name in JOINTS}
                    for register in safety.REQUIRED_SAFETY_REGISTERS
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="UTC"):
        safety.load_hardware_safety_profile(path)


def test_no_motion_vendor_adapter_closes_cameras_and_bus_without_writes() -> None:
    client = _hiwonder_client()
    robot = _FakeVendorRobot()
    adapter = client.HiwonderSO101IO(robot, joint_names=JOINTS)

    adapter.close()

    assert robot.bus.sync_writes == []
    assert robot.bus.enable_calls == 0
    assert robot.bus.disable_calls == 0
    assert robot.bus.disconnect_calls == [False]
    assert [camera.disconnect_calls for camera in robot.cameras.values()] == [1, 1]


def test_vendor_adapter_reads_positions_in_policy_joint_order() -> None:
    client = _hiwonder_client()
    robot = _FakeVendorRobot()
    robot.bus.registers["Present_Position"] = {
        "gripper": 60,
        "wrist_roll": 5,
        "wrist_flex": 4,
        "elbow_flex": 3,
        "shoulder_lift": 2,
        "shoulder_pan": 1,
    }
    adapter = client.HiwonderSO101IO(robot, joint_names=JOINTS)

    positions = adapter.read_positions()

    np.testing.assert_array_equal(positions, np.array([1, 2, 3, 4, 5, 60]))


def test_vendor_adapter_refuses_motion_without_operator_safety_profile() -> None:
    client = _hiwonder_client()
    robot = _FakeVendorRobot()
    adapter = client.HiwonderSO101IO(robot, joint_names=JOINTS)

    with pytest.raises(RuntimeError, match="hardware safety profile"):
        adapter.prepare_motion(move_time_ms=200)

    assert robot.bus.enable_calls == 0
    assert robot.bus.sync_writes == []


def test_vendor_adapter_refuses_limit_mismatch_before_torque_enable() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    profile, registers = _matching_profile(safety)
    registers["Maximum_Acceleration"]["elbow_flex"] = 11
    robot = _FakeVendorRobot()
    robot.bus.registers.update(registers)
    adapter = client.HiwonderSO101IO(
        robot,
        joint_names=JOINTS,
        safety_profile=profile,
        robot_serial="5C82107541",
    )

    with pytest.raises(RuntimeError, match="Maximum_Acceleration.elbow_flex"):
        adapter.prepare_motion(move_time_ms=200)

    assert robot.bus.enable_calls == 0


def test_vendor_adapter_enables_torque_only_after_exact_limit_readback() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    profile, registers = _matching_profile(safety)
    robot = _FakeVendorRobot()
    robot.bus.registers.update(registers)
    adapter = client.HiwonderSO101IO(
        robot,
        joint_names=JOINTS,
        safety_profile=profile,
        robot_serial="5C82107541",
    )

    adapter.prepare_motion(move_time_ms=200)

    assert robot.bus.enable_calls == 1
    assert adapter.armed is True


def test_vendor_adapter_rejects_position_write_until_armed() -> None:
    client = _hiwonder_client()
    robot = _FakeVendorRobot()
    adapter = client.HiwonderSO101IO(robot, joint_names=JOINTS)

    with pytest.raises(RuntimeError, match="not armed"):
        adapter.write_positions(np.zeros(6), move_time_ms=200)

    assert robot.bus.sync_writes == []


def test_position_write_uses_public_units_and_dwell_not_goal_time_register() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    profile, registers = _matching_profile(safety)
    robot = _FakeVendorRobot()
    robot.bus.registers.update(registers)
    sleeps: list[float] = []
    adapter = client.HiwonderSO101IO(
        robot,
        joint_names=JOINTS,
        safety_profile=profile,
        robot_serial="5C82107541",
        sleep=sleeps.append,
    )
    adapter.prepare_motion(move_time_ms=200)

    adapter.write_positions(np.array([1, 2, 3, 4, 5, 60]), move_time_ms=200)

    assert robot.bus.sync_writes == [
        (
            "Goal_Position",
            {
                "shoulder_pan": 1.0,
                "shoulder_lift": 2.0,
                "elbow_flex": 3.0,
                "wrist_flex": 4.0,
                "wrist_roll": 5.0,
                "gripper": 60.0,
            },
        )
    ]
    assert sleeps == [0.2]


def test_position_write_records_timestamp_before_bus_write_and_dwell() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    profile, registers = _matching_profile(safety)
    clock = _FakeClock()
    robot = _FakeVendorRobot()
    robot.bus.registers.update(registers)

    def advance(seconds: float) -> None:
        clock.now += seconds

    adapter = client.HiwonderSO101IO(
        robot,
        joint_names=JOINTS,
        safety_profile=profile,
        robot_serial="5C82107541",
        sleep=advance,
        clock=clock,
    )
    adapter.prepare_motion(move_time_ms=200)
    clock.now = 1.25

    adapter.write_positions(np.array([1, 2, 3, 4, 5, 60]), move_time_ms=200)

    assert adapter.last_write_monotonic == pytest.approx(1.25)
    assert clock.now == pytest.approx(1.45)


@pytest.mark.parametrize(
    ("target", "move_time_ms"),
    [
        (np.zeros((1, 6)), 200),
        (np.array([np.nan, 0, 0, 0, 0, 50]), 200),
        (np.zeros(6), 199),
    ],
)
def test_armed_vendor_adapter_rejects_malformed_write(
    target: np.ndarray, move_time_ms: int
) -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    profile, registers = _matching_profile(safety)
    robot = _FakeVendorRobot()
    robot.bus.registers.update(registers)
    adapter = client.HiwonderSO101IO(
        robot,
        joint_names=JOINTS,
        safety_profile=profile,
        robot_serial="5C82107541",
    )
    adapter.prepare_motion(move_time_ms=200)

    with pytest.raises(ValueError):
        adapter.write_positions(target, move_time_ms=move_time_ms)

    assert robot.bus.sync_writes == []


def test_vendor_adapter_disables_torque_with_readback_verification() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    profile, registers = _matching_profile(safety)
    robot = _FakeVendorRobot()
    robot.bus.registers.update(registers)
    adapter = client.HiwonderSO101IO(
        robot,
        joint_names=JOINTS,
        safety_profile=profile,
        robot_serial="5C82107541",
    )
    adapter.prepare_motion(move_time_ms=200)

    adapter.disable_torque()

    assert robot.bus.disable_calls == 1
    assert adapter.armed is False


def test_closing_armed_vendor_adapter_disables_before_port_close() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    profile, registers = _matching_profile(safety)
    robot = _FakeVendorRobot()
    robot.bus.registers.update(registers)
    adapter = client.HiwonderSO101IO(
        robot,
        joint_names=JOINTS,
        safety_profile=profile,
        robot_serial="5C82107541",
    )
    adapter.prepare_motion(move_time_ms=200)

    adapter.close()

    assert robot.bus.disable_calls == 1
    assert adapter.armed is False
    assert robot.bus.disconnect_calls == [False]


def test_close_releases_cameras_and_port_even_if_torque_disable_errors() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    profile, registers = _matching_profile(safety)
    robot = _FakeVendorRobot()
    robot.bus.registers.update(registers)
    adapter = client.HiwonderSO101IO(
        robot,
        joint_names=JOINTS,
        safety_profile=profile,
        robot_serial="5C82107541",
    )
    adapter.prepare_motion(move_time_ms=200)
    robot.bus.fail_disable = True

    with pytest.raises(BaseExceptionGroup, match="hardware close failed"):
        adapter.close()

    assert robot.bus.disconnect_calls == [False]
    assert [camera.disconnect_calls for camera in robot.cameras.values()] == [1, 1]


def test_vendor_observation_rejects_an_all_black_camera_frame() -> None:
    client = _hiwonder_client()
    robot = _FakeVendorRobot()
    robot.frames["wrist_camera"] = np.zeros((4, 5, 3), dtype=np.uint8)
    adapter = client.HiwonderSO101IO(robot, joint_names=JOINTS)

    with pytest.raises(RuntimeError, match="wrist_camera.*all black"):
        adapter.read_observation()


def test_vendor_observation_requires_fresh_rgb_frames_from_both_cameras() -> None:
    client = _hiwonder_client()
    robot = _FakeVendorRobot()
    adapter = client.HiwonderSO101IO(robot, joint_names=JOINTS)

    observation = adapter.read_observation()

    assert observation["wrist_camera"].shape == (4, 5, 3)
    assert observation["top_camera"].shape == (4, 5, 3)
    assert robot.bus.sync_writes == []


def test_four_rpc_client_exchanges_real_lerobot_messages_over_grpc() -> None:
    from concurrent import futures
    import pickle

    import grpc
    import torch
    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
    from lerobot.transport import services_pb2, services_pb2_grpc

    class RecordingService(services_pb2_grpc.AsyncInferenceServicer):
        def __init__(self) -> None:
            self.ready_calls = 0
            self.policy_config = None
            self.observation = None

        def Ready(self, request, context):  # noqa: N802
            self.ready_calls += 1
            return services_pb2.Empty()

        def SendPolicyInstructions(self, request, context):  # noqa: N802
            self.policy_config = pickle.loads(request.data)
            return services_pb2.Empty()

        def SendObservations(self, request_iterator, context):  # noqa: N802
            payload = b"".join(message.data for message in request_iterator)
            self.observation = pickle.loads(payload)
            return services_pb2.Empty()

        def GetActions(self, request, context):  # noqa: N802
            action = TimedAction(
                timestamp=10.0,
                timestep=7,
                action=torch.tensor([1, 2, 3, 4, 5, 60], dtype=torch.float32),
            )
            return services_pb2.Actions(data=pickle.dumps([action]))

    service = RecordingService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    client = _hiwonder_client()
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"{name}.pos" for name in JOINTS],
        },
        "observation.images.wrist_camera": {"dtype": "video", "shape": (3, 4, 5)},
        "observation.images.top_camera": {"dtype": "video", "shape": (3, 4, 5)},
    }
    transport = client.LeRobotFourRPCClient(
        server_address=f"127.0.0.1:{port}",
        checkpoint="/local/reviewed/checkpoint",
        lerobot_features=features,
        rename_map={
            "observation.images.wrist_camera": "observation.images.camera1",
            "observation.images.top_camera": "observation.images.camera2",
        },
        rpc_timeout_seconds=0.5,
    )
    observation = {
        **{f"{name}.pos": value for name, value in zip(JOINTS, [0, 0, 0, 0, 0, 50], strict=True)},
        "wrist_camera": np.ones((4, 5, 3), dtype=np.uint8),
        "top_camera": np.ones((4, 5, 3), dtype=np.uint8),
        "task": "hold position",
    }

    try:
        transport.start()
        transport.send_observation(observation, timestep=7, timestamp=9.5)
        chunk = transport.receive_action_chunk()
    finally:
        transport.close()
        server.stop(grace=0).wait()

    assert service.ready_calls == 1
    assert isinstance(service.policy_config, RemotePolicyConfig)
    assert service.policy_config.actions_per_chunk == 1
    assert service.policy_config.rename_map == transport.rename_map
    assert isinstance(service.observation, TimedObservation)
    assert service.observation.must_go is True
    np.testing.assert_array_equal(chunk, np.array([[1, 2, 3, 4, 5, 60]]))


def test_four_rpc_deadline_is_reported_as_watchdog_timeout() -> None:
    client_module = _hiwonder_client()

    class Deadline(Exception):
        def code(self):
            return "deadline"

    class Stub:
        def GetActions(self, _request, *, timeout):  # noqa: N802
            assert timeout == 0.5
            raise Deadline

    transport = object.__new__(client_module.LeRobotFourRPCClient)
    transport.started = True
    transport.rpc_timeout_seconds = 0.5
    transport.stub = Stub()
    transport._services_pb2 = SimpleNamespace(Empty=lambda: object())
    transport._grpc = SimpleNamespace(RpcError=Deadline, StatusCode=SimpleNamespace(DEADLINE_EXCEEDED="deadline"))

    assert transport.receive_action_chunk() is None


def test_four_rpc_non_deadline_error_is_not_suppressed() -> None:
    client_module = _hiwonder_client()

    class Unavailable(Exception):
        def code(self):
            return "unavailable"

    class Stub:
        def GetActions(self, _request, *, timeout):  # noqa: N802
            raise Unavailable

    transport = object.__new__(client_module.LeRobotFourRPCClient)
    transport.started = True
    transport.rpc_timeout_seconds = 0.5
    transport.stub = Stub()
    transport._services_pb2 = SimpleNamespace(Empty=lambda: object())
    transport._grpc = SimpleNamespace(RpcError=Unavailable, StatusCode=SimpleNamespace(DEADLINE_EXCEEDED="deadline"))

    with pytest.raises(Unavailable):
        transport.receive_action_chunk()


def test_four_rpc_start_uses_setup_timeout_not_action_watchdog() -> None:
    client_module = _hiwonder_client()
    calls: list[tuple[str, float]] = []

    class Future:
        def result(self, *, timeout):
            calls.append(("channel", timeout))

    class Grpc:
        @staticmethod
        def channel_ready_future(_channel):
            return Future()

    class Stub:
        def Ready(self, _request, *, timeout):  # noqa: N802
            calls.append(("ready", timeout))

        def SendPolicyInstructions(self, _request, *, timeout):  # noqa: N802
            calls.append(("setup", timeout))

    transport = object.__new__(client_module.LeRobotFourRPCClient)
    transport._grpc = Grpc()
    transport.channel = object()
    transport.stub = Stub()
    transport._services_pb2 = SimpleNamespace(
        Empty=lambda: object(),
        PolicySetup=lambda **kwargs: kwargs,
    )
    transport._policy_config = {"checkpoint": "local"}
    transport.rpc_timeout_seconds = 0.5
    transport.setup_timeout_seconds = 600.0
    transport.started = False

    transport.start()

    assert calls == [("channel", 600.0), ("ready", 600.0), ("setup", 600.0)]
    assert transport.started is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"server_address": "192.168.1.50:8080"},
        {"server_address": "127.0.0.1:8080", "rpc_timeout_seconds": 0.501},
        {"server_address": "127.0.0.1:8080", "rpc_timeout_seconds": 0.0},
        {"server_address": "127.0.0.1:8080", "checkpoint": ""},
    ],
)
def test_four_rpc_client_rejects_remote_or_loose_watchdog_configuration(
    kwargs: dict[str, object],
) -> None:
    client = _hiwonder_client()
    config = {
        "server_address": "127.0.0.1:8080",
        "checkpoint": "/local/checkpoint",
        "lerobot_features": {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": [f"{name}.pos" for name in JOINTS],
            }
        },
        "rename_map": {},
        "rpc_timeout_seconds": 0.5,
    }
    config.update(kwargs)

    with pytest.raises(ValueError):
        client.LeRobotFourRPCClient(**config)


def test_session_telemetry_counts_chunks_clips_rate_limits_rejections_and_timeouts() -> None:
    safety = _hardware_safety()
    session = safety.SafetySession(
        _FakeRobot(),
        _envelope(safety),
        safety.SafetySessionConfig(),
    )

    with session:
        session.process_chunk(np.array([[90.0, 0.0, 0.0, 0.0, 0.0, 90.0]]))
        session.process_chunk(np.array([[np.nan, 0.0, 0.0, 0.0, 0.0, 50.0]]))
        session.record_timeout()

    assert session.telemetry.chunks == 2
    assert session.telemetry.clipped_joints == 1
    assert session.telemetry.rate_limited_joints == 2
    assert session.telemetry.rejected_chunks == 1
    assert session.telemetry.holds == 2
    assert session.telemetry.timeouts == 1
    assert session.telemetry.rejection_reasons == {"non_finite": 1}


def test_no_motion_control_loop_runs_to_duration_with_latency_summary() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    clock = _FakeClock()
    robot = _FakeVendorRobot()
    adapter = client.HiwonderSO101IO(robot, joint_names=JOINTS)
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]])
    transport = _FakeTransport(clock, [action.copy() for _ in range(5)])

    result = client.run_control_loop(
        adapter=adapter,
        transport=transport,
        envelope=_envelope(safety),
        config=safety.SafetySessionConfig(no_motion=True, duration_seconds=1.0),
        task="hold position",
        fps=5,
        clock=clock,
        wall_clock=lambda: 1000.0 + clock.now,
        sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
    )

    assert result.stop_reason == "duration_limit"
    assert result.chunks == 5
    assert result.observation_to_chunk_median_ms == pytest.approx(50.0)
    assert result.observation_to_chunk_p95_ms == pytest.approx(50.0)
    assert result.observation_to_motion_median_ms is None
    assert result.camera_sample_fps == pytest.approx(5.0)
    assert len(transport.sent) == 5
    assert all(sent[0]["task"] == "hold position" for sent in transport.sent)
    assert transport.closed is True
    assert robot.bus.sync_writes == []


def test_control_loop_closes_transport_when_start_fails() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    adapter = client.HiwonderSO101IO(_FakeVendorRobot(), joint_names=JOINTS)

    class FailingTransport:
        closed = False

        def start(self) -> None:
            raise RuntimeError("server unavailable")

        def close(self) -> None:
            self.closed = True

    transport = FailingTransport()

    with pytest.raises(RuntimeError, match="server unavailable"):
        client.run_control_loop(
            adapter=adapter,
            transport=transport,
            envelope=_envelope(safety),
            config=safety.SafetySessionConfig(),
            task="hold position",
        )

    assert transport.closed is True
    assert adapter.robot.bus.disconnect_calls == [False]


def test_control_loop_finishes_server_setup_before_enabling_torque() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    profile, registers = _matching_profile(safety)
    robot = _FakeVendorRobot()
    robot.bus.registers.update(registers)
    adapter = client.HiwonderSO101IO(
        robot,
        joint_names=JOINTS,
        safety_profile=profile,
        robot_serial="5C82107541",
    )

    class FailingTransport:
        closed = False

        def start(self) -> None:
            assert robot.bus.enable_calls == 0
            raise RuntimeError("model setup failed")

        def close(self) -> None:
            self.closed = True

    transport = FailingTransport()

    with pytest.raises(RuntimeError, match="model setup failed"):
        client.run_control_loop(
            adapter=adapter,
            transport=transport,
            envelope=_envelope(safety),
            config=safety.SafetySessionConfig(no_motion=False),
            task="hold position",
        )

    assert robot.bus.enable_calls == 0
    assert robot.bus.disable_calls == 0
    assert robot.bus.disconnect_calls == [False]
    assert transport.closed is True


def test_motion_latency_ends_at_first_motor_write_not_after_dwell() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()
    profile, registers = _matching_profile(safety)
    clock = _FakeClock()
    robot = _FakeVendorRobot()
    robot.bus.registers.update(registers)
    adapter = client.HiwonderSO101IO(
        robot,
        joint_names=JOINTS,
        safety_profile=profile,
        robot_serial="5C82107541",
        sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        clock=clock,
    )
    action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]])
    transport = _FakeTransport(clock, [action])

    result = client.run_control_loop(
        adapter=adapter,
        transport=transport,
        envelope=_envelope(safety),
        config=safety.SafetySessionConfig(no_motion=False, chunk_limit=1),
        task="hold position",
        fps=5,
        clock=clock,
        wall_clock=lambda: 1000.0 + clock.now,
        sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
    )

    assert result.observation_to_motion_median_ms == pytest.approx(50.0)
    assert result.observation_to_motion_p95_ms == pytest.approx(50.0)
    assert result.duration_seconds == pytest.approx(0.25)
    assert result.camera_sample_fps == pytest.approx(4.0)


def test_vendor_factory_rejects_wrong_checkout_before_import_or_device_access(
    tmp_path: Path,
) -> None:
    client = _hiwonder_client()

    with pytest.raises(FileNotFoundError, match="vendor checkout"):
        client.open_hiwonder_follower(
            vendor_root=tmp_path,
            follower_port="/dev/tty.never-opened",
            calibration_id="follower",
            wrist_camera=1,
            fixed_camera=2,
        )


def test_vendor_factory_uses_manual_read_only_connect_path(tmp_path: Path) -> None:
    client = _hiwonder_client()
    source = tmp_path / "src/lerobot"
    (source / "robots/so_follower").mkdir(parents=True)
    (source / "motors/hiwonder").mkdir(parents=True)
    (source / "robots/so_follower/so_follower.py").write_text("# test fixture\n")
    (source / "motors/hiwonder/hiwonder.py").write_text("# test fixture\n")

    class Bus(_FakeBus):
        def __init__(self) -> None:
            super().__init__()
            self.connect_calls: list[bool] = []
            self.is_connected = False

        def connect(self, handshake: bool = True) -> None:
            self.connect_calls.append(handshake)
            self.is_connected = True

    class Camera(_FakeCamera):
        def __init__(self) -> None:
            super().__init__()
            self.is_connected = False
            self.connect_calls = 0

        def connect(self) -> None:
            self.connect_calls += 1
            self.is_connected = True

    class CameraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class RobotConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class Robot:
        def __init__(self, config) -> None:
            self.config = config
            self.bus = Bus()
            self.cameras = {
                name: Camera() for name in config.kwargs["cameras"]
            }
            self.calibration = {name: object() for name in JOINTS}
            self.is_calibrated = True

        def get_observation(self):
            return {
                **{f"{name}.pos": value for name, value in zip(JOINTS, [0, 0, 0, 0, 0, 50], strict=True)},
                "wrist_camera": np.ones((480, 640, 3), dtype=np.uint8),
                "top_camera": np.ones((480, 640, 3), dtype=np.uint8),
            }

    adapter = client.open_hiwonder_follower(
        vendor_root=tmp_path,
        follower_port="/dev/tty.follower-only",
        calibration_id="hiwonder_follower",
        wrist_camera=1,
        fixed_camera=2,
        _vendor_loader=lambda _source: (CameraConfig, RobotConfig, Robot),
    )

    assert adapter.robot.bus.connect_calls == [True]
    assert adapter.robot.bus.sync_writes == []
    assert adapter.robot.bus.enable_calls == 0
    assert adapter.robot.bus.disable_calls == 0
    assert [camera.connect_calls for camera in adapter.robot.cameras.values()] == [1, 1]


def test_vendor_adapter_derives_envelope_from_loaded_calibration() -> None:
    client = _hiwonder_client()
    robot = _FakeVendorRobot()
    robot.calibration = {
        name: SimpleNamespace(range_min=1000, range_max=3000)
        for name in JOINTS
    }
    adapter = client.HiwonderSO101IO(robot, joint_names=JOINTS)

    ranges = adapter.calibration_ranges()

    assert ranges["shoulder_pan"].upper == pytest.approx(1000 * 360 / 4095)
    assert ranges["shoulder_pan"].lower == pytest.approx(-1000 * 360 / 4095)
    assert ranges["gripper"].lower == 0.0
    assert ranges["gripper"].upper == 100.0


def test_hardware_extra_provides_vendor_io_and_rpc_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["optional-dependencies"]["hardware"] == [
        "lerobot[async,hardware]==0.6.1; python_version >= '3.12'",
    ]


def test_hardware_telemetry_is_exclusive_private_and_rejects_payloads(
    tmp_path: Path,
) -> None:
    client = _hiwonder_client()
    path = tmp_path / "client.jsonl"

    with client.HardwareTelemetryRecorder(path) as recorder:
        recorder.write_event("session_start", {"mode": "no-motion"})
        recorder.write_event("session_result", {"observations": 12, "chunks": 11})
        with pytest.raises(ValueError, match="payload"):
            recorder.write_event("bad", {"action": [1, 2, 3]})

    assert os.stat(path).st_mode & 0o777 == 0o600
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["session_start", "session_result"]
    assert all("recorded_at" in record for record in records)
    with pytest.raises(FileExistsError):
        client.HardwareTelemetryRecorder(path)


def test_so101_feature_contract_maps_two_physical_cameras() -> None:
    client = _hiwonder_client()

    features, rename_map = client.so101_lerobot_contract()

    assert features["observation.state"] == {
        "dtype": "float32",
        "shape": (6,),
        "names": [f"{name}.pos" for name in JOINTS],
    }
    assert features["observation.images.wrist_camera"]["shape"] == (3, 480, 640)
    assert features["observation.images.top_camera"]["shape"] == (3, 480, 640)
    assert rename_map == {
        "observation.images.wrist_camera": "observation.images.camera1",
        "observation.images.top_camera": "observation.images.camera2",
    }


def test_so101_public_action_domain_is_explicit() -> None:
    safety = _hardware_safety()
    client = _hiwonder_client()

    ranges = client.so101_public_action_ranges()

    assert ranges == {
        **{name: safety.JointRange(-180.0, 180.0) for name in JOINTS[:-1]},
        "gripper": safety.JointRange(0.0, 100.0),
    }


def test_standalone_modes_resolve_to_non_overridable_safety_caps(tmp_path: Path) -> None:
    standalone = _standalone_client()
    safety = _hardware_safety()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "robot_serial": "5C82107541",
                "verified_at": "2026-09-02T12:00:00Z",
                "procedure": "Operator verified the low limits and physical power cut.",
                "expected_registers": {
                    register: {name: 10 for name in JOINTS}
                    for register in safety.REQUIRED_SAFETY_REGISTERS
                },
            }
        ),
        encoding="utf-8",
    )
    base = [
        "--vendor-root", str(tmp_path),
        "--follower-port", "/dev/tty.usbmodem5C821075411",
        "--calibration-id", "follower",
        "--robot-serial", "5C82107541",
        "--wrist-camera", "1",
        "--fixed-camera", "2",
        "--checkpoint", str(tmp_path),
        "--task", "hold position",
        "--telemetry", str(tmp_path / "telemetry.jsonl"),
    ]

    no_motion = standalone.resolve_run_config(standalone.build_parser().parse_args(base))
    assert no_motion.no_motion is True
    assert no_motion.duration_seconds == 60.0

    continuous = standalone.resolve_run_config(
        standalone.build_parser().parse_args(
            ["--continuous", "--hardware-safety-profile", str(profile_path), *base]
        )
    )
    assert continuous.no_motion is False
    assert continuous.duration_seconds == 90.0
    assert continuous.chunk_limit == 20

    single = standalone.resolve_run_config(
        standalone.build_parser().parse_args(
            ["--single-action", "--hardware-safety-profile", str(profile_path), *base]
        )
    )
    assert single.no_motion is False
    assert single.chunk_limit == 1


def test_standalone_motion_requires_profile_and_matching_serial(tmp_path: Path) -> None:
    standalone = _standalone_client()
    arguments = standalone.build_parser().parse_args(
        [
            "--single-action",
            "--vendor-root", str(tmp_path),
            "--follower-port", "/dev/tty.usbmodem5C821075411",
            "--calibration-id", "follower",
            "--robot-serial", "5C82107541",
            "--wrist-camera", "1",
            "--fixed-camera", "2",
            "--checkpoint", str(tmp_path),
            "--task", "hold position",
            "--telemetry", str(tmp_path / "telemetry.jsonl"),
        ]
    )

    with pytest.raises(ValueError, match="safety profile"):
        standalone.resolve_run_config(arguments)


def test_lightweight_protocol_shim_is_pickle_compatible_with_lerobot() -> None:
    import pickle

    import torch
    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction

    make_payload = r'''
import base64
import pickle
import sys
import types
from mlx_smolvla.hiwonder_client import install_lerobot_protocol_shim
root = types.ModuleType("lerobot")
root.__path__ = []
sys.modules["lerobot"] = root
RemotePolicyConfig, _, _ = install_lerobot_protocol_shim()
value = RemotePolicyConfig("smolvla", "/checkpoint", {"state": {}}, 1, "cpu", {"a": "b"})
print(base64.b64encode(pickle.dumps(value)).decode())
'''
    made = subprocess.run(
        [sys.executable, "-c", make_payload],
        check=False,
        capture_output=True,
        text=True,
    )
    assert made.returncode == 0, made.stderr
    config = pickle.loads(base64.b64decode(made.stdout.strip()))
    assert isinstance(config, RemotePolicyConfig)
    assert config.rename_map == {"a": "b"}

    action_payload = base64.b64encode(
        pickle.dumps(TimedAction(1.25, 7, torch.tensor([1.0, 2.0])))
    ).decode()
    read_payload = r'''
import base64
import json
import pickle
import sys
import types
from mlx_smolvla.hiwonder_client import install_lerobot_protocol_shim
root = types.ModuleType("lerobot")
root.__path__ = []
sys.modules["lerobot"] = root
install_lerobot_protocol_shim()
value = pickle.loads(base64.b64decode(sys.argv[1]))
print(json.dumps({"timestep": value.get_timestep(), "action": value.get_action().tolist()}))
'''
    read = subprocess.run(
        [sys.executable, "-c", read_payload, action_payload],
        check=False,
        capture_output=True,
        text=True,
    )
    assert read.returncode == 0, read.stderr
    assert json.loads(read.stdout) == {"timestep": 7, "action": [1.0, 2.0]}
