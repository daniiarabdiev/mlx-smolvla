#!/usr/bin/env python3
"""Fail-closed Hiwonder SO-101 client for mlx-smolvla's audited RPC service."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
from typing import Sequence

from mlx_smolvla.hardware_safety import (
    SafetyEnvelope,
    SafetySessionConfig,
    load_hardware_safety_profile,
)
from mlx_smolvla.hiwonder_client import (
    HardwareTelemetryRecorder,
    LeRobotFourRPCClient,
    open_hiwonder_follower,
    run_control_loop,
    so101_lerobot_contract,
    so101_public_action_ranges,
    validate_physical_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Supervised Hiwonder SO-101 client. The default and --no-motion modes "
            "never issue actuator or torque writes."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-motion", dest="mode", action="store_const", const="no-motion")
    mode.add_argument("--single-action", dest="mode", action="store_const", const="single-action")
    mode.add_argument("--continuous", dest="mode", action="store_const", const="continuous")
    parser.set_defaults(mode="no-motion")
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--robot-serial", required=True)
    parser.add_argument("--wrist-camera", type=int, required=True)
    parser.add_argument("--fixed-camera", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--chunk-limit", type=int, default=20)
    parser.add_argument("--hardware-safety-profile", type=Path)
    parser.add_argument(
        "--arming-mode",
        choices=("explicit-torque", "goal-write"),
        default="explicit-torque",
        help="verified controller arming behavior; goal-write must be selected explicitly",
    )
    parser.add_argument("--telemetry", type=Path, required=True)
    return parser


def resolve_run_config(arguments: argparse.Namespace) -> SafetySessionConfig:
    """Resolve each graduated mode to immutable safety ceilings."""

    if not arguments.task.strip():
        raise ValueError("task must be a non-empty string")
    if not arguments.robot_serial.strip():
        raise ValueError("robot serial must be a non-empty string")
    if arguments.robot_serial not in Path(arguments.follower_port).name:
        raise ValueError("follower port does not contain the verified robot serial")
    arguments.vendor_root = arguments.vendor_root.expanduser().resolve(strict=True)
    arguments.checkpoint = arguments.checkpoint.expanduser().resolve(strict=True)
    if not arguments.telemetry.expanduser().resolve().parent.is_dir():
        raise FileNotFoundError("hardware telemetry parent directory does not exist")

    no_motion = arguments.mode == "no-motion"
    if no_motion:
        duration = 60.0 if arguments.duration_seconds is None else arguments.duration_seconds
        if duration != 60.0:
            raise ValueError("the graduated no-motion run is fixed at 60 seconds")
        chunk_limit = arguments.chunk_limit
        stop_on_valid_action = False
    else:
        if arguments.hardware_safety_profile is None:
            raise ValueError("motion requires an operator-verified hardware safety profile")
        validate_physical_checkpoint(arguments.checkpoint)
        profile = load_hardware_safety_profile(arguments.hardware_safety_profile)
        if profile.robot_serial != arguments.robot_serial:
            raise ValueError("hardware safety profile robot serial does not match --robot-serial")
        arguments.hardware_safety_profile = (
            arguments.hardware_safety_profile.expanduser().resolve(strict=True)
        )
        duration = 90.0 if arguments.duration_seconds is None else arguments.duration_seconds
        chunk_limit = arguments.chunk_limit
        stop_on_valid_action = arguments.mode == "single-action"

    return SafetySessionConfig(
        no_motion=no_motion,
        move_time_ms=200,
        return_move_time_ms=1000,
        watchdog_seconds=0.5,
        timeout_limit=3,
        duration_seconds=duration,
        chunk_limit=chunk_limit,
        stop_on_valid_action=stop_on_valid_action,
    )


def _sigterm_as_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = resolve_run_config(arguments)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    profile = (
        None
        if config.no_motion
        else load_hardware_safety_profile(arguments.hardware_safety_profile)
    )
    previous_sigterm = signal.signal(signal.SIGTERM, _sigterm_as_interrupt)
    adapter = None
    loop_invoked = False
    try:
        with HardwareTelemetryRecorder(arguments.telemetry) as telemetry:
            telemetry.write_event(
                "session_start",
                {
                    "mode": arguments.mode,
                    "checkpoint": str(arguments.checkpoint),
                    "follower_port": arguments.follower_port,
                    "robot_serial": arguments.robot_serial,
                    "wrist_camera": arguments.wrist_camera,
                    "fixed_camera": arguments.fixed_camera,
                    "duration_seconds": config.duration_seconds,
                    "chunk_limit": config.chunk_limit,
                    "stop_on_valid_action": config.stop_on_valid_action,
                    "watchdog_seconds": config.watchdog_seconds,
                    "move_time_ms": config.move_time_ms,
                    "return_move_time_ms": config.return_move_time_ms,
                    "profile_verified_at": None if profile is None else profile.verified_at,
                },
            )
            try:
                adapter = open_hiwonder_follower(
                    vendor_root=arguments.vendor_root,
                    follower_port=arguments.follower_port,
                    calibration_id=arguments.calibration_id,
                    wrist_camera=arguments.wrist_camera,
                    fixed_camera=arguments.fixed_camera,
                    safety_profile=profile,
                    robot_serial=arguments.robot_serial,
                    arming_mode=arguments.arming_mode,
                )
                envelope = SafetyEnvelope(
                    adapter.joint_names,
                    adapter.calibration_ranges(),
                    normalized_ranges=so101_public_action_ranges(),
                )
                features, rename_map = so101_lerobot_contract()
                transport = LeRobotFourRPCClient(
                    server_address=arguments.server_address,
                    checkpoint=str(arguments.checkpoint),
                    lerobot_features=features,
                    rename_map=rename_map,
                    rpc_timeout_seconds=config.watchdog_seconds,
                )
                loop_invoked = True
                result = run_control_loop(
                    adapter=adapter,
                    transport=transport,
                    envelope=envelope,
                    config=config,
                    task=arguments.task,
                    fps=5,
                )
            except KeyboardInterrupt:
                telemetry.write_event(
                    "session_error",
                    {"error_type": "KeyboardInterrupt", "message": "operator interrupt"},
                )
                return 130
            except BaseException as error:
                telemetry.write_event(
                    "session_error",
                    {"error_type": type(error).__name__, "message": str(error)},
                )
                print(f"hardware session failed: {error}", file=sys.stderr)
                return 1
            finally:
                if adapter is not None and not loop_invoked:
                    adapter.close()
            summary = asdict(result)
            telemetry.write_event("session_result", summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
