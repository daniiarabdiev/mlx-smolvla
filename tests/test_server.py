"""LeRobot 0.6.1 async-inference protocol and native server contracts."""

from __future__ import annotations

from concurrent import futures
import hashlib
import json
import pickle
import threading
import time
from pathlib import Path

import grpc
import mlx.core as mx
import numpy as np
import pytest
import torch

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


_REFERENCE_DESCRIPTOR_SHA256 = (
    "e116fbf44dd1fc65b67ff255c04857000c28e69055211af5ef3df85ac8d81f8d"
)


def _features() -> dict[str, dict[str, object]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": tuple(f"joint_{index}" for index in range(6)),
        },
        "observation.images.camera1": {
            "dtype": "image",
            "shape": (3, 256, 256),
            "names": ("height", "width", "channels"),
        },
        "observation.images.camera2": {
            "dtype": "image",
            "shape": (3, 256, 256),
            "names": ("height", "width", "channels"),
        },
    }


def _raw_observation(value: float = 0.0) -> dict[str, object]:
    result: dict[str, object] = {
        f"joint_{index}": np.float32(value + index) for index in range(6)
    }
    result.update(
        {
            "camera1": np.full((256, 256, 3), value, dtype=np.float32),
            "camera2": np.full((256, 256, 3), value / 2, dtype=np.float32),
            "task": "put the block in the box",
        }
    )
    return result


class _FakePreprocessor:
    def unnormalize_actions(self, value):
        return value


class _FakeConfig:
    chunk_size = 4
    n_action_steps = 4
    action_dim = 2
    image_keys = (
        "observation.images.camera1",
        "observation.images.camera2",
    )
    image_shapes = (
        ("observation.images.camera1", (3, 256, 256)),
        ("observation.images.camera2", (3, 256, 256)),
    )


class _FakePolicy:
    def __init__(self, *, delay: float = 0.0, fail: bool = False) -> None:
        self.config = _FakeConfig()
        self.preprocessor = _FakePreprocessor()
        self.delay = delay
        self.fail = fail
        self.active = 0
        self.max_active = 0
        self.started = threading.Event()
        self.reset_count = 0

    def predict_action_chunk(self, observation):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("deliberate inference failure")
            first = float(np.asarray(observation["observation.state"])[0])
            values = np.arange(8, dtype=np.float32).reshape(1, 4, 2) + first
            return mx.array(values)
        finally:
            self.active -= 1

    def reset(self) -> None:
        self.reset_count += 1


def _policy_config(*, actions_per_chunk: int = 3) -> RemotePolicyConfig:
    return RemotePolicyConfig(
        policy_type="smolvla",
        pretrained_name_or_path="recorded/fake-smolvla",
        lerobot_features=_features(),
        actions_per_chunk=actions_per_chunk,
        device="cpu",
    )


def _send_setup(stub, policy_config: object) -> None:
    stub.SendPolicyInstructions(
        services_pb2.PolicySetup(data=pickle.dumps(policy_config)), timeout=5
    )


def _send_observation(stub, observation: TimedObservation) -> None:
    messages = send_bytes_in_chunks(
        pickle.dumps(observation), services_pb2.Observation, silent=True
    )
    stub.SendObservations(messages, timeout=5)


def _start_fake_server(fake_policy: _FakePolicy, **overrides):
    from smolvla_mlx.server import ServeConfig, create_server

    values = dict(
        host="127.0.0.1",
        port=0,
        fps=20,
        inference_latency=0,
        obs_queue_timeout=0.15,
    )
    values.update(overrides)
    config = ServeConfig(**values)
    grpc_server, servicer, port = create_server(
        config, policy_loader=lambda **_kwargs: fake_policy
    )
    grpc_server.start()
    channel = grpc.insecure_channel(
        f"127.0.0.1:{port}", options=grpc_channel_options(enable_retries=False)
    )
    grpc.channel_ready_future(channel).result(timeout=5)
    return grpc_server, servicer, channel, services_pb2_grpc.AsyncInferenceStub(channel)


def test_pinned_reference_schema_is_exact() -> None:
    descriptor = services_pb2.DESCRIPTOR
    assert descriptor.package == "transport"
    assert hashlib.sha256(descriptor.serialized_pb).hexdigest() == _REFERENCE_DESCRIPTOR_SHA256

    service = descriptor.services_by_name["AsyncInference"]
    assert {
        method.name: (
            method.input_type.full_name,
            method.output_type.full_name,
            method.client_streaming,
            method.server_streaming,
        )
        for method in service.methods
    } == {
        "SendObservations": ("transport.Observation", "transport.Empty", True, False),
        "GetActions": ("transport.Empty", "transport.Actions", False, False),
        "SendPolicyInstructions": (
            "transport.PolicySetup",
            "transport.Empty",
            False,
            False,
        ),
        "Ready": ("transport.Empty", "transport.Empty", False, False),
    }
    assert {
        value.name: value.number
        for value in descriptor.enum_types_by_name["TransferState"].values
    } == {
        "TRANSFER_UNKNOWN": 0,
        "TRANSFER_BEGIN": 1,
        "TRANSFER_MIDDLE": 2,
        "TRANSFER_END": 3,
    }


def test_server_configuration_defaults_to_loopback_and_requires_remote_opt_in() -> None:
    from smolvla_mlx.server import ServeConfig

    config = ServeConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 8080
    assert config.environment_dt == 1 / config.fps
    assert config.quantization is None
    assert config.latency_log is None

    assert ServeConfig(quantization="vlm-8bit").quantization == "vlm-8bit"
    assert ServeConfig(quantization="vlm-4bit").quantization == "vlm-4bit"

    with pytest.raises(ValueError, match="allow_remote"):
        ServeConfig(host="0.0.0.0")
    remote = ServeConfig(host="0.0.0.0", allow_remote=True)
    assert remote.allow_remote

    for invalid in (
        {"port": -1},
        {"port": 65536},
        {"fps": 0},
        {"inference_latency": -1},
        {"obs_queue_timeout": -1},
        {"max_workers": 0},
        {"seed": -1},
        {"quantization": "everything-4bit"},
        {"quantization": "vlm-8bit", "dtype": "float32"},
        {"quantization": "vlm-4bit", "execution_mode": "strict"},
        {"latency_log": ""},
    ):
        with pytest.raises(ValueError):
            ServeConfig(**invalid)


def test_server_passes_quantization_opt_in_to_policy_loader() -> None:
    from smolvla_mlx.server import ServeConfig, create_server

    captured: dict[str, object] = {}
    policy = _FakePolicy()

    def loader(**kwargs):
        captured.update(kwargs)
        return policy

    grpc_server, _servicer, port = create_server(
        ServeConfig(host="127.0.0.1", port=0, quantization="vlm-4bit"),
        policy_loader=loader,
    )
    grpc_server.start()
    channel = grpc.insecure_channel(
        f"127.0.0.1:{port}", options=grpc_channel_options(enable_retries=False)
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        _send_setup(stub, _policy_config())
        assert captured["quantization"] == "vlm-4bit"
    finally:
        channel.close()
        grpc_server.stop(grace=0).wait()


def test_reference_transport_round_trip_preserves_latest_observation_and_timing() -> None:
    policy = _FakePolicy()
    server, servicer, channel, stub = _start_fake_server(policy)
    try:
        stub.Ready(services_pb2.Empty(), timeout=5)
        _send_setup(stub, _policy_config(actions_per_chunk=3))
        _send_observation(
            stub,
            TimedObservation(
                timestamp=100.0,
                timestep=7,
                observation=_raw_observation(1.0),
            ),
        )
        _send_observation(
            stub,
            TimedObservation(
                timestamp=200.0,
                timestep=11,
                observation=_raw_observation(4.0),
                must_go=True,
            ),
        )

        response = stub.GetActions(services_pb2.Empty(), timeout=5)
        actions = pickle.loads(response.data)

        assert len(actions) == 3
        assert [action.timestep for action in actions] == [11, 12, 13]
        assert [action.timestamp for action in actions] == [200.0, 200.05, 200.1]
        assert all(action.action.device.type == "cpu" for action in actions)
        np.testing.assert_array_equal(
            torch.stack([action.action for action in actions]).numpy(),
            np.arange(8, dtype=np.float32).reshape(4, 2)[:3] + 4,
        )
        assert servicer.actions_per_chunk == 3
    finally:
        channel.close()
        server.stop(grace=0).wait()


def test_invalid_setup_and_inference_errors_are_explicit_grpc_statuses() -> None:
    policy = _FakePolicy(fail=True)
    server, _servicer, channel, stub = _start_fake_server(policy)
    try:
        stub.Ready(services_pb2.Empty(), timeout=5)
        with pytest.raises(grpc.RpcError) as invalid:
            _send_setup(stub, {"not": "RemotePolicyConfig"})
        assert invalid.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        _send_setup(stub, _policy_config())
        _send_observation(
            stub,
            TimedObservation(
                timestamp=1.0,
                timestep=1,
                observation=_raw_observation(),
                must_go=True,
            ),
        )
        with pytest.raises(grpc.RpcError) as inference:
            stub.GetActions(services_pb2.Empty(), timeout=5)
        assert inference.value.code() == grpc.StatusCode.INTERNAL
        assert "deliberate inference failure" in inference.value.details()
    finally:
        channel.close()
        server.stop(grace=0).wait()


def test_get_actions_is_cancellable_and_inference_is_serialized() -> None:
    policy = _FakePolicy(delay=0.1)
    server, _servicer, channel, stub = _start_fake_server(
        policy, obs_queue_timeout=1.0, max_workers=4
    )
    executor = futures.ThreadPoolExecutor(max_workers=2)
    try:
        stub.Ready(services_pb2.Empty(), timeout=5)
        _send_setup(stub, _policy_config(actions_per_chunk=1))

        waiting = stub.GetActions.future(services_pb2.Empty(), timeout=5)
        assert waiting.cancel()
        with pytest.raises(grpc.FutureCancelledError):
            waiting.result(timeout=1)

        _send_observation(
            stub,
            TimedObservation(1.0, 1, _raw_observation(1.0), must_go=True),
        )
        first = executor.submit(stub.GetActions, services_pb2.Empty(), timeout=5)
        assert policy.started.wait(timeout=2)
        _send_observation(
            stub,
            TimedObservation(2.0, 2, _raw_observation(2.0), must_go=True),
        )
        second = executor.submit(stub.GetActions, services_pb2.Empty(), timeout=5)
        assert first.result(timeout=5).data
        assert second.result(timeout=5).data
        assert policy.max_active == 1
    finally:
        executor.shutdown(wait=True)
        channel.close()
        server.stop(grace=0).wait()


def test_cancelled_waiter_restores_observation_consumed_during_cancellation() -> None:
    from smolvla_mlx.server import NativeMLXPolicyServer, ServeConfig

    class CancelAfterDequeueContext:
        def __init__(self) -> None:
            self.active_checks = 0

        def is_active(self) -> bool:
            self.active_checks += 1
            return self.active_checks == 1

        def abort(self, _code, details: str) -> None:
            raise RuntimeError(details)

    servicer = NativeMLXPolicyServer(
        ServeConfig(host="127.0.0.1", port=0, obs_queue_timeout=0.01),
        policy_loader=lambda **_kwargs: _FakePolicy(),
    )
    observation = TimedObservation(1.0, 9, _raw_observation(), must_go=True)
    servicer.observation_queue.put_nowait(observation)

    with pytest.raises(RuntimeError, match="cancelled"):
        servicer._next_observation(CancelAfterDequeueContext())

    assert servicer.observation_queue.get_nowait() is observation


def test_latency_log_records_one_successful_loopback_chunk_without_payloads(
    tmp_path: Path,
) -> None:
    policy = _FakePolicy(delay=0.01)
    latency_log = tmp_path / "latency.jsonl"
    server, _servicer, channel, stub = _start_fake_server(
        policy,
        latency_log=latency_log,
    )
    client_timestamp = time.time()
    try:
        stub.Ready(services_pb2.Empty(), timeout=5)
        _send_setup(stub, _policy_config(actions_per_chunk=2))
        _send_observation(
            stub,
            TimedObservation(
                client_timestamp,
                17,
                _raw_observation(1.0),
                must_go=True,
            ),
        )
        response = stub.GetActions(services_pb2.Empty(), timeout=5)
        assert response.data
    finally:
        channel.close()
        server.stop(grace=0).wait()

    records = [json.loads(line) for line in latency_log.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["format_version"] == 1
    assert record["event"] == "observation_to_action_chunk"
    assert record["sequence"] == 0
    assert record["observation"] == {
        "client_timestamp": client_timestamp,
        "must_go": True,
        "timestep": 17,
    }
    assert record["chunk"] == {
        "action_count": 2,
        "first_timestep": 17,
        "last_timestep": 18,
    }
    assert record["policy"] == {
        "dtype": "bfloat16",
        "execution_mode": "production",
        "pretrained_name_or_path": "recorded/fake-smolvla",
        "quantization": None,
        "type": "smolvla",
    }
    assert record["latency_ms"]["inference"] >= 5
    assert record["latency_ms"]["server_receive_to_chunk"] >= record["latency_ms"]["inference"]
    assert record["latency_ms"]["client_observation_to_chunk"] >= record["latency_ms"]["server_receive_to_chunk"]
    assert set(record["server"]) == {"chunk_ready_at_utc", "received_at_utc"}
    serialized = json.dumps(record)
    assert "camera" not in serialized
    assert "action\"" not in serialized


def test_latency_log_refuses_to_mix_with_an_existing_session(tmp_path: Path) -> None:
    from smolvla_mlx.server import ServeConfig, create_server

    path = tmp_path / "existing.jsonl"
    path.write_text("existing session\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        create_server(
            ServeConfig(host="127.0.0.1", port=0, latency_log=path),
            policy_loader=lambda **_kwargs: _FakePolicy(),
        )


@pytest.mark.slow
def test_recorded_loopback_chunk_equals_direct_select_action(
    checkpoint_dir: Path,
    base_vlm_dir: Path,
) -> None:
    """Use reference client transport and real recorded data; never touch hardware."""

    from smolvla_mlx.server import ServeConfig, create_server

    config = ServeConfig(
        host="127.0.0.1",
        port=0,
        cache_dir=Path(".cache/smolvla_mlx/server-loopback"),
        tokenizer_dir=base_vlm_dir,
        dtype="float32",
        execution_mode="strict",
        fps=30,
        inference_latency=0,
        obs_queue_timeout=1,
        seed=20260831,
    )
    grpc_server, servicer, port = create_server(config)
    grpc_server.start()
    channel = grpc.insecure_channel(
        f"127.0.0.1:{port}", options=grpc_channel_options(enable_retries=False)
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        stub.Ready(services_pb2.Empty(), timeout=5)
        _send_setup(
            stub,
            RemotePolicyConfig(
                policy_type="smolvla",
                pretrained_name_or_path=str(checkpoint_dir),
                lerobot_features=_features(),
                actions_per_chunk=3,
                device="cpu",
            ),
        )

        raw = _raw_observation()
        state = np.load("tests/golden/sample_000/raw/state.npy").astype(np.float32)
        for index, value in enumerate(state):
            raw[f"joint_{index}"] = value
        for camera in ("camera1", "camera2"):
            chw = np.load(f"tests/golden/sample_000/raw/{camera}.npy")
            raw[camera] = np.moveaxis(chw, 0, -1)
        raw["task"] = "pink lego brick into the transparent box"
        timed = TimedObservation(
            timestamp=1_700_000_000.0,
            timestep=23,
            observation=raw,
            must_go=True,
        )
        native_observation = servicer.prepare_observation(timed)

        _send_observation(stub, timed)
        served = pickle.loads(stub.GetActions(services_pb2.Empty(), timeout=30).data)

        assert servicer.policy is not None
        servicer.policy.reset()
        mx.random.seed(20260831)
        direct = np.stack(
            [servicer.policy.select_action(native_observation) for _ in range(3)]
        )
        transported = torch.stack([item.action for item in served]).numpy()

        np.testing.assert_array_equal(transported, direct)
        assert hashlib.sha256(np.ascontiguousarray(transported).tobytes()).hexdigest() == (
            "46a4b2809975a6f14925db404d55b7595d45df9ef578f1f2cd1ae760ce137981"
        )
        assert [item.timestep for item in served] == [23, 24, 25]
        assert [item.timestamp for item in served] == pytest.approx(
            [1_700_000_000.0 + index / 30 for index in range(3)], abs=0, rel=0
        )
    finally:
        channel.close()
        grpc_server.stop(grace=0).wait()
