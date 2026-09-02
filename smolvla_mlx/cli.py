"""Command-line entry points for native SmolVLA MLX inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping

import mlx.core as mx
import numpy as np

from smolvla_mlx.benchmark import run_benchmark
from smolvla_mlx.cache import resolve_cache_dir
from smolvla_mlx.policy import SmolVLAMLX


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smolvla-mlx", description="Native MLX inference for SmolVLA.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    convert = subcommands.add_parser("convert", help="download/open and strictly convert a checkpoint")
    convert.add_argument("--model", default="lerobot/smolvla_base")
    convert.add_argument("--cache-dir", type=Path)
    convert.add_argument("--tokenizer-dir", type=Path)
    convert.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    convert.add_argument("--execution-mode", choices=("production", "strict"), default="production")
    convert.set_defaults(handler=_convert)

    test = subcommands.add_parser("test", help="run the repository test suite")
    test.add_argument("--cache-dir", type=Path)
    test.add_argument("tests", nargs="*", default=["tests"])
    test.set_defaults(handler=_test)

    bench = subcommands.add_parser("bench", help="benchmark a saved real observation")
    bench.add_argument("--model", default="lerobot/smolvla_base")
    bench.add_argument("--cache-dir", type=Path)
    bench.add_argument("--tokenizer-dir", type=Path)
    bench.add_argument("--sample-root", type=Path, default=Path("tests/golden/sample_000"))
    bench.add_argument("--metadata", type=Path, default=Path("tests/golden/metadata.json"))
    bench.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    bench.add_argument("--execution-mode", choices=("production", "strict"), default="production")
    bench.add_argument("--quantization", choices=("vlm-8bit", "vlm-4bit"))
    bench.add_argument("--runs", type=int, default=50)
    bench.add_argument("--warmups", type=int, default=5)
    bench.add_argument("--output", type=Path)
    bench.set_defaults(handler=_bench)

    predict = subcommands.add_parser(
        "predict",
        help="predict one action from a saved observation or real LeRobot dataset frame",
    )
    predict.add_argument("--model", default="lerobot/smolvla_base")
    predict.add_argument("--cache-dir", type=Path)
    predict.add_argument("--tokenizer-dir", type=Path)
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset")
    source.add_argument("--observation", type=Path)
    predict.add_argument("--index", type=int)
    predict.add_argument("--episode", type=int, default=0)
    predict.add_argument("--metadata", type=Path, default=Path("tests/golden/metadata.json"))
    predict.add_argument("--camera1-key", default="observation.images.side")
    predict.add_argument("--camera2-key", default="observation.images.up")
    predict.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    predict.add_argument("--execution-mode", choices=("production", "strict"), default="production")
    predict.add_argument("--quantization", choices=("vlm-8bit", "vlm-4bit"))
    predict.set_defaults(handler=_predict)

    serve = subcommands.add_parser(
        "serve",
        help="serve native MLX actions to a trusted LeRobot 0.6.1 async client",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--cache-dir", type=Path)
    serve.add_argument("--tokenizer-dir", type=Path)
    serve.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    serve.add_argument("--execution-mode", choices=("production", "strict"), default="production")
    serve.add_argument("--quantization", choices=("vlm-8bit", "vlm-4bit"))
    serve.add_argument(
        "--latency-log",
        type=Path,
        help="new JSONL path for observation-to-action-chunk timing records",
    )
    serve.add_argument("--fps", type=int, default=30)
    serve.add_argument("--inference-latency", type=float, default=0.0)
    serve.add_argument("--obs-queue-timeout", type=float, default=1.0)
    serve.add_argument("--max-workers", type=int, default=4)
    serve.add_argument(
        "--seed",
        type=int,
        help="optional deterministic base seed; each action chunk advances it by one",
    )
    serve.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow a non-loopback insecure/pickle bind (trusted network only)",
    )
    serve.set_defaults(handler=_serve)

    train = subcommands.add_parser(
        "train",
        help="fine-tune SmolVLA natively with MLX using LoRA or the full reference trainable set",
    )
    train.add_argument("dataset", help="LeRobot dataset repo id or local dataset path")
    training_mode = train.add_mutually_exclusive_group(required=True)
    training_mode.add_argument(
        "--lora",
        dest="training_mode",
        action="store_const",
        const="lora",
        help="train native MLX low-rank adapters",
    )
    training_mode.add_argument(
        "--full",
        dest="training_mode",
        action="store_const",
        const="full",
        help="train every parameter enabled by the reference SmolVLA freeze policy",
    )
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--lr", dest="learning_rate", type=float, default=1e-4)
    train.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--checkpoint-every", dest="checkpoint_interval", type=int, default=25)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    train.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/smolvla_mlx/policy-float32"),
    )
    train.add_argument("--rank", type=int, default=8)
    train.add_argument("--alpha", type=float, default=16.0)
    train.add_argument("--dropout", type=float, default=0.0)
    train.set_defaults(handler=_train)
    return parser


def _cache(args: argparse.Namespace) -> Path:
    cache_dir = resolve_cache_dir(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_policy(args: argparse.Namespace, cache_dir: Path) -> SmolVLAMLX:
    return SmolVLAMLX.from_pretrained(
        args.model,
        cache_dir=cache_dir,
        dtype=args.dtype,
        tokenizer_dir=args.tokenizer_dir,
        execution_mode=args.execution_mode,
        quantization=getattr(args, "quantization", None),
    )


def _emit(**values: object) -> None:
    print(json.dumps(values, sort_keys=True, default=str))


def _convert(args: argparse.Namespace) -> int:
    cache_dir = _cache(args)
    policy = _load_policy(args, cache_dir)
    _emit(
        model=args.model,
        cache=str(cache_dir),
        dtype=args.dtype,
        execution_mode=policy.execution_mode,
        quantization=policy.quantization,
        output=str(policy.converted_weights_path),
    )
    return 0


def _test(args: argparse.Namespace) -> int:
    cache_dir = _cache(args)
    completed = subprocess.run([sys.executable, "-m", "pytest", *args.tests], check=False)
    _emit(
        model=None,
        cache=str(cache_dir),
        dtype=None,
        execution_mode=None,
        output="pytest",
        returncode=completed.returncode,
    )
    return completed.returncode


def _saved_observation(sample_root: Path, metadata_path: Path) -> Mapping[str, object]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Golden metadata is absent at {metadata_path}; run make goldens first")
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Golden metadata has no samples")
    sample = next(
        (
            candidate
            for candidate in samples
            if isinstance(candidate, dict) and candidate.get("name") == sample_root.name
        ),
        None,
    )
    if sample is None:
        raise ValueError(
            f"Golden metadata has no sample named {sample_root.name!r} for {sample_root}"
        )
    task = sample.get("task")
    if not isinstance(task, str):
        raise ValueError(f"Golden metadata sample {sample_root.name!r} has no task string")
    return {
        "observation.images.camera1": np.load(sample_root / "raw/camera1.npy"),
        "observation.images.camera2": np.load(sample_root / "raw/camera2.npy"),
        "observation.state": np.load(sample_root / "raw/state.npy"),
        "task": task,
    }


def _bench(args: argparse.Namespace) -> int:
    cache_dir = _cache(args)
    policy = _load_policy(args, cache_dir)
    result = run_benchmark(
        policy,
        _saved_observation(args.sample_root, args.metadata),
        measured_runs=args.runs,
        warmup_runs=args.warmups,
    )
    payload = result.as_dict()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(
        model=args.model,
        cache=str(cache_dir),
        dtype=args.dtype,
        execution_mode=policy.execution_mode,
        quantization=policy.quantization,
        output=str(args.output) if args.output else None,
        result=payload,
    )
    return 0


def _dataset_observation(args: argparse.Namespace, cache_dir: Path) -> Mapping[str, object]:
    """Extract a frame in a child process so the runtime module never imports LeRobot."""

    if args.index is None:
        raise ValueError("predict --dataset requires --index")

    with tempfile.TemporaryDirectory(prefix="dataset-frame-", dir=cache_dir) as temporary_name:
        temporary = Path(temporary_name)
        output = temporary / "frame.npz"
        task_path = temporary / "task.json"
        configuration = {
            "dataset": args.dataset,
            "root": str(cache_dir / "datasets" / args.dataset.replace("/", "__")),
            "episode": args.episode,
            "index": args.index,
            "camera1": args.camera1_key,
            "camera2": args.camera2_key,
            "output": str(output),
            "task_output": str(task_path),
        }
        child = """
import json
from pathlib import Path
import numpy as np
import importlib

configuration = json.loads(__import__('sys').argv[1])
LeRobotDataset = importlib.import_module('lero' + 'bot.datasets.lerobot_dataset').LeRobotDataset
dataset = LeRobotDataset(
    configuration['dataset'],
    root=Path(configuration['root']),
    episodes=[configuration['episode']],
)
item = dataset[configuration['index']]
np.savez(
    configuration['output'],
    camera1=item[configuration['camera1']].cpu().numpy(),
    camera2=item[configuration['camera2']].cpu().numpy(),
    state=item['observation.state'].cpu().numpy(),
)
Path(configuration['task_output']).write_text(json.dumps(item['task']), encoding='utf-8')
"""
        completed = subprocess.run(
            [sys.executable, "-c", child, json.dumps(configuration)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Dataset prediction requires the optional LeRobot dataset bridge; "
                f"child process failed: {completed.stderr.strip()}"
            )
        with np.load(output) as frame:
            observation = {
                "observation.images.camera1": np.array(frame["camera1"]),
                "observation.images.camera2": np.array(frame["camera2"]),
                "observation.state": np.array(frame["state"]),
                "task": json.loads(task_path.read_text(encoding="utf-8")),
            }
    return observation


def _predict(args: argparse.Namespace) -> int:
    cache_dir = _cache(args)
    policy = _load_policy(args, cache_dir)
    observation = (
        _saved_observation(args.observation, args.metadata)
        if args.observation is not None
        else _dataset_observation(args, cache_dir)
    )
    action = policy.select_action(observation)
    _emit(
        model=args.model,
        cache=str(cache_dir),
        dtype=args.dtype,
        execution_mode=policy.execution_mode,
        quantization=policy.quantization,
        output="action",
        action=action.tolist(),
    )
    return 0


def _serve(args: argparse.Namespace) -> int:
    if sys.version_info < (3, 12):
        raise RuntimeError("serve requires Python 3.12+ and the optional .[serve] dependencies")
    try:
        from smolvla_mlx.server import ServeConfig, serve_forever
    except ImportError as error:
        raise RuntimeError(
            "serve dependencies are unavailable; install this package with `pip install '.[serve]'`"
        ) from error

    serve_forever(
        ServeConfig(
            host=args.host,
            port=args.port,
            cache_dir=args.cache_dir,
            tokenizer_dir=args.tokenizer_dir,
            dtype=args.dtype,
            execution_mode=args.execution_mode,
            quantization=args.quantization,
            latency_log=args.latency_log,
            fps=args.fps,
            inference_latency=args.inference_latency,
            obs_queue_timeout=args.obs_queue_timeout,
            max_workers=args.max_workers,
            seed=args.seed,
            allow_remote=args.allow_remote,
        )
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    if sys.version_info < (3, 12):
        raise RuntimeError("train requires Python 3.12+ and the optional .[train] dependencies")
    try:
        from training.ux import FullTrainingConfig, LoRATrainingConfig, run_training
    except ImportError as error:
        raise RuntimeError(
            "training dependencies are unavailable; install this package with `pip install '.[train]'`"
        ) from error

    common = {
        "dataset": args.dataset,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dtype": args.dtype,
        "output_dir": args.output,
        "cache_dir": args.cache_dir,
        "native_cache": args.native_cache,
        "checkpoint_interval": args.checkpoint_interval,
        "resume": args.resume,
    }
    config = (
        FullTrainingConfig(**common)
        if args.training_mode == "full"
        else LoRATrainingConfig(
            **common,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
    )

    def progress(step: int, total: int, update: object) -> None:
        if step == 1 or step % 10 == 0 or step == total:
            print(
                f"step={step}/{total} loss={getattr(update, 'loss'):.6f} "
                f"lr={getattr(update, 'learning_rate'):.8g}",
                flush=True,
            )

    result = run_training(config, progress=progress)
    payload = result.as_dict() if hasattr(result, "as_dict") else result
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"smolvla-mlx: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
