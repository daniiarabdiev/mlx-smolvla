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
    predict.set_defaults(handler=_predict)
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
        output="action",
        action=action.tolist(),
    )
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
