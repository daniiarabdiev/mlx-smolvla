#!/usr/bin/env python3
"""Benchmark or execute the fixed Stage T3 native MLX LoRA run."""

from __future__ import annotations

import os
import sys


if __name__ == "__main__" and not (sys.flags.isolated and sys.flags.no_site):
    raise RuntimeError(
        "fine-tune Python entrypoint must be started by scripts/finetune_lora "
        "with an already isolated -I -S interpreter"
    )


import argparse
import hashlib
import importlib.machinery
import json
from pathlib import Path
import stat
from types import CodeType, ModuleType


def _code_signature(code: CodeType) -> tuple[object, ...]:
    """Return an exact semantic signature without marshal intern-table noise."""

    def constant_signature(value: object) -> object:
        if isinstance(value, CodeType):
            return ("code", _code_signature(value))
        if isinstance(value, tuple):
            return ("tuple", tuple(constant_signature(item) for item in value))
        if isinstance(value, frozenset):
            return (
                "frozenset",
                tuple(sorted(repr(constant_signature(item)) for item in value)),
            )
        return (type(value).__qualname__, value)

    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        tuple(constant_signature(item) for item in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_name,
        code.co_qualname,
        code.co_firstlineno,
        code.co_linetable,
        code.co_exceptiontable,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _require_real_directory(path: Path, *, label: str) -> Path:
    """Return one canonical directory whose complete ancestry has no symlink."""

    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        identity = os.lstat(current)
        if stat.S_ISLNK(identity.st_mode) or not stat.S_ISDIR(identity.st_mode):
            raise RuntimeError(f"{label} has unsafe ancestry: {current}")
    if Path(os.path.realpath(path)) != path:
        raise RuntimeError(f"{label} has symlinked ancestry: {path}")
    return path


if __name__ == "__main__":
    if not (sys.flags.isolated and sys.flags.no_site):
        raise RuntimeError("fine-tune launcher failed to enter isolated no-site mode")
    REPOSITORY_ROOT = _require_real_directory(
        REPOSITORY_ROOT,
        label="fine-tune repository root",
    )
    virtual_environment = Path(os.path.abspath(sys.executable)).parent.parent
    virtual_site_packages = (
        virtual_environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    virtual_environment = _require_real_directory(
        virtual_environment,
        label="isolated launcher virtual environment",
    )
    virtual_site_packages = _require_real_directory(
        virtual_site_packages,
        label="isolated launcher site-packages",
    )
    sys.prefix = str(virtual_environment)
    sys.exec_prefix = str(virtual_environment)
    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.append(str(virtual_site_packages))
elif str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

if __name__ == "__main__":
    def stable_bootstrap_source(
        path: Path,
        *,
        code_filename: str | None = None,
    ) -> tuple[Path, bytes, object]:
        path = Path(os.path.abspath(path))
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"bootstrap source is not regular: {path}")
            payload = bytearray()
            while chunk := os.read(descriptor, 1024 * 1024):
                payload.extend(chunk)
            after = os.fstat(descriptor)
            named = os.stat(path, follow_symlinks=False)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
                or len(payload) != before.st_size
            ):
                raise RuntimeError(f"bootstrap source changed while read: {path}")
        finally:
            os.close(descriptor)
        source = bytes(payload)
        return path, source, compile(
            source,
            str(path) if code_filename is None else code_filename,
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )

    def execute_bootstrap_module(
        fullname: str,
        source: tuple[Path, bytes, object],
        *,
        package: bool,
    ) -> ModuleType:
        path, _, code = source
        module = ModuleType(fullname)
        module.__file__ = str(path)
        module.__package__ = fullname if package else fullname.rpartition(".")[0]
        module.__loader__ = None
        module.__spec__ = importlib.machinery.ModuleSpec(
            fullname,
            loader=None,
            is_package=package,
        )
        if package:
            module.__path__ = [str(path.parent)]
            module.__spec__.submodule_search_locations = module.__path__
        sys.modules[fullname] = module
        try:
            exec(code, module.__dict__)
        except BaseException:
            sys.modules.pop(fullname, None)
            raise
        return module

    bootstrap_sources = {
        "__main__": stable_bootstrap_source(
            Path(__file__),
            code_filename=sys._getframe().f_code.co_filename,
        ),
        "mlx_smolvla._lab.training": stable_bootstrap_source(REPOSITORY_ROOT / "training" / "__init__.py"),
        "mlx_smolvla._lab.training.runtime_provenance": stable_bootstrap_source(
            REPOSITORY_ROOT / "training" / "runtime_provenance.py"
        ),
    }
    if _code_signature(sys._getframe().f_code) != _code_signature(
        bootstrap_sources["__main__"][2]
    ):
        raise RuntimeError(
            "executed launcher code differs from the captured bootstrap source"
        )
    training_package = execute_bootstrap_module(
        "mlx_smolvla._lab.training",
        bootstrap_sources["mlx_smolvla._lab.training"],
        package=True,
    )
    provenance_module = execute_bootstrap_module(
        "mlx_smolvla._lab.training.runtime_provenance",
        bootstrap_sources["mlx_smolvla._lab.training.runtime_provenance"],
        package=False,
    )
    training_package.runtime_provenance = provenance_module
    provenance_module.install_runtime_provenance(
        repository_root=REPOSITORY_ROOT,
        bootstrap_sources=bootstrap_sources,
    )

from mlx_smolvla._lab.training.finetune import (  # noqa: E402
    ADAPTIVE_BUDGET_MODE,
    FIXED_BUDGET_MODE,
    FineTuneConfig,
    benchmark_lora_updates,
    finetune_implementation_hashes,
    prepare_lora_finetune_launch,
    run_lora_finetune,
    write_run_state,
)
from mlx_smolvla._lab.training.lora import EXPERT_ONLY_SCOPE, LEGACY_FULL_SCOPE  # noqa: E402
from mlx_smolvla._lab.training.runtime_provenance import runtime_provenance_evidence  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/hf"))
    parser.add_argument(
        "--native-cache",
        type=Path,
        default=Path(".cache/mlx_smolvla/policy-float32"),
    )
    parser.add_argument("--output", type=Path, default=Path(".cache/training/t3"))
    parser.add_argument("--nominal-steps", type=int, default=3_000)
    parser.add_argument("--effective-batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-scope",
        choices=(LEGACY_FULL_SCOPE, EXPERT_ONLY_SCOPE),
        default=LEGACY_FULL_SCOPE,
    )
    parser.add_argument(
        "--budget-mode",
        choices=(ADAPTIVE_BUDGET_MODE, FIXED_BUDGET_MODE),
        default=ADAPTIVE_BUDGET_MODE,
    )
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing incomplete output from its latest validated checkpoint",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--benchmark-only", action="store_true")
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze and hash the exact T3B configuration without training",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path(".cache/training/t3-benchmark.json"),
    )
    parser.add_argument(
        "--launch-config",
        type=Path,
        help="immutable T3B launch commitment (defaults to OUTPUT/launch.json)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="safely bind stdout/stderr to OUTPUT/training.log inside the held lease",
    )
    parser.add_argument(
        "--verify-runtime-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.verify_runtime_only:
        verified_runtime_hashes = finetune_implementation_hashes()
        verified_runtime_evidence = runtime_provenance_evidence()
        implementation_digest = hashlib.sha256(
            json.dumps(
                verified_runtime_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        print(
            json.dumps(
                {
                    "runtime_provenance_verified": True,
                    "implementation_hash_count": len(verified_runtime_hashes),
                    "implementation_sha256": implementation_digest,
                    "isolated": bool(sys.flags.isolated),
                    "no_site": bool(sys.flags.no_site),
                    "runtime_provenance_frozen": verified_runtime_evidence["frozen"],
                    "runtime_provenance_module_count": len(
                        verified_runtime_evidence["modules"]
                    ),
                    "native_dependency_scope": verified_runtime_evidence[
                        "native_dependency_scope"
                    ],
                    "sys_path": list(sys.path),
                },
                sort_keys=True,
            )
        )
        return
    config = FineTuneConfig(
        cache_dir=args.cache_dir,
        native_cache=args.native_cache,
        output_dir=args.output,
        nominal_steps=args.nominal_steps,
        effective_batch_size=args.effective_batch_size,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        lora_scope=args.lora_scope,
        budget_mode=args.budget_mode,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    requires_launch = (
        config.lora_scope == EXPERT_ONLY_SCOPE
        and config.budget_mode == FIXED_BUDGET_MODE
    )
    launch_path = args.launch_config
    if (requires_launch or args.prepare_only) and launch_path is None:
        launch_path = config.output_dir / "launch.json"
    if args.prepare_only:
        if args.resume:
            parser.error("--prepare-only cannot be combined with --resume")
        assert launch_path is not None
        document, digest = prepare_lora_finetune_launch(
            config,
            output_path=launch_path,
        )
        topology = document["lora_topology"]
        print(
            json.dumps(
                {
                    "launch_config": str(launch_path.resolve()),
                    "launch_file_sha256": digest,
                    "configuration_sha256": document["configuration_sha256"],
                    "run_config_sha256": document["run_config_sha256"],
                    "adapter_count": topology["adapter_count"],
                    "trainable_tensor_count": topology["trainable_tensor_count"],
                    "trainable_scalar_count": topology["trainable_scalar_count"],
                },
                indent=2,
            )
        )
        return
    if args.benchmark_only:
        if config.budget_mode == FIXED_BUDGET_MODE:
            parser.error("fixed_steps mode forbids budget-selection timing")
        result = benchmark_lora_updates(config)
        digest = write_run_state(args.benchmark_output, result.as_dict())
        print(json.dumps({**result.as_dict(), "report_sha256": digest}, indent=2))
        return

    if requires_launch and args.log_file is None:
        parser.error("fixed expert-only training requires --log-file OUTPUT/training.log")
    if args.log_file is not None:
        if not requires_launch:
            parser.error("--log-file is only valid for fixed expert-only training")
        expected_log_path = Path(
            os.path.abspath(config.output_dir.expanduser())
        ) / "training.log"
        if Path(os.path.abspath(args.log_file.expanduser())) != expected_log_path:
            parser.error("--log-file must be OUTPUT/training.log")
    def progress(step: int, total: int, update) -> None:
        if step == 1 or step % 25 == 0 or step == total:
            print(
                f"step={step}/{total} loss={update.loss:.6f} "
                f"lr={update.learning_rate:.8g} seconds={update.seconds:.3f}",
                flush=True,
            )

    result = run_lora_finetune(
        config,
        launch_config_path=launch_path,
        progress=progress,
        training_log_path=args.log_file,
    )
    print(
        json.dumps(
            {
                "selected_steps": result.selected_steps,
                "training_seconds": result.training_seconds,
                "final_loss": result.final_loss,
                "final_smoothed_loss": result.final_smoothed_loss,
                "peak_memory_bytes": result.peak_memory_bytes,
                "adapter_sha256": result.adapter_sha256,
                "export_dir": str(result.export_dir),
                "run_state_sha256": result.run_state_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
