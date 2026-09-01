"""Stdlib-only import bootstrap for binding direct Python and extension loads.

The guard routes each imported extension's own Mach-O file through a retained
descriptor.  Its installed-distribution inventory also hashes ``.dylib`` files,
but dyld opens transitive dependencies independently; this module does not claim
load-time identity attestation for those transitive images.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
from importlib.metadata import PackageNotFoundError, distribution
import json
import os
from pathlib import Path
import stat
import sys
from threading import RLock
from types import CodeType, ModuleType
from typing import Mapping


_DEFAULT_DISTRIBUTIONS = (
    "lerobot",
    "datasets",
    "pyarrow",
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "av",
    "huggingface-hub",
    "safetensors",
    "numpy",
    "mlx",
    "mlx-metal",
    "pillow",
    "pandas",
    "packaging",
    "einops",
    "fsspec",
)


def _stable_regular_payload(path: Path) -> bytes:
    """Read one exact regular inode and reject name or byte changes."""

    path = Path(os.path.abspath(path))
    try:
        named_before = os.lstat(path)
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(named_before.st_mode):
        raise FileNotFoundError(f"runtime source is a symlink: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FileNotFoundError(f"runtime source is not regular: {path}")
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
            raise RuntimeError(f"runtime source changed while captured: {path}")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_code_value(value: object) -> object:
    """Encode code constants without process-local marshal intern/reference flags."""

    if isinstance(value, CodeType):
        return {
            "type": "code",
            "argcount": value.co_argcount,
            "posonlyargcount": value.co_posonlyargcount,
            "kwonlyargcount": value.co_kwonlyargcount,
            "nlocals": value.co_nlocals,
            "stacksize": value.co_stacksize,
            "flags": value.co_flags,
            "bytecode": value.co_code.hex(),
            "constants": [_canonical_code_value(item) for item in value.co_consts],
            "names": list(value.co_names),
            "varnames": list(value.co_varnames),
            "freevars": list(value.co_freevars),
            "cellvars": list(value.co_cellvars),
            "name": value.co_name,
            "qualname": value.co_qualname,
            "firstlineno": value.co_firstlineno,
            "linetable": value.co_linetable.hex(),
            "exceptiontable": value.co_exceptiontable.hex(),
        }
    if value is None or value is Ellipsis or value is NotImplemented:
        return {"type": type(value).__qualname__}
    if type(value) in {bool, int, str}:
        return {"type": type(value).__qualname__, "value": value}
    if isinstance(value, float):
        return {"type": "float", "value": value.hex()}
    if isinstance(value, complex):
        return {
            "type": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_canonical_code_value(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_canonical_code_value(item) for item in value]
        return {
            "type": "frozenset",
            "items": sorted(
                items,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        }
    raise TypeError(f"unsupported Python code constant: {type(value).__qualname__}")


def _code_sha256(code: CodeType) -> str:
    payload = json.dumps(
        _canonical_code_value(code),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(payload)


class _RuntimeProvenanceState:
    def __init__(
        self,
        *,
        repository_root: Path,
        include_installed: bool,
        bootstrap_sources: Mapping[str, tuple[Path, bytes, CodeType]],
    ) -> None:
        self.repository_root = repository_root.resolve(strict=True)
        self.expected_files: dict[Path, str] = {}
        self.guarded_roots: set[Path] = {self.repository_root}
        self.modules: dict[str, list[dict[str, object]]] = {}
        self.frozen = False
        self.lock = RLock()
        self._capture_repository()
        if include_installed:
            self._capture_installed()
        self._record_bootstrap_sources(bootstrap_sources)
        self._audit_preloaded_modules(frozenset(bootstrap_sources))

    def _resolve_guarded_origin(self, path: Path) -> Path | None:
        """Resolve one guarded name only when no guarded component is a symlink."""

        absolute = Path(os.path.abspath(path))
        lexical_root = next(
            (root for root in self.guarded_roots if absolute.is_relative_to(root)),
            None,
        )
        if lexical_root is not None:
            current = lexical_root
            for component in absolute.relative_to(lexical_root).parts:
                current /= component
                try:
                    identity = os.lstat(current)
                except FileNotFoundError:
                    return None
                if stat.S_ISLNK(identity.st_mode):
                    raise ImportError(f"guarded runtime source is a symlink: {absolute}")
            resolved = absolute.resolve(strict=True)
            if not resolved.is_relative_to(lexical_root):
                raise ImportError(f"guarded runtime source escapes its root: {absolute}")
            return resolved
        try:
            resolved = absolute.resolve(strict=True)
        except FileNotFoundError:
            return None
        if any(resolved.is_relative_to(root) for root in self.guarded_roots):
            raise ImportError(f"guarded runtime source uses a symlink alias: {absolute}")
        return None

    def _record_bootstrap_sources(
        self,
        sources: Mapping[str, tuple[Path, bytes, CodeType]],
    ) -> None:
        for fullname, (origin, payload, code) in sources.items():
            resolved = self._resolve_guarded_origin(origin)
            if resolved is None:
                raise RuntimeError(
                    f"bootstrap source is outside guarded roots: {fullname}: {origin}"
                )
            digest = _sha256(payload)
            if self.expected_files.get(resolved) != digest:
                raise RuntimeError(
                    f"bootstrap source differs from captured bytes: {fullname}: {resolved}"
                )
            self.record_source(fullname, resolved, digest, code)

    def _audit_preloaded_modules(self, bootstrap_names: frozenset[str]) -> None:
        for fullname, module in tuple(sys.modules.items()):
            origin_value = getattr(module, "__file__", None)
            if not isinstance(origin_value, str):
                continue
            try:
                guarded_origin = self._resolve_guarded_origin(Path(origin_value))
            except ImportError as error:
                raise RuntimeError(
                    f"preloaded guarded module has an unsafe origin: {fullname}"
                ) from error
            if guarded_origin is None:
                continue
            if fullname not in bootstrap_names:
                raise RuntimeError(
                    "preloaded guarded module lacks bootstrap evidence: "
                    f"{fullname}: {guarded_origin}"
                )

    def _capture_repository(self) -> None:
        for path in sorted(self.repository_root.rglob("*.py")):
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"runtime repository source is unsafe: {path}")
            resolved = path.resolve(strict=True)
            self.expected_files[resolved] = _sha256(
                _stable_regular_payload(resolved)
            )

    def _capture_installed(self) -> None:
        for entry in sys.path:
            if not entry:
                continue
            candidate = Path(entry)
            if candidate.is_dir() and candidate.name in {
                "site-packages",
                "dist-packages",
            }:
                self.guarded_roots.add(candidate.resolve(strict=True))
        for distribution_name in _DEFAULT_DISTRIBUTIONS:
            try:
                installed = distribution(distribution_name)
            except PackageNotFoundError as error:
                raise FileNotFoundError(
                    f"required runtime distribution is missing: {distribution_name}"
                ) from error
            files = installed.files
            if files is None:
                raise FileNotFoundError(
                    f"runtime distribution has no file inventory: {distribution_name}"
                )
            for item in files:
                suffix = Path(str(item)).suffix
                if suffix not in {".py", ".so", ".dylib"}:
                    continue
                unresolved = Path(installed.locate_file(item))
                path = self._resolve_guarded_origin(unresolved)
                if path is None:
                    raise ValueError(
                        "runtime distribution file escapes guarded roots: "
                        f"{unresolved}"
                    )
                self.expected_files[path] = _sha256(
                    _stable_regular_payload(path)
                )

    def guards(self, path: Path) -> bool:
        return self._resolve_guarded_origin(path) is not None

    def prepare_load(self, fullname: str, origin: Path, payload: bytes) -> str:
        return self.prepare_digest(fullname, origin, _sha256(payload))

    def prepare_digest(self, fullname: str, origin: Path, digest: str) -> str:
        resolved = self._resolve_guarded_origin(origin)
        if resolved is None:
            raise ImportError(f"runtime source escaped guarded roots: {origin}")
        expected = self.expected_files.get(resolved)
        if expected is not None and digest != expected:
            raise ImportError(
                f"runtime source changed after provenance capture: {resolved}"
            )
        with self.lock:
            if self.frozen and fullname not in self.modules:
                raise ImportError(
                    f"guarded module imported after provenance freeze: {fullname}"
                )
        return digest

    def record_source(
        self,
        fullname: str,
        origin: Path,
        payload_sha256: str,
        code: object,
    ) -> None:
        evidence = {
            "origin": str(self._resolve_guarded_origin(origin)),
            "kind": "source",
            "file_sha256": payload_sha256,
            "code_sha256": _code_sha256(code),
        }
        with self.lock:
            generations = self.modules.setdefault(fullname, [])
            if evidence not in generations:
                generations.append(evidence)

    def record_binary(
        self,
        fullname: str,
        origin: Path,
        payload_sha256: str,
    ) -> None:
        evidence = {
            "origin": str(self._resolve_guarded_origin(origin)),
            "kind": "extension",
            "file_sha256": payload_sha256,
            "code_sha256": payload_sha256,
        }
        with self.lock:
            generations = self.modules.setdefault(fullname, [])
            if evidence not in generations:
                generations.append(evidence)

    def evidence(self) -> dict[str, object]:
        with self.lock:
            return {
                "format_version": 1,
                "frozen": self.frozen,
                "native_dependency_scope": (
                    "direct-extension-origin-bound; "
                    "transitive-dyld-images-inventory-hashed-only"
                ),
                "modules": {
                    name: [dict(item) for item in value]
                    for name, value in sorted(self.modules.items())
                },
            }


class _GuardedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(
        self,
        fullname: str,
        path: str,
        state: _RuntimeProvenanceState,
    ) -> None:
        super().__init__(fullname, path)
        self._state = state

    def get_code(self, fullname: str):
        origin = Path(self.path)
        payload = _stable_regular_payload(origin)
        payload_sha256 = self._state.prepare_load(fullname, origin, payload)
        code = self.source_to_code(
            payload,
            str(origin),
            _optimize=sys.flags.optimize,
        )
        self._state.record_source(fullname, origin, payload_sha256, code)
        return code


class _GuardedBinaryLoader(importlib.abc.Loader):
    def __init__(
        self,
        fullname: str,
        origin: Path,
        delegate: importlib.abc.Loader,
        state: _RuntimeProvenanceState,
    ) -> None:
        self._fullname = fullname
        self._origin = origin
        self._delegate = delegate
        self._state = state
        self._descriptor: int | None = None
        self._identity: tuple[int, int, int, int] | None = None
        self._digest: str | None = None
        self._authority_origin: str | None = None

    def _capture_authority(self) -> None:
        if self._descriptor is not None:
            return
        descriptor = os.open(
            self._origin,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ImportError(f"guarded extension is not regular: {self._fullname}")
            digest = hashlib.sha256()
            byte_count = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            after = os.fstat(descriptor)
            named = os.stat(self._origin, follow_symlinks=False)
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            if (
                not stat.S_ISREG(named.st_mode)
                or identity
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or (named.st_dev, named.st_ino) != identity[:2]
                or byte_count != before.st_size
            ):
                raise ImportError(
                    f"guarded extension changed before loading: {self._fullname}"
                )
            payload_sha256 = digest.hexdigest()
            prepared = self._state.prepare_digest(
                self._fullname,
                self._origin,
                payload_sha256,
            )
            if prepared != payload_sha256:
                raise AssertionError("guarded extension digest preparation failed")
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        self._identity = identity
        self._digest = payload_sha256
        self._authority_origin = f"/dev/fd/{descriptor}"
        self._state.record_binary(self._fullname, self._origin, payload_sha256)

    def _delegate_with_authority(self, spec, callback):
        if self._authority_origin is None:
            raise RuntimeError("guarded extension authority path is unavailable")
        original_origin = spec.origin
        original_loader = spec.loader
        assert self._descriptor is not None
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        spec.origin = self._authority_origin
        spec.loader = self._delegate
        try:
            return callback()
        finally:
            spec.loader = original_loader
            spec.origin = original_origin

    def _verify_authority(self) -> None:
        if self._descriptor is None or self._identity is None or self._digest is None:
            raise RuntimeError("guarded extension authority was not captured")
        opened = os.fstat(self._descriptor)
        try:
            named = os.stat(self._origin, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ImportError(
                f"guarded extension changed while loading: {self._fullname}"
            ) from error
        if (
            not stat.S_ISREG(named.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            != self._identity
            or (named.st_dev, named.st_ino) != self._identity[:2]
        ):
            raise ImportError(
                f"guarded extension changed while loading: {self._fullname}"
            )
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(self._descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != self._digest:
            raise ImportError(
                f"guarded extension bytes changed while loading: {self._fullname}"
            )

    def _close_authority(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def create_module(self, spec):
        self._capture_authority()
        self._verify_authority()
        creator = getattr(self._delegate, "create_module", None)
        try:
            module = (
                None
                if creator is None
                else self._delegate_with_authority(spec, lambda: creator(spec))
            )
            self._verify_authority()
            return module
        except BaseException:
            self._close_authority()
            raise

    def exec_module(self, module: ModuleType) -> None:
        self._capture_authority()
        try:
            self._verify_authority()
            executor = getattr(self._delegate, "exec_module", None)
            if executor is None:
                raise ImportError(f"guarded extension has no executor: {self._fullname}")
            module_spec = getattr(module, "__spec__", None)
            if module_spec is None:
                executor(module)
            else:
                self._delegate_with_authority(
                    module_spec,
                    lambda: executor(module),
                )
            self._verify_authority()
        finally:
            self._close_authority()


class _GuardFinder(importlib.abc.MetaPathFinder):
    def __init__(self, state: _RuntimeProvenanceState) -> None:
        self._state = state

    def find_spec(self, fullname: str, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.origin in {None, "built-in", "frozen"}:
            return spec
        origin = Path(spec.origin)
        if not self._state.guards(origin):
            return spec
        if isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = _GuardedSourceLoader(fullname, str(origin), self._state)
        elif isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
            spec.loader = _GuardedBinaryLoader(
                fullname,
                origin,
                spec.loader,
                self._state,
            )
        else:
            raise ImportError(
                "guarded runtime origin uses an unsupported loader: "
                f"{fullname}: {type(spec.loader).__name__}"
            )
        return spec


_STATE: _RuntimeProvenanceState | None = None
_FINDER: _GuardFinder | None = None


def install_runtime_provenance(
    *,
    repository_root: str | Path,
    include_installed: bool = True,
    bootstrap_sources: Mapping[
        str,
        tuple[Path, bytes, CodeType],
    ] | None = None,
) -> Mapping[str, object]:
    """Capture guarded bytes and install the import loader before ML modules load."""

    global _FINDER, _STATE
    repository_root = Path(repository_root).resolve(strict=True)
    if _STATE is not None:
        if _STATE.repository_root != repository_root:
            raise RuntimeError("runtime provenance is already installed for another root")
        return _STATE.evidence()
    state = _RuntimeProvenanceState(
        repository_root=repository_root,
        include_installed=include_installed,
        bootstrap_sources={} if bootstrap_sources is None else bootstrap_sources,
    )
    finder = _GuardFinder(state)
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, finder)
    _STATE = state
    _FINDER = finder
    return state.evidence()


def freeze_runtime_provenance() -> Mapping[str, object]:
    """Reject any later import from guarded repository or environment roots."""

    if _STATE is None:
        return {
            "format_version": 1,
            "frozen": False,
            "native_dependency_scope": (
                "direct-extension-origin-bound; "
                "transitive-dyld-images-inventory-hashed-only"
            ),
            "modules": {},
        }
    with _STATE.lock:
        _STATE.frozen = True
    return _STATE.evidence()


def runtime_provenance_evidence() -> Mapping[str, object]:
    """Return exact source/direct-extension evidence and the native scope marker."""

    if _STATE is None:
        return {
            "format_version": 1,
            "frozen": False,
            "native_dependency_scope": (
                "direct-extension-origin-bound; "
                "transitive-dyld-images-inventory-hashed-only"
            ),
            "modules": {},
        }
    return _STATE.evidence()
