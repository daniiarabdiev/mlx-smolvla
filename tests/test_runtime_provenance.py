"""Fresh-process gates for the T3B import-provenance bootstrap."""

from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.slow


def _run_probe(repository_root: Path, script: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_import_bootstrap_rejects_source_swapped_after_capture(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    victim = guarded / "victim.py"
    victim.write_text("VALUE = 'captured'\n", encoding="utf-8")
    probe = _run_probe(
        repository_root,
        f"""
import sys
from pathlib import Path
from training.runtime_provenance import install_runtime_provenance
root = Path({str(guarded)!r})
sys.path.insert(0, str(root))
install_runtime_provenance(repository_root=root, include_installed=False)
(root / 'victim.py').write_text("VALUE = 'replacement'\\n", encoding='utf-8')
try:
    import victim
except ImportError as error:
    print(type(error).__name__, str(error))
else:
    raise SystemExit(f'replacement executed: {{victim.VALUE}}')
""",
    )
    assert probe.returncode == 0, probe.stderr
    assert "changed after provenance capture" in probe.stdout


def test_import_bootstrap_rejects_new_guarded_modules_after_freeze(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    (guarded / "first.py").write_text("VALUE = 1\n", encoding="utf-8")
    (guarded / "late.py").write_text("VALUE = 2\n", encoding="utf-8")
    probe = _run_probe(
        repository_root,
        f"""
import sys
from pathlib import Path
from training.runtime_provenance import (
    freeze_runtime_provenance,
    install_runtime_provenance,
    runtime_provenance_evidence,
)
root = Path({str(guarded)!r})
sys.path.insert(0, str(root))
install_runtime_provenance(repository_root=root, include_installed=False)
import first
before = runtime_provenance_evidence()
assert 'first' in before['modules']
freeze_runtime_provenance()
try:
    import late
except ImportError as error:
    print(type(error).__name__, str(error))
else:
    raise SystemExit(f'late module executed: {{late.VALUE}}')
""",
    )
    assert probe.returncode == 0, probe.stderr
    assert "after provenance freeze" in probe.stdout


def test_import_bootstrap_rejects_a_preloaded_guarded_module(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    (guarded / "preloaded.py").write_text("VALUE = 'already ran'\n", encoding="utf-8")
    probe = _run_probe(
        repository_root,
        f"""
import sys
from pathlib import Path
from training.runtime_provenance import install_runtime_provenance
root = Path({str(guarded)!r})
sys.path.insert(0, str(root))
import preloaded
try:
    install_runtime_provenance(repository_root=root, include_installed=False)
except RuntimeError as error:
    print(type(error).__name__, str(error))
else:
    raise SystemExit('preloaded guarded module was accepted')
""",
    )
    assert probe.returncode == 0, probe.stderr
    assert "preloaded guarded module" in probe.stdout


def test_binary_loader_verifies_origin_before_delegate_create_module(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    origin = guarded / "native.so"
    origin.write_bytes(b"captured native bytes")
    probe = _run_probe(
        repository_root,
        f"""
from pathlib import Path
from types import SimpleNamespace
import training.runtime_provenance as provenance
root = Path({str(guarded)!r})
origin = root / 'native.so'
provenance.install_runtime_provenance(repository_root=root, include_installed=False)
events = []
class Delegate:
    def create_module(self, spec):
        events.append('delegate-create')
        origin.rename(root / 'captured.so')
        origin.write_bytes(b'replacement native bytes')
        return SimpleNamespace()
    def exec_module(self, module):
        events.append('delegate-exec')
loader = provenance._GuardedBinaryLoader(
    'native', origin, Delegate(), provenance._STATE
)
spec = SimpleNamespace(origin=str(origin), loader=loader)
try:
    loader.create_module(spec)
except ImportError as error:
    print(type(error).__name__, str(error), events)
else:
    raise SystemExit(f'create-time replacement was accepted: {{events}}')
""",
    )
    assert probe.returncode == 0, probe.stderr
    assert "changed while loading" in probe.stdout
    assert "delegate-create" in probe.stdout


def test_binary_loader_delegates_through_the_retained_inode_during_aba(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    origin = guarded / "native.so"
    origin.write_bytes(b"captured native bytes")
    probe = _run_probe(
        repository_root,
        f"""
from pathlib import Path
from types import SimpleNamespace
import training.runtime_provenance as provenance
root = Path({str(guarded)!r})
origin = root / 'native.so'
provenance.install_runtime_provenance(repository_root=root, include_installed=False)
executed = []
class Delegate:
    def create_module(self, spec):
        original = root / 'captured.so'
        origin.rename(original)
        origin.write_bytes(b'malicious replacement')
        executed.append(Path(spec.origin).read_bytes())
        origin.unlink()
        original.rename(origin)
        return SimpleNamespace()
    def exec_module(self, module):
        pass
loader = provenance._GuardedBinaryLoader(
    'native', origin, Delegate(), provenance._STATE
)
spec = SimpleNamespace(origin=str(origin), loader=loader)
loader.create_module(spec)
assert executed == [b'captured native bytes'], executed
assert spec.origin == str(origin)
print('retained-authority-ok')
""",
    )
    assert probe.returncode == 0, probe.stderr
    assert "retained-authority-ok" in probe.stdout


def test_import_bootstrap_rejects_a_guarded_sourceless_loader_after_freeze(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    probe = _run_probe(
        repository_root,
        f"""
import py_compile
import sys
from pathlib import Path
from training.runtime_provenance import (
    freeze_runtime_provenance,
    install_runtime_provenance,
)
root = Path({str(guarded)!r})
source = root / 'late.py'
source.write_text("VALUE = 'sourceless executed'\\n", encoding='utf-8')
py_compile.compile(str(source), cfile=str(root / 'late.pyc'), doraise=True)
source.unlink()
sys.path.insert(0, str(root))
install_runtime_provenance(repository_root=root, include_installed=False)
freeze_runtime_provenance()
try:
    import late
except ImportError as error:
    print(type(error).__name__, str(error))
else:
    raise SystemExit(f'sourceless guarded module executed: {{late.VALUE}}')
""",
    )
    assert probe.returncode == 0, probe.stderr
    assert "unsupported loader" in probe.stdout


def test_isolated_launcher_verifies_the_real_runtime_inventory() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    launcher = repository_root / "scripts" / "finetune_lora"
    documents = []
    for _ in range(2):
        probe = subprocess.run(
            [
                str(launcher),
                "--verify-runtime-only",
            ],
            cwd=repository_root,
            env=dict(os.environ),
            text=True,
            capture_output=True,
            check=False,
        )
        assert probe.returncode == 0, probe.stderr
        documents.append(json.loads(probe.stdout))
    assert all(document["runtime_provenance_verified"] is True for document in documents)
    assert all(document["runtime_provenance_frozen"] is True for document in documents)
    assert all(document["runtime_provenance_module_count"] > 0 for document in documents)
    assert all(
        document["native_dependency_scope"]
        == "direct-extension-origin-bound; transitive-dyld-images-inventory-hashed-only"
        for document in documents
    )
    assert len(documents[0]["implementation_sha256"]) == 64
    assert documents[0] == documents[1]


def test_shell_launcher_preserves_caller_cwd_and_argument_boundaries(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    probe_root = tmp_path / "probe-repository"
    scripts = probe_root / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / "finetune_lora"
    shutil.copyfile(repository_root / "scripts" / "finetune_lora", launcher)
    launcher.chmod(0o755)
    entrypoint = scripts / "finetune_lora.py"
    entrypoint.write_text("# probe entrypoint\n", encoding="utf-8")
    python = probe_root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/sh\n"
        "printf '<cwd=%s>\\n' \"$PWD\"\n"
        "for argument in \"$@\"; do printf '<arg=%s>\\n' \"$argument\"; done\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    caller = tmp_path / "caller directory"
    caller.mkdir()

    probe = subprocess.run(
        [str(launcher), "--probe", "value with spaces"],
        cwd=caller,
        text=True,
        capture_output=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == [
        f"<cwd={caller}>",
        "<arg=-I>",
        "<arg=-S>",
        f"<arg={entrypoint}>",
        "<arg=--probe>",
        "<arg=value with spaces>",
    ]


def test_python_entrypoint_rejects_unisolated_startup() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "finetune_lora.py"),
            "--verify-runtime-only",
        ],
        cwd=repository_root,
        env=dict(os.environ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert probe.returncode != 0
    assert "scripts/finetune_lora" in probe.stderr
    assert "-I -S" in probe.stderr


def test_fixed_t3b_public_apis_reject_an_unisolated_uninstalled_runtime(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    probe = _run_probe(
        repository_root,
        f"""
from pathlib import Path
from training.finetune import (
    FIXED_BUDGET_MODE,
    FineTuneConfig,
    prepare_lora_finetune_launch,
    run_lora_finetune,
)
config = FineTuneConfig(
    output_dir=Path({str(tmp_path / 'run')!r}),
    lora_scope='expert_only',
    budget_mode=FIXED_BUDGET_MODE,
)
for name, callback in (
    ('prepare', lambda: prepare_lora_finetune_launch(config)),
    ('run', lambda: run_lora_finetune(config, training_log_path=config.output_dir / 'training.log')),
):
    try:
        callback()
    except RuntimeError as error:
        assert 'isolated -I -S launcher' in str(error), error
        print(name, 'rejected')
    else:
        raise SystemExit(f'{{name}} accepted an unisolated runtime')
""",
    )
    assert probe.returncode == 0, probe.stderr
    assert "prepare rejected" in probe.stdout
    assert "run rejected" in probe.stdout


def test_isolated_launcher_ignores_pythonpath_shadow_modules(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    shadow_root = tmp_path / "pythonpath-shadow"
    shadow_root.mkdir()
    marker = tmp_path / "shadow-imported.txt"
    mutation_target = tmp_path / "must-remain-unchanged.txt"
    mutation_target.write_text("original\n", encoding="utf-8")
    (shadow_root / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['SMOLVLA_SHADOW_MARKER']).write_text('sitecustomize executed\\n')\n"
        "Path(os.environ['SMOLVLA_MUTATION_TARGET']).write_text('mutated\\n')\n",
        encoding="utf-8",
    )
    (shadow_root / "mlx.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['SMOLVLA_SHADOW_MARKER']).write_text('executed\\n')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(shadow_root)
    environment["SMOLVLA_SHADOW_MARKER"] = str(marker)
    environment["SMOLVLA_MUTATION_TARGET"] = str(mutation_target)
    probe = subprocess.run(
        [
            str(repository_root / "scripts" / "finetune_lora"),
            "--verify-runtime-only",
        ],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    document = json.loads(probe.stdout)
    assert document["isolated"] is True
    assert document["no_site"] is True
    assert str(shadow_root) not in document["sys_path"]
    assert not marker.exists()
    assert mutation_target.read_text(encoding="utf-8") == "original\n"
