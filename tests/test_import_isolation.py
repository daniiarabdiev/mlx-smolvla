import json
import subprocess
import sys


def test_runtime_import_excludes_reference_frameworks() -> None:
    code = """
import json
import sys

import mlx_smolvla
import mlx_smolvla.config
import mlx_smolvla.benchmark
import mlx_smolvla.connector
import mlx_smolvla.cli
import mlx_smolvla.expert
import mlx_smolvla.flow
import mlx_smolvla.language
import mlx_smolvla.policy
import mlx_smolvla.preprocessing
import mlx_smolvla.rmsnorm
import mlx_smolvla.statistical
import mlx_smolvla.types
import mlx_smolvla.vision
import training

loaded = {name.split('.', 1)[0] for name in sys.modules}
print(json.dumps({
    'forbidden': sorted(loaded & {'grpc', 'google', 'torch', 'lerobot', 'transformers'}),
    'training_api': sorted(name for name in vars(training) if not name.startswith('_')),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["forbidden"] == []
    assert payload["training_api"] == []
