import json
import subprocess
import sys


def test_runtime_import_excludes_reference_frameworks() -> None:
    code = """
import json
import sys

import smolvla_mlx
import smolvla_mlx.config
import smolvla_mlx.connector
import smolvla_mlx.expert
import smolvla_mlx.flow
import smolvla_mlx.language
import smolvla_mlx.policy
import smolvla_mlx.preprocessing
import smolvla_mlx.rmsnorm
import smolvla_mlx.types
import smolvla_mlx.vision

loaded = {name.split('.', 1)[0] for name in sys.modules}
print(json.dumps(sorted(loaded & {'torch', 'lerobot', 'transformers'})))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
