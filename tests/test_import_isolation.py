import json
import subprocess
import sys


def test_runtime_import_excludes_reference_frameworks() -> None:
    code = """
import json
import sys

import smolvla_mlx

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
