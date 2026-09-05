"""Prevent internal runbook vocabulary from returning to the public pitch."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DISALLOWED_PHRASES = (
    "operator",
    "the measured host",
    "pinned test case",
    "release surface",
    "authorization to actuate",
    "bounded integration evidence",
    "until then",
    "when the operator",
)


def test_readme_voice():
    readme = (ROOT / "README.md").read_text().lower()
    found = [phrase for phrase in DISALLOWED_PHRASES if phrase in readme]
    assert not found, f"Internal runbook language in README: {found}"
