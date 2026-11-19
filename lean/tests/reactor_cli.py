"""`uv run reactor`, run as a user runs it, for the tests that check what it
prints and writes."""

import subprocess
from pathlib import Path

# Root of the `lean` project (where its pyproject.toml lives).
PKG_ROOT = Path(__file__).parent.parent


def reactor(*args) -> subprocess.CompletedProcess:
    """`uv run reactor *args` from the package root, its output captured."""
    return subprocess.run(
        ["uv", "run", "reactor", *args],
        capture_output=True,
        text=True,
        cwd=PKG_ROOT,
    )
