"""`reactor`, run as a user runs it -- a process of its own, from the
package root -- for the tests that check what it prints and writes.

It runs as `python -m reactor_lean.pipeline` under the interpreter running
the tests, not as `uv run reactor`: `uv run` syncs the environment first,
which can rebuild the `reactor` extension in the middle of a suite, and
costs its start-up on every run. `test_reactor_script.py` runs the script
the package installs.
"""

import subprocess
import sys

from testdata import PKG_ROOT


def reactor(*args) -> subprocess.CompletedProcess:
    """`reactor *args` from the package root, its output captured."""
    return subprocess.run(
        [sys.executable, "-m", "reactor_lean.pipeline", *args],
        capture_output=True,
        text=True,
        cwd=PKG_ROOT,
    )
