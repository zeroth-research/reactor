"""The `reactor` script the package installs. The other tests run the command
as `python -m reactor_lean.pipeline` (`reactor_cli`); this one runs the
entry point a user's `reactor` is."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_the_installed_reactor_script_runs():
    script = shutil.which("reactor", path=str(Path(sys.executable).parent))
    if script is None:
        pytest.skip("no `reactor` script beside this interpreter")
    r = subprocess.run([script, "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("usage: reactor")
