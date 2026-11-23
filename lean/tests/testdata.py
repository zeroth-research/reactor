"""Where the tests' inputs are, and the modules they define.

Every test names a fixture through these, so the layout of `tests/` is
written down once.
"""

from __future__ import annotations

from pathlib import Path

TESTS = Path(__file__).resolve().parent
PKG_ROOT = TESTS.parent                 # the `lean` project: its pyproject.toml
FIXTURES = TESTS / "fixtures"           # Python inputs, one module each
SMV = TESTS / "smv_fixtures"            # SMV inputs
MODS = TESTS / "limits" / "mods"        # the `m_*.py` modules of the limit matrix
LAKE_PROJECT = TESTS / "lean"           # the Lake project the slow tests build in


def fixture(name: str):
    """The module `fixtures/<name>.py` defines, loaded as an input is."""
    from reactor_lean.inputs.python import load_module

    return load_module(FIXTURES / f"{name}.py")
