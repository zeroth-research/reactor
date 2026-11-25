"""Where the tests' inputs are, and the modules they define.

Every test names a fixture through these, so the layout of `tests/` is
written down once.
"""

from __future__ import annotations

from pathlib import Path

TESTS = Path(__file__).resolve().parent
PKG_ROOT = TESTS.parent                 # the `lean` project: its pyproject.toml
FIXTURES = TESTS / "fixtures"           # Python inputs, one module each
MODS = TESTS / "limits" / "mods"        # the `m_*.py` modules of the limit matrix
LAKE_PROJECT = TESTS / "lean"           # the Lake project the slow tests build in


def fixture(name: str):
    """The module `fixtures/<name>.py` defines, loaded as an input is."""
    from reactor_lean.parse import load_module

    return load_module(FIXTURES / f"{name}.py")


def all_encodings(module) -> str:
    """The functional encoding, the scalar and the scalar-relational one
    of `module`, as one text."""
    from reactor_lean import LeanContext, ModuleToLean4

    m2l = ModuleToLean4(LeanContext(module))
    return "\n\n".join([m2l.to_lean_functional(), m2l.to_lean_scalar(),
                        m2l.to_lean_rel()])


def relu(e: str, cmp: str = ">=") -> str:
    """`max(e, 0)` in SMT-LIB, the test written with `cmp`."""
    return f"(ite ({cmp} {e} 0) {e} 0)"


def dense(layers: int, cmp: str = ">=") -> str:
    """A ranking over `s0`: `layers` hidden layers of two ReLU units, each
    reading both units below, so its printed form shares subterms behind
    `let`."""
    a, b = "s0", "s0"
    for _ in range(layers):
        a, b = relu(f"(+ {a} {b})", cmp), relu(f"(+ {a} (- {b}))", cmp)
    return f"(+ {a} {b})"
