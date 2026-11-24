"""The names the generated Lean project is made of, in one place.

Codegen writes the project's files and modules under these names, and the
rest of the pipeline refers to them: the lake build target and its `sorry`
filter, among others. Each is defined here once, so a reference cannot
drift from what was written. A leaf: it imports nothing from the package,
so anything can use it. The templates under `templates/project/` spell the
same names, and `tests/test_lean_names.py` holds the two together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    """Where one route puts an encoding and what Lean calls it.

    `base` is the name of the row with no suffix -- the functional encoding
    -- and every other row hangs off it. In the project the files live in
    their own `System/` directory and are named by suffix alone; a
    `--cert-file` writes them next to whatever else is in the output
    directory, so there the names carry `base` too (`flat`).
    """

    base: str
    prefix: str = ""            # Lean module prefix, e.g. "System."
    flat: bool = False          # file names carry `base`
    directory: Path = Path()    # only `path` uses it

    def stem(self, suffix: str) -> str:
        return f"{self.base}{suffix}" if self.flat else (suffix or self.base)

    def path(self, suffix: str) -> Path:
        return self.directory / f"{self.stem(suffix)}.lean"

    def lean_module(self, suffix: str) -> str:
        return f"{self.prefix}{self.stem(suffix)}"


# The project's own layout, also the one `generate_root` imports through.
SYSTEM_LAYOUT = Layout(base="System", prefix="System.")


# The certificate's library: the `Certificate/` directory and the lean_lib
# that globs it, so every file in it is built.
CERT_LIB = "Certificate"
DATA = "Data"                   # the predicates (`Data.lean.j2`)
MODULE = "Module"               # `RM` and `lts`, fixed text (`Module.lean.j2`)
PROOFS = "Proofs"               # the obligations' proofs (`Proofs.lean.j2`)
CERTIFICATE = "Certificate"     # the certificate, fixed text (`Certificate.lean.j2`)


# The linters every generated Lean file turns off, after its imports. Each
# proof is one tactic cascade for every goal shape, so on any one goal some
# of its steps do nothing and some of its simp lemmas go unused; and the
# encodings' binders are positional, used or not. Those five linters said
# so 1063 times over the test suite's certificates, which buried the one
# warning a build is read for: `declaration uses 'sorry'`.
LINTERS_OFF = "\n".join(
    f"set_option linter.{name} false"
    for name in (
        "unusedVariables", "unusedSimpArgs", "unusedTactic",
        "unreachableTactic", "unnecessarySeqFocus",
    )
)


def cert_file(stem: str) -> str:
    """The certificate library's file `stem`, relative to the project."""
    return f"{CERT_LIB}/{stem}.lean"

