"""Reading the input: the module file (`module`), and the claims made of
it, into reactor's terms.

`load` reads a Python file into a `Problem`: the module, the text it was
read from, and the names a claim may call its variables by. A claim is
written in those names; `names` rewrites it to the internal ones every
later stage reads (`to_internal`).
`native` then reads a predicate's text into reactor's terms, the IR a
module is written in, which `proof.from_ir` prints: the Python predicate
syntax first (`dsl`), and SMT-LIB for text that is no Python (`sexpr` and
`smtlib`).
"""

from .module import InputError, Named, Problem, SourceText, load, load_module
from .names import ClaimError, Names, to_internal

__all__ = [
    "ClaimError", "InputError", "Named", "Names", "Problem", "SourceText",
    "load", "load_module", "to_internal",
]
