"""The error every stage of the pipeline refuses a run with."""

from __future__ import annotations


class Refused(ValueError):
    """This run cannot be carried out, and the message says why.

    A refusal is a decision the generator makes about its input -- an op with
    no Lean form, a state element the executable cannot print, a name the
    property does not have. It is not a defect, so `main` prints it as
    `error: ...` and exits; everything else keeps its traceback, which is
    what tells the two apart.

    It is a `ValueError`: what is refused is a value the input gave.
    """


class Unreadable(Refused):
    """Text that is no predicate of this module in either dialect: the
    Python predicate syntax and wrong in it, SMT-LIB its parser cannot
    read, or nothing at all. It is what a parser raises about its text,
    and the one thing a caller turns into "cannot read"; anything else
    raised under it is a bug, and keeps its traceback."""


class PredicateRefused(Refused):
    """A refusal about one of the claim's predicates, named by its `role`:
    "property" (of `kind` "safety" or "buchi"), "invariant", "ranking",
    or an assumption on the inputs: "extl-assume" (of both blocks),
    "extl-assume-init" or "extl-assume-update".
    The layers that raise it know predicates by role; the refusal shows the
    command-line flag the predicate came from, which is where a user can
    act on it."""

    def __init__(self, role: str, message: str, *, kind: str = "buchi"):
        super().__init__(message)
        self.role, self.kind, self.message = role, kind, message

    @property
    def flag(self) -> str:
        return f"--{self.kind}" if self.role == "property" else f"--{self.role}"

    def __str__(self) -> str:
        return f"{self.flag}: {self.message}"
