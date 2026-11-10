"""The error every stage of the pipeline refuses a run with."""

from __future__ import annotations


class Refused(ValueError):
    """This run cannot be carried out, and the message says why.

    A refusal is a decision the generator makes about its input -- an op with
    no Lean form, a state element the executable cannot print, a name the
    property does not have. It is not a defect, so `main` prints it as
    `error: ...` and exits; everything else keeps its traceback, which is
    what tells the two apart.

    `ValueError` is the base because that is what these sites raised before
    the type existed, and callers that catch it still do.
    """


class PredicateRefused(Refused):
    """A refusal about one of the claim's predicates, named by its `role`:
    "property" (of `kind` "safety" or "buchi"), "pre", "invariant" or
    "ranking" -- or any other label, for a term made up along the way.
    The layers that raise it know predicates by role; a claim's predicate
    is shown as the command-line flag it came from, which is where a user
    can act on it, and this is the one place that spelling lives."""

    def __init__(self, role: str, message: str, *, kind: str = "buchi"):
        super().__init__(message)
        self.role, self.kind, self.message = role, kind, message

    @property
    def flag(self) -> str:
        if self.role == "property":
            return f"--{self.kind}"
        if self.role in ("pre", "invariant", "ranking"):
            return f"--{self.role}"
        return self.role

    def __str__(self) -> str:
        return f"{self.flag}: {self.message}"
