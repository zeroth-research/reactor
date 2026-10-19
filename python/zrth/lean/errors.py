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

