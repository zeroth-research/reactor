"""The certificate: the module checked for a Lean form (`lean_context`),
and the predicates rendered as codegen takes them (`render`).
"""

from __future__ import annotations

from ..codegen.context import LeanContext
from ..codegen.terms import lean_gaps
from ..core.certificate import Certificate, Claim
from ..core.errors import Refused
from ..parse.native import read_predicates
from ..proof.render import LeanCertificate, render_certificate


def render(module, claim: "Claim | None",
           cert: Certificate = Certificate()) -> "LeanCertificate | None":
    """The certificate as codegen takes it: the predicates parsed once and
    printed as Lean. `None` without a claim: a bare run parses nothing."""
    if claim is None:
        return None
    return render_certificate(read_predicates(module, claim, cert))


def lean_context(module) -> LeanContext:
    """The module's `LeanContext`, refusing a module with no Lean form.

    An operator with no Lean form is a fact about the module, said here
    once, before any of the project exists, rather than as a missing table
    cell in the middle of a file.

    Built once and threaded: its discovery pass interns every init/update
    term, and everything that prints the module needs the same one.
    """
    ctx = LeanContext(module)
    gaps = lean_gaps(ctx)
    if gaps:
        raise Refused("this module has no Lean form: " + "; also ".join(gaps))
    return ctx
