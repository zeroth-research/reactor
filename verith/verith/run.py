"""Running the route a command selected, and reporting its refusals.

`route_precheck` puts the module to the route's own shape check before any
of the project is written; `run_route` runs it against the project and
checks what it handed back against what its row declares; `refused` says
a refusal under the flag it is about, and `note_refusal` leaves it in
`artifacts/` for the next run; `show_certificate` prints what it found in
the input's names. Both commands that run a route -- `verith` and
`certify` -- run it through these.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from reactor.lean.core.certificate import Certificate, Claim
from reactor.lean.core.errors import Refused

from .common.record import GaveUp
from .io import InferInput, InferResult, ProjectHandle
from .route import InferRoute

if TYPE_CHECKING:
    from reactor.lean.inputs import Names


def refused(route: "InferRoute | None", error: Exception) -> SystemExit:
    """A refusal, under the flag it is about: its own `flag` when it names
    one -- `--infer sygus`, `--ic3ia`, `--pre` -- and else the route's.
    With no route there is no flag to name: the refusal is about the
    module.
    """
    if route is None or getattr(error, "flag", None):
        return SystemExit(f"error: {error}")
    return SystemExit(f"error: --infer {route.name}: {error}")


def note_refusal(handle: ProjectHandle, route, e: Refused) -> None:
    """Leave *why* a route gave up in `artifacts/`.

    `NO-CERT` is three measurements wearing one name -- a space proved
    empty, a budget that ran out, a shape the route never looked at -- and
    the prose that distinguishes them is a dozen wordings per route, which
    is why reading it back is guesswork. The status is not: a route that
    searched a bounded shape and refuted it writes `no_solution`, one whose
    budget ran out writes `unknown`, and everything else is a decline.

    A route that searched and has more to say than this raises `GaveUp`,
    which carries its note -- `smt-linear`, `sygus`, `houdini` and
    `vampire` do -- and that note is written instead.

    A `setup` refusal is not about the module at all -- a package or a key
    is missing -- so it is left unrecorded rather than filed against a run
    that never started.
    """
    store = handle.artifacts
    if store is None or e.setup:
        return
    if isinstance(e, GaveUp):
        store.note(e.text, status=e.status, what=e.what, why=e.why)
        return
    searched = e.searched
    store.note(
        f"{e}\n",
        status="unknown" if searched else "declined",
        what=(f"A search by `--infer {route.name}` that did not finish."
              if searched else
              f"`--infer {route.name}` declined this module and property."),
        why=("The budget ran out first; nothing is known about the space."
             if searched else
             "Nothing was searched, so this says what the route does not "
             "take rather than anything about the module's difficulty."),
    )


def run_route(
    settings, module, handle: ProjectHandle, claim: Claim,
    given: Certificate, prechecked, smt=None,
) -> "InferResult":
    """Run the selected route against the generated project.

    Returns what it handed back: predicates, or -- for a route whose row
    says `returns="files"` -- whole files, with no predicates for the
    project to be regenerated from.

    Everything route-specific is a field of the row: what of the project it
    may read and write, which artifact roles it resumes from, and which
    languages its answer comes back in. `settings` is the command's:
    `route`, `route_opts`, `route_config`, `budget`.
    """
    route = settings.route
    try:
        result = route.run(
            InferInput(
                project=handle,
                module=module,
                claim=claim,
                given=given,
                opts=settings.route_opts,
                config=settings.route_config,
                prechecked=prechecked,
                budget=settings.budget,
                log=print,
                smt=smt,
            )
        )
    except Refused as e:
        note_refusal(handle, route, e)
        raise refused(route, e) from e

    # The row *declares* what comes back and the result carries it, so the
    # two are checked against each other once rather than each reader
    # trusting whichever is nearer to hand.
    agrees = {
        "smt": not result.cert.empty and result.lean is None,
        "lean": result.lean is not None,
        "files": bool(result.files) and result.cert.empty
                 and result.lean is None,
    }[route.returns]
    if not agrees:
        if result.cert.empty and result.lean is None:
            handed = ("files and no predicates" if result.files
                      else "no predicates")
        else:
            handed = "predicates"
            handed += " in Lean" if result.lean is not None else " in SMT-LIB"
            if result.files:
                handed += " and files"
        raise SystemExit(
            f"error: --infer {route.name}: the route's row declares "
            f"returns={route.returns!r} but it handed back {handed}"
        )
    for rel in result.files:
        if not handle.owns_path(rel):
            raise SystemExit(
                f"error: --infer {route.name}: the route handed back {rel}, "
                f"which its row does not own"
            )
    return result


def route_precheck(settings, module, claim: "Claim | None", smt):
    """The route's own shape check, before a line of Lean is written; what
    it computed reaches the route as `InferInput.prechecked`.

    Left until the route runs, a module its encoding cannot express is met
    first by `create_project` -- which fails about the functional encoding,
    in a traceback, for a route that was never going to use it. The check
    reads the claim as well: `--pre` on `--infer fbk-proveit` is answerable
    only by the encoding, so it is answered where the encoding is made.
    """
    route = settings.route
    if route is None or route.precheck is None:
        return None
    try:
        return route.precheck(
            module, settings.route_opts, settings.route_config, claim, smt
        )
    except Refused as e:
        raise refused(route, e) from e


def show_certificate(cert: Certificate, names: "Names") -> None:
    """A certificate in the input's names, as `--invariant` and
    `--ranking` take it: what one run found, the next can be given."""
    from reactor.lean.inputs import to_surface

    shown = [(flag, text) for flag, text in (("--invariant", cert.inv),
                                             ("--ranking", cert.ranking))
             if text]
    if not shown:
        return
    print(".. The certificate, in the input's names:")
    for flag, text in shown:
        print(f"   {flag} {shlex.quote(to_surface(text, names))}")
