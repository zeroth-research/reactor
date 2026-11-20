"""Running the route a command selected, reporting its refusals, and
proving what it found.

`route_precheck` puts the module to the route's own shape check before any
of the project is written; `run_route` runs it against the project and
checks what it handed back against what its row declares; `refused` says
a refusal under the flag it is about, and `note_refusal` leaves it in
`artifacts/` for the next run; `show_certificate` prints what it found in
the input's names; `regenerate` rewrites the project's certificate from it,
with its proofs, unless the route's answer is files of its own (`proves`);
`lakefile_parts` is what the project's lakefile needs for both. Both
commands that run a route -- `verith` and `certify` -- run it through
these.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from reactor_lean.core.certificate import Certificate, Claim
from reactor_lean.core.errors import Refused
from reactor_lean.pipeline.certificate import record_predicates, render
from reactor_lean.project.lakefile import LakefileParts

from .common.record import GaveUp, note
from .io import InferInput, InferResult, ProjectHandle
from .proofs import LAKEFILE, hint_budget, pre_check, prove
from .route import InferRoute

if TYPE_CHECKING:
    from reactor_lean.inputs import Names
    from reactor_lean.proof.render import LeanCertificate


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
    from reactor_lean.inputs import to_surface

    shown = [(flag, text) for flag, text in (("--invariant", cert.inv),
                                             ("--ranking", cert.ranking))
             if text]
    if not shown:
        return
    print(".. The certificate, in the input's names:")
    for flag, text in shown:
        print(f"   {flag} {shlex.quote(to_surface(text, names))}")


def proves(route: "InferRoute | None") -> bool:
    """Whether the run proves the certificate: unless its route answers
    with whole files, whose certificate is its own."""
    return route is None or route.returns != "files"


def lakefile_parts(route: "InferRoute | None", route_config,
                   project_name: str) -> LakefileParts:
    """What the project's lakefile needs beyond what `reactor` writes: the
    route's libraries and requires, and the hammer the proofs call."""
    parts = LakefileParts()
    if route is not None and route.lakefile is not None:
        parts += route.lakefile(route_config, project_name)
    if proves(route):
        parts += LAKEFILE
    return parts


def note_unrenderable(store, route, e: Refused) -> None:
    """Leave in `artifacts/` that a route's answer has no Lean form.

    Not a refusal by the route: it answered, and what failed was printing
    the answer as Lean, after the route had returned. So no note of the
    route's own can say this, and without one the next run reads a found
    predicate in `artifacts/` and nothing about why no certificate came of
    it -- and an LLM route resuming from it proposes the same shape again.
    """
    note(
        store, f"{e}\n",
        status="unsupported",
        what=f"An answer by `--infer {route.name}` that has no Lean form.",
        why=("The route answered; what failed is writing that answer as "
             "Lean, so the same shape proposed again fails the same way."),
    )


def regenerate(settings, smt, handle: ProjectHandle, claim: Claim,
               result: InferResult) -> "LeanCertificate | None":
    """The project's certificate, rewritten from `result` and proved:
    from the route's answer, or, without a route, from the certificate the
    run was given; and the files the route handed back, written. Returns
    what the certificate was rendered from, or `None` if nothing was.

    The certificate is that alone, under the claim the run was given, and
    its proofs are written for it. The definitions (`Data.lean`) and the
    proofs (`Proofs.lean`), read off their shape, are rewritten together,
    through the handle, so a route that writes more of the project and the
    proofs go through one writer. An answer in Lean has no terms for cvc5
    to read, so it is not pre-checked. Without a project (`verith` with no
    `--project`) there is nowhere to write, and only the pre-check runs. A
    route whose answer is files has proved its certificate itself, and only
    those are written.

    `settings` is the command's: `route`, `pre_check`, `smt_tactics`,
    `budget`.
    """
    cert, lean = result.cert, result.lean
    route = settings.route
    if not proves(route):
        handle.write_files(result.files)
        return None
    store = handle.artifacts
    if settings.pre_check == "cvc5":
        pre_check(smt, claim, lean or cert, settings.budget,
                  artifacts=store)
    if route is not None:
        record_predicates(store, claim, cert, inferred=True)
    if handle.dir is None:
        return None
    try:
        lean_cert = render(smt, claim, cert, verbatim=lean)
    except Refused as e:
        if route is None:
            raise SystemExit(f"error: {e}") from e
        note_unrenderable(store, route, e)
        # Under the route, whatever flag the refusal names: the predicate
        # it is about is the route's answer, not one the user passed.
        raise SystemExit(f"error: --infer {route.name}: {e}") from e
    proofs = prove(smt, handle.ctx, claim, cert, lean_cert,
                   budget=hint_budget(settings), restated=route is not None,
                   verbatim=lean)
    handle.write_certificate(lean_cert, proofs)
    handle.write_files(result.files)
    return lean_cert
