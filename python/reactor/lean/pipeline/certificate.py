"""The certificate: the module checked for a Lean form (`lean_context`),
the project's `artifacts/` as a run sees it (`workspace`), and the
predicates recorded there and rendered as codegen takes them
(`record_predicates`, `render`).
"""

from __future__ import annotations


from pathlib import Path

from ..artifacts import ARTIFACTS_DIR, ArtifactStore, module_digest
from ..cli import Settings
from ..codegen.context import LeanContext
from ..codegen.terms import lean_gaps
from ..core.certificate import Certificate, Claim, LeanAnswer
from ..core.errors import Refused
from ..proof.facts import solver_hints
from ..proof.render import LeanCertificate, prepare
from ..smt.limits import SmtBudget
from ..smt.precheck import predicate_terms


# Each predicate recorded: its role, the `Certificate` field it is read from
# (`None` for the claim's property), and what it is.
_PREDICATES = (
    ("property", None,
     "The property this run was asked to prove, as it was given to "
     "`--safety` / `--buchi`."),
    ("inv", "inv",
     "The invariant the certificate is generated from."),
    ("ranking", "ranking",
     "The ranking function the certificate is generated from."),
)


def record_predicates(store, claim: "Claim | None", cert: Certificate, *,
                      inferred: bool = False) -> None:
    """Leave the certificate's predicates in `artifacts/`, as SMT-LIB.

    They are what every obligation beside them is stated against, so a reader
    comparing `system.smt2` with a refuted obligation has the third side of it
    here rather than having to dig it out of `Certificate/Data.lean`, where it is
    Lean and no longer the text any solver was given. An answer that is Lean
    only (`LeanAnswer`) has no SMT-LIB to record. `inferred` marks the
    certificate's predicates, never the property, as found by the run
    rather than given to it.
    """
    if store is None or claim is None:
        return
    for role, field, what in _PREDICATES:
        src = claim.prp if field is None else getattr(cert, field)
        if not (src and src.strip()):
            continue
        if inferred and role != "property":
            what += " Inferred by this run's route."
        store.encoded(role, role, src.strip(), language="smt-src", what=what)


def workspace(project_dir: Path, module, claim: "Claim | None", *,
              kind: str, mode: str = "use",
              producer: str = "") -> ArtifactStore:
    """The project's `artifacts/`, as this run sees it.

    `artifacts/` outlives the run that wrote it, so a sequence of runs in
    one `-o` continues rather than restarts. The three fields that are not
    the directory are what stop a stale artifact being trusted: an invariant
    found for another property, another proof rule or another module is not
    about this run, and `usable()` says so rather than silently omitting it.
    `mode` is `--artifacts`; `producer` names what writes, if anything
    other than the run itself does.
    """
    store = ArtifactStore(
        dir=project_dir / ARTIFACTS_DIR,
        module_digest=module_digest(module),
        kind=kind,
        prp=(claim.prp if claim else None) or "",
        producer=producer,
        enabled=mode == "use",
        log=print,
    )
    if mode == "reset":
        gone = store.reset()
        print(f".. artifacts: cleared {gone} file(s) from {store.dir}")
    return store


def hint_budget(settings: Settings) -> "SmtBudget | None":
    """What cvc5's solver hints are asked under: the run's budget with
    `--smt-tactics cvc5`, and `None`, no hints, without it."""
    return settings.budget if settings.smt_tactics == "cvc5" else None


def render(smt, claim: "Claim | None", cert: Certificate, *,
           hints: "SmtBudget | None", restated: bool = False,
           verbatim: "LeanAnswer | None" = None) -> "LeanCertificate | None":
    """The certificate as codegen takes it. `None` without a claim: a bare
    run asks cvc5 nothing. `smt` is the run's `smt.encoding.lazy` encoding.

    The predicates are parsed once, and that one parse is printed as Lean,
    read for its shape (`predicate_facts`) and, given a `hints` budget, put
    to `solver_hints` under it (`proof.render.prepare` says how the two
    meet). `verbatim` is an answer that is Lean already.
    """
    if claim is None:
        return None
    q = predicate_terms(smt, claim, cert)
    h = None
    if hints is not None:
        restating = ", restated over the inferred predicates" if restated else ""
        print(
            f".. SMT-informed tactics (cvc5){restating}: "
            f"<={hints.per_call_ms} ms per query, "
            f"<={hints.phase_ms} ms total"
        )
        h = solver_hints(q, hints)
    lean = prepare(q, h, verbatim)
    if h is not None:
        reasons = [ln.removeprefix("-- cvc5: ")
                   for ln in lean.plan.hints.why().splitlines()]
        for reason in reasons or ["nothing cvc5 can add to the plan"]:
            print(f"   {reason}")
    return lean


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
