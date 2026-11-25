"""The certificate: the module checked for a Lean form (`lean_context`),
the project's `artifacts/` as a run sees it (`workspace`), and the
predicates recorded there and rendered as codegen takes them
(`record_predicates`, `render`).
"""

from __future__ import annotations


from collections.abc import Iterable
from pathlib import Path

from ..artifacts import ARTIFACTS_DIR, ArtifactStore, module_digest
from ..codegen.context import LeanContext
from ..codegen.terms import lean_gaps
from ..core.certificate import Certificate, Claim, LeanAnswer
from ..core.errors import Refused
from ..parse import PARSERS
from ..parse.native import read_predicates
from ..project.scaffold import selected_encodings
from ..proof.render import LeanCertificate, render_certificate
from ..smt.queries import predicate_terms


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


def render(smt, claim: "Claim | None", cert: Certificate = Certificate(), *,
           verbatim: "LeanAnswer | None" = None,
           parser: str = PARSERS[0]) -> "LeanCertificate | None":
    """The certificate as codegen takes it: the predicates parsed once, by
    `parser` (one of `parse.PARSERS`), and printed as Lean. `None` without
    a claim: a bare run parses nothing. `smt` is the module, its encoding
    or a factory for one; `verbatim` is an answer that is Lean already."""
    if claim is None:
        return None
    if parser == "native":
        q = read_predicates(_module_of(smt), claim, cert)
    elif parser == "cvc5":
        q = predicate_terms(smt, claim, cert)
    else:
        raise ValueError(f"no parser {parser!r}: one of {', '.join(PARSERS)}")
    return render_certificate(q, verbatim=verbatim, names=claim.names)


def _module_of(smt):
    """The module `render`'s `smt` is, or is the encoding of."""
    module = getattr(smt, "module", None)
    if module is not None:
        return module
    return smt().module if callable(smt) else smt


def lean_context(module, encodings: "Iterable[str] | None" = None) -> LeanContext:
    """The module's `LeanContext`, refusing a module with no Lean form, or,
    given the names of the `encodings` the run writes, with no form in one
    of them (`project.scaffold.selected_encodings`).

    An operator with no Lean form is a fact about the module, said here
    once, before any of the project exists, rather than as a missing table
    cell in the middle of a file. So is a module of several atoms, which
    only the functional encoding composes.

    Built once and threaded: its discovery pass interns every init/update
    term, and everything that prints the module needs the same one.
    """
    ctx = LeanContext(module)
    gaps = lean_gaps(ctx)
    if gaps:
        raise Refused("this module has no Lean form: " + "; also ".join(gaps))
    if encodings is not None:
        selected_encodings(ctx, encodings)
    return ctx
