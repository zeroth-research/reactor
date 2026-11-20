"""The certificate's proofs: what replaces the `sorry`s `reactor` leaves.

`reactor` generates the project with each obligation stated in
`Certificate/Proofs.lean` and proved by `sorry`. This package writes the
proofs. `scripts` reads the tactic plan off the module and the certificate
(`tactics`), with what the predicates' terms say and what cvc5 settles in
advance (`facts`), and hands it to `reactor` as `ProofScripts`, which puts
each script under its statement. The tactics the scripts call are the
hammer's (`ZerothHammer.lean`), which goes into the project with them.
`precheck` asks cvc5 whether the obligations are true at all, and `build`
builds the certificate.

The stages the two commands that write proofs -- `verith` and `certify` --
share are here: `prove` (with `hint_budget`), `hammer_files`, `build_cert`
and `hammer_only`, with `LAKEFILE`, what the project's lakefile needs for
them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from reactor_lean.project.lakefile import LakefileParts, LeanLib
from reactor_lean.proof.render import ProofScripts

from ..common.encoding import predicate_terms
from .build import LakeBuildError, build_certificate
from .facts import solver_hints
from .precheck import pre_check
from .scripts import HAMMER_LIB, print_hints, proof_scripts, render

if TYPE_CHECKING:
    from reactor_lean.codegen.context import LeanContext
    from reactor_lean.core.certificate import Certificate, Claim, LeanAnswer
    from reactor_lean.proof.render import LeanCertificate

    from ..common.limits import SmtBudget

__all__ = [
    "HAMMER", "LAKEFILE", "build_cert", "hammer_files", "hammer_only",
    "hint_budget", "pre_check", "prove",
]


# The tactics the proofs call, at the project's root.
HAMMER = f"{HAMMER_LIB}.lean"

# What the lakefile needs for the proofs: the hammer, as a lean_lib.
LAKEFILE = LakefileParts(libs=(
    LeanLib(HAMMER_LIB, comment=(
        "The tactics the certificate's proofs call: `zeroth_hammer`, "
        "`lra_close`."
    )),
))


def generate_zeroth_hammer_lean() -> str:
    """ZerothHammer.lean: the hammer's tactics, the same for every project."""
    return render("ZerothHammer.lean.j2")


def hammer_files() -> dict[str, str]:
    """The hammer, by project-relative path: what a project whose proofs
    are these needs besides `Certificate/`."""
    return {HAMMER: generate_zeroth_hammer_lean()}


def hint_budget(settings) -> "SmtBudget | None":
    """What cvc5's solver hints are asked under: the run's budget with
    `--smt-tactics cvc5`, and `None`, no hints, without it."""
    return settings.budget if settings.smt_tactics == "cvc5" else None


def prove(smt, ctx: "LeanContext", claim: "Claim | None",
          cert: "Certificate", lean_cert: "LeanCertificate | None", *,
          budget: "SmtBudget | None" = None,
          verbatim: "LeanAnswer | None" = None,
          restated: bool = False) -> ProofScripts:
    """The proofs of the certificate `lean_cert` prints, read off the module
    (`ctx`) and the certificate's terms: `claim` and `cert` as `reactor`
    parsed them, against `smt`, the run's encoding. Without a claim there
    are no terms, and the plan reads the module alone.

    With a `budget` (`--smt-tactics cvc5`) cvc5 is asked under it what it
    can settle in advance, and each thing it settled is said; `restated`
    says the predicates are the inferred ones. `verbatim` is an answer that
    is Lean already, which no term stands behind, so cvc5 is asked nothing.
    """
    if claim is None:
        return proof_scripts(ctx, lean_cert)
    q = predicate_terms(smt, claim, cert)
    printed = None
    if budget is not None and verbatim is None:
        restating = ", restated over the inferred predicates" if restated else ""
        print(
            f".. SMT-informed tactics (cvc5){restating}: "
            f"<={budget.per_call_ms} ms per query, "
            f"<={budget.phase_ms} ms total"
        )
        printed = print_hints(solver_hints(q, budget), q.msmt.ctrl_next)
        reasons = [ln.removeprefix("-- cvc5: ")
                   for ln in printed.why().splitlines()]
        for reason in reasons or ["nothing cvc5 can add to the plan"]:
            print(f"   {reason}")
    return proof_scripts(ctx, lean_cert, q, printed, verbatim)


def build_cert(project_dir: Path, lean_cert: "LeanCertificate | None") -> None:
    """`--build-cert`, once the certificate is written.

    `lean_cert` is what the project's certificate was generated from, or
    `None` when its files came from elsewhere. The predicates are re-checked
    here rather than only at parse time because the parse-time check sees
    the flags, not the certificate a caller hands over: one with `inv` or
    `ranking` unset would build as a proof of `sorry`.
    """
    if lean_cert is not None:
        # A safety certificate has no ranking function to be missing.
        required = [("property", lean_cert.prp), ("invariant", lean_cert.inv)]
        if not lean_cert.is_safety:
            required.append(("ranking", lean_cert.ranking))
        missing = [name for name, value in required if not value]
        if missing:
            raise SystemExit(
                f"error: --build-cert: the certificate has no "
                f"{', '.join(missing)}, so every obligation it carries is "
                "`sorry`."
            )
    print(f".. Building the certificate in {project_dir}")
    try:
        build_certificate(project_dir)
    except LakeBuildError as e:
        raise SystemExit(f"error: --build-cert: {e}") from e


def hammer_only(hammer_file: str) -> None:
    """`--hammer-file`: write ZerothHammer.lean and return. No module needed."""
    out = Path(hammer_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_zeroth_hammer_lean())
    except OSError as e:
        raise SystemExit(f"error: --hammer-file: {e}") from e
    print(f"Wrote {out}")
