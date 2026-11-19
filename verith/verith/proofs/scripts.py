"""The certificate's proofs, as `reactor` takes them (`ProofScripts`).

`reactor` states the obligations in `Certificate/Proofs.lean` and proves
each by `sorry`; this is what replaces the `sorry`s. `proof_scripts` reads
the tactic plan off the module and the certificate (`tactics.plan_for`),
with what the predicates' terms say (`facts.predicate_facts`) and what cvc5
settled in advance (`facts.solver_hints`, printed by `print_hints`), and
renders it (`templates/scripts.lean.j2`): the budgets the proofs run under,
the `simp` macros they share, and each obligation's tactic proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import FileSystemLoader

from reactor_lean.core.certificate import LeanAnswer
from reactor_lean.core.errors import Refused
from reactor_lean.project.templates import env as reactor_env
from reactor_lean.proof.render import LeanCertificate, ProofScripts

from .facts import predicate_facts
from .tactics import DSIMP_ERASE, MAT_SIMP, plan_for

if TYPE_CHECKING:
    from reactor_lean.codegen.context import LeanContext
    from reactor_lean.smt.queries import ModuleQueries

    from .facts import SolverHints


TEMPLATES = Path(__file__).parent / "templates"

# `reactor`'s environment over these templates: the scripts are spliced
# into what it renders, so they are rendered with its whitespace rules.
_env = reactor_env.overlay(loader=FileSystemLoader(str(TEMPLATES)))


def render(template: str, **context: object) -> str:
    """Render one of `templates/`."""
    return _env.get_template(template).render(**context)


# The hammer: the library of the tactics the scripts call.
HAMMER_LIB = "ZerothHammer"

# What the scripts use beyond the statements: the `Fin` sums `simp_mat`
# reduces with, and the hammer's tactics.
IMPORTS = ("Mathlib.Algebra.BigOperators.Fin", HAMMER_LIB)


# ══════════════════════════════════════════════════════════════════════════
# What cvc5 settled, as Lean
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class LeanHints:
    """`SolverHints` printed for the tactic plan: a settled condition as
    Lean over the proofs' state binder `s`, a product as `nlinarith` hint
    terms."""

    # (Lean text of the condition, the value `inv` forces it to).
    determined: list = field(default_factory=list)
    # None when no obligation was proved, so nothing may be concluded.
    linear_proof: "bool | None" = None
    # Lean terms to hand `nlinarith`.
    nlinarith_hints: list = field(default_factory=list)
    # A real precondition that no obligation's refutation needed.
    pre_unused: bool = False
    notes: list = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(
            self.determined or self.nlinarith_hints or self.pre_unused
        ) or (self.linear_proof is not None)

    def why(self) -> str:
        bits = []
        for text, value in self.determined:
            bits.append(f"inv forces `{text}` to be {str(value).lower()}")
        if self.linear_proof is False:
            bits.append(f"cvc5 needed {len(self.nlinarith_hints)} product hint(s)")
        if self.pre_unused:
            bits.append("no refutation needed the precondition: clearing it")
        bits += self.notes
        return "\n".join(f"-- cvc5: {b}" for b in bits)


def print_hints(hints: "SolverHints", ctrl_next) -> LeanHints:
    """`hints` printed. A condition or a factor the printer cannot express
    is left out: a hint is an optimisation, never needed for the proof."""
    from reactor_lean.proof.from_smt import smt_to_lean_body

    determined = []
    for cond, value in hints.determined:
        try:
            # Every proof binds the state as `s` (`scripts.lean.j2`).
            text = smt_to_lean_body(cond, ctrl_next, param_name="s")
        except Refused:
            continue
        determined.append((text, value))

    # `nlinarith` looks for degree-2 certificates by multiplying pairs of
    # hypotheses; naming the squares it needs up front turns that search
    # into a check. `mul_self_nonneg (a - b)` and `(a + b)` between them
    # span the products of two atoms, which is what the arithmetic rules
    # produce.
    seen: set[str] = set()
    nlinarith: list[str] = []
    for a_term, b_term in hints.products:
        try:
            a = smt_to_lean_body(a_term, ctrl_next, param_name="s")
            b = smt_to_lean_body(b_term, ctrl_next, param_name="s")
        except Refused:
            continue
        for hint in (f"mul_self_nonneg ({a} - {b})",
                     f"mul_self_nonneg ({a} + {b})"):
            if hint not in seen:
                seen.add(hint)
                nlinarith.append(hint)
    return LeanHints(determined, hints.linear_proof, nlinarith,
                     hints.pre_unused, list(hints.notes))


# ══════════════════════════════════════════════════════════════════════════
# The scripts
# ══════════════════════════════════════════════════════════════════════════


def _predicate_text(lean_cert: LeanCertificate) -> str:
    """The Lean text of every certificate predicate, for shape detection."""
    ranking = None if lean_cert.is_safety else lean_cert.ranking
    return "\n".join(v for v in (lean_cert.init_pre, lean_cert.update_pre,
                                 lean_cert.inv, lean_cert.prp, ranking) if v)


def _all_defs(ctx: "LeanContext", lean_cert: LeanCertificate) -> str:
    """What the two `simp` macros unfold: the certificate's definitions and
    the module's constants.

    A Buchi certificate always defines `ranking` (as `sorry` when nothing
    supplied one); a safety certificate defines none, and naming it in a
    simp set would be an unknown identifier rather than an unused one.
    """
    names = ["RM", "init", "update", "inv", "init_pre", "update_pre", "P"]
    if not lean_cert.is_safety:
        names.append("ranking")
    names += ctx.constants.names()
    return ", ".join(names)


def proof_scripts(
    ctx: "LeanContext",
    lean_cert: "LeanCertificate | None" = None,
    q: "ModuleQueries | None" = None,
    hints: "LeanHints | None" = None,
    verbatim: "LeanAnswer | None" = None,
) -> ProofScripts:
    """The proofs of `lean_cert`'s obligations, as `reactor` puts them
    under its statements.

    `q` is the certificate as `reactor` parsed it, whose terms the plan
    reads the predicates' shape off; `hints`, what cvc5 settled about them;
    `verbatim`, an answer that is Lean already, with no term behind it, so
    the plan reads its text instead.

    The definitions are printed shared whatever the hints settle: a
    settled condition is spliced into a tactic in expanded form, and it
    still matches the `if` it settles, because `simp_defs` unfolds the
    definitions with `simp`'s default `zeta`, which substitutes the `let`s
    away before the hint's rewrite runs.
    """
    from reactor_lean.proof.from_smt import smt_to_lean_body

    lean_cert = lean_cert or LeanCertificate()
    facts = predicate_facts(q)
    floors: list[str] = []
    for arg in facts.floors if facts is not None else ():
        text = smt_to_lean_body(arg, q.msmt.ctrl_next, param_name="s")
        if text not in floors:
            floors.append(text)
    plan = plan_for(
        ctx,
        _predicate_text(lean_cert),
        facts=facts,
        hints=hints,
        verbatim=("\n".join(t for t in (verbatim.inv, verbatim.ranking) if t)
                  if verbatim is not None else ""),
        floors=tuple(floors),
    )
    macros = _env.get_template("scripts.lean.j2").make_module(dict(
        plan=plan,
        all_defs=_all_defs(ctx, lean_cert),
        mat_simp=", ".join(MAT_SIMP),
        dsimp_erase=", ".join(f"-{n}" for n in DSIMP_ERASE),
    ))

    def part(name: str) -> str:
        return str(getattr(macros, name)()).strip("\n")

    return ProofScripts(
        imports=IMPORTS,
        options=part("options"),
        preamble=part("preamble"),
        scripts={name: part(name) for name in lean_cert.obligations},
    )
