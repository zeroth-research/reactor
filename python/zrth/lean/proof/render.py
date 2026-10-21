"""The certificate as Lean: its predicates parsed once and printed
(`render_certificate`), with what the tactic plan reads of them (`prepare`,
which every certificate goes through), and the files that come of it --
the project's two (`certificate_files`) and the one `--cert-file` writes
(`generate_standalone_cert_lean`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from zrth.lean.codegen.encodings import ModuleToLean4
from zrth.lean.codegen.terms import extl_native_type, product_type
from zrth.lean.core.certificate import Certificate, LeanAnswer
from zrth.lean.core.errors import PredicateRefused, Refused
from zrth.lean.codegen.context import LeanContext
from zrth.lean.project.names import DATA, MAT_SIMP, PROOF, cert_file, rm_def
from zrth.lean.proof.tactics import plan_for
from zrth.lean.project.templates import render

if TYPE_CHECKING:
    from zrth import Module


# ══════════════════════════════════════════════════════════════════════════
# The certificate as Lean
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LeanPredicates:
    """The certificate's definitions as Lean text, `fun s => ...` over the
    state and `fun e => ...` over the inputs; `None` where the template's
    placeholder stands. Built by `render_certificate` alone."""

    kind: str = "buchi"
    prp: str | None = None
    inv: str | None = None
    ranking: str | None = None
    init_pre: str | None = None
    update_pre: str | None = None

    @property
    def is_safety(self) -> bool:
        return self.kind == "safety"


@dataclass(frozen=True)
class LeanCertificate:
    """Everything codegen needs of a certificate."""

    defs: LeanPredicates = field(default_factory=LeanPredicates)


def render_certificate(
    q, *, share: bool = True,
    verbatim: LeanAnswer | None = None,
) -> LeanPredicates:
    """The one place certificate predicates are printed as Lean.

    `share=False` prints repeated subterms in full instead of `let`-binding
    them, which Lean reads the same once `simp` has zeta-reduced the
    `let`s; it is for reading a definition, not for proving with it.
    `verbatim` is an answer that is Lean already; its `inv` and `ranking`
    are taken as they are.
    """

    msmt = q.msmt

    def translate(term, mode: str) -> str | None:
        if term is None:
            return None

    def named(term, role: str) -> str | None:
        """`translate`, with the predicate's role on any refusal.

        """
        try:
            return translate(term, role)
        except Refused as e:
            raise PredicateRefused(role, str(e), kind=q.kind) from e

    prp = named(q.prp, "property")
    init_pre = named(q.init_pre, "pre")
    update_pre = named(q.update_pre, "pre")
    if verbatim is not None:
        inv, ranking = verbatim.inv, verbatim.ranking
    else:
        inv = named(q.inv, "invariant")
        ranking = named(q.ranking, "ranking")
    return LeanPredicates(
        kind=q.kind, prp=prp, inv=inv, ranking=ranking,
        init_pre=init_pre, update_pre=update_pre,
    )


def prepare(
    q,
    verbatim: LeanAnswer | None = None,
) -> LeanCertificate:
    """`q`'s certificate as codegen takes it: the predicates printed, the
    facts their terms give, `hints`, and the text of a `verbatim` answer.

    The definitions are printed shared whatever the hints settle: a
    settled condition is spliced into a tactic in expanded form, and it
    still matches the `if` it settles, because `simp_defs` unfolds the
    definitions with `simp`'s default `zeta`, which substitutes the `let`s
    away before the hint's `if_pos` runs.
    """

    floors: list[str] = []


def _cert_def_context(ctx: LeanContext, lean_cert: LeanCertificate) -> dict:
    """Build a Jinja2 context dict for the Data.lean.j2 template.

    Each predicate is an ``_expr`` (a Lean `fun` from `render_certificate`)
    or `None`, in which case the template uses the placeholder value.
    """
    d = lean_cert.defs
    return dict(
        extl_type=extl_native_type(ctx),
        ctrl_type=product_type(ctx.ctrl_next),
        noncomp="noncomputable " if ctx.uses_real else "",
        is_safety=d.is_safety,
        init_pre_expr=d.init_pre,
        update_pre_expr=d.update_pre,
        inv_expr=d.inv,
        p_expr=d.prp,
        ranking_expr=None if d.is_safety else d.ranking,
    )


def _predicate_text(ctx: LeanContext, lean_cert: LeanCertificate) -> str:
    """The Lean text of every certificate predicate, for shape detection."""
    c = _cert_def_context(ctx, lean_cert)
    return "\n".join(str(v) for k, v in c.items() if v and k.endswith("_expr"))


def generate_data_lean(
    ctx: LeanContext,
    lean_cert: LeanCertificate | None = None,
) -> str:
    """Generate Certificate/Data.lean: init_pre, update_pre, inv, P, ranking.

    Imports only ``Core.Basic``.  ``Certificate.lean`` imports this file so that
    only the data file needs regeneration when the certificate data changes
    (e.g. after --infer).

    It sits under ``Certificate/`` rather than ``System/`` because that is what
    it is about: ``System/`` is the module, which is the same whatever is being
    proved about it, while these five definitions are the certificate's own and
    change with every ``--infer``.  The two files a route rewrites are then the
    two files in one directory.
    """
    return render(
        "project/Certificate/Data.lean.j2",
        **_cert_def_context(ctx, lean_cert or LeanCertificate()),
    )


def generate_certificate_lean(
    ctx: LeanContext,
    lean_cert: LeanCertificate | None = None,
    *,
    module_inline: str | None = None,
) -> str:
    """Generate Certificate.lean.

    Args:
        module_inline: When set, inline this Lean source (the ``init`` /
            ``update`` functions) for standalone certs instead of importing
            ``System.System``.  Should be the output of
            ``ModuleToLean4.to_lean_functional()``.
    """
    lean_cert = lean_cert or LeanCertificate()
    defs = lean_cert.defs

    extl_native = extl_native_type(ctx)
    ctrl_native = product_type(ctx.ctrl_next)

    const_names = ctx.constants.names()
    const_list = ", ".join(const_names) if const_names else ""


    # Inline definitions when used standalone (no separate Certificate/Data.lean).
    inline_defs = (
        render("project/Certificate/_defs.lean.j2",
               **_cert_def_context(ctx, lean_cert)).strip("\n")
        if module_inline is not None else None
    )
    # A Buchi certificate always defines `ranking` (as `sorry` when nothing
    # supplied one); a safety certificate defines none, and naming it in a
    # simp set would be an unknown identifier rather than an unused one.
    has_ranking = not defs.is_safety

    all_defs = "RM, init, update, inv, init_pre, update_pre, P"
    if has_ranking:
        all_defs += ", ranking"
    if const_list:
        all_defs += f", {const_list}"

    plan = plan_for(
        ctx,
        _predicate_text(ctx, lean_cert),
        facts=lean_cert.plan.facts,
        hints=lean_cert.plan.hints,
        verbatim=lean_cert.plan.verbatim,
        floors=lean_cert.plan.floors,
    )

    return render(
        "project/Certificate/Certificate.lean.j2",
        module_inline=module_inline,
        inline_defs=inline_defs,
        noncomp="noncomputable " if ctx.uses_real else "",
        rm_def=rm_def(extl_native, ctrl_native, noncomputable=ctx.uses_real),
        all_defs=all_defs,
        mat_simp=", ".join(MAT_SIMP),
        is_safety=defs.is_safety,
        plan=plan,
    )


def certificate_files(
    ctx: LeanContext, lean_cert: LeanCertificate | None = None,
) -> dict[str, str]:
    """The project's two certificate files by project-relative path: the
    definitions (`Data.lean`) and the proof whose tactics are read off
    their shape (`Certificate.lean`). Placeholders where `lean_cert` has no
    predicate, or is `None`.

    Rendered together because they are rewritten together: a file of
    either kind left over from other predicates no longer matches the
    other.
    """
    return {
        cert_file(DATA): generate_data_lean(ctx, lean_cert),
        cert_file(PROOF): generate_certificate_lean(ctx, lean_cert),
    }


def generate_standalone_cert_lean(
    module: "Module",
    lean_cert: LeanCertificate | None = None,
) -> str:
    """Generate a self-contained certificate Lean file with init/update inlined.

    Unlike the project's ``Certificate.lean``, the functional module code
    (init, update) is inlined directly rather than imported.  ZerothHammer
    is always imported from the project-level ``ZerothHammer.lean``.

    Suitable for placing a single .lean file inside an existing lake project
    such as ``tests/lean/Certs/``.
    """
    ctx = LeanContext(module)
    module_code = ModuleToLean4(ctx).to_lean_functional()
    return generate_certificate_lean(ctx, lean_cert, module_inline=module_code)
