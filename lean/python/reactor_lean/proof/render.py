"""The certificate as Lean: its predicates parsed once and printed
(`render_certificate`), and the files that come of it -- the project's four
(`certificate_files`) and the one `--cert-file` writes
(`generate_standalone_cert_lean`).

The obligations are stated here and proved elsewhere. `Proofs.lean` states
each one again and proves it by `sorry`, for a prover to replace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reactor_lean.codegen.encodings import ModuleToLean4
from reactor_lean.codegen.terms import extl_native_type, product_type
from reactor_lean.core.errors import PredicateRefused, Refused
from reactor_lean.codegen.context import LeanContext
from reactor_lean.project.names import (
    CERTIFICATE, DATA, MODULE, PROOFS, cert_file,
)
from reactor_lean.project.templates import render

if TYPE_CHECKING:
    from reactor import Module

    from reactor_lean.parse.native import NativePredicates

# ══════════════════════════════════════════════════════════════════════════
# The certificate as Lean
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LeanCertificate:
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


def render_certificate(q: "NativePredicates") -> LeanCertificate:
    """The one place certificate predicates are printed as Lean.

    `q` is what the parser read (`parse.native.NativePredicates`), printed
    by `from_ir` with repeated subterms `let`-bound, which Lean reads the
    same once `simp` has zeta-reduced the `let`s.
    """
    from .from_ir import ir_to_lean as prop
    from .from_ir import ir_to_lean_nat as nat

    wires = q.wires

    def translate(term, mode: str) -> str | None:
        if term is None:
            return None
        if mode in ("property", "invariant"):
            return prop(term, wires.ctrl_next, param_name="s")
        if mode == "ranking":
            return nat(term, wires.ctrl_next, param_name="s")
        # preconditions
        return prop(
            term,
            state_wires=[],
            param_name="e",
            extra=[
                ("e", "e.2", wires.extl_next),
                ("el", "e.1", wires.extl_latched),
            ],
        )

    def named(term, role: str) -> str | None:
        """`translate`, with the predicate's role on any refusal.

        The printer walks a term and knows nothing of where it came from;
        `--pre (= s0 0)` parses and then fails on a name preconditions do
        not bind, and "Unknown free variable `s0`" alone does not say which
        of five sources asked for it.
        """
        try:
            return translate(term, role)
        except Refused as e:
            raise PredicateRefused(role, str(e), kind=q.kind) from e

    prp = named(q.prp, "property")
    init_pre = named(q.init_pre, "pre")
    update_pre = named(q.update_pre, "pre")
    inv = named(q.inv, "invariant")
    ranking = named(q.ranking, "ranking")
    return LeanCertificate(
        kind=q.kind, prp=prp, inv=inv, ranking=ranking,
        init_pre=init_pre, update_pre=update_pre,
    )


def _cert_def_context(ctx: LeanContext, lean_cert: LeanCertificate) -> dict:
    """Build a Jinja2 context dict for the Data.lean.j2 template.

    Each predicate is an ``_expr`` (a Lean `fun` from `render_certificate`)
    or `None`, in which case the template uses the placeholder value.
    """
    return dict(
        extl_type=extl_native_type(ctx),
        ctrl_type=product_type(ctx.ctrl_next),
        noncomp="noncomputable " if ctx.uses_real else "",
        is_safety=lean_cert.is_safety,
        init_pre_expr=lean_cert.init_pre,
        update_pre_expr=lean_cert.update_pre,
        inv_expr=lean_cert.inv,
        p_expr=lean_cert.prp,
        ranking_expr=None if lean_cert.is_safety else lean_cert.ranking,
    )


def generate_data_lean(
    ctx: LeanContext,
    lean_cert: LeanCertificate | None = None,
) -> str:
    """Generate Certificate/Data.lean: init_pre, update_pre, inv, P, ranking.

    Imports only ``Relational.Basic``.  ``Module.lean`` imports it, and
    the proofs and the certificate through that.

    It sits under ``Certificate/`` rather than ``System/`` because that is what
    it is about: ``System/`` is the module, which is the same whatever is being
    proved about it, while these five definitions are the certificate's own and
    change with every certificate.  The two files a new certificate rewrites,
    this one and ``Proofs.lean``, are then in one directory with the
    certificate.
    """
    return render(
        "project/Certificate/Data.lean.j2",
        **_cert_def_context(ctx, lean_cert or LeanCertificate()),
    )


# ══════════════════════════════════════════════════════════════════════════
# The certificate's other files
# ══════════════════════════════════════════════════════════════════════════


def generate_module_lean() -> str:
    """Generate Certificate/Module.lean: `RM` and `lts`, the same text for
    every module and certificate."""
    return render("project/Certificate/Module.lean.j2")


def generate_proofs_lean(lean_cert: LeanCertificate | None = None) -> str:
    """Generate Certificate/Proofs.lean: each obligation of `lean_cert`'s
    kind stated again, and proved by `sorry`."""
    return render(
        "project/Certificate/Proofs.lean.j2",
        is_safety=(lean_cert or LeanCertificate()).is_safety,
    )


def generate_certificate_lean(
    ctx: LeanContext,
    lean_cert: LeanCertificate | None = None,
    *,
    module_inline: str | None = None,
) -> str:
    """Generate Certificate.lean: the certificate.

    In a project it is the same text for every certificate of a property of
    one kind (safety or Büchi, `lean_cert`'s): it states the obligations and
    takes their proofs from `Proofs.lean`, which with `Data.lean` is what
    changes with the module and the certificate.

    Args:
        module_inline: When set, inline this Lean source (the ``init`` /
            ``update`` functions) for standalone certs instead of importing
            ``System.System``, and the definitions, ``Module.lean`` and the
            proofs with it. Should be the output of
            ``ModuleToLean4.to_lean_functional()``.
    """
    lean_cert = lean_cert or LeanCertificate()
    inline = {}
    if module_inline is not None:
        inline = dict(
            inline_defs=render(
                "project/Certificate/_defs.lean.j2",
                **_cert_def_context(ctx, lean_cert)).strip("\n"),
            inline_module=render(
                "project/Certificate/_module.lean.j2").strip("\n"),
            inline_proofs=render(
                "project/Certificate/_proofs.lean.j2",
                is_safety=lean_cert.is_safety).strip("\n"),
        )
    return render(
        "project/Certificate/Certificate.lean.j2",
        is_safety=lean_cert.is_safety,
        module_inline=module_inline,
        **inline,
    )


def certificate_files(
    ctx: LeanContext, lean_cert: LeanCertificate | None = None,
) -> dict[str, str]:
    """The project's four certificate files by project-relative path: the
    definitions (`Data.lean`); `RM` and `lts` over them (`Module.lean`), the
    same text for every certificate; the obligations stated again and
    proved by `sorry` (`Proofs.lean`); and the certificate, which is fixed
    text for the property's kind and takes its proofs from `Proofs.lean`
    (`Certificate.lean`). Placeholders where `lean_cert` has no predicate,
    or is `None`.

    Rendered together because they are rewritten together: a definitions
    file left over from other predicates no longer matches the proofs, and
    the certificate follows the kind of the property they are for.
    """
    return {
        cert_file(DATA): generate_data_lean(ctx, lean_cert),
        cert_file(MODULE): generate_module_lean(),
        cert_file(PROOFS): generate_proofs_lean(lean_cert),
        cert_file(CERTIFICATE): generate_certificate_lean(ctx, lean_cert),
    }


def generate_standalone_cert_lean(
    src: "LeanContext | Module",
    lean_cert: LeanCertificate | None = None,
) -> str:
    """Generate a self-contained certificate Lean file with init/update inlined.

    Unlike the project's ``Certificate.lean``, the functional module code
    (init, update) is inlined directly rather than imported, and so are the
    definitions and the proofs.

    Suitable for placing a single .lean file inside an existing lake project
    such as ``tests/lean/Certs/``. `src` is the run's `LeanContext`, or a
    module to build one for.
    """
    ctx = src if isinstance(src, LeanContext) else LeanContext(src)
    module_code = ModuleToLean4(ctx).to_lean_functional()
    return generate_certificate_lean(ctx, lean_cert, module_inline=module_code)
