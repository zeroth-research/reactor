"""The certificate as Lean: its predicates parsed once and printed
(`render_certificate`), and the files that come of it -- the project's four
(`certificate_files`) and the one `--cert-file` writes
(`generate_standalone_cert_lean`).

The obligations are stated here and proved elsewhere. `Proofs.lean` states
each one again and proves it by `sorry`, unless a prover hands its proofs
over as `ProofScripts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reactor.lean.codegen.encodings import ModuleToLean4
from reactor.lean.codegen.terms import extl_native_type, product_type
from reactor.lean.core.certificate import LeanAnswer
from reactor.lean.core.errors import PredicateRefused, Refused
from reactor.lean.codegen.context import LeanContext
from reactor.lean.project.names import (
    CERTIFICATE, DATA, MODULE, PROOFS, cert_file,
)
from reactor.lean.project.templates import render

if TYPE_CHECKING:
    from reactor import Module

# ══════════════════════════════════════════════════════════════════════════
# The certificate as Lean
# ══════════════════════════════════════════════════════════════════════════


# Every obligation a certificate states, by the name of its `_proof`
# theorem without the suffix: the first three of a safety certificate, or
# the first two and `hrank` of a Buchi one (`LeanCertificate.obligations`).
OBLIGATIONS = ("init_inv", "step_inv", "inv_imp_P", "hrank")


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

    @property
    def obligations(self) -> tuple[str, ...]:
        """The obligations this certificate states, by the names
        `ProofScripts.scripts` is keyed by."""
        return ("init_inv", "step_inv",
                "inv_imp_P" if self.is_safety else "hrank")


def render_certificate(
    q, *, share: bool = True,
    verbatim: LeanAnswer | None = None,
) -> LeanCertificate:
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
# The proofs
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ProofScripts:
    """The proofs of a certificate's obligations, as a prover hands them
    over: this package states the obligations and proves none of them.

    `scripts` maps an obligation's name -- `init_inv`, `step_inv`, and
    `hrank` or `inv_imp_P` -- to the tactic proof of its `_proof` theorem,
    the lines after `:= by`; an obligation it leaves out is `sorry`.
    `imports` are the modules the scripts need, `options` the `set_option`s
    they run under, set before anything else is elaborated (a `--cert-file`
    certificate sets them before its inlined module), and `preamble` what
    they share, put before the first of them.
    """

    imports: tuple[str, ...] = ()
    options: str = ""
    preamble: str = ""
    scripts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A misspelt name would otherwise leave its obligation `sorry`,
        # found only when `lake build` warns about it.
        unknown = sorted(set(self.scripts) - set(OBLIGATIONS))
        if unknown:
            raise ValueError(
                f"no obligation is named {', '.join(unknown)}: the "
                f"obligations are {', '.join(OBLIGATIONS)}"
            )

    def script(self, name: str) -> str:
        """The proof of obligation `name`: its script, or `sorry`."""
        return self.scripts.get(name, "sorry")


# Every obligation proved by `sorry`: what this package writes on its own.
SORRY = ProofScripts()


def generate_module_lean() -> str:
    """Generate Certificate/Module.lean: `RM` and `lts`, the same text for
    every module and certificate."""
    return render("project/Certificate/Module.lean.j2")


def generate_proofs_lean(
    lean_cert: LeanCertificate | None = None,
    proofs: ProofScripts = SORRY,
) -> str:
    """Generate Certificate/Proofs.lean: each obligation of `lean_cert`'s
    kind stated again, and proved by `proofs`."""
    return render(
        "project/Certificate/Proofs.lean.j2",
        is_safety=(lean_cert or LeanCertificate()).is_safety,
        proofs=proofs,
    )


def generate_certificate_lean(
    ctx: LeanContext,
    lean_cert: LeanCertificate | None = None,
    *,
    module_inline: str | None = None,
    proofs: ProofScripts = SORRY,
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
            proofs (`proofs`) with it. Should be the output of
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
                is_safety=lean_cert.is_safety, proofs=proofs).strip("\n"),
        )
    return render(
        "project/Certificate/Certificate.lean.j2",
        is_safety=lean_cert.is_safety,
        module_inline=module_inline,
        proofs=proofs,
        **inline,
    )


def certificate_files(
    ctx: LeanContext, lean_cert: LeanCertificate | None = None,
    proofs: ProofScripts = SORRY,
) -> dict[str, str]:
    """The project's four certificate files by project-relative path: the
    definitions (`Data.lean`); `RM` and `lts` over them (`Module.lean`), the
    same text for every certificate; the obligations stated again and
    proved by `proofs` (`Proofs.lean`); and the certificate, which is fixed
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
        cert_file(PROOFS): generate_proofs_lean(lean_cert, proofs),
        cert_file(CERTIFICATE): generate_certificate_lean(ctx, lean_cert),
    }


def generate_standalone_cert_lean(
    src: "LeanContext | Module",
    lean_cert: LeanCertificate | None = None,
    proofs: ProofScripts = SORRY,
) -> str:
    """Generate a self-contained certificate Lean file with init/update inlined.

    Unlike the project's ``Certificate.lean``, the functional module code
    (init, update) is inlined directly rather than imported, and so are the
    definitions and the proofs, `proofs`' imports aside.

    Suitable for placing a single .lean file inside an existing lake project
    such as ``tests/lean/Certs/``. `src` is the run's `LeanContext`, or a
    module to build one for.
    """
    ctx = src if isinstance(src, LeanContext) else LeanContext(src)
    module_code = ModuleToLean4(ctx).to_lean_functional()
    return generate_certificate_lean(ctx, lean_cert, module_inline=module_code,
                                     proofs=proofs)
