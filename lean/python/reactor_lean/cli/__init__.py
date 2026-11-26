"""`uv run reactor`: a reactive module in, a Lean 4 certificate project out.

`main` runs the stages in order: `parse_args` reads the command line
(`args`); `parse.load` reads the module, and `claim_from_flags` what is
to be proved of it; `codegen.context.lean_context` checks that the module
has a Lean form, and `proof.render.lean_certificate` renders the
certificate; `generate` writes the project, or `standalone` the one file
asked for. Each stage refuses what it cannot carry out (`Refused`), and
`main` prints that as one `error:` line.

The certificate states its obligations and proves each by `sorry`: the
proofs are a prover's to write, into `Certificate/Proofs.lean`, which
nothing else in the project depends on but the certificate's `exact`s.

Quick start
-----------
A *module file* is a Python file that defines a ``module()`` function (or
class) returning a :class:`reactor.Module` (or a ``reactor_lean.parse.Named``
one, for claims in its own names).

Generate a bare Lean project (all certificate fields left as ``sorry``)::

    uv run reactor mymodule.py -o out/ -p MyProject

Pass the property so the certificate knows what to prove.  Which flag it
goes under is the choice of proof rule, and so of what the certificate
contains::

    # `G (F x == 0)` -- holds infinitely often: invariant + ranking function
    uv run reactor mymodule.py --buchi "(= (X x) 0)" -o out/ -p MyProject

    # `G (x <= 100)` -- holds in every reachable state: invariant alone
    uv run reactor mymodule.py --safety "(<= (X x) 100)" -o out/ -p MyProject

A claim is written in the module's own terms: ``(X x)`` is the value of ``x``
after the round.  Names like ``x`` come from a module file that returns
``reactor_lean.parse.Named(module, {"x": var})``, and ``(X @intf.0)`` names
interface variable 0 by position, which works for any module.

Supply the certificate's predicates::

    uv run reactor mymodule.py --buchi "(= (X x) 0)" \\
        --invariant "(<= (X x) 10)" --ranking "(- 10 (X x))" \\
        -o out/ -p MyProject

Module file format
------------------
The file must expose a callable named ``module`` (override with ``-d``) that
returns a :class:`reactor.Module`::

    # mymodule.py
    from reactor import Module, Int, Var, X
    from reactor.analyzer import convert_method

    def init():
        return 0

    def update(old_x):
        x = old_x + 1
        if x == 10:
            return 0
        return x

    def module() -> Module:
        state = Var(Int([1, 1]))
        init_terms   = convert_method(init,   {},               [X(state)], number=Int)
        update_terms = convert_method(update, {"old_x": state}, [X(state)], number=Int)
        return Module.sequential([state], init_terms, update_terms)
"""

from __future__ import annotations

from pathlib import Path

from .. import parse
from ..codegen.context import LeanContext, lean_context
from ..codegen.encodings import ModuleToLean4
from ..errors import Refused
from ..parse import Certificate, Claim
from ..project.lake_cache import Packages, project_packages
from ..project.main_lean import check_runnable
from ..project.scaffold import ENCODINGS, Layout, create_project, write_encoding
from ..proof.render import (
    LeanCertificate,
    certificate_files,
    generate_standalone_cert_lean,
    lean_certificate,
)
from .args import Settings, parse_args

__all__ = ["Settings", "main", "parse_args"]


def main():
    settings = parse_args()
    # What the module file's own code raises is not a refusal, and keeps
    # its traceback.
    try:
        # `$REACTLIB` is a checkout the project's lakefile will require, and
        # `$REACTOR_LAKE_CACHE` the packages it will build against, holding
        # what the lakefile requires from git at its revision: a bad one is
        # refused here, before any of the project is written.
        packages = None if settings.cert_file else project_packages()
        problem = parse.load(settings.module_file, settings.module_def)
        module = problem.module
        print(module)
        claim, given = claim_from_flags(settings, problem.names)
        ctx = lean_context(module)
        if settings.executable:
            check_runnable(ctx)
        lean_cert = lean_certificate(module, claim, given)
    except Refused as e:
        raise SystemExit(f"error: {e}") from e

    if settings.cert_file:
        standalone(settings.cert_file, ctx, lean_cert)
        return

    project_dir = generate(settings, ctx, lean_cert, problem.source, packages)
    print(f"\nProject ready at: {project_dir}")


# ══════════════════════════════════════════════════════════════════════
# What is to be proved of the module
# ══════════════════════════════════════════════════════════════════════


def claim_from_flags(
    settings: Settings, names: "parse.Names",
) -> "tuple[Claim | None, Certificate]":
    """What the flags ask to be proved, and what of its certificate they
    supply. No claim when no predicate flag was passed.

    The flags are in the module's own terms (`parse.names`); the claim
    is in the internal names every later stage reads, and each flag is
    printed in them once, since that is how the logs and
    any refusal will spell it."""
    if not (settings.prp or settings.extl_assume or settings.extl_assume_init
            or settings.extl_assume_update or settings.invariant
            or settings.ranking):
        return None, Certificate()
    prp_flag = "--safety" if settings.is_safety else "--buchi"

    def internal(text: "str | None", flag: str) -> "str | None":
        if not text:
            return None
        spelled = parse.to_internal(text, names, flag)
        print(f".. {flag}, in internal names: {spelled}")
        return spelled

    prp = internal(settings.prp, prp_flag)
    # `--extl-assume` is both blocks' assumption; it excludes the other two.
    both = internal(settings.extl_assume, "--extl-assume")
    init = internal(settings.extl_assume_init, "--extl-assume-init") or both
    update = (internal(settings.extl_assume_update, "--extl-assume-update")
              or both)
    claim = Claim(prp=prp, kind=settings.kind, init_assume=init,
                  update_assume=update)
    return claim, Certificate(internal(settings.invariant, "--invariant"),
                              internal(settings.ranking, "--ranking"))


# ══════════════════════════════════════════════════════════════════════
# What the run writes: the project, or a standalone certificate; both
# prove the obligations by `sorry`
# ══════════════════════════════════════════════════════════════════════


def standalone(cert_file: str, ctx: LeanContext,
               lean_cert: LeanCertificate) -> None:
    """`--cert-file`: a self-contained certificate and its sibling encodings.

    No project: the predicates are the ones supplied.
    """
    out = Path(cert_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_standalone_cert_lean(ctx, lean_cert))
        print(f"Wrote standalone certificate: {out}")
        # The same encoding table the project walks, so these files and
        # the ones in a generated project carry the same headers and the
        # same imports. The functional encoding is already inlined in the
        # standalone certificate above, which is the base they import.
        m2l = ModuleToLean4(ctx)
        layout = Layout(base=out.stem, prefix=lake_module_prefix(out.parent),
                        flat=True, directory=out.parent)
        for enc in ENCODINGS:
            if enc.cert_file:
                path = write_encoding(enc, m2l, layout, out.stem)
                print(f"Wrote {enc.title.lower()}: {path}")
    # What cannot be written is the flag's; a refusal of the module reads
    # as it does without it.
    except Refused as e:
        raise SystemExit(f"error: {e}") from e
    except OSError as e:
        raise SystemExit(f"error: --cert-file: {e}") from e


def lake_module_prefix(directory: Path) -> str:
    """What Lake prefixes the module of a file in `directory` with: the
    file's path from the nearest directory above that has a lakefile, so
    `Certs.` in `tests/lean/Certs/` under `tests/lean/lakefile.toml`.
    Nothing when no directory above has one."""
    directory = directory.resolve()
    for root in (directory, *directory.parents):
        if any((root / f).is_file() for f in ("lakefile.lean", "lakefile.toml")):
            return "".join(f"{p}." for p in directory.relative_to(root).parts)
    return ""


def generate(settings: Settings, ctx: LeanContext,
             lean_cert: LeanCertificate, source: str,
             packages: Packages) -> Path:
    """The Lean project, with the certificate as it stands; `source` is
    the input's text, which the project keeps in `dbg/`, and `packages`
    what its lakefile requires."""
    print(".. Generating lean code")
    certificate = certificate_files(ctx, lean_cert)
    try:
        return create_project(
            output_dir=settings.output_dir,
            ctx=ctx,
            project_name=settings.project_name,
            executable=settings.executable,
            certificate=certificate,
            source=source,
            packages=packages,
        )
    except (Refused, OSError) as e:
        raise SystemExit(f"error: {e}") from e
