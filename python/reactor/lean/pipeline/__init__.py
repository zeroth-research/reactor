"""`uv run reactor`: a reactive module in, a Lean 4 certificate project out.

`main` runs the stages in order. `claim` reads the module and what is to
be proved of it; `certificate` checks the module and renders the
certificate; `outputs` writes the project, or the one file asked for, and
builds it.

Quick start
-----------
A *module file* is a Python file that defines a ``module()`` function (or
class) returning a :class:`reactor.Module` (or a ``reactor.lean.inputs.Named``
one, for claims in its own names), or another format ``--format`` lists.

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
``reactor.lean.inputs.Named(module, {"x": var})``, and ``(X @intf.0)`` names
interface variable 0 by position, which works for any module.

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
        init_terms   = convert_method(init,   {},               [X(state)])
        update_terms = convert_method(update, {"old_x": state}, [X(state)])
        return Module.sequential([state], init_terms, update_terms)
"""

from __future__ import annotations

from ..cli import parse_args
from ..core.errors import Refused
from ..project.lakefile import reactor_lean_checkout
from .certificate import lean_context, render
from .claim import claim_from_flags, load
from .outputs import build_cert, generate, hammer_only, standalone


def main():
    settings = parse_args()

    if settings.hammer_file:
        hammer_only(settings.hammer_file)
        return

    # `$REACTOR_LEAN` is a checkout the project's lakefile will require, so
    # a bad one is refused here, before any of the project is written.
    if not settings.cert_file:
        try:
            reactor_lean_checkout()
        except Refused as e:
            raise SystemExit(f"error: {e}") from e

    problem = load(settings)
    module = problem.module
    print(module)
    claim, given = claim_from_flags(settings, problem.names)
    try:
        ctx = lean_context(module)
    except Refused as e:
        raise SystemExit(f"error: {e}") from e


    try:
        lean_cert = render(module, claim, given)
    except Refused as e:
        raise SystemExit(f"error: {e}") from e

    if settings.cert_file:
        standalone(settings.cert_file, ctx, lean_cert)
        return

    project_dir = generate(settings, module, ctx, lean_cert, problem.source)

    if settings.build_cert:
        build_cert(project_dir, lean_cert)

    print(f"\nProject ready at: {project_dir}")
