"""`uv run verith`: a reactive module in, a Lean 4 certificate project out.

`main` runs the stages in order. `claim` reads the module and what is to
be proved of it; `certify` checks the module, renders the certificate,
runs the route and renders the certificate again from its answer;
`outputs` writes the project, or the one file asked for, and builds it.

Quick start
-----------
A *module file* is a Python file that defines a ``module()`` function (or
class) returning a :class:`reactor.Module` (or a ``reactor.lean.inputs.Named``
one, for claims in its own names), or another format ``--format`` lists.
The LLM routes prompt on the module's generated Lean; a route that declares
``reads_source`` is also handed the input file itself.

Generate a bare Lean project (all certificate fields left as ``sorry``)::

    uv run verith mymodule.py -o out/ -p MyProject

Pass the property so the certificate knows what to prove.  Which flag it
goes under is the choice of proof rule, and so of what the certificate
contains::

    # `G (F x == 0)` -- holds infinitely often: invariant + ranking function
    uv run verith mymodule.py --buchi "(= (X x) 0)" -o out/ -p MyProject

    # `G (x <= 100)` -- holds in every reachable state: invariant alone
    uv run verith mymodule.py --safety "(<= (X x) 100)" -o out/ -p MyProject

A claim is written in the module's own terms: ``(X x)`` is the value of ``x``
after the round.  Names like ``x`` come from a module file that returns
``reactor.lean.inputs.Named(module, {"x": var})``, and ``(X @intf.0)`` names
interface variable 0 by position, which works for any module.

Ask the AI to infer the certificate automatically (the ranking function too,
for a Buchi property)::

    uv run verith mymodule.py --buchi "(= (X x) 0)" --infer -o out/ -p MyProject

Or infer it without an LLM at all: ``--infer nuterm`` trains a ranking function
on rollouts of the module and hands it over only once a Farkas/CEGAR decision
procedure has certified it, with the invariant inferred by Houdini.  It needs
no API key, and it reads modules whose state is scalar integers::

    uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer nuterm -o out/ -p MyProject

Use a local LLM via Ollama instead of Claude::

    uv run verith mymodule.py --buchi "(= (X x) 0)" --infer \\
        --model qwen3-coder --base-url http://localhost:11434/v1 \\
        -o out/ -p MyProject

AI inference requirements
-------------------------
* **Claude (default)**: set ``ANTHROPIC_API_KEY`` and install
  ``pip install reactor[ai]``.
* **Local LLM (Ollama, vLLM, …)**: install ``pip install reactor[ai-local]``
  and provide ``--base-url`` pointing to an OpenAI-compatible endpoint.
* **Hosted OpenAI-compatible provider (OpenRouter, …)**: as above, plus set
  ``OPENROUTER_API_KEY`` (or ``OPENAI_API_KEY``), e.g.
  ``--base-url https://openrouter.ai/api/v1 --model anthropic/claude-haiku-4.5``.

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
from ..infer import ProjectHandle
from .certify import (
    lean_context,
    regenerate,
    render,
    route_precheck,
    run_route,
)
from .claim import claim_from_flags, load, show_certificate
from .outputs import build_cert, generate, hammer_only, standalone


def main():
    settings = parse_args()

    if settings.hammer_file:
        hammer_only(settings.hammer_file)
        return

    problem = load(settings)
    module = problem.module
    print(module)
    claim, given = claim_from_flags(settings, problem.names)
    prechecked = route_precheck(settings, module, claim)
    ctx = lean_context(settings, module)


    try:
        lean_cert = render(
            module, claim, given)
    except Refused as e:
        raise SystemExit(f"error: {e}") from e

    if settings.cert_file:
        standalone(settings.cert_file, ctx, lean_cert)
        return

    project_dir = generate(settings, module, ctx, lean_cert, problem.source)

    if settings.route is not None:
        print(f".. Inferring the certificate with --infer {settings.route.name}")
        handle = ProjectHandle(
            dir=project_dir,
            name=settings.project_name,
            ctx=ctx,
            reads=frozenset(settings.route.reads),
            owns=settings.route.owns,
        )
        result = run_route(settings, module, handle, claim, given, prechecked,
                           problem=problem)
        # `files`: the certificate is the route's own, so there is nothing
        # of ours to be complete about and nothing to regenerate from.
        lean_cert = (
            None if settings.route.returns == "files"
            else regenerate(settings, module, handle, claim, result)
        )
        handle.write_files(result.files)
        show_certificate(result.cert, problem.names)

    if settings.build_cert:
        build_cert(project_dir, lean_cert)

    print(f"\nProject ready at: {project_dir}")

