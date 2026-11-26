"""The `uv run certify` command line: `reactor`'s flags, `--infer` and
the proofs' flags.

`reactor`'s groups, checks and fields (`reactor_lean.cli`), with the
routes' and the proofs' flags and rules (`verith.flags`) in the places
today's command line has them. What the pipeline gets back is
:class:`Settings`: `reactor`'s, the route the run selected, and what is
asked of the proofs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from reactor_lean.cli import Settings as ReactorSettings
from reactor_lean.cli import claim_deferred, settings_fields
from reactor_lean.cli.checks import (
    check_blanks,
    check_claim_flags,
    check_input_options,
    check_outputs,
    input_adapter,
)
from reactor_lean.cli.parser import add_claim, add_input, add_outputs
from reactor_lean.core.encodings import ENCODING_NAMES
from verith.common.limits import SmtBudget
from verith import InferRoute
from verith.flags import (
    add_proofs,
    add_routes,
    check_budget,
    check_build,
    check_route,
    check_route_options,
    proof_fields,
    resolve_aliases,
    resolve_route,
    route_fields,
    selected_route,
)


_EPILOG = """\
examples:
  # bare project — no property, so `P` and `ranking` are sorry
  uv run certify mymodule.py -o out/ -p MyProject

  # a Buchi property, `G (F (= (X @intf.0) 0))` (certificate stub with known P)
  uv run certify mymodule.py --buchi "(= (X @intf.0) 0)" -o out/ -p MyProject

  # a safety property, `G (<= (X @intf.0) 100)`: invariant only, no ranking function
  uv run certify mymodule.py --safety "(<= (X @intf.0) 100)" -o out/ -p MyProject

  # AI inference with Claude (requires ANTHROPIC_API_KEY + pip install verith[ai])
  uv run certify mymodule.py --buchi "(= (X @intf.0) 0)" --infer -o out/ -p MyProject

  # ... and the same loop for a safety property (cvc5-checked route only)
  uv run certify mymodule.py --safety "(<= (X @intf.0) 100)" --infer ai-cegis -o out/ -p MyProject

  # no LLM: learn a ranking function and certify it before it is offered
  uv run certify mymodule.py --buchi "(= (X @intf.0) 0)" --infer nuterm -o out/ -p MyProject

  # no LLM: synthesise the invariant, congruences included (`x` even), and
  # leave it in artifacts/ for the next run to take as given
  uv run certify mymodule.py --safety "(not (= (X @intf.0) 1))" --infer sygus -o out/ -p MyProject

  # no LLM: one query per shape -- and a refuted shape is a *proof* that no
  # certificate of that shape exists, which --infer ai-cegis reads next run
  uv run certify mymodule.py --buchi "(= (X @intf.0) 0)" --infer smt-linear -o out/ -p MyProject

  # no LLM: Houdini and a ranking search, each candidate decided by cvc5 --
  # a candidate it cannot prove comes back with the counterexample that kills it
  uv run certify mymodule.py --buchi "(= (X @intf.0) 0)" --infer houdini -o out/ -p MyProject

  # ... the same search, refuted by the Vampire prover instead
  uv run certify mymodule.py --buchi "(= (X @intf.0) 0)" --infer houdini \\
      --houdini-solver vampire --vampire ~/vampire/vampire -o out/ -p MyProject

  # no LLM, and nothing proposed: the obligations are stated with the invariant
  # left open and Vampire's answer literals return its coefficients
  uv run certify mymodule.py --safety "(<= (X @intf.0) 100)" --infer vampire \\
      --vampire ~/vampire/vampire -o out/ -p MyProject

  # AI inference with Ollama (requires pip install verith[ai-local])
  uv run certify mymodule.py --buchi "(= (X @intf.0) 0)" --infer \\
      --model qwen3-coder --base-url http://localhost:11434/v1 -o out/ -p MyProject

  # ask cvc5 whether the obligations are true before spending a lake build on them
  uv run certify mymodule.py --buchi "(= (X @intf.0) 0)" --invariant "(<= (X @intf.0) 100)" --ranking "(X @intf.0)" \\
      --pre-check cvc5 -o out/ -p MyProject

  # certify through lean-ltl-certifying's proveit.py (lean2vmt -> ic3ia -> vmt2lean)
  uv run certify mymodule.py --safety "(not (= (X @intf.0) 15))" -o out/ -p MyProject \\
      --infer fbk-proveit --proveit-dir ~/lean-ltl-certifying --ic3ia ~/ic3ia/build/ic3ia

  # build the certificate of any route that fills it: `lake build Certificate`
  uv run certify mymodule.py --buchi "(= (X @intf.0) 0)" --invariant "(<= (X @intf.0) 100)" --ranking "(X @intf.0)" \\
      --build-cert -o out/ -p MyProject
"""


# The encodings a `certify` project carries: every one (`reads` on a
# route's row; fbk-proveit's bridge imports the scalar one). `reactor`'s
# `--encodings`, fixed: `certify` has no such flag.
CERTIFY_ENCODINGS: tuple[str, ...] = ENCODING_NAMES


@dataclass
class Settings(ReactorSettings):
    """`reactor`'s settings, with the encodings `CERTIFY_ENCODINGS`; the
    inference route: which one (`route`), its options (`route_opts`) and
    what it resolved at parse time (`route_config`); and what cvc5 is asked
    about the certificate, under what budget, and whether it is built."""

    encodings: tuple[str, ...] = CERTIFY_ENCODINGS

    route: InferRoute | None = None
    route_opts: dict = field(default_factory=dict)
    route_config: object | None = None

    hammer_file: str | None = None
    build_cert: bool = False
    pre_check: str = "none"
    smt_tactics: str = "none"
    budget: SmtBudget = field(default_factory=SmtBudget)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="certify",
        description=(
            "Generate a Lean4 certificate project from a reactive module, read "
            "from a Python file or another format --format names. "
            "The property goes under --safety (`G FORMULA`: every reachable "
            "state) or --buchi (`G (F FORMULA)`: infinitely often); with --infer "
            "the invariant is searched for -- by an LLM, by learning a ranking "
            "function and certifying it, or by an external prover -- along with "
            "the ranking function a Buchi certificate also needs. "
            "`reactor` is this command without --infer, and `verith` is "
            "--infer without the project."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_input(parser, optional="Optional when --hammer-file is used alone.")
    add_claim(parser)
    add_routes(parser)
    add_outputs(parser)
    parser.add_argument(
        "--hammer-file",
        default=None,
        help="Write a standalone ZerothHammer.lean to this path.",
    )
    add_proofs(parser)
    return parser


def parse_args(argv: "list[str] | None" = None, *,
               defer_claim: bool = True) -> Settings:
    """The command line as `Settings`; `defer_claim` as in
    `reactor_lean.cli.parse_args`."""
    parser = build_parser()
    args = parser.parse_args(argv)

    check_blanks(parser, args)
    resolve_aliases(parser, args)
    check_claim_flags(parser, args)
    check_budget(parser, args)

    adapter = input_adapter(parser, args)
    check_input_options(parser, args, adapter)

    route = selected_route(args)
    check_route_options(parser, args, route)

    settings = Settings(
        **settings_fields(args, adapter),
        **route_fields(args, route),
        **proof_fields(args),
        hammer_file=args.hammer_file,
    )
    settings.claim_deferred = claim_deferred(settings, adapter, defer_claim)
    if route is not None and not settings.claim_deferred:
        check_route(parser, settings)
    check_no_project(parser, settings)

    check_outputs(parser, settings,
                  one_file=(("--hammer-file", settings.hammer_file),))
    check_hammer_file(parser, settings)
    check_build(parser, settings)
    resolve_route(parser, settings)
    return settings


def check_hammer_file(parser: argparse.ArgumentParser,
                      settings: Settings) -> None:
    """`--hammer-file` and `--cert-file` each write one file and return:
    neither builds a project, and `--hammer-file` alone needs no module.
    (That neither has an `artifacts/` is `check_outputs`'s to say.)"""
    which = ("--cert-file" if settings.cert_file
             else "--hammer-file" if settings.hammer_file else None)
    if settings.build_cert and which:
        parser.error(
            f"--build-cert is incompatible with {which}: that writes a "
            "file and generates no project to build"
        )
    if not settings.module_file and not settings.hammer_file:
        parser.error("module_file is required (unless --hammer-file is used alone)")


def check_no_project(parser: argparse.ArgumentParser,
                     settings: Settings) -> None:
    """A route and an output that generates no project do not go together.

    Inference runs against the generated project, and `--cert-file` and
    `--hammer-file` each write one file and return before any of that -- so
    the route would not run at all. Every route, not the ones that declare
    they read or write the project: a route that declares neither still
    never runs. Not deferred with the claim, because a `--hammer-file` run
    returns before the claim is read again.
    """
    route = settings.route
    if route is None:
        return
    if settings.cert_file:
        parser.error(
            f"--infer {route.name} is incompatible with --cert-file: "
            f"the standalone certificate is written from the predicates as "
            f"supplied, before a route could infer any"
        )
    if settings.hammer_file:
        parser.error(
            f"--infer {route.name} is incompatible with --hammer-file: that "
            f"writes ZerothHammer.lean and returns, and generates no "
            f"certificate to infer"
        )
