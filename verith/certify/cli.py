"""The `uv run certify` command line: `reactor`'s flags and `--infer`.

`reactor`'s groups, checks and fields (`reactor.lean.cli`), with the
routes' flags and rules (`verith.flags`) in the places today's command
line has them. What the pipeline gets back is :class:`Settings`:
`reactor`'s, and the route the run selected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from reactor.lean.cli import Settings as ReactorSettings
from reactor.lean.cli import claim_deferred, settings_fields
from reactor.lean.cli.checks import (
    check_blanks,
    check_budget,
    check_claim_flags,
    check_input_options,
    check_outputs,
    input_adapter,
)
from reactor.lean.cli.parser import (
    add_budget,
    add_claim,
    add_input,
    add_outputs,
    add_smt_checks,
)
from verith import InferRoute
from verith.flags import (
    add_routes,
    check_route,
    check_route_options,
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


@dataclass
class Settings(ReactorSettings):
    """`reactor`'s settings, and the inference route: which one (`route`),
    its options (`route_opts`), what it resolved at parse time
    (`route_config`), and the model the LLM routes ask."""

    route: InferRoute | None = None
    route_opts: dict = field(default_factory=dict)
    route_config: object | None = None
    model: str | None = None
    base_url: str | None = None


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
    add_input(parser)
    add_claim(parser)
    add_routes(parser)
    add_outputs(parser)
    add_smt_checks(parser)
    add_budget(parser)
    return parser


def parse_args(argv: "list[str] | None" = None, *,
               defer_claim: bool = True) -> Settings:
    """The command line as `Settings`; `defer_claim` as in
    `reactor.lean.cli.parse_args`."""
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

    settings = Settings(**settings_fields(args, adapter),
                        **route_fields(args, route))
    settings.claim_deferred = claim_deferred(settings, adapter, defer_claim)
    if route is not None and not settings.claim_deferred:
        check_route(parser, args, settings)
        check_cert_file(parser, settings)

    check_outputs(parser, settings, infer=route.name if route else None)
    resolve_route(parser, settings)
    return settings


def check_cert_file(parser: argparse.ArgumentParser, settings: Settings) -> None:
    """A route and `--cert-file` do not go together."""
    # Inference runs against the generated project, and `--cert-file` writes
    # a standalone file and returns before any of that -- so the route would
    # not run at all. Every route, not the ones that declare they read or
    # write the project: a route that declares neither still never runs.
    if settings.cert_file:
        parser.error(
            f"--infer {settings.route.name} is incompatible with --cert-file: "
            f"the standalone certificate is written from the predicates as "
            f"supplied, before a route could infer any"
        )
