"""`uv run verith`: infer a module's certificate, and prove it.

A module and a claim in, the invariant -- and, for a Buchi property, the
ranking function -- out, in the input's names and in the form `reactor`'s
--invariant and --ranking take them::

    uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer houdini

The module and the claim are read with the same flags `reactor` reads them
with (`reactor_lean.cli`), and the route, the SMT budget and what is asked
of the proofs with the flags `certify` takes (`verith.flags`), so a command
line moves between the three by adding or dropping flags.

`--project DIR` names a project `reactor` generated from the same module,
whose obligations it left proved by `sorry`. With one, the certificate --
the one the route found, or without `--infer` the one --invariant and
--ranking give -- is written into it with its proofs (`Certificate/`, and
the hammer the proofs call), and `--build-cert` builds it. A route whose
row reads or writes the project -- the LLM routes prompt on its Lean,
`fbk-proveit` installs a certificate of its own in it -- needs one. With a
project the route also reads and writes its `artifacts/`, so a run
continues where the last one in that project left off.

`--json` prints the answer as one JSON object on stdout, and everything the
route logs on stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from reactor_lean import inputs
from reactor_lean.artifacts import ARTIFACTS_DIR
from reactor_lean.cli import claim_deferred
from reactor_lean.cli.checks import (
    check_blanks,
    check_claim_flags,
    check_input_options,
    input_adapter,
)
from reactor_lean.cli.parser import add_claim, add_input
from reactor_lean.core.certificate import Claim
from reactor_lean.core.errors import Refused
from reactor_lean.pipeline.certificate import (
    lean_context,
    record_predicates,
    workspace,
)
from reactor_lean.pipeline.claim import claim_from_flags, load, stated_claim
from reactor_lean.project.lakefile import LakefileParts
from reactor_lean.project.scaffold import write_lakefile

from .common.encoding import lazy
from .common.limits import SmtBudget
from .flags import (
    add_proofs,
    add_routes,
    check_budget,
    check_route,
    check_route_options,
    proof_fields,
    resolve_aliases,
    resolve_route,
    route_fields,
    selected_route,
    supplied_complete,
)
from .io import InferResult, ProjectHandle, declared
from .proofs import build_cert
from .route import InferRoute
from .run import (
    lakefile_parts,
    regenerate,
    route_precheck,
    run_route,
    show_certificate,
)


_EPILOG = """\
examples:
  # no LLM: Houdini and a ranking search, each candidate decided by cvc5
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer houdini

  # no LLM: synthesise the invariant of a safety property with cvc5's SyGuS
  uv run verith mymodule.py --safety "(not (= (X @intf.0) 1))" --infer sygus

  # the answer as JSON on stdout, the route's log on stderr
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer smt-linear --json

  # an LLM route prompts on the generated Lean, so it needs the project;
  # the certificate it finds is written into it, with its proofs
  uv run reactor mymodule.py --buchi "(= (X @intf.0) 0)" -o out/ -p MyProject
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer ai-cegis \\
      --project out/MyProject

  # no --infer: prove the certificate given, ask cvc5 first, and build it
  uv run reactor mymodule.py --buchi "(= (X @intf.0) 0)" \\
      --invariant '...' --ranking '...' -o out/ -p MyProject
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" \\
      --invariant '...' --ranking '...' --project out/MyProject \\
      --pre-check cvc5 --smt-tactics cvc5 --build-cert

`certify` runs reactor and verith in one go: it takes reactor's flags,
--infer and the proofs' flags.
"""


@dataclass
class Settings:
    """A validated `verith` command line.

    The claim's fields are named as `reactor_lean.cli.Settings` names them,
    so the stages that read a claim (`reactor_lean.pipeline.claim`) and the
    route checks (`verith.flags`) read this one too.
    """

    module_file: str
    input_format: str | None
    prp: str | None
    kind: str
    pre: str | None
    invariant: str | None
    ranking: str | None
    budget: SmtBudget
    input_opts: dict = field(default_factory=dict)

    route: InferRoute | None = None
    route_opts: dict = field(default_factory=dict)
    route_config: object | None = None

    project: Path | None = None         # `--project`
    artifacts: str = "use"              # "use" | "ignore" | "reset"
    json: bool = False

    pre_check: str = "none"
    smt_tactics: str = "none"
    build_cert: bool = False

    claim_deferred: bool = False

    @property
    def is_safety(self) -> bool:
        return self.kind == "safety"

    @property
    def project_name(self) -> str | None:
        """The project's name, which is its directory's: what a route that
        names Lean modules after it (`fbk-proveit`) checks."""
        return self.project.name if self.project else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verith",
        description=(
            "Infer the certificate of a reactive module's property -- the "
            "invariant, plus the ranking function a --buchi property also "
            "needs -- and print it as reactor's --invariant and --ranking "
            "take it; and prove it, in the project --project names. The "
            "module and the claim are read as reactor reads them; --infer "
            "selects how the certificate is searched for, and without it "
            "the certificate is the one --invariant and --ranking give."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_input(parser, project=False)
    add_claim(parser)
    add_routes(parser)
    parser.add_argument(
        "--project",
        default=None,
        metavar="DIR",
        help=(
            "A Lean project `reactor` (or `certify`) generated from the same "
            "module: the project directory, `-o DIR -p NAME`'s DIR/NAME. "
            "The certificate is written into it with its proofs. Required "
            "by a route that reads or writes the project; with one, every "
            f"route reads and writes its `{ARTIFACTS_DIR}/`."
        ),
    )
    parser.add_argument(
        "--artifacts",
        default="use",
        choices=["use", "ignore", "reset"],
        help=(
            f"What to do with --project's `{ARTIFACTS_DIR}/`: `use` "
            f"(default) lets the route start from what an earlier run left "
            f"there, `ignore` searches from scratch and still records what "
            f"this run finds, `reset` empties it first."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print the answer as one JSON object on stdout, and the route's "
            "log on stderr."
        ),
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
        module_file=args.module_file,
        input_format=adapter.name if adapter else None,
        input_opts=(
            {o.dest: getattr(args, o.dest) for o in adapter.options}
            if adapter else {}
        ),
        prp=args.safety or args.buchi,
        kind="safety" if args.safety else "buchi",
        pre=args.pre,
        invariant=args.invariant,
        ranking=args.ranking,
        project=Path(args.project) if args.project else None,
        artifacts=args.artifacts,
        json=args.json,
        **proof_fields(args),
        **route_fields(args, route),
    )
    settings.claim_deferred = claim_deferred(settings, adapter, defer_claim)

    check_project(parser, settings)
    if not settings.claim_deferred:
        if route is not None:
            check_route(parser, settings)
        else:
            # Which also leaves --build-cert nothing to refuse: the
            # certificate given is complete.
            check_given(parser, settings)
    resolve_route(parser, settings)
    return settings


def check_given(parser: argparse.ArgumentParser, settings: Settings) -> None:
    """Without --infer, the certificate is the one given, and the run has
    something to do with it: prove it in --project, or ask cvc5 about it."""
    if settings.project is None and settings.pre_check == "none":
        parser.error(
            "nothing to do: pass --infer to search for a certificate, "
            "--project to prove the one --invariant and --ranking give, or "
            "--pre-check cvc5 to ask cvc5 about it"
        )
    if not supplied_complete(settings):
        parser.error(
            "without --infer, the certificate is the one given: pass "
            "--safety with --invariant, or --buchi with --invariant and "
            "--ranking"
        )


def check_project(parser: argparse.ArgumentParser, settings: Settings) -> None:
    """`--project` names a generated project, and a route that reads or
    writes one, or a flag about the proofs written into one, is given it."""
    route, project = settings.route, settings.project
    if project is None:
        needs = [w for w, given in (("reads", route and route.reads),
                                    ("writes", route and route.owns)) if given]
        if needs:
            parser.error(
                f"--infer {route.name} {' and '.join(needs)} the generated "
                f"Lean project: generate one with `reactor MODULE -o DIR -p "
                f"NAME` and pass --project DIR/NAME, or run `certify`, which "
                f"does both"
            )
        if settings.artifacts != "use":
            parser.error(
                f"--artifacts {settings.artifacts} is about --project's "
                f"`{ARTIFACTS_DIR}/`, and no --project was given"
            )
        for flag, given in (("--smt-tactics", settings.smt_tactics != "none"),
                            ("--build-cert", settings.build_cert)):
            if given:
                parser.error(
                    f"{flag} is about the proofs written into --project, "
                    f"and no --project was given"
                )
        return
    if not (project / "lakefile.toml").is_file():
        parser.error(
            f"--project {project} is not a generated Lean project: it has no "
            f"lakefile.toml"
        )


def project_handle(settings: Settings, module,
                   claim: Claim) -> ProjectHandle:
    """The project the route is handed and the proofs are written into:
    `--project`, or none.

    The project has to be about this module -- a route reads its Lean and
    writes files that import it, and the proofs are of its obligations --
    and `dbg/system.txt` is the module it was generated from, as
    `str(module)` renders it (which is also what `artifacts/` tells modules
    apart by), compared up to the numbering of its wires (`same_module`).
    """
    route = settings.route
    project = settings.project
    if project is None:
        return ProjectHandle(dir=None, name="", ctx=None, artifacts=None,
                             **declared(route))

    stamp = project / "dbg" / "system.txt"
    if not stamp.is_file() or not same_module(stamp.read_text(), str(module)):
        raise SystemExit(
            f"error: --project {project} was not generated from this module "
            f"({stamp} {'differs' if stamp.is_file() else 'is missing'}); "
            f"generate it again with `reactor`"
        )
    try:
        ctx = lean_context(module)
    except Refused as e:                # about the module, not the route
        raise SystemExit(f"error: {e}") from e
    store = workspace(project, module, claim, kind=settings.kind,
                      mode=settings.artifacts,
                      producer=route.name if route else "")
    # The lakefile `reactor` generates carries no route's libraries or
    # requires, nor the hammer the proofs call; what this run needs is
    # added now, before the route writes the files that need it.
    parts = lakefile_parts(route, settings.route_config, project.name)
    if parts != LakefileParts():
        try:
            write_lakefile(
                project, project.name,
                executable=(project / "Main.lean").exists(),
                parts=parts,
            )
        except Refused as e:            # a bad `$REACTLIB` or Lake cache
            raise SystemExit(f"error: {e}") from e
    return ProjectHandle(dir=project, name=project.name, ctx=ctx,
                         artifacts=store, **declared(route))


def same_module(a: str, b: str) -> bool:
    """Whether two renderings of a module are one module.

    A wire prints as `#<id>`, and an id is a process-wide counter: the same
    file loaded twice in one process renders with other ids, though loaded
    once in a fresh process it renders the same. So each rendering's ids are
    renumbered in the order they first appear, which keeps which wire feeds
    which and drops where the counter happened to be.
    """
    def canonical(text: str) -> str:
        ids: dict[str, int] = {}
        return re.sub(r"#(\d+)",
                      lambda m: f"#{ids.setdefault(m.group(1), len(ids))}",
                      text)

    return canonical(a) == canonical(b)


def answer_json(settings: Settings, result: InferResult,
                names: "inputs.Names") -> dict:
    """The answer as `--json` prints it: the predicates in the input's
    names (what `reactor` takes) and in the internal ones (what
    `artifacts/` records), the Lean of a route that answers in Lean, and
    the files a route that answers with files wrote."""
    cert = result.cert

    def surface(text: "str | None") -> "str | None":
        return inputs.to_surface(text, names) if text else None

    route = settings.route
    return {
        "route": route.name if route else None,
        "kind": settings.kind,
        "returns": route.returns if route else None,
        "invariant": surface(cert.inv),
        "ranking": surface(cert.ranking),
        "smt": {"inv": cert.inv, "ranking": cert.ranking},
        "lean": (
            {"inv": result.lean.inv, "ranking": result.lean.ranking}
            if result.lean is not None else None
        ),
        "files": [str(settings.project / rel) for rel in result.files],
    }


def show_answer(settings: Settings, result: InferResult,
                names: "inputs.Names") -> None:
    """The answer, for a reader: as --invariant and --ranking take it, or
    the Lean a route answered in, or the files it wrote."""
    show_certificate(result.cert, names)
    if result.lean is not None:
        print(".. The certificate, in Lean:")
        print(f"   inv := {result.lean.inv}")
        if result.lean.ranking is not None:
            print(f"   ranking := {result.lean.ranking}")
    if result.files:
        print(".. The certificate, as the files it wrote:")
        for rel in result.files:
            print(f"   {settings.project / rel}")


def main(argv: "list[str] | None" = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    settings = parse_args(argv)
    if not settings.json:
        infer(settings, argv)
        return
    # stdout is the answer's alone; everything else is the log.
    with contextlib.redirect_stdout(sys.stderr):
        settings, result, names = infer(settings, argv)
    print(json.dumps(answer_json(settings, result, names), indent=2))


def infer(settings: Settings, argv: list[str]):
    """Run the route the command line selects, if it selects one, and prove
    the certificate: the route's, or the one given. Returns the settings
    (with the claim the input states, if it states one), the answer and the
    names it is shown in."""
    problem = load(settings)
    settings = stated_claim(settings, problem, argv, reparse=parse_args)
    module = problem.module
    # The run's one cvc5 encoding, built by whichever stage asks first.
    smt = lazy(module)
    claim, given = claim_from_flags(settings, problem.names)
    route = settings.route
    prechecked = route_precheck(settings, module, claim, smt)
    handle = project_handle(settings, module, claim)
    # As `certify` records them: what the run was given before the route,
    # what it found after. Without a project there is nowhere to.
    record_predicates(handle.artifacts, claim, given)

    result = InferResult(cert=given)
    if route is not None:
        print(f".. Inferring the certificate with --infer {route.name}")
        result = run_route(settings, module, handle, claim, given, prechecked,
                           smt=smt)
    lean_cert = regenerate(settings, smt, handle, claim, result)
    if route is not None and not settings.json:
        show_answer(settings, result, problem.names)
    if settings.build_cert:
        build_cert(handle.dir, lean_cert)
    return settings, result, problem.names
