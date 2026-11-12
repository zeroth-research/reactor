"""`uv run certify`: `reactor` and `verith` in one run.

A reactive module in, a Lean 4 certificate project out, with the
certificate inferred by the route `--infer` selects and proved: the command
`uv run verith` was before `reactor` and `verith` were split apart, and the
same command line. It runs `reactor`'s stages (`reactor.lean.pipeline`),
`verith`'s route (`verith.run`) and `verith`'s proofs (`verith.proofs`) in
one process, so the route is handed the project as it is generated, and
the certificate is rewritten from its answer, with its proofs
(`regenerate`).

The stages, in order: the module and the claim are read; the route checks
the module's shape (`route_precheck`); the project is generated with the
route's lakefile entries and the proofs'; the route runs against it
(`run_route`); the certificate is rendered again from the answer, cvc5
asked about it (`--pre-check`) and its proofs written; and `--build-cert`
builds it. Without a route, the certificate is the one supplied and is
proved as the project is generated.
"""

from __future__ import annotations

import sys

from reactor.lean.core.errors import Refused
from reactor.lean.project.lakefile import LakefileParts, reactor_lean_checkout
from reactor.lean.project.scaffold import write_files
from reactor.lean.pipeline.certificate import (
    lean_context,
    record_predicates,
    render,
    workspace,
)
from reactor.lean.pipeline.claim import claim_from_flags, load, stated_claim
from reactor.lean.pipeline.outputs import generate, standalone
from reactor.lean.proof.render import SORRY
from reactor.lean.smt.encoding import lazy
from verith import ProjectHandle
from verith.proofs import (
    LAKEFILE,
    build_cert,
    hammer_files,
    hammer_only,
    hint_budget,
    pre_check,
    prove,
)
from verith.run import regenerate, route_precheck, run_route, show_certificate

from .cli import parse_args


def main(argv: "list[str] | None" = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    settings = parse_args(argv)

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
    settings = stated_claim(settings, problem, argv, reparse=parse_args)
    module = problem.module
    print(module)
    # The run's one cvc5 encoding, built by whichever stage asks first.
    smt = lazy(module)
    claim, given = claim_from_flags(settings, problem.names)
    prechecked = route_precheck(settings, module, claim, smt)
    try:
        ctx = lean_context(module)
    except Refused as e:                # about the module, not the route
        raise SystemExit(f"error: {e}") from e

    # The workspace belongs to the project, not to the route: `--artifacts
    # reset` is what to pass when a bad artifact is being inherited by every
    # run, and a run that selects no route is one of those runs. Built
    # before the pre-check, the run's first encoding, so what that encodes
    # is kept.
    store = (
        None if settings.cert_file
        else workspace(
            settings.output_dir / settings.project_name, module, claim,
            kind=settings.kind, mode=settings.artifacts,
            producer=settings.route.name if settings.route else "")
    )
    record_predicates(store, claim, given)

    # With a route the certificate is the inferred one, and `regenerate`
    # checks that instead.
    if settings.pre_check == "cvc5" and settings.route is None:
        if claim is None:
            print(".. SMT pre-check: nothing to check (no certificate data)")
        else:
            pre_check(smt, claim, given, settings.budget, artifacts=store)

    # With a route this rendering is a placeholder the answer replaces, and
    # no route reads `Certificate/`, so it is proved only once there is an
    # answer: cvc5 is asked for hints only about the certificate the project
    # ends up with.
    try:
        lean_cert = render(smt, claim, given)
        proofs = SORRY if settings.route is not None else prove(
            smt, ctx, claim, given, lean_cert, budget=hint_budget(settings))
    except Refused as e:
        raise SystemExit(f"error: {e}") from e

    if settings.cert_file:
        standalone(settings.cert_file, ctx, lean_cert, proofs)
        return

    route = settings.route
    project_dir = generate(
        settings, module, ctx, lean_cert, problem.source,
        lakefile=(
            (route.lakefile(settings.route_config, settings.project_name)
             if route is not None and route.lakefile else LakefileParts())
            # A route that answers with files proves its certificate itself.
            + (LAKEFILE if route is None or route.returns != "files"
               else LakefileParts())
        ),
        proofs=proofs,
    )
    if route is None:
        write_files(project_dir, hammer_files())

    if settings.route is not None:
        print(f".. Inferring the certificate with --infer {settings.route.name}")
        handle = ProjectHandle(
            dir=project_dir,
            name=settings.project_name,
            ctx=ctx,
            reads=frozenset(settings.route.reads),
            owns=settings.route.owns,
            resumes=frozenset(settings.route.reads_artifacts),
            artifacts=store,
        )
        result = run_route(settings, module, handle, claim, given, prechecked,
                           smt=smt)
        # `files`: the certificate is the route's own, so there is nothing
        # of ours to be complete about and nothing to regenerate from.
        lean_cert = (
            None if settings.route.returns == "files"
            else regenerate(settings, smt, handle, claim, result)
        )
        handle.write_files(result.files)
        show_certificate(result.cert, problem.names)

    if settings.build_cert:
        build_cert(project_dir, lean_cert)

    print(f"\nProject ready at: {project_dir}")
