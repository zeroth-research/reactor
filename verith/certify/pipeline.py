"""`uv run certify`: `reactor` and `verith` in one run.

A reactive module in, a Lean 4 certificate project out, with the
certificate inferred by the route `--infer` selects: the command `uv run
verith` was before `reactor` and `verith` were split apart, and the same
command line. It runs `reactor`'s stages (`reactor.lean.pipeline`) and
`verith`'s route (`verith.run`) in one process, so the route is handed the
project as it is generated, and the certificate is rewritten from its
answer (`regenerate`).

The stages, in order: the module and the claim are read; the route checks
the module's shape (`route_precheck`); the project is generated with the
route's lakefile entries; the route runs against it (`run_route`); the
certificate is rendered again from the answer; and `--build-cert` builds
it.
"""

from __future__ import annotations

import sys

from reactor.lean.core.certificate import Claim
from reactor.lean.core.errors import Refused
from reactor.lean.project.lakefile import reactor_lean_checkout
from reactor.lean.pipeline.certificate import (
    hint_budget,
    lean_context,
    record_predicates,
    render,
    workspace,
)
from reactor.lean.pipeline.claim import claim_from_flags, load, stated_claim
from reactor.lean.pipeline.outputs import (
    build_cert,
    generate,
    hammer_only,
    standalone,
)
from verith.common.record import note
from reactor.lean.proof.render import LeanCertificate
from reactor.lean.smt.encoding import lazy
from reactor.lean.smt.precheck import pre_check
from verith import InferResult, ProjectHandle
from verith.run import refused, route_precheck, run_route, show_certificate

from .cli import Settings, parse_args


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
    except Refused as e:
        raise refused(settings.route, e) from e

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
    # no route reads `Certificate/`, so cvc5 is asked for hints only about
    # the certificate the project ends up with.
    try:
        lean_cert = render(
            smt, claim, given,
            hints=hint_budget(settings) if settings.route is None else None)
    except Refused as e:
        raise SystemExit(f"error: {e}") from e

    if settings.cert_file:
        standalone(settings.cert_file, ctx, lean_cert)
        return

    project_dir = generate(
        settings, module, ctx, lean_cert, problem.source,
        lakefile=(
            settings.route.lakefile(settings.route_config, settings.project_name)
            if settings.route is not None and settings.route.lakefile
            else None
        ),
    )

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
                           smt=smt, problem=problem)
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


def note_unrenderable(store, route, e: Refused) -> None:
    """Leave in `artifacts/` that a route's answer has no Lean form.

    Not a refusal by the route: it answered, and what failed was printing
    the answer as Lean, after the route had returned. So no note of the
    route's own can say this, and without one the next run reads a found
    predicate in `artifacts/` and nothing about why no certificate came of
    it -- and an LLM route resuming from it proposes the same shape again.
    """
    note(
        store, f"{e}\n",
        status="unsupported",
        what=f"An answer by `--infer {route.name}` that has no Lean form.",
        why=("The route answered; what failed is writing that answer as "
             "Lean, so the same shape proposed again fails the same way."),
    )


def regenerate(settings: Settings, smt, handle: ProjectHandle,
               claim: Claim, result: InferResult) -> LeanCertificate:
    """The project's certificate, rewritten from the route's answer.

    The certificate is that answer alone, under the claim the run was given,
    and its tactic plan is restated over it. Both files are rewritten --
    `Data.lean` holds the definitions and `Certificate.lean`'s tactics are
    generated from their shape -- through the handle, so a route that
    writes more of the project and this pipeline go through one writer.
    """
    cert, lean = result.cert, result.lean
    store = handle.artifacts
    if settings.pre_check == "cvc5":
        pre_check(smt, claim, lean or cert, settings.budget,
                  artifacts=store)
    record_predicates(store, claim, cert, inferred=True)
    try:
        lean_cert = render(smt, claim, cert,
                           hints=hint_budget(settings) if lean is None else None,
                           restated=True, verbatim=lean)
    except Refused as e:
        note_unrenderable(store, settings.route, e)
        raise refused(settings.route, e) from e
    handle.write_predicates(lean_cert)
    return lean_cert
