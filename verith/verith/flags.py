"""The inference routes' flags, the proofs' flags, and the rules about them.

Shared by the two commands that run a route and write the proofs: `verith`
(:mod:`verith.cli`) and `certify`. Each builds its command line from
`reactor_lean.cli`'s groups and adds these: `add_routes` for the flags,
then, once parsed, `resolve_aliases`, `check_route_options`, `route_fields`,
`check_route` and `resolve_route`, in that order. Every rule reads a field
of the route's row, so none of them knows a route by name.

The proofs' flags are added by `add_proofs` -- whether the certificate is
built, what cvc5 is asked about it and under what budget -- and read by
`check_budget`, `proof_fields`, `supplied_complete` and `check_build`.
"""

from __future__ import annotations

import argparse

from reactor_lean.cli.checks import check_row_options
from reactor_lean.core.options import given
from reactor_lean.core.errors import Refused
from .common.limits import DEFAULT_CALL_MS, DEFAULT_PHASE_MS, SmtBudget

from . import (
    DEFAULT_ROUTE,
    ROUTES,
    InferRoute,
    all_route_aliases,
    route_by_name,
    route_names,
    shared_route_options,
)

# One row per predicate flag: the predicate it supplies, by its short name;
# the `Settings` attribute that carries it; and the flag to name in a
# refusal. The first two differ for `inv`, which is why the attribute
# is in the table rather than patched at the one place that reads it.
SEED_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("inv", "invariant", "--invariant"),
    ("ranking", "ranking", "--ranking"),
    ("pre", "pre", "--pre"),
)


def infer_help() -> str:
    """`--infer`'s help, assembled from the rows."""
    lines = [
        "Infer the certificate for --safety or --buchi: the invariant, plus "
        "the ranking function --buchi needs. One of:",
    ]
    for r in ROUTES:
        kinds = " and ".join(f"--{k}" for k in sorted(r.kinds))
        lines.append(f"`{r.name}` ({kinds}) -- {r.summary}.")
    lines.append(f"Passed without a value, `{DEFAULT_ROUTE}`.")
    return " ".join(lines)


def add_routes(parser: argparse.ArgumentParser) -> None:
    """The route that infers the certificate, and every route's own
    options, grouped by the routes that declare them."""
    parser.add_argument(
        "--infer",
        nargs="?",
        const=DEFAULT_ROUTE,
        default=None,
        choices=list(route_names()),
        help=infer_help(),
    )

    # One group per option, named for every route that declares it, so
    # `--help` shows whose flag is whose and says when a flag is two routes'.
    groups: dict = {}
    for opt, owners in shared_route_options():
        title = ", ".join(f"--infer {r.name}" for r in owners)
        if title not in groups:
            groups[title] = parser.add_argument_group(title)
        # `default=None` so that "given" is distinguishable from "not
        # given" for the stray-option check; the row's own default is
        # put back in `check_route_options` once the route is known.
        kwargs = dict(opt.kwargs)
        kwargs["default"] = None
        groups[title].add_argument(*opt.flags, **kwargs)

    # Deprecated spellings, from the rows that own them: each selects its
    # route and fills one of that route's options, which is why it cannot be
    # an `Opt` of its own.
    for route, alias in all_route_aliases():
        parser.add_argument(
            alias.flag, default=None, metavar="DIR", help=alias.help
        )


def resolve_aliases(parser: argparse.ArgumentParser, args) -> None:
    """Rewrite each deprecated spelling into the route and option it means.

    One loop over the rows, so a route's older flag goes away with the route
    and a second one is a row entry rather than another branch here.
    """
    for route, alias in all_route_aliases():
        given = getattr(args, alias.flag.lstrip("-").replace("-", "_"), None)
        if given is None:
            continue
        if args.infer not in (None, route.name):
            parser.error(
                f"{alias.flag} selects `--infer {route.name}`, so it cannot be "
                f"combined with --infer {args.infer}: a run takes one route"
            )
        explicit = getattr(args, alias.fills, None)
        flag = next(
            o.flags[-1] for o in route.options if o.dest == alias.fills
        )
        if explicit is not None and explicit != given:
            parser.error(
                f"{alias.flag} and {flag} name the same thing and were given "
                f"different values"
            )
        args.infer = route.name
        setattr(args, alias.fills, given)


def check_route_options(
    parser: argparse.ArgumentParser, args, route: "InferRoute | None"
) -> None:
    """A route-scoped option is only meaningful with its route.

    Given without it, the flag would do nothing -- which is the failure mode
    worth reporting: the run would succeed having ignored what was asked.
    An option several routes declare is stray only when *none* of them was
    selected: `--vampire` is meaningful to two routes and a no-op under the
    other five.
    """
    check_row_options(
        parser, args,
        [(o, [f"--infer {r.name}" for r in owners])
         for o, owners in shared_route_options()
         if route not in owners and given(o, args)],
        lambda rows: f"together with {', '.join(rows)}",
        f"--infer {route.name}" if route else "",
        route.options if route else None,
    )


def check_route(parser: argparse.ArgumentParser, settings) -> None:
    """Every rule about inference, read off the route's row.

    `settings` is the command's own: the claim's fields, named as
    `reactor_lean.cli.Settings` names them, and `route`, `route_opts`."""
    route = settings.route
    if not settings.prp:
        # Naming the kinds this route has rather than both of them: a route
        # that only certifies `G P` asking for "--safety or --buchi" would
        # name a flag it goes on to refuse.
        kinds = " or ".join(f"--{k}" for k in sorted(route.kinds, reverse=True))
        parser.error(f"--infer {route.name} requires {kinds}")

    # Which proof rules this route can produce a certificate for.
    if settings.kind not in route.kinds:
        parser.error(
            route.kinds_refusal
            or (
                f"--infer {route.name} does not infer a {settings.kind} "
                f"certificate (it infers: {', '.join(sorted(route.kinds))})"
            )
        )

    # A predicate the route will not seed from would be discarded. Saying so
    # beats dropping it silently: the run would succeed, having proved
    # something about a certificate the user did not describe.
    discarded = [
        flag
        for field, attr, flag in SEED_FLAGS
        if getattr(settings, attr) and field not in route.seeds
    ]
    if discarded:
        because = f": {route.seeds_refusal}" if route.seeds_refusal else ""
        parser.error(
            f"--infer {route.name} is incompatible with "
            f"{', '.join(discarded)}{because}"
        )

    if route.extra_check is not None:
        problem = route.extra_check(settings)
        if problem:
            parser.error(problem)


def selected_route(args) -> "InferRoute | None":
    """The row `--infer` names, if it was given."""
    return route_by_name(args.infer) if args.infer else None


def route_fields(args, route: "InferRoute | None") -> dict:
    """The fields a command's `Settings` carries about the route: the row,
    and its options by `dest` (defaulted by `check_route_options`)."""
    return dict(
        route=route,
        route_opts=(
            {o.dest: getattr(args, o.dest) for o in route.options}
            if route else {}
        ),
    )


def resolve_route(parser: argparse.ArgumentParser, settings) -> None:
    """The route's `resolve` hook, into `settings.route_config`.

    Last, because it is the only check that touches the filesystem: a
    route whose lakefile names a path has to reject a bad one before any
    of the project exists, and the lakefile is written while it is created.
    """
    route = settings.route
    if route is not None and route.resolve is not None:
        try:
            settings.route_config = route.resolve(settings.route_opts)
        except Refused as e:
            parser.error(str(e))


# ══════════════════════════════════════════════════════════════════════════
# The proofs
# ══════════════════════════════════════════════════════════════════════════


def add_proofs(parser: argparse.ArgumentParser) -> None:
    """The proofs' flags: whether the certificate is built, what cvc5 is
    asked about it, and under what budget."""
    add_build(parser)
    add_smt_checks(parser)
    add_budget(parser)


def proof_fields(args) -> dict:
    """The fields a command's `Settings` carries about the proofs."""
    return dict(
        pre_check=args.pre_check,
        smt_tactics=args.smt_tactics,
        build_cert=args.build_cert,
        budget=SmtBudget(per_call_ms=args.smt_timeout,
                         phase_ms=args.smt_budget),
    )


def add_smt_checks(parser: argparse.ArgumentParser) -> None:
    """What cvc5 is asked about the certificate."""
    parser.add_argument(
        "--pre-check",
        default="none",
        choices=["none", "cvc5"],
        help=(
            "Before writing the proofs, ask cvc5 whether the certificate's "
            "obligations are actually true. A refuted obligation means the invariant or "
            "ranking is wrong, which a failing `lake build` cannot tell you "
            "apart from tactics that are merely too weak. Bounded "
            "by --smt-timeout / --smt-budget; anything cvc5 cannot answer is "
            "reported as unknown and changes nothing."
        ),
    )
    parser.add_argument(
        "--smt-tactics",
        default="none",
        choices=["none", "cvc5"],
        help=(
            "Let cvc5 inform the proofs' tactics: discharge the branch "
            "conditions the invariant already settles instead of splitting "
            "on them, hand `nlinarith` the products cvc5's own refutation "
            "needed, and drop a precondition no refutation used. Bounded by "
            "--smt-timeout / --smt-budget; strictly additive, so anything "
            "cvc5 cannot answer leaves the plan as it would have been."
        ),
    )


def add_budget(parser: argparse.ArgumentParser) -> None:
    """The time cvc5 is given: per query, and per phase of queries."""
    parser.add_argument(
        "--smt-timeout",
        type=int,
        default=DEFAULT_CALL_MS,
        metavar="MS",
        help=(
            f"Wall-clock limit for a single SMT query, in ms "
            f"(default: {DEFAULT_CALL_MS})."
        ),
    )
    parser.add_argument(
        "--smt-budget",
        type=int,
        default=DEFAULT_PHASE_MS,
        metavar="MS",
        help=(
            f"Wall-clock limit for one phase of SMT queries, in ms "
            f"(default: {DEFAULT_PHASE_MS}). A phase that fans out over many "
            f"small queries still cannot exceed this."
        ),
    )


def add_build(parser: argparse.ArgumentParser) -> None:
    """Whether the certificate, once proved, is built."""
    parser.add_argument(
        "--build-cert",
        action="store_true",
        help=(
            "Build the certificate once its proofs are written: `lake "
            "update` then `lake build Certificate` in the project. Needs a "
            "certificate with predicates in it, so pass --invariant and "
            "--ranking or --infer; an obligation left as `sorry` is an "
            "error, as is one lake cannot close. Off by default because the "
            "build resolves and compiles cslib, Mathlib, lean-smt and cvc5 "
            "-- minutes against a warm package cache, much longer against a "
            "cold one."
        ),
    )


def check_budget(parser: argparse.ArgumentParser, args) -> None:
    """`--smt-timeout` and `--smt-budget` are positive."""
    # `phase_left_ms` is `max(0, phase_ms - spent)` and a phase is exhausted
    # at zero, so a non-positive budget is not "unlimited": it is a phase
    # that never runs a query and says nothing about why.
    for flag, value in (
        ("--smt-timeout", args.smt_timeout),
        ("--smt-budget", args.smt_budget),
    ):
        if value <= 0:
            parser.error(
                f"{flag} must be a positive number of milliseconds, got {value}"
            )


def supplied_complete(settings) -> bool:
    """Whether the flags alone carry a certificate: the property and the
    invariant, and the ranking function unless the property is a safety
    one."""
    return bool(settings.prp and settings.invariant
                and (settings.is_safety or settings.ranking))


def check_build(parser: argparse.ArgumentParser, settings) -> None:
    """--build-cert needs a certificate worth building: one the run's route
    finds, or one the flags supply. Whether there is a project to build it
    in is each command's own rule.

    A bare project's unsupplied fields default to `True` (`inv`, the
    preconditions) or to `sorry` (`P`, `ranking`), and its obligations are
    still handed to the tactics -- which cannot close one stated over a
    `sorry`, so `lake build` fails on it rather than compiling to a proof
    of nothing. Asking for that is a mistake."""
    if not settings.build_cert:
        return
    if not (settings.route is not None or supplied_complete(settings)
            or settings.claim_deferred):
        parser.error(
            "--build-cert needs a certificate to build: pass --safety "
            "with --invariant, or --buchi with --invariant and "
            "--ranking. Without them the certificate's predicates are "
            "`True` or `sorry`, and its obligations cannot be discharged."
        )
