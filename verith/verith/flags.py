"""The inference routes' flags, and the rules about them.

Shared by the two commands that run a route: `verith` (:mod:`verith.cli`)
and `certify`. Each builds its command line from `reactor.lean.cli`'s
groups and adds these: `add_routes` for the flags, then, once parsed,
`resolve_aliases`, `check_route_options`, `route_fields`, `check_route`
and `resolve_route`, in that order. Every rule reads a field of the
route's row, so none of them knows a route by name.
"""

from __future__ import annotations

import argparse

from reactor.lean.cli.checks import check_row_options
from reactor.lean.core.options import given
from reactor.lean.core.errors import Refused

from . import (
    DEFAULT_ROUTE,
    ROUTES,
    InferRoute,
    all_route_aliases,
    route_by_name,
    route_names,
    shared_route_options,
)
from .common.llm import DEFAULT_MODEL

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


def add_routes(parser: argparse.ArgumentParser, *,
               required: bool = False) -> None:
    """The route that infers the certificate, the model the LLM routes
    ask, and every route's own options, grouped by the routes that
    declare them. `required` for a command that does nothing else."""
    parser.add_argument(
        "--infer",
        nargs="?",
        const=DEFAULT_ROUTE,
        default=None,
        required=required,
        choices=list(route_names()),
        help=infer_help(),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model to use for inference (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL, e.g. http://localhost:11434/v1 for Ollama "
            "or https://openrouter.ai/api/v1 for OpenRouter. Hosted providers read "
            "the API key from OPENROUTER_API_KEY or OPENAI_API_KEY; local servers "
            "need no key."
        ),
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


def check_route(parser: argparse.ArgumentParser, args, settings) -> None:
    """Every rule about inference, read off the route's row.

    `settings` is the command's own: the claim's fields, named as
    `reactor.lean.cli.Settings` names them, and `route`, `route_opts`."""
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

    # `--model` and `--base-url` are for the routes that call an LLM. Named
    # as the pair they are even when only one was passed: which of the two
    # was given is not the point, that this route calls no LLM is.
    if not route.uses_llm and (
        args.model != parser.get_default("model") or args.base_url
    ):
        parser.error(
            f"--model and --base-url are for the LLM routes; --infer "
            f"{route.name} calls none"
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
    its options by `dest` (defaulted by `check_route_options`), and the
    model the LLM routes ask."""
    return dict(
        route=route,
        route_opts=(
            {o.dest: getattr(args, o.dest) for o in route.options}
            if route else {}
        ),
        model=args.model,
        base_url=args.base_url,
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
