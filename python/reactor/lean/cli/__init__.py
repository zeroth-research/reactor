"""The `uv run verith` command line: the flags, and what combinations mean.

Apart from the pipeline (:mod:`reactor.lean.pipeline`) so that the pipeline
and the argument grammar are two things.  What the pipeline gets back is
:class:`Settings` -- validated, with the property and its kind already
decided and the inference route already looked up -- rather than an
`argparse.Namespace` to re-derive facts from.  `parser` is the flags
(`build_parser`), `checks` the rules about which go together, `settings`
what they are read into; `parse_args` runs the three.

Nothing here knows a route by name.  Every cross-flag rule about inference
reads a field of the route's row in :mod:`reactor.lean.infer`: the check
is generic, and the sentence printed when it fails comes from the row, so a
refusal still says why *that* route refuses.  Adding a route is a row there
and no edit here.
"""

from __future__ import annotations

from pathlib import Path

from ..core.errors import Refused
from ..infer import route_by_name
from .checks import (
    check_blanks,
    resolve_aliases,
    input_adapter,
    check_input_options,
    check_route_options,
    check_route,
    check_outputs,
)
from .parser import build_parser
from .settings import SEED_FLAGS, Settings

__all__ = ["SEED_FLAGS", "Settings", "build_parser", "parse_args"]


def parse_args(argv: "list[str] | None" = None) -> Settings:
    """The command line as `Settings`."""
    parser = build_parser()
    args = parser.parse_args(argv)

    check_blanks(parser, args)
    resolve_aliases(parser, args)

    # A certificate proves one property under one proof rule, and which rule
    # decides what the certificate even consists of -- a ranking function or
    # no ranking function. So the two flags are not a pair of properties to
    # be combined; they are the choice.
    if args.safety and args.buchi:
        parser.error(
            "--safety and --buchi are mutually exclusive: a certificate "
            "proves `G FORMULA` or `G (F FORMULA)`, under one proof rule"
        )

    # `G FORMULA` needs no ranking function -- `rule_globally` has nowhere to
    # put one. Taking it and dropping it would look like it had been used.
    if args.safety and args.ranking:
        parser.error(
            "--ranking is meaningless with --safety: `G FORMULA` is proved "
            "by an invariant that implies FORMULA, and `rule_globally` takes "
            "no ranking function. Ranking functions belong to --buchi."
        )

    adapter = input_adapter(parser, args)
    check_input_options(parser, args, adapter)

    route = route_by_name(args.infer) if args.infer else None
    check_route_options(parser, args, route)

    settings = Settings(
        module_file=args.module_file,
        output_dir=Path(args.output_dir),
        project_name=args.project_name,
        input_format=adapter.name if adapter else None,
        input_opts=(
            {o.dest: getattr(args, o.dest) for o in adapter.options}
            if adapter else {}
        ),
        executable=args.executable,
        prp=args.safety or args.buchi,
        kind="safety" if args.safety else "buchi",
        pre=args.pre,
        invariant=args.invariant,
        ranking=args.ranking,
        route=route,
        route_opts=(
            {o.dest: getattr(args, o.dest) for o in route.options} if route else {}
        ),
        cert_file=args.cert_file,
        hammer_file=args.hammer_file,
        build_cert=args.build_cert,
        model=args.model,
        base_url=args.base_url,
    )

    if route is not None:
        check_route(parser, args, settings)

    check_outputs(parser, settings)

    # Last, because it is the only check that touches the filesystem: a
    # route whose lakefile names a path has to reject a bad one before any
    # of the project exists, and the lakefile is written while it is created.
    if route is not None and route.resolve is not None:
        try:
            settings.route_config = route.resolve(settings.route_opts)
        except Refused as e:
            parser.error(str(e))

    return settings
