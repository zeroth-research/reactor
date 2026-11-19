"""The `uv run reactor` command line: the flags, and what combinations mean.

Apart from the pipeline (:mod:`reactor.lean.pipeline`) so that the pipeline
and the argument grammar are two things.  What the pipeline gets back is
:class:`Settings` -- validated, with the property and its kind already
decided -- rather than an `argparse.Namespace` to re-derive facts from.
`parser` is the flags (`build_parser`), `checks` the rules about which go
together, `settings` what they are read into; `parse_args` runs the three.
A command built on this one assembles its command line from the same
groups (`parser.add_*`), checks and fields (`settings_fields`), and adds
flags and rules of its own.
"""

from __future__ import annotations

from pathlib import Path

from ..inputs import InputAdapter
from .checks import (
    check_blanks,
    check_claim_flags,
    check_input_options,
    check_outputs,
    input_adapter,
)
from .parser import build_parser
from .settings import Settings

__all__ = [
    "Settings", "build_parser", "claim_deferred",
    "parse_args", "settings_fields",
]


def parse_args(argv: "list[str] | None" = None, *,
               defer_claim: bool = True) -> Settings:
    """The command line as `Settings`. With `defer_claim`, an input whose
    format can state its own property and no --safety/--buchi leaves the
    checks that need the claim to the second parse, once it is read."""
    parser = build_parser()
    args = parser.parse_args(argv)

    check_blanks(parser, args)
    check_claim_flags(parser, args)

    adapter = input_adapter(parser, args)
    check_input_options(parser, args, adapter)

    settings = Settings(**settings_fields(args, adapter), parser=args.parser)
    settings.claim_deferred = claim_deferred(settings, adapter, defer_claim)
    check_outputs(parser, settings)
    return settings


def settings_fields(args, adapter: "InputAdapter | None") -> dict:
    """`Settings`' fields, read off a command line built from
    `build_parser`'s groups."""
    return dict(
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
        cert_file=args.cert_file,
        artifacts=args.artifacts,
    )


def claim_deferred(settings, adapter: "InputAdapter | None",
                   defer_claim: bool) -> bool:
    """No --safety/--buchi, and an input whose format can state its own
    property: the checks that need the claim wait until it is read."""
    return bool(
        defer_claim and adapter is not None and adapter.states_claims
        and not settings.prp
    )
