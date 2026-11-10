"""The rules about which flags go together, each read off the rows it is
about: an input format's options, the claim, the SMT budget and the
outputs a run can write.

Public so that a command built on this one, which takes these flags too,
applies the same rules.
"""

from __future__ import annotations

import argparse

from ..core.options import given, missing, put_defaults
from ..inputs import ADAPTERS, InputAdapter, adapter_by_name, adapter_for
from .settings import Settings


def check_blanks(parser: argparse.ArgumentParser, args) -> None:
    """An option given an empty string is the error, not "not passed".

    `--invariant "$INV"` with `INV` unset -- which is how a copied
    benchmark command reads -- reads as "not passed" to every
    truthiness test below.  The invariant would be dropped and the run
    would *succeed*, having quietly generated a project with no
    certificate instead.  No option here has a meaningful empty value.
    """
    blank = [
        action.option_strings[-1]
        for action in parser._actions
        if action.option_strings and getattr(args, action.dest, None) == ""
    ]
    if blank:
        verb = "was" if len(blank) == 1 else "were"
        parser.error(
            f"{', '.join(blank)} {verb} given an empty value (an unset "
            "shell variable?). Pass a real one or drop the flag."
        )


def input_adapter(parser: argparse.ArgumentParser,
                  args) -> InputAdapter | None:
    """The format the input is read in: `--format`, else its suffix's."""
    if args.format:
        return adapter_by_name(args.format)
    if not args.module_file:
        return None
    adapter = adapter_for(args.module_file)
    if adapter is None:
        parser.error(
            f"cannot tell the format of {args.module_file} from its suffix; "
            f"pass --format ({', '.join(a.name for a in ADAPTERS)})"
        )
    return adapter


def check_row_options(parser: argparse.ArgumentParser, args, stray,
                       meaningful, row: str, options) -> None:
    """The rules every row's options follow (`core.options`): an option of
    a row that was not selected is refused, `stray` naming each with the
    rows it belongs to and `meaningful` phrasing those rows; and for the
    selected row, `row`'s `options`, a required one has to be given and
    the others get their row default."""
    if stray:
        flags = [o.flags[-1] for o, _ in stray]
        rows = sorted({r for _, rs in stray for r in rs})
        verb = "is" if len(flags) == 1 else "are"
        parser.error(
            f"{', '.join(flags)} {verb} only meaningful {meaningful(rows)}")
    if options is None:
        return
    absent = [o.flags[-1] for o in missing(options, args)]
    if absent:
        parser.error(f"{row} requires {', '.join(absent)}")
    put_defaults(options, args)


def check_input_options(parser: argparse.ArgumentParser, args,
                        adapter: InputAdapter | None) -> None:
    """A format's option is only meaningful for input in that format; one
    its row requires has to be given; and its row default is put back once
    the format is known."""
    check_row_options(
        parser, args,
        [(o, [a.name]) for a in ADAPTERS for o in a.options
         if a is not adapter and given(o, args)],
        lambda rows: f"for {' or '.join(rows)} input",
        f"{adapter.name} input" if adapter else "",
        adapter.options if adapter else None,
    )


def check_claim_flags(parser: argparse.ArgumentParser, args) -> None:
    """The property is one flag, and a ranking function is for --buchi."""
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


def check_outputs(parser: argparse.ArgumentParser, settings: Settings, *,
                  finds_certificate: bool = False) -> None:
    """What the run writes, against what it was given. `finds_certificate`
    is whether a command built on this one finds the certificate itself:
    then --build-cert has one to build without --invariant."""
    # --build-cert needs a certificate worth building. A bare project's
    # unsupplied fields default to `True`
    # (`inv`, the preconditions) or to `sorry` (`P`, `ranking`), and its
    # obligations are still handed to the tactics -- which cannot close one
    # stated over a `sorry`, so `lake build` fails on it rather than
    # compiling to a proof of nothing. Asking for that is a mistake.
    if settings.build_cert:
        if settings.cert_file or settings.hammer_file:
            which = "--cert-file" if settings.cert_file else "--hammer-file"
            parser.error(
                f"--build-cert is incompatible with {which}: that writes a "
                "file and generates no project to build"
            )
        if not (finds_certificate or settings.supplied_complete):
            parser.error(
                "--build-cert needs a certificate to build: pass --safety "
                "with --invariant, or --buchi with --invariant and "
                "--ranking. Without them the certificate's predicates are "
                "`True` or `sorry`, and its obligations cannot be discharged."
            )

    # `--hammer-file` alone writes one tactic file and needs no module.
    if not settings.module_file and not settings.hammer_file:
        parser.error("module_file is required (unless --hammer-file is used alone)")
