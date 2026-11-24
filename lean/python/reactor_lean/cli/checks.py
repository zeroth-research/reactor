"""The rules about which flags go together."""

from __future__ import annotations

import argparse


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

