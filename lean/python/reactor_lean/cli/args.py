"""The command line: the flags (`build_parser`), the rules about which go
together (`check_blanks`, `check_claim_flags`), and `Settings`, what
`parse_args` reads them into -- validated, with the property and its kind
decided, rather than an `argparse.Namespace` to re-derive facts from.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """A validated command line.

    The decisions the grammar makes are made here, once: which proof rule
    (`kind`), say.  The pipeline reads; it does not re-derive.
    """

    module_file: str
    module_def: str                     # the callable in it that makes the module
    output_dir: Path
    project_name: str
    executable: bool

    prp: str | None                     # the property, whichever flag carried it
    kind: str                           # "safety" | "buchi"

    pre: str | None
    invariant: str | None
    ranking: str | None

    cert_file: str | None = None

    @property
    def is_safety(self) -> bool:
        return self.kind == "safety"


_EPILOG = """\
examples:
  # bare project — no property, so `P` and `ranking` are sorry
  uv run reactor mymodule.py -o out/ -p MyProject

  # a Buchi property, `G (F (= (X @intf.0) 0))` (certificate stub with known P)
  uv run reactor mymodule.py --buchi "(= (X @intf.0) 0)" -o out/ -p MyProject

  # a safety property, `G (<= (X @intf.0) 100)`: invariant only, no ranking function
  uv run reactor mymodule.py --safety "(<= (X @intf.0) 100)" -o out/ -p MyProject

  # the certificate's predicates: its obligations are stated, and proved by `sorry`
  uv run reactor mymodule.py --buchi "(= (X @intf.0) 0)" --invariant "(<= (X @intf.0) 100)" --ranking "(X @intf.0)" \\
      -o out/ -p MyProject
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reactor",
        description=(
            "Generate a Lean4 certificate project from a reactive module, read "
            "from a Python file. "
            "The property goes under --safety (`G FORMULA`: every reachable "
            "state) or --buchi (`G (F FORMULA)`: infinitely often), and the "
            "certificate's predicates under --invariant and --ranking. The "
            "certificate states its obligations and proves each by `sorry`, "
            "for a prover to replace in Certificate/Proofs.lean."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_input(parser)
    add_claim(parser)
    add_outputs(parser)
    return parser


def add_input(parser: argparse.ArgumentParser) -> None:
    """The input: the module file and the callable in it that makes the
    module, and where the project is written."""
    parser.add_argument(
        "module_file",
        help=(
            "The input: a Python file containing a callable that returns a "
            "Module."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory where the Lean project will be created (default: current directory).",
    )
    parser.add_argument(
        "-p",
        "--project-name",
        default="Rea",
        help="Name of the Lean package / library (default: Rea).",
    )
    parser.add_argument(
        "-d",
        "--module-def",
        default="module",
        metavar="NAME",
        help=(
            "Name of the function or class in the Python file that "
            "produces the Module (default: module)."
        ),
    )
    parser.add_argument(
        "-x",
        "--executable",
        action="store_true",
        help="Generate Main.lean and add [[lean_exe]] to lakefile for a runnable binary.",
    )


def add_claim(parser: argparse.ArgumentParser) -> None:
    """The claim, as SMT-LIB in the module's names: the property under
    the proof rule it is proved by, and the predicates the certificate
    is given."""
    parser.add_argument(
        "--safety",
        default=None,
        metavar="FORMULA",
        help=(
            "Safety property `G FORMULA`: an SMT-LIB 2 Bool expression that "
            "must hold in *every* reachable state, over the state after each "
            "round: `(X x)` for a variable the input names `x`, `(X @intf.N)` "
            "for interface variable N. Certified with `rule_globally`, which "
            "needs an invariant implying FORMULA and no ranking function at "
            "all. Example: '(not (= (X @intf.0) 15))'."
        ),
    )
    parser.add_argument(
        "--buchi",
        default=None,
        metavar="FORMULA",
        help=(
            "Buchi property `G (F FORMULA)`: an SMT-LIB 2 Bool expression "
            "over the state after each round (`(X x)`, `(X @intf.N)`) that "
            "must hold infinitely often. "
            "Certified with `rule_buchi`, which needs an invariant *and* a "
            "ranking function that decreases in every state where FORMULA "
            "is false. Example: '(= (X @intf.0) 0)'."
        ),
    )
    parser.add_argument(
        "--pre",
        default=None,
        help=(
            "Precondition as an SMT-LIB 2 Bool expression over the inputs "
            "of a round: `(X u)` / `(X @extl.N)` for the input read in it, "
            "`u` / `@extl.N` for the one read in the round before. Tuple "
            "(matrix) elements use '((_ tuple.select k) (X u))'. "
            "Applied to both init_pre and update_pre. "
            "Example: '(>= ((_ tuple.select 0) (X @extl.0)) 1.0)'."
        ),
    )
    parser.add_argument(
        "--invariant",
        default=None,
        help=(
            "Invariant as an SMT-LIB 2 Bool expression over the state, "
            "named as --safety names it. Example: "
            "'(= (X @intf.0) (X @intf.5))'."
        ),
    )
    parser.add_argument(
        "--ranking",
        default=None,
        help=(
            "Ranking as an SMT-LIB 2 Int expression over the state, named "
            "as --buchi names it, for --buchi only. Must satisfy `ranking "
            ">= 0` under the invariant. Example: "
            "'(ite (X @intf.5) (ite (X @intf.4) 1 2) 0)'."
        ),
    )


def add_outputs(parser: argparse.ArgumentParser) -> None:
    """What the run writes besides the project."""
    parser.add_argument(
        "--cert-file",
        default=None,
        help=(
            "Write a standalone, self-contained certificate .lean file to this path "
            "instead of creating a full project.  The file inlines init/update, "
            "the definitions and the obligations' proofs.  The other encodings of the "
            "same module are written beside it under the same stem (Rel, Scalar, "
            "ScalarRel), as they are in a project."
        ),
    )


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


def parse_args(argv: "list[str] | None" = None) -> Settings:
    """The command line as `Settings`."""
    parser = build_parser()
    args = parser.parse_args(argv)

    check_blanks(parser, args)
    check_claim_flags(parser, args)
    return Settings(
        module_file=args.module_file,
        module_def=args.module_def,
        output_dir=Path(args.output_dir),
        project_name=args.project_name,
        executable=args.executable,
        prp=args.safety or args.buchi,
        kind="safety" if args.safety else "buchi",
        pre=args.pre,
        invariant=args.invariant,
        ranking=args.ranking,
        cert_file=args.cert_file,
    )
