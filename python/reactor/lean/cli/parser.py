"""The flags: `build_parser`, with the help each input format contributes
assembled from its rows.

Each group is its own `add_*` so that a command built on this one
assembles the same flags rather than copies of them.
"""

from __future__ import annotations

import argparse

from ..artifacts import ARTIFACTS_DIR
from ..inputs import ADAPTERS


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


def _format_help() -> str:
    """`--format`'s help, assembled from the rows."""
    rows = " ".join(
        f"`{a.name}` ({', '.join(a.suffixes)}) -- {a.summary}." for a in ADAPTERS
    )
    return (
        "The input's format; by default the one its suffix names. One of: "
        + rows
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reactor",
        description=(
            "Generate a Lean4 certificate project from a reactive module, read "
            "from a Python file or another format --format names. "
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


def add_input(parser: argparse.ArgumentParser, *, project: bool = True,
              optional: str = "") -> None:
    """The input: the module file, the format it is read in and that
    format's own options, and, with `project`, where the project is
    written. The module file is required, unless `optional` says when a
    command runs without one."""
    parser.add_argument(
        "module_file",
        nargs="?" if optional else None,
        help=(
            "The input: a file stating the module, in the format --format "
            "names (by default the one its suffix names; a Python file must "
            "contain a callable that returns a Module)."
            + (f" {optional}" if optional else "")
        ),
    )
    if project:
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
        "--format",
        choices=[a.name for a in ADAPTERS],
        default=None,
        help=_format_help(),
    )
    # A format's own options, one group per format, parsed `default=None`
    # so "given" is distinguishable from "not given".
    for adapter in ADAPTERS:
        if adapter.options:
            group = parser.add_argument_group(f"--format {adapter.name}")
            for opt in adapter.options:
                group.add_argument(*opt.flags, **{**opt.kwargs, "default": None})
    if project:
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
    """What the run writes besides the project, and what it does with
    `artifacts/`."""
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
    parser.add_argument(
        "--artifacts",
        default="use",
        choices=["use", "ignore", "reset"],
        help=(
            f"What to do with the project's `{ARTIFACTS_DIR}/` directory, "
            f"which is how a sequence of runs in one -o continues rather "
            f"than restarts: what a previous run found or asked is left "
            f"there. `use` (default) lets the run read it -- an artifact "
            f"written for a different module, property or proof rule is "
            f"skipped, and what is picked up is printed, because two "
            f"identical command lines must not quietly mean different "
            f"things. `ignore` reads nothing back and still records what "
            f"this run finds. `reset` empties it first, which is what to "
            f"pass when a bad artifact is being inherited by every run."
        ),
    )
