"""The project's lakefile: the dependencies every project requires
(Mathlib, Cslib, and the ReactLib repository's `ReactLib`) at the
revisions it pins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..errors import Refused
from .templates import render


MATHLIB_REV = "v4.34.0"
CSLIB_REV = MATHLIB_REV
# Lean toolchain version — should match Mathlib's requirement
LEAN_TOOLCHAIN = f"leanprover/lean4:{MATHLIB_REV}"

# The Lean libraries the generated files import are the Lake package
# `ReactLib`, in the directory of the ReactLib repository of that name:
# `Relational.*`, which the encodings and the certificate are stated in,
# and `Spec.*`, the LTL their proof rules are stated in.
REACTLIB_URL = "https://github.com/zeroth-research/ReactLib"
# A commit rather than a branch, so a project cannot drift from the
# generator that wrote it.
REACTLIB_REV = "01cb44e77112a6b5a0e6e793723885d42c303a50"
REACTLIB_PACKAGE = "ReactLib"
# A ReactLib checkout to require by path instead of from git, to work on
# the libraries and the generator together.
REACTLIB_ENV = "REACTLIB"


@dataclass(frozen=True)
class Require:
    """A `[[require]]`: from a local checkout at `path`, or else at `rev`
    from the git repository `git`, or from Reservoir under `scope`, the
    package being its `sub_dir` if given."""

    name: str
    path: str | None = None
    comment: str = ""
    git: str | None = None
    rev: str | None = None
    sub_dir: str | None = None
    scope: str | None = None


def reactlib_checkout() -> Path | None:
    """`$REACTLIB` as a ReactLib checkout, or None when it is unset.

    Refused when it is set to something that is not a checkout: the
    lakefile would require packages that are not there, and lake would say
    so only at build time, long after the project was written.
    """
    value = os.environ.get(REACTLIB_ENV)
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    if not (root / REACTLIB_PACKAGE / "lakefile.toml").is_file():
        raise Refused(
            f"${REACTLIB_ENV}: {root} is not a ReactLib checkout: "
            f"it has no {REACTLIB_PACKAGE}/lakefile.toml",
        )
    return root


_REACTLIB_COMMENT = (
    "ReactLib's `ReactLib` package: `Relational.*`, what the encodings and "
    "the certificate are stated in, and `Spec.*`, the LTL of its proof "
    "rules.\n"
    f"Set ${REACTLIB_ENV} to a ReactLib checkout to require it by path."
)


# What every project requires besides ReactLib: Cslib, which
# brings Mathlib at the revision it pins.
FIXED_REQUIRES = (
    Require("cslib", scope="leanprover", rev=CSLIB_REV),
)


def reactlib_require() -> Require:
    """The require on ReactLib: by path from `$REACTLIB` when it is set,
    from git at `REACTLIB_REV` otherwise."""
    checkout = reactlib_checkout()
    if checkout is not None:
        return Require(REACTLIB_PACKAGE, path=str(checkout / REACTLIB_PACKAGE),
                       comment=_REACTLIB_COMMENT)
    return Require(REACTLIB_PACKAGE, git=REACTLIB_URL, rev=REACTLIB_REV,
                   sub_dir=REACTLIB_PACKAGE, comment=_REACTLIB_COMMENT)


def project_requires() -> tuple[Require, ...]:
    """Everything the project's lakefile requires, in its order: the fixed
    packages, then ReactLib's."""
    return (*FIXED_REQUIRES, reactlib_require())


def generate_lakefile(project_name: str, requires: tuple[Require, ...],
                      executable: bool = False) -> str:
    """Render the project's lakefile, which requires `requires`."""
    return render(
        "project/lakefile.toml.j2",
        project_name=project_name,
        executable=executable,
        requires=requires,
    )
