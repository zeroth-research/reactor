"""The project's lakefile: the dependencies every project requires
(Mathlib, Cslib, lean-smt, and reactor-lean's `Relational` and `LeanAI`)
at the revisions it pins, and what an inference route adds to them
(`LakefileParts`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import Refused
from .templates import render


MATHLIB_URL = "https://github.com/leanprover-community/mathlib4"
CSLIB_URL = "https://github.com/leanprover-community/cslib"
PROOFS_URL = "https://github.com/zeroth/proof-prototyping"

MATHLIB_REV = "v4.34.0"
CSLIB_REV = MATHLIB_REV
# Lean toolchain version — should match Mathlib's requirement
LEAN_TOOLCHAIN = f"leanprover/lean4:{MATHLIB_REV}"

# The Lean libraries the generated files import -- `Relational.*`, which the
# encodings and the certificate are stated in, and `LeanAI`, the `ai_solve`
# tactic -- are Lake packages of their own, each in its own directory of the
# reactor-lean repository, which is also what each package is named.
REACTOR_LEAN_URL = "https://github.com/zeroth-research/reactor-lean"
# A commit rather than a branch, so a project cannot drift from the
# generator that wrote it.
REACTOR_LEAN_REV = "ef04b5b93ee7fd06c1fdd1fdc44ed99234bc73b6"
REACTOR_LEAN_PACKAGES = ("Relational", "LeanAI")
# A reactor-lean checkout to require by path instead of from git, to work on
# the libraries and the generator together.
REACTOR_LEAN_ENV = "REACTOR_LEAN"


@dataclass(frozen=True)
class LeanLib:
    """A `[[lean_lib]]` a route adds: `name` is the module name its files are
    imported under, `src_dir` the project-relative directory they are in."""

    name: str
    src_dir: str
    comment: str = ""


@dataclass(frozen=True)
class Require:
    """A `[[require]]`: from a local checkout at `path`, or else from the git
    repository `git` at `rev`, the package being its `sub_dir` if given."""

    name: str
    path: str | None = None
    comment: str = ""
    git: str | None = None
    rev: str | None = None
    sub_dir: str | None = None


@dataclass(frozen=True)
class LakefileParts:
    """What an inference route adds to the lakefile: libraries after
    `System`, requires after the fixed ones, each preceded by its `comment`."""

    libs: tuple[LeanLib, ...] = ()
    requires: tuple[Require, ...] = ()


def reactor_lean_checkout() -> Path | None:
    """`$REACTOR_LEAN` as a reactor-lean checkout, or None when it is unset.

    Refused when it is set to something that is not a checkout: the
    lakefile would require packages that are not there, and lake would say
    so only at build time, long after the project was written.
    """
    value = os.environ.get(REACTOR_LEAN_ENV)
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    missing = [f"{p}/lakefile.toml" for p in REACTOR_LEAN_PACKAGES
               if not (root / p / "lakefile.toml").is_file()]
    if missing:
        raise Refused(
            f"${REACTOR_LEAN_ENV}: {root} is not a reactor-lean checkout: "
            f"it has no {', '.join(missing)}",
            setup=True,
        )
    return root


_REACTOR_LEAN_COMMENTS = {
    "Relational": (
        "reactor-lean's `Relational.*`: what the encodings and the certificate "
        "are stated in.\n"
        f"Set ${REACTOR_LEAN_ENV} to a reactor-lean checkout to require its "
        "packages by path."
    ),
    "LeanAI": "reactor-lean's `LeanAI`: the `ai_solve` tactic.",
}


def reactor_lean_requires() -> tuple[Require, ...]:
    """The requires on reactor-lean's packages: by path from
    `$REACTOR_LEAN` when it is set, from git at `REACTOR_LEAN_REV`
    otherwise."""
    checkout = reactor_lean_checkout()
    if checkout is not None:
        return tuple(Require(p, path=str(checkout / p),
                             comment=_REACTOR_LEAN_COMMENTS[p])
                     for p in REACTOR_LEAN_PACKAGES)
    return tuple(
        Require(p, git=REACTOR_LEAN_URL, rev=REACTOR_LEAN_REV, sub_dir=p,
                comment=_REACTOR_LEAN_COMMENTS[p])
        for p in REACTOR_LEAN_PACKAGES
    )


def with_reactor_lean(manifest: dict, checkout: Path) -> dict:
    """`manifest` (a parsed `lake-manifest.json`) with reactor-lean's
    packages entered as path dependencies on `checkout`.

    `lake build` materializes a path dependency from the manifest's `dir`,
    not from the lakefile's `path`, so a project that is built against a
    manifest it did not resolve itself -- the test runners copy
    `tests/lean`'s, whose `dir`s are relative to `tests/lean` -- needs
    these entries rewritten for it.
    """
    entries = [
        {"type": "path", "scope": "", "name": p,
         "manifestFile": "lake-manifest.json", "inherited": False,
         "dir": str(checkout / p), "configFile": "lakefile.toml"}
        for p in REACTOR_LEAN_PACKAGES
    ]
    rest = [e for e in manifest["packages"]
            if e["name"] not in REACTOR_LEAN_PACKAGES]
    return {**manifest, "packages": rest + entries}


def generate_lakefile(
    project_name: str,
    executable: bool = False,
    parts: LakefileParts = LakefileParts(),
) -> str:
    """Render the project's lakefile, with `parts` added to the fixed ones.

    A route cannot amend the lakefile after the fact -- it is written while
    the project is created, before any route runs -- so what the route needs
    in it is given here, from its row's `lakefile` hook.
    """
    return render(
        "project/lakefile.toml.j2",
        project_name=project_name,
        executable=executable,
        cslib_rev=CSLIB_REV,
        parts=parts,
        requires=reactor_lean_requires() + parts.requires,
    )
