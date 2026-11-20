"""The project's lakefile: the dependencies every project requires
(Mathlib, Cslib, and the ReactLib repository's `ReactLib` and
`LeanAI`) at the revisions it pins, and what a caller adds to them
(`LakefileParts`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import Refused
from .templates import render


MATHLIB_REV = "v4.34.0"
CSLIB_REV = MATHLIB_REV
# Lean toolchain version — should match Mathlib's requirement
LEAN_TOOLCHAIN = f"leanprover/lean4:{MATHLIB_REV}"

# The Lean libraries the generated files import come in two Lake packages,
# each in its own directory of the ReactLib repository, which is also
# what each package is named: `ReactLib`, whose libraries are
# `Relational.*`, which the encodings and the certificate are stated in,
# and `Spec.*`, the LTL their proof rules are stated in; and `LeanAI`, the
# `ai_solve` tactic.
REACTLIB_URL = "https://github.com/zeroth-research/ReactLib"
# A commit rather than a branch, so a project cannot drift from the
# generator that wrote it.
REACTLIB_REV = "97144c6538db206fdefa10825d850219901c673b"
REACTLIB_PACKAGES = ("ReactLib", "LeanAI")
# A ReactLib checkout to require by path instead of from git, to work on
# the libraries and the generator together.
REACTLIB_ENV = "REACTLIB"


@dataclass(frozen=True)
class LeanLib:
    """A `[[lean_lib]]` a caller adds: `name` is the module name its files are
    imported under, `src_dir` the project-relative directory they are in,
    the project's own when empty."""

    name: str
    src_dir: str = ""
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
    """What a caller adds to the lakefile: libraries after
    `System`, requires after the fixed ones, each preceded by its `comment`."""

    libs: tuple[LeanLib, ...] = ()
    requires: tuple[Require, ...] = ()

    def __add__(self, other: "LakefileParts") -> "LakefileParts":
        """Both callers' additions, `self`'s first."""
        return LakefileParts(self.libs + other.libs,
                             self.requires + other.requires)


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
    missing = [f"{p}/lakefile.toml" for p in REACTLIB_PACKAGES
               if not (root / p / "lakefile.toml").is_file()]
    if missing:
        raise Refused(
            f"${REACTLIB_ENV}: {root} is not a ReactLib checkout: "
            f"it has no {', '.join(missing)}",
        )
    return root


_REACTLIB_COMMENTS = {
    "ReactLib": (
        "ReactLib's `ReactLib` package: `Relational.*`, what the encodings and "
        "the certificate are stated in, and `Spec.*`, the LTL of its proof "
        "rules.\n"
        f"Set ${REACTLIB_ENV} to a ReactLib checkout to require its "
        "packages by path."
    ),
    "LeanAI": "ReactLib's `LeanAI`: the `ai_solve` tactic.",
}


def reactlib_requires() -> tuple[Require, ...]:
    """The requires on ReactLib's packages: by path from
    `$REACTLIB` when it is set, from git at `REACTLIB_REV`
    otherwise."""
    checkout = reactlib_checkout()
    if checkout is not None:
        return tuple(Require(p, path=str(checkout / p),
                             comment=_REACTLIB_COMMENTS[p])
                     for p in REACTLIB_PACKAGES)
    return tuple(
        Require(p, git=REACTLIB_URL, rev=REACTLIB_REV, sub_dir=p,
                comment=_REACTLIB_COMMENTS[p])
        for p in REACTLIB_PACKAGES
    )


def with_reactlib(manifest: dict, checkout: Path) -> dict:
    """`manifest` (a parsed `lake-manifest.json`) with ReactLib's
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
        for p in REACTLIB_PACKAGES
    ]
    rest = [e for e in manifest["packages"]
            if e["name"] not in REACTLIB_PACKAGES]
    return {**manifest, "packages": rest + entries}


def generate_lakefile(
    project_name: str,
    executable: bool = False,
    parts: LakefileParts = LakefileParts(),
) -> str:
    """Render the project's lakefile, with `parts` added to the fixed ones.

    The lakefile is written while the project is created and not amended
    after the fact, so what a caller needs in it is given here.
    """
    return render(
        "project/lakefile.toml.j2",
        project_name=project_name,
        executable=executable,
        cslib_rev=CSLIB_REV,
        parts=parts,
        requires=reactlib_requires() + parts.requires,
    )
