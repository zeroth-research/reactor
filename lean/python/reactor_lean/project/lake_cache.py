"""A Lake package cache: the packages every project requires, fetched and
built once and shared by the projects that point at it.

A cache is a directory holding `packages/` -- Cslib, Mathlib and what
they require, built -- and `lake-manifest.json`, the manifest they
were resolved by. `$REACTOR_LAKE_CACHE` names one. A project whose lakefile
is written while it is set (`scaffold.write_lakefile`) gets `.lake/packages`
as a link to the cache's and a manifest made from the cache's, so `lake
build` neither fetches nor compiles what the cache holds. Without a cache, a
project's first `lake update` fetches its packages into its own `.lake`,
and its first build compiles them: about an hour for Mathlib.

A cache is made from a project built without one: its `.lake/packages`
moved to `<cache>/packages`, and its `lake-manifest.json` copied beside it.
Only the cache's git packages are shared. A path require -- ReactLib's
packages from `$REACTLIB`, or one a caller adds -- is the project's own,
and its manifest entry is made from the require.

A build that needs a module of a cached package that is not compiled yet
compiles it into the cache, where every project then finds it. A project
that already has packages of its own, from a run without the cache, keeps
them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..errors import Refused
from .lakefile import Require, project_requires

LAKE_CACHE_ENV = "REACTOR_LAKE_CACHE"
MANIFEST = "lake-manifest.json"


@dataclass(frozen=True)
class Packages:
    """What a project's lakefile requires, and the cache it builds against
    (None: it fetches into its own `.lake`)."""

    requires: tuple[Require, ...]
    cache: Path | None


def project_packages() -> Packages:
    """The packages of a project written now: `project_requires`, and
    `$REACTOR_LAKE_CACHE` checked to hold them (`lake_cache`). `cli.main`
    works them out before any of the project is written, and the project
    is written with them."""
    requires = project_requires()
    return Packages(requires, lake_cache(requires))


def lake_cache(requires: Iterable[Require] = ()) -> Path | None:
    """`$REACTOR_LAKE_CACHE` as a Lake package cache, or None when it is
    unset.

    Refused when it is set to something that is not a cache, or to one
    that does not hold a git require of `requires` at the revision the
    lakefile asks for (`held_packages`).
    """
    value = os.environ.get(LAKE_CACHE_ENV)
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    missing = [n for n in ("packages", MANIFEST) if not (root / n).exists()]
    if missing:
        raise Refused(
            f"${LAKE_CACHE_ENV}: {root} is not a Lake package cache: "
            f"it has no {' or '.join(missing)}",
        )
    held_packages(root, requires)
    return root


def held_packages(cache: Path, requires: Iterable[Require]) -> "tuple[dict, dict[str, dict]]":
    """The cache's manifest, and its entry for each git package it holds,
    by name.

    Refused for a git require of `requires` that the cache does not hold
    at the revision the lakefile asks for: lake would fetch it into the
    cache, or fetch another revision over the one every other project
    builds against. Refused too when the manifest cannot be read as one.
    """
    try:
        cached = json.loads((cache / MANIFEST).read_text())
        held = {e["name"]: e for e in cached["packages"] if e["type"] == "git"}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Refused(
            f"${LAKE_CACHE_ENV}: {cache / MANIFEST} is not a Lake manifest "
            f"({type(e).__name__}: {e})",
        ) from e
    for r in requires:
        if r.path is not None:
            continue
        entry = held.get(r.name)
        if entry is None:
            raise Refused(
                f"${LAKE_CACHE_ENV}: {cache} does not hold {r.name}, which "
                f"the project requires from git; build a project without "
                f"the cache to fetch it (see `project.lake_cache`)",
            )
        if r.rev not in (entry.get("inputRev"), entry.get("rev")):
            raise Refused(
                f"${LAKE_CACHE_ENV}: {cache} holds {r.name} at "
                f"{entry.get('inputRev') or entry.get('rev')}, and the project "
                f"requires it at {r.rev}",
            )
    return cached, held


def cached_manifest(cache: Path, requires: Iterable[Require], *,
                    name: str, project: Path) -> dict:
    """The manifest of the project `name` at `project`, which requires
    `requires` and builds against `cache`: the cache's entry for every git
    package it holds, and an entry made from each path require. Refused as
    `held_packages` refuses."""
    requires = tuple(requires)
    cached, held = held_packages(cache, requires)
    paths = [_path_entry(r, project) for r in requires if r.path is not None]
    own = {e["name"] for e in paths}
    return {
        **cached,
        "name": name,
        "packagesDir": ".lake/packages",
        "packages": [e for e in held.values() if e["name"] not in own] + paths,
    }


def _path_entry(r: Require, project: Path) -> dict:
    """The manifest entry of the path require `r`. Lake builds a path
    dependency from the manifest's `dir`, so it is the lakefile's `path`."""
    root = project / r.path
    config = ("lakefile.lean" if (root / "lakefile.lean").is_file()
              and not (root / "lakefile.toml").is_file() else "lakefile.toml")
    return {"type": "path", "scope": "", "name": r.name,
            "manifestFile": MANIFEST, "inherited": False,
            "dir": r.path, "configFile": config}


def use_lake_cache(project: Path, requires: Iterable[Require], *,
                   name: str, cache: Path) -> bool:
    """Build the project `name` at `project` against `cache`: its
    `.lake/packages` a link to the cache's, and its manifest
    `cached_manifest`'s. Whether it does: a project with packages of its
    own -- one an earlier run built without the cache -- keeps them, and
    its manifest, and builds as it did.
    """
    link = project / ".lake" / "packages"
    if link.exists() and not link.is_symlink():
        return False
    manifest = cached_manifest(cache, requires, name=name, project=project)
    if link.is_symlink():
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(cache / "packages", target_is_directory=True)
    (project / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return True
