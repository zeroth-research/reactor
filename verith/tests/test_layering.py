"""Which module may import what, read off the source.

`verith` is built on `reactor.lean`, and imports only the layers of it the
routes were built on when they were one of its packages: the SMT layer,
the Lean code generators, the project and the proof. The commands over the
routes (`verith.cli`, `verith.flags`, `verith.run`) also read the input
formats and `reactor`'s own command line, whose flags they share. `certify`
is the one module that runs `reactor` and a route together; nothing in
`verith` imports it. (`reactor.lean` imports none of the three:
`python/tests/test_layering.py`.) An import under `if TYPE_CHECKING:` is
exempt: it names a type for the reader and loads nothing.

A route is a package of `verith`: its row in `__init__.py` and its engine
beside it. No route imports another; what two share lives in
`verith.common` or under `reactor.lean.smt`. A route's row imports its
engine inside the function that runs it, so `verith --help` loads no engine.

And a name with a leading underscore is its module's own: no module here
imports one from another module of `reactor`, `verith` or `certify`, nor
reads one off an imported module.
"""

from __future__ import annotations

import ast
import functools
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]          # the `verith/` project
VERITH = ROOT / "verith"
CERTIFY = ROOT / "certify"

PACKAGE = sorted([*VERITH.rglob("*.py"), *CERTIFY.rglob("*.py")])

# The layers of `reactor.lean` the routes are built on, and what the
# commands over them read besides.
ROUTE_LAYERS = frozenset({
    "core", "lowering", "artifacts", "codegen", "smt", "project", "proof"})
COMMAND_LAYERS = ROUTE_LAYERS | {"inputs", "cli", "pipeline"}
COMMANDS = {"verith.cli", "verith.flags", "verith.run"}


@functools.cache
def _tree(path: Path) -> ast.Module:
    """`path` parsed, once: every test below reads the same files."""
    return ast.parse(path.read_text(), filename=str(path))


def _module(path: Path) -> str:
    """`path` as a dotted module name: `verith.houdini.engine`."""
    parts = path.relative_to(ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


MODULES = {_module(p) for p in PACKAGE}


def _package(path: Path) -> list[str]:
    """`path`'s package as a dotted list: `["verith", "houdini"]`."""
    return list(path.parent.relative_to(ROOT).parts)


class Import(NamedTuple):
    module: str          # absolute
    name: str | None     # what a `from` import takes; `None` for `import`
    bound: str           # the name the import binds
    line: int
    top: bool            # runs when the module is imported
    typing: bool         # under `if TYPE_CHECKING:`

    @property
    def target(self) -> str:
        """The module this import loads: `from verith import io` loads
        `verith.io`, and so does `from verith.io import ProjectHandle`."""
        if self.name and f"{self.module}.{self.name}" in MODULES:
            return f"{self.module}.{self.name}"
        return self.module


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            or isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _import_nodes(body, *, top: bool, typing: bool):
    """`(node, top, typing)` for every import statement under `body`."""
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, top, typing
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            yield from _import_nodes(node.body, top=top, typing=True)
            yield from _import_nodes(node.orelse, top=top, typing=typing)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _import_nodes(node.body, top=False, typing=typing)
        else:
            for field in ("body", "orelse", "finalbody", "handlers"):
                yield from _import_nodes(getattr(node, field, ()), top=top,
                                         typing=typing)


def _imports(path: Path):
    """Every import in `path`, the module absolute."""
    pkg = _package(path)
    for node, top, typing in _import_nodes(_tree(path).body, top=True,
                                           typing=False):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield Import(a.name, None, a.asname or a.name.split(".")[0],
                             node.lineno, top, typing)
        else:
            if node.level:
                base = pkg[:len(pkg) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            for a in node.names:
                yield Import(module, a.name, a.asname or a.name, node.lineno,
                             top, typing)


def _name(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _reactor_layer(module: str) -> str | None:
    """`smt` for `reactor.lean.smt.encode`; `None` outside `reactor.lean`."""
    parts = module.split(".")
    if parts[:2] == ["reactor", "lean"] and len(parts) > 2:
        return parts[2]
    return None


# Every route: a package of `verith` other than `common`.
ROUTE_DIRS = sorted(d for d in VERITH.iterdir()
                    if (d / "__init__.py").exists() and d.name != "common")
ROUTES = {d.name for d in ROUTE_DIRS}


def _route(module: str) -> str | None:
    """The route `module` belongs to: `houdini` for `verith.houdini.engine`."""
    parts = module.split(".")
    if parts[0] == "verith" and len(parts) > 1:
        return parts[1] if parts[1] in ROUTES else None
    return None


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


VERITH_MODULES = [p for p in PACKAGE if p.is_relative_to(VERITH)]


# ── what of `reactor.lean` ──────────────────────────────────────────────────


@pytest.mark.parametrize("path", VERITH_MODULES, ids=_name)
def test_verith_imports_only_the_layers_it_is_built_on(path):
    me = _module(path)
    allowed = COMMAND_LAYERS if me in COMMANDS else ROUTE_LAYERS
    bad = []
    for i in _imports(path):
        if i.typing:
            continue
        layer = _reactor_layer(i.target)
        if layer is not None and layer not in allowed:
            bad.append(f"{_name(path)}:{i.line} imports {i.target} "
                       f"(reactor.lean.{layer})")
        if i.module.split(".")[0] == "certify":
            bad.append(f"{_name(path)}:{i.line} imports {i.module}, which is "
                       f"built on it")
    assert not bad, "\n".join(bad)


def test_every_command_module_exists():
    """A command module renamed takes its entry with it; a stale one would
    check nothing."""
    assert COMMANDS <= MODULES, sorted(COMMANDS - MODULES)


# ── the routes ──────────────────────────────────────────────────────────────


def test_the_routes_are_found():
    assert ROUTES >= {
        "ai", "ai_cegis", "fbk_proveit", "houdini", "nuterm", "smt_linear",
        "sygus", "vampire"}


@pytest.mark.parametrize("route", ROUTE_DIRS, ids=lambda d: d.name)
def test_every_route_has_an_engine(route):
    """The row is `__init__.py` and what it runs is `engine.py`, so the rule
    below names one module."""
    assert (route / "engine.py").exists()


@pytest.mark.parametrize("path", [p for p in VERITH_MODULES
                                  if p != VERITH / "__init__.py"], ids=_name)
def test_no_route_imports_another(path):
    """A route imports its own modules and what no route owns; a module
    that belongs to no route imports none. The registry, which lists the
    rows, is the one module that knows them all."""
    me = _route(_module(path))
    bad = [f"{_name(path)}:{i.line} imports {i.target}"
           for i in _imports(path)
           if _route(i.target) not in (None, me)]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("route", ROUTE_DIRS, ids=lambda d: d.name)
def test_a_route_row_imports_its_engine_only_when_it_runs(route):
    """The commands import every row to print `--help`; an engine imported
    with its row would be loaded by every run, whichever route it asks for."""
    path = route / "__init__.py"
    engine = f"verith.{route.name}.engine"
    bad = [f"{_name(path)}:{i.line} imports {i.target} at module level"
           for i in _imports(path)
           if i.top and not i.typing and i.target == engine]
    assert not bad, "\n".join(bad)


def test_the_registry_loads_no_engine():
    """What the rule above is for, observed: importing every row and both
    command lines, as `verith --help` and `certify --help` do, imports no
    engine and no module an engine imports at its top."""
    probe = ("import torch, sys, verith.cli, certify.cli; print(sorted(m for m "
             "in sys.modules if m.endswith('.engine') or m == 'cvc5'))")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", r.stdout


# ── private names ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", PACKAGE, ids=_name)
def test_no_private_name_crosses_a_module(path):
    bad = []
    imported = set()
    for i in _imports(path):
        if i.module.split(".")[0] not in ("reactor", "verith", "certify"):
            continue
        imported.add(i.bound)
        if i.name and _private(i.name):
            bad.append(f"{_name(path)}:{i.line} imports {i.module}.{i.name}")
    # `mod._name`, where `mod` is a module or a name this one imported.
    for node in ast.walk(_tree(path)):
        if (isinstance(node, ast.Attribute) and _private(node.attr)
                and isinstance(node.value, ast.Name)
                and node.value.id in imported):
            bad.append(f"{_name(path)}:{node.lineno} reads "
                       f"{node.value.id}.{node.attr}")
    assert not bad, "\n".join(bad)
