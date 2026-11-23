"""Which module may import what, read off the source.

The package is layered. `LAYERS` names the modules of each layer and the
layers it may import besides itself, and a module imports nothing else of
`reactor_lean`. An import under `if TYPE_CHECKING:` is exempt: it names a type
for the reader and loads nothing.

The inference routes are not here: they are the `verith` package, in the
`verith/` project beside this one, which is built on this package and has
layering rules of its own (`verith/tests/test_layering.py`). Nothing in
`reactor_lean` imports it, nor the `certify` command or the `nuterm`
learner beside it.

And a name with a leading underscore is its module's own: no module of
`reactor_lean` imports one from another, nor reads one off an imported module.
A name another module needs is part of its module's interface, and is
spelled without the underscore.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path
from typing import NamedTuple

import pytest

from testdata import PKG_ROOT

LEAN = PKG_ROOT / "python" / "reactor_lean"

# Every module of the package: the SMT layer the routes are built from, and
# the Lean code generators.
PACKAGE = sorted(LEAN.rglob("*.py"))

# Each layer is the package `reactor_lean.<layer>` and everything under it, and
# maps to the layers it may import besides itself. A module no layer names
# is not checked, but a checked module may import it only through a layer.
LAYERS: dict[str, frozenset[str]] = {
    "core": frozenset(),
    "lowering": frozenset({"core"}),
    "artifacts": frozenset({"core"}),
    "codegen": frozenset({"core", "lowering"}),
    # Reading a predicate's text: the parsers, cvc5's aside.
    "parse": frozenset({"core", "lowering"}),
    "smt": frozenset({"core", "lowering", "artifacts", "parse"}),
    # The project on disk. It writes the certificate's files as the text it
    # is handed, so it sits below `proof`, which renders them.
    "project": frozenset({"core", "lowering", "codegen"}),
    "proof": frozenset({"core", "lowering", "codegen", "smt", "project", "parse"}),
    # The input formats; one encodes a module through `smt` as it reads it.
    "inputs": frozenset({"core", "smt"}),
    # The command line: read off the input formats' rows.
    "cli": frozenset({"core", "artifacts", "smt", "inputs", "parse"}),
    # `uv run reactor`: every stage, in order.
    "pipeline": frozenset({
        "core", "lowering", "artifacts", "codegen", "smt", "project", "proof",
        "inputs", "cli", "parse"}),
}


@functools.cache
def _tree(path: Path) -> ast.Module:
    """`path` parsed, once: every test below reads the same files."""
    return ast.parse(path.read_text(), filename=str(path))


def _module(path: Path) -> str:
    """`path` as a dotted module name: `reactor_lean.smt.encode`."""
    parts = path.relative_to(LEAN.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


MODULES = {_module(p) for p in PACKAGE}


def _package(path: Path) -> list[str]:
    """`path`'s package as a dotted list: `["reactor_lean", "smt"]`."""
    return list(path.parent.relative_to(LEAN.parent).parts)


class Import(NamedTuple):
    module: str          # absolute
    name: str | None     # what a `from` import takes; `None` for `import`
    bound: str           # the name the import binds
    line: int
    top: bool            # runs when the module is imported
    typing: bool         # under `if TYPE_CHECKING:`

    @property
    def target(self) -> str:
        """The module this import loads: `from reactor_lean.core import errors`
        loads `reactor_lean.core.errors`, and so does `from reactor_lean.core.errors
        import Refused`."""
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


def _layer(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] == "reactor_lean" and len(parts) > 1 and parts[1] in LAYERS:
        return parts[1]
    return None


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _name(path: Path) -> str:
    return str(path.relative_to(LEAN))


LAYERED = [p for p in PACKAGE if _layer(_module(p))]


@pytest.mark.parametrize("layer", sorted(LAYERS))
def test_every_module_a_layer_names_exists(layer):
    """A package moved or renamed takes its layer entry with it; a stale
    entry would check nothing."""
    assert f"reactor_lean.{layer}" in MODULES
    assert LAYERS[layer] <= set(LAYERS), sorted(LAYERS[layer] - set(LAYERS))


@pytest.mark.parametrize("path", LAYERED, ids=_name)
def test_a_layer_imports_only_the_layers_below_it(path):
    me = _layer(_module(path))
    allowed = {me} | LAYERS[me]
    bad = []
    for i in _imports(path):
        if i.typing or not i.target.startswith("reactor_lean."):
            continue
        layer = _layer(i.target)
        if layer not in allowed:
            bad.append(f"{_name(path)}:{i.line} ({me}) imports {i.target} "
                       f"({layer or 'in no layer'})")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("path", PACKAGE, ids=_name)
def test_no_private_name_crosses_a_module(path):
    bad = []
    imported = set()
    for i in _imports(path):
        if not i.module.startswith("reactor"):
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


# What is built on this package, and so never imported by it.
ABOVE = ("verith", "certify", "nuterm")


@pytest.mark.parametrize("path", PACKAGE, ids=_name)
def test_nothing_imports_what_is_built_on_it(path):
    """Not at module level and not inside a function: `reactor` runs, and
    its project is generated, without the `verith/` project installed.
    What an inference route needs of the project -- its lakefile entries,
    the files it owns -- it is handed as data (`pipeline.outputs.generate`).
    """
    bad = [f"{_name(path)}:{i.line} imports {i.module}"
           for i in _imports(path)
           if i.module.split(".")[0] in ABOVE]
    assert not bad, "\n".join(bad)
