"""The input: a Python file whose `module()` -- or the callable `-d`
names -- returns the reactive module, read into a `Problem` (`load`).

A Gymnasium environment comes in this way too: whatever the callable
returns that is not a `Module` is made one by `reactor.Env`.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reactor import Module

from ..errors import Refused
from .names import Names


class InputError(Refused):
    """The input is not a module reactor can read; the message says why."""


@dataclass(frozen=True)
class Problem:
    """What the input states: the module the run is about, the text it
    was read from, and the names a claim may call its variables by
    (positional only when the input gives none)."""

    module: Module
    source: str
    names: Names


@dataclass(frozen=True)
class Named:
    """A module, with the names a claim about it may call its variables
    by. What a module file's `module()` returns to be spoken of in its own
    terms rather than by position::

        def module():
            x, u = Var(Int([1, 1])), Var(Int([1, 1]))
            ...
            return Named(Module.sequential([x, u], init, update),
                         {"x": x, "u": u})

    after which `--safety "(<= (X x) 9)"` means `(<= (X @intf.0) 9)`. A
    variable the mapping leaves out is named by position only.
    """

    module: object
    names: Mapping[str, object]


def load(path: str | Path, module_def: str = "module") -> Problem:
    """The `Problem` the Python file at `path` states: the module its
    callable `module_def` returns, and the names a `Named` one gives.

    The file not being there or not being readable, not being a Python
    file and not defining `module_def` are `InputError`s; what the file's
    own code raises is not caught.
    """
    path = _existing(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise InputError(f"cannot read {path}: {e}") from e
    module, names = _as_module(_call(path, module_def))
    return Problem(
        module, text,
        Names.of(module, names) if names else Names.positional(module),
    )


def _existing(path: str | Path) -> Path:
    path = Path(path).resolve()
    if not path.exists():
        raise InputError(f"Module file not found: {path}")
    return path


def _as_module(result) -> "tuple[Any, Mapping[str, Any] | None]":
    """The module `result` is, and the names a `Named` one gives it.

    A Gymnasium environment is made a `Module` by `reactor.Env`. A `Named`
    one is refused: its variables are made by `Env`, so a name given
    beside the environment cannot be one of them.
    """
    from reactor import Env

    if isinstance(result, Named):
        if not isinstance(result.module, Module):
            raise InputError(
                f"`Named` takes a `Module`, not `{type(result.module).__name__}`"
                ": make an environment one with `reactor.Env(env)`, and name "
                "that module's variables"
            )
        return result.module, result.names
    if isinstance(result, Module):
        return result, None
    return Env(result), None


def _call(path: Path, module_def: str):
    """What the Python file at `path` makes of its `module_def`: a function
    returning the `Module` (or a `Named` one), or a `Module` subclass whose
    constructor takes no arguments."""
    spec = importlib.util.spec_from_file_location("_module_file", path)
    if spec is None or spec.loader is None:
        raise InputError(f"{path} is no Python file")
    module = importlib.util.module_from_spec(spec)
    # In `sys.modules` while it runs, as an imported module is, for the
    # file's own code that looks it up there: `dataclasses` does, for a
    # class whose annotations are strings.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        if not hasattr(module, module_def):
            raise InputError(
                f"Module file {path} must define `{module_def}` which is a "
                "function returning `Module` or a sub-class of `Module` with "
                "constructor without arguments."
            )
        return getattr(module, module_def)()
    finally:
        del sys.modules[spec.name]
