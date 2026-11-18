"""Type stubs for the `reactor.reactor` native extension.

They support type checking (mypy) and editor hints only, with no runtime effect, and
are maintained by hand: PyO3 does not generate stubs, so adding a sort means updating
this file too (see https://pyo3.rs/v0.29.0/python-typing-hints). The stubs of the
operations, in `reactor/op/`, are generated from those of `theory::op`.

The runtime API is the theory: the sorts `Real` / `Int` / `Bool` / `Word` / `Zero`,
the operations of each theory in `op.real` / `op.int` / `op.bool` / `op.word` (a
class per operation, taken apart via `match`/unpacking) and the logics `op.LRA`,
`op.LIA`, `op.BV`, ...; and the structural layer `Wire` / `Var` / `Term` / `Atom` /
`Module`. A `Var` bundles the latched, next (`X(v)`), and derivative (`d(v)`) wires
of a state variable and stands for its latched wire wherever a wire is expected; its
derivative wire has the tangent of its sort: `Real(shape, degree + 1)` for reals, and
`Zero()` for the discrete sorts.
"""

from typing import Callable, Iterable, Mapping, Sequence, override

from . import op as op
from .op import Op

# A `hide` argument selects the variables to make private: a predicate over
# variables, or any iterable of them (preferably a set); None hides nothing.
type Hide = Callable[[Var], bool] | Iterable[Var] | None


# ---------------------------------------------------------------------------
# Sorts
# ---------------------------------------------------------------------------

# `reactor.sort` registers the sorts as virtual subclasses of its ABC `Sort`. A sort
# takes its fields in order and lists them in `__match_args__`; sorts are immutable,
# hashable, and compare by value.

class Real:
    """Real matrices of the given shape; of degree `d > 0`, their rates of change."""

    __match_args__ = ("shape", "degree")
    def __init__(self, shape: list[int], degree: int = 0) -> None: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def degree(self) -> int: ...


class Int:
    """Integer matrices of the given shape."""

    __match_args__ = ("shape",)
    def __init__(self, shape: list[int]) -> None: ...
    @property
    def shape(self) -> list[int]: ...


class Bool:
    """Boolean matrices of the given shape."""

    __match_args__ = ("shape",)
    def __init__(self, shape: list[int]) -> None: ...
    @property
    def shape(self) -> list[int]: ...


class Word:
    """Matrices of `width`-bit words of the given shape; `width` is positive."""

    __match_args__ = ("width", "shape")
    def __init__(self, width: int, shape: list[int]) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def shape(self) -> list[int]: ...


class Zero:
    """The singleton sort: the rates of change of the discrete sorts."""

    __match_args__ = ()
    def __init__(self) -> None: ...


type Sort = Real | Int | Bool | Word | Zero


# ---------------------------------------------------------------------------
# Wires / Variables / Terms
# ---------------------------------------------------------------------------

class Wire:
    def __init__(self, dtype: Sort) -> None: ...

    @property
    def id(self) -> int: ...

    @property
    def dtype(self) -> Sort: ...

    @override
    def __eq__(self, other: object) -> bool: ...

    def __lt__(self, other: Wire) -> bool: ...

    def __le__(self, other: Wire) -> bool: ...

    def __gt__(self, other: Wire) -> bool: ...

    def __ge__(self, other: Wire) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @override
    def __repr__(self) -> str: ...


class Var:
    """A state variable: three wires (latched, next, derivative).

    A variable stands for its latched wire — attribute lookups (`id`, `dtype`),
    equality, hashing, and ordering all go through it, so a `Var`
    and its latched `Wire` are interchangeable e.g. as dictionary keys.
    """

    def __init__(self, dtype: Sort) -> None: ...

    @property
    def id(self) -> int: ...

    @property
    def dtype(self) -> Sort: ...

    @override
    def __eq__(self, other: object) -> bool: ...

    def __lt__(self, other: Var | Wire) -> bool: ...

    def __le__(self, other: Var | Wire) -> bool: ...

    def __gt__(self, other: Var | Wire) -> bool: ...

    def __ge__(self, other: Var | Wire) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @override
    def __str__(self) -> str: ...

    @override
    def __repr__(self) -> str: ...


def X(var: Var) -> Wire:
    """The variable's next wire."""
    ...


def d(var: Var) -> Wire:
    """The variable's derivative wire."""
    ...


class Term:
    @staticmethod
    def function(
            itype: Op,
            write: Sequence[Wire | Var],
            read: Sequence[Wire | Var],
    ) -> Term: ...

    @staticmethod
    def constant(itype: Op, write: Sequence[Wire | Var]) -> Term: ...

    def __init__(
            self,
            itype: Op,
            write: Sequence[Wire | Var],
            read: Sequence[Wire | Var] | None = None,
    ) -> None: ...

    @property
    def write(self) -> Sequence[Wire]: ...

    @property
    def read(self) -> Sequence[Wire]: ...

    @property
    def itype(self) -> Op: ...

    @override
    def __str__(self) -> str: ...

    @override
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

class Atom:
    def __init__(
            self,
            vars: Iterable[Var],
            init: Iterable[Term] | None = None,
            delay: Iterable[Term] | None = None,
            update: Iterable[Term] | None = None,
    ) -> None: ...

    @staticmethod
    def sequential(
            vars: Iterable[Var], init: Iterable[Term], update: Iterable[Term]
    ) -> Atom: ...

    @staticmethod
    def differential(
            vars: Iterable[Var], init: Iterable[Term], delay: Iterable[Term]
    ) -> Atom: ...

    @staticmethod
    def hybrid(
            vars: Iterable[Var],
            init: Iterable[Term],
            update: Iterable[Term],
            delay: Iterable[Term],
    ) -> Atom: ...

    @staticmethod
    def jump(vars: Iterable[Var], update: Iterable[Term]) -> Atom: ...

    @staticmethod
    def uninitialized(
            vars: Iterable[Var], update: Iterable[Term], delay: Iterable[Term]
    ) -> Atom: ...

    @staticmethod
    def constant(vars: Iterable[Var], init: Iterable[Term]) -> Atom: ...

    @staticmethod
    def hold(vars: Iterable[Var]) -> Atom: ...

    @staticmethod
    def flow(vars: Iterable[Var], delay: Iterable[Term]) -> Atom: ...

    @staticmethod
    def combinatorial(vars: Iterable[Var], assign: Iterable[Term]) -> Atom: ...

    @property
    def read(self) -> Sequence[Var]: ...

    @property
    def ctrl(self) -> Sequence[Var]: ...

    @property
    def wait(self) -> Sequence[Var]: ...

    @property
    def init(self) -> Sequence[Term]: ...

    @property
    def update(self) -> Sequence[Term]: ...

    @property
    def delay(self) -> Sequence[Term]: ...

    def show(self, names: Mapping[Var, str]) -> str: ...

    @override
    def __str__(self) -> str: ...

    @override
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class Module:
    def __init__(
            self,
            *args: Atom | Module,
            init: Iterable[Term] | None = None,
            delay: Iterable[Term] | None = None,
            update: Iterable[Term] | None = None,
            vars: Iterable[Var] | None = None,
            hide: Hide = None,
    ) -> None: ...

    @staticmethod
    def atomic(
            vars: Iterable[Var],
            init: Iterable[Term] | None = None,
            delay: Iterable[Term] | None = None,
            update: Iterable[Term] | None = None,
            *,
            hide: Hide = None,
    ) -> Module: ...

    @staticmethod
    def sequential(
            vars: Iterable[Var],
            init: Iterable[Term],
            update: Iterable[Term],
            *,
            hide: Hide = None,
    ) -> Module: ...

    @staticmethod
    def combinatorial(
            vars: Iterable[Var], assign: Iterable[Term], *, hide: Hide = None
    ) -> Module: ...

    @staticmethod
    def differential(
            vars: Iterable[Var],
            init: Iterable[Term],
            delay: Iterable[Term],
            *,
            hide: Hide = None,
    ) -> Module: ...

    @staticmethod
    def hybrid(
            vars: Iterable[Var],
            init: Iterable[Term],
            update: Iterable[Term],
            delay: Iterable[Term],
            *,
            hide: Hide = None,
    ) -> Module: ...

    @staticmethod
    def jump(
            vars: Iterable[Var], update: Iterable[Term], *, hide: Hide = None
    ) -> Module: ...

    @staticmethod
    def uninitialized(
            vars: Iterable[Var],
            update: Iterable[Term],
            delay: Iterable[Term],
            *,
            hide: Hide = None,
    ) -> Module: ...

    @staticmethod
    def hold(vars: Iterable[Var], *, hide: Hide = None) -> Module: ...

    @staticmethod
    def flow(
            vars: Iterable[Var], delay: Iterable[Term], *, hide: Hide = None
    ) -> Module: ...

    @staticmethod
    def constant(
            vars: Iterable[Var], init: Iterable[Term], *, hide: Hide = None
    ) -> Module: ...

    @staticmethod
    def compose(*modules: Module, hide: Hide = None) -> Module: ...

    def hide(self, hide: Hide) -> Module: ...

    def __mul__(self, other: Module) -> Module: ...

    @property
    def atoms(self) -> Sequence[Atom]: ...

    @property
    def extl(self) -> Sequence[Var]: ...

    @property
    def intf(self) -> Sequence[Var]: ...

    @property
    def prvt(self) -> Sequence[Var]: ...

    @property
    def obs(self) -> Sequence[Var]: ...

    @property
    def ctrl(self) -> Sequence[Var]: ...

    def closed(self) -> bool: ...

    def open(self) -> bool: ...

    def with_varnames(self, names: Mapping[Var, str]) -> str: ...

    @override
    def __str__(self) -> str: ...

    @override
    def __repr__(self) -> str: ...
