"""Type stubs for the `reactor.reactor` native extension.

They support type checking (mypy) and editor hints only, with no runtime effect, and
are maintained by hand: PyO3 does not generate stubs (see
https://pyo3.rs/v0.29.0/python-typing-hints).

The extension holds the structural layer `Wire` / `Var` / `Term` / `Atom` / `Module`;
the structural operations `Havoc` and `Skip`, which belong to no theory; and the
namespaces of the sorts and the other operations, mirroring the `theory` crate: `zero`
holds the singleton sort `Real0`, shared by all theories, and the operation `Zero`
writing its only value; `real`, `int`, `bool` and `word` (with `word.signed` and
`word.unsigned`, and each with `.broadcast`, the choice of whole matrices by a scalar
condition) hold the sort of each theory (`real.Real([rows, cols])`,
`word.Word(width, [rows, cols])`) and its operations, each built from the parameters
its sorts do not determine, then its domain, a single sort bare and several as a list,
and optionally its codomain, which must agree (`real.Add([x, y])`,
`word.Extract(i, j, x)`); an operation that reads nothing is built from its codomain
(`real.Zero(x)`, `Havoc(x)`), except `zero.Zero()`, whose codomain, the singleton sort,
carries no information. The sorts and operations of the namespaces are typed loosely
here, as `Sort` and `Op`: a stub file cannot describe the nested namespaces of a single
extension module. A `Var` bundles the latched, next (`X(v)`), and derivative (`d(v)`)
wires of a state variable and stands for its latched wire wherever a wire is expected.
"""

import builtins
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence, override

# A `hide` argument selects the variables to make private: a predicate over
# variables, or any iterable of them (preferably a set); None hides nothing.
type Hide = Callable[[Var], builtins.bool] | Iterable[Var] | None

# A sort, of one of the namespaces, and an operation, `Havoc`, `Skip` or of one of the
# namespaces.
type Sort = Any
type Op = Any

# The namespaces of the sorts and operations; `int` and `bool` shadow the builtins here,
# which this stub names `builtins.int` and `builtins.bool`.
zero: ModuleType
real: ModuleType
int: ModuleType
bool: ModuleType
word: ModuleType


class Havoc:
    """The nondeterministic choice of a value of the sort `cod`, reading nothing."""

    def __init__(self, cod: Sort) -> None: ...

    @property
    def cod(self) -> Sort: ...

    @property
    def sort(self) -> Sort: ...


class Skip:
    """The identity of the sort `dom`, whose codomain `cod`, when given, must be it."""

    def __init__(self, dom: Sort, cod: Sort | None = None) -> None: ...

    @property
    def dom(self) -> Sort: ...

    @property
    def cod(self) -> Sort: ...

    @property
    def sort(self) -> Sort: ...


class Wire:
    def __init__(self, dtype: Sort) -> None: ...

    @property
    def id(self) -> builtins.int: ...

    @property
    def dtype(self) -> Sort: ...

    @override
    def __eq__(self, other: object) -> builtins.bool: ...

    def __lt__(self, other: Wire) -> builtins.bool: ...

    def __le__(self, other: Wire) -> builtins.bool: ...

    def __gt__(self, other: Wire) -> builtins.bool: ...

    def __ge__(self, other: Wire) -> builtins.bool: ...

    @override
    def __hash__(self) -> builtins.int: ...

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
    def id(self) -> builtins.int: ...

    @property
    def dtype(self) -> Sort: ...

    @override
    def __eq__(self, other: object) -> builtins.bool: ...

    def __lt__(self, other: Var | Wire) -> builtins.bool: ...

    def __le__(self, other: Var | Wire) -> builtins.bool: ...

    def __gt__(self, other: Var | Wire) -> builtins.bool: ...

    def __ge__(self, other: Var | Wire) -> builtins.bool: ...

    @override
    def __hash__(self) -> builtins.int: ...

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


def T(sort: Sort) -> Sort:
    """The tangent sort: the sort of the rate of change of the values of `sort`, one
    degree higher for a real, and the singleton `zero.Real0()` for a discrete sort."""
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

    def closed(self) -> builtins.bool: ...

    def open(self) -> builtins.bool: ...

    def with_varnames(self, names: Mapping[Var, str]) -> str: ...

    @override
    def __str__(self) -> str: ...

    @override
    def __repr__(self) -> str: ...
