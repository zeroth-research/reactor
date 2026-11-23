"""Type stubs for the `reactor.reactor` native extension.

They support type checking (mypy) and editor hints only, with no runtime effect, and
are maintained by hand: PyO3 does not generate stubs (see
https://pyo3.rs/v0.29.0/python-typing-hints).

The extension holds the structural layer `Wire` / `Var` / `Term` / `Atom` / `Module`,
and the namespace `tensor` of the operations and sorts, mirroring the `theory` crate:
`tensor` holds the structural operations `Havoc`, `Skip` and `Zero`, and the namespaces
`tensor.real`, `.int`, `.bool`, `.word` (with `.signed` and `.unsigned`) hold the
operations of each theory, sized by their shapes (`tensor.real.Add([m, n])`),
`tensor.cast` the casts between their sorts (`tensor.cast.ToReal([m, n])`), and
`tensor.sort` the sorts (`Real([rows, cols])`, `Word(width, [rows, cols])`, and the
rates of change `Real([rows, cols], degree=d)`). The constants, products with a constant
and affine maps there (`tensor.real.Const(t)`, `tensor.real.Scale(c)`,
`tensor.real.Linear(a, b)`) hold `torch.Tensor`s, as do the operations of `Term`, `Atom`
and `Module`. The namespace `jax` holds their counterparts over `jax.Array`s: the classes
`jax.Term`, `jax.Atom` and `jax.Module`, and the constants, products and affine maps
`jax.real.Const`, `jax.real.Scale`, `jax.real.Linear`, and so on; the namespace `mixed`
holds those over either, a module of them holding torch tensors and JAX arrays alike. The
operations holding no tensors are those of `tensor`. Those are
typed loosely here, as `Sort` and `Op`: a stub file cannot describe the nested
namespaces of a single extension module. A `Var` bundles the latched, next (`X(v)`),
and derivative (`d(v)`) wires of a state variable and stands for its latched wire
wherever a wire is expected.
"""

from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence, override

# A `hide` argument selects the variables to make private: a predicate over
# variables, or any iterable of them (preferably a set); None hides nothing.
type Hide = Callable[[Var], bool] | Iterable[Var] | None

# A sort of `tensor.sort`, and an operation of `tensor` or of one of its namespaces.
type Sort = Any
type Op = Any

# The namespace of the operations and sorts.
tensor: ModuleType
# The namespace of the terms, atoms, modules and operations over JAX arrays.
jax: ModuleType
# The namespace of the terms, atoms, modules and operations over torch tensors and JAX
# arrays alike.
mixed: ModuleType


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
    """The variable's derivative wire, of the tangent of its sort: `Real(shape, degree=1)`
    for `Real(shape)`, and `Zero()` for the discrete sorts."""
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
