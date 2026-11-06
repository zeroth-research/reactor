"""Type stubs for the `reactor.reactor` native extension (theory_pyo3).

They support type checking (mypy) and editor hints only, with no runtime effect, and
are maintained by hand: PyO3 does not generate stubs, so adding a theory or an op means
updating this file too (see https://pyo3.rs/v0.29.0/python-typing-hints).

The runtime API is the per-theory IR: `Sort` (Bool/Int/Real/BitVec, and the sorts of
their derivatives TReal/Zero), the per-theory
op enums `LRA` / `LIA` / `BV` (complex enums: each variant is its own subclass,
payloads are read via `match`/unpacking), the distinguished ops `Combinatorial` /
`Sequential` / `Differential` (`HAVOC`, `SKIP`, `ZERO`, `DISCARD`), and the structural
layer `Wire` / `Term` / `Block` / `Var` / `Atom` / `Module`. A `Var` is a name and a
sort, and holds no wires: an atom binds its variables to the boundaries of its blocks
by position, where a variable stands for its latched value, `X(v)` for its next value
and `d(v)` for its derivative.
"""

from torch import Tensor as TorchTensor
from typing import Callable, Iterable, Iterator, Mapping, Sequence, override

# A `hide` argument selects the variables to make private: a predicate over
# variables, or any iterable of them (preferably a set); None hides nothing.
type Hide = Callable[[Var], bool] | Iterable[Var] | None

# The ops a term may carry: those of a base theory, or a distinguished one.
type Op = LRA | LIA | BV | Combinatorial | Sequential | Differential


# ---------------------------------------------------------------------------
# Sorts
# ---------------------------------------------------------------------------

# `reactor.sort` shadows `Sort` with an ABC and registers the extension's enum
# against it, so the variants present as plain subclasses of `Sort`.
class Sort:
    @override
    def __eq__(self, other: object) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @override
    def __str__(self) -> str: ...

    @override
    def __repr__(self) -> str: ...


class Bool(Sort):
    def __init__(self, shape: list[int]) -> None: ...


class Int(Sort):
    def __init__(self, shape: list[int]) -> None: ...


class Real(Sort):
    def __init__(self, shape: list[int]) -> None: ...


class BitVec(Sort):
    def __init__(self, width: int, shape: list[int]) -> None: ...


class TReal(Sort):
    """The derivatives (of the given degree >= 1) of `Real`."""

    def __init__(self, shape: list[int], degree: int) -> None: ...


class Zero(Sort):
    """The (zero) derivatives of `Bool`, `Int`, and `BitVec`."""

    def __init__(self, shape: list[int]) -> None: ...


# ---------------------------------------------------------------------------
# Theory op enums (each variant is a subclass; construct e.g. `LRA.Add(rows, cols)`).
# The ops carry their parameters (the shapes of the matrices, the ranks of derivatives,
# the widths of bit-vectors), so that each op has a fixed boundary; see `reactor.ops.op`.
# ---------------------------------------------------------------------------

class LRA:
    class Real(LRA):
        def __init__(self, tensor: TorchTensor) -> None: ...

    class Bool(LRA):
        def __init__(self, tensor: TorchTensor) -> None: ...

    class And(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Or(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Xor(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Not(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Le(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Lt(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Ge(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Gt(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Eq(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Ne(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Linear(LRA):
        def __init__(self, a: TorchTensor, b: TorchTensor, batch: int) -> None: ...

    class Add(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Sub(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class ReLU(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Argmax(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Min(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Max(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class TransposeReal(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class TransposeBool(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class IteReal(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class IteBool(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class IdReal(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class IdBool(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class UninterpretedReal(LRA):
        def __init__(self, name: str, rows: int, cols: int) -> None: ...

    class UninterpretedBool(LRA):
        def __init__(self, name: str, rows: int, cols: int) -> None: ...

    class AnyReal(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class AnyBool(LRA):
        def __init__(self, rows: int, cols: int) -> None: ...

    # the operations on the derivatives of reals (of the rank >= 1)
    class TAdd(LRA):
        def __init__(self, rows: int, cols: int, rank: int) -> None: ...

    class TSub(LRA):
        def __init__(self, rows: int, cols: int, rank: int) -> None: ...

    class TLinear(LRA):
        def __init__(self, a: TorchTensor, b: TorchTensor, batch: int, rank: int) -> None: ...

    class TIte(LRA):
        def __init__(self, rows: int, cols: int, rank: int) -> None: ...

    class TId(LRA):
        def __init__(self, rows: int, cols: int, rank: int) -> None: ...

    class RealZerograd(LRA):  # unstable
        def __init__(self, rows: int, cols: int, rank: int) -> None: ...

    class BoolZerograd(LRA):  # unstable
        def __init__(self, rows: int, cols: int) -> None: ...


class LIA:
    class Int(LIA):
        def __init__(self, tensor: TorchTensor) -> None: ...

    class Bool(LIA):
        def __init__(self, tensor: TorchTensor) -> None: ...

    class And(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Or(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Xor(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Not(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Le(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Lt(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Ge(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Gt(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Eq(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Ne(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Linear(LIA):
        def __init__(self, a: TorchTensor, b: TorchTensor, batch: int) -> None: ...

    class Add(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Sub(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class ReLU(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Argmax(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Min(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class Max(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class TransposeInt(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class TransposeBool(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class IteInt(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class IteBool(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class IdInt(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class IdBool(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class UninterpretedInt(LIA):
        def __init__(self, name: str, rows: int, cols: int) -> None: ...

    class UninterpretedBool(LIA):
        def __init__(self, name: str, rows: int, cols: int) -> None: ...

    class AnyInt(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...

    class AnyBool(LIA):
        def __init__(self, rows: int, cols: int) -> None: ...


class BV:
    class Const(BV):
        def __init__(self, tensor: TorchTensor, width: int) -> None: ...

    class Add(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Sub(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Mul(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class UDiv(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class SDiv(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class UMod(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class SMod(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Neg(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Abs(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class MatMul(BV):
        def __init__(self, width: int, m: int, k: int, n: int) -> None: ...

    class And(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Or(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Xor(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Not(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class ULe(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class ULt(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class UGe(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class UGt(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class SLe(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class SLt(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class SGe(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class SGt(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Eq(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Ne(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Ite(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class Id(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class BVToBool(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...

    class BitSelect(BV):
        def __init__(self, *, high: int, low: int, width: int, rows: int, cols: int) -> None: ...

    class Extend(BV):
        def __init__(self, *, extra: int, width: int, rows: int, cols: int) -> None: ...

    class Uninterpreted(BV):
        def __init__(self, name: str, width: int, rows: int, cols: int) -> None: ...

    class Havoc(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...


# ---------------------------------------------------------------------------
# Distinguished operations (theory-agnostic)
# ---------------------------------------------------------------------------

class Combinatorial:
    class HAVOC(Combinatorial):
        """Any value of the sort: `I -> sort`."""
        def __init__(self, sort: Sort) -> None: ...

    class DISCARD(Combinatorial):
        """Consumes a value of the sort: `sort -> I`."""
        def __init__(self, sort: Sort) -> None: ...


class Sequential:
    class HAVOC(Sequential):
        def __init__(self, sort: Sort) -> None: ...

    class SKIP(Sequential):
        """Keeps a value of the sort: `sort -> sort`."""
        def __init__(self, sort: Sort) -> None: ...

    class DISCARD(Sequential):
        def __init__(self, sort: Sort) -> None: ...


class Differential:
    class ZERO(Differential):
        """The zero derivative of the sort: `I -> sort`."""
        def __init__(self, sort: Sort) -> None: ...

    class DISCARD(Differential):
        def __init__(self, sort: Sort) -> None: ...


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
    """A state variable: a name, and the sort of the values it holds.

    A variable holds no wires: the blocks of an atom take and give the values
    of its variables at the positions of their boundaries. A variable stands
    for its latched value, `X(var)` for its next value, and `d(var)` for its
    derivative. Equality, hashing and ordering go through the name.
    """

    def __init__(self, dtype: Sort) -> None:
        """A fresh variable of the sort `dtype`, under a name no other
        variable has."""
        ...

    @property
    def name(self) -> int: ...

    @property
    def dtype(self) -> Sort: ...

    @override
    def __eq__(self, other: object) -> bool: ...

    def __lt__(self, other: Var) -> bool: ...

    def __le__(self, other: Var) -> bool: ...

    def __gt__(self, other: Var) -> bool: ...

    def __ge__(self, other: Var) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @override
    def __repr__(self) -> str: ...


class View:
    """The next value, or the derivative, of a variable. A view is no wire: it
    names the value a block takes or gives at a position of its boundary."""

    @property
    def var(self) -> Var: ...

    @property
    def dtype(self) -> Sort:
        """The sort of the variable for the next value, its tangent for the
        derivative: `TReal(shape, 1)` for a `Real` variable, `Zero(shape)` for
        the others."""
        ...

    @property
    def next(self) -> bool: ...

    @property
    def derivative(self) -> bool: ...

    @override
    def __eq__(self, other: object) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @override
    def __repr__(self) -> str: ...


def X(var: Var) -> View:
    """The next value of the variable."""
    ...


def d(var: Var) -> View:
    """The derivative of the variable."""
    ...


class Term:
    @staticmethod
    def function(
            itype: Op,
            write: Sequence[Wire],
            read: Sequence[Wire],
    ) -> Term: ...

    @staticmethod
    def constant(itype: Op, write: Sequence[Wire]) -> Term: ...

    def __init__(
            self,
            itype: Op,
            write: Sequence[Wire],
            read: Sequence[Wire] | None = None,
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
# Blocks
# ---------------------------------------------------------------------------

class Block:
    """Terms computing the outputs `write` from the inputs `read`, both in
    order. Every input is read (an input the block has no use for is consumed
    by a `DISCARD` term), and every output written exactly once."""

    def __init__(
            self, read: Sequence[Wire], write: Sequence[Wire], terms: Iterable[Term]
    ) -> None: ...

    @property
    def read(self) -> Sequence[Wire]: ...

    @property
    def write(self) -> Sequence[Wire]: ...

    def __getitem__(self, index: int) -> Term: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Term]: ...

    @override
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

class Atom:
    """An atom: its interfaces, and its blocks bound to them by position.

    The interfaces list the variables in the order of their names, and so do
    the boundaries of the blocks: `init` maps `X(wait)` to `X(ctrl)`, `update`
    maps `read` followed by `X(wait)` to `X(ctrl)`, and `delay` maps `read`
    followed by `d(wait)` to `d(ctrl)`.
    """

    def __init__(
            self,
            ctrl: Iterable[Var],
            wait: Iterable[Var],
            read: Iterable[Var],
            init: Block,
            update: Block,
            delay: Block,
    ) -> None: ...

    @property
    def read(self) -> Sequence[Var]: ...

    @property
    def ctrl(self) -> Sequence[Var]: ...

    @property
    def wait(self) -> Sequence[Var]: ...

    @property
    def init(self) -> Block: ...

    @property
    def update(self) -> Block: ...

    @property
    def delay(self) -> Block: ...

    def show(self, names: Mapping[Var, str]) -> str: ...

    @override
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class Module:
    def __init__(self, *args: Atom | Module, hide: Hide = None) -> None:
        """The module of the atoms `args`, or the parallel composition of the
        modules `args`, hiding the `hide` variables; the empty module, given
        nothing."""
        ...

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
