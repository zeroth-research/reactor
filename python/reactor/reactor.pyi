"""Type stubs for the `reactor.reactor` native extension (theory_pyo3).

They support type checking (mypy) and editor hints only, with no runtime effect, and
are maintained by hand: PyO3 does not generate stubs, so adding a theory or an op means
updating this file too (see https://pyo3.rs/v0.29.0/python-typing-hints).

The runtime API is the per-theory IR: `Sort` (Bool/Int/Real/BitVec, with derivatives
typed by a Real of degree >= 1 or by Zero), the per-theory
op enums `LRA` / `LIA` / `BV` (complex enums: each variant is its own subclass,
payloads are read via `match`/unpacking), and the structural layer `Wire` / `Var` /
`Term` / `Atom` / `Module`. A `Var` bundles the latched, next (`X(v)`), and
derivative (`d(v)`) wires of a state variable and stands for its latched wire
wherever a wire is expected. A wire's sort says whether it carries a value or a
derivative: `d(v)` has the tangent of `v`'s sort (`Real(shape, 1)` for `Real(shape)`).
"""

from torch import Tensor as TorchTensor
from typing import Callable, Iterable, Mapping, Sequence, override

# A `hide` argument selects the variables to make private: a predicate over
# variables, or any iterable of them (preferably a set); None hides nothing.
type Hide = Callable[[Var], bool] | Iterable[Var] | None


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
    """A real of the given shape and `degree`: a value at degree `0` (the default), the
    sort of its `degree`-th derivative at `degree >= 1`."""

    def __init__(self, shape: list[int], degree: int = 0) -> None: ...


class BitVec(Sort):
    def __init__(self, width: int, shape: list[int]) -> None: ...


class Zero(Sort):
    """The (zero) derivatives of `Bool`, `Int`, and `BitVec`."""

    def __init__(self, shape: list[int]) -> None: ...


# ---------------------------------------------------------------------------
# Theory op enums (each variant is a subclass; construct e.g. `LRA.Add(rows, cols)`).
# The ops carry their parameters (the shapes of the matrices, the degrees of derivatives,
# the widths of bit-vectors), so that each op has a fixed boundary; see `reactor.ops.op`.
# The parameters are checked when the op is built: an ill-formed one raises a `ValueError`
# (e.g., the tensor of a literal must be 2D, and the degree of a derivative at least 1).
# ---------------------------------------------------------------------------

class Affine:
    """The constants of the affine map `x ↦ a·x + b`: `a` is a 2D tensor `[outputs, inputs]`,
    and `b` is either empty (no bias) or a column vector `[outputs, 1]`."""
    __match_args__ = ("a", "b")
    def __init__(self, a: TorchTensor, b: TorchTensor) -> None: ...
    @property
    def a(self) -> TorchTensor: ...
    @property
    def b(self) -> TorchTensor: ...


class BVConst:
    """A constant matrix of bit-vectors of the width `width` (at most 64): a 2D tensor of the
    values that fit into (unsigned) bit-vectors of the width."""
    __match_args__ = ("tensor", "width")
    def __init__(self, tensor: TorchTensor, width: int) -> None: ...
    @property
    def tensor(self) -> TorchTensor: ...
    @property
    def width(self) -> int: ...


class Bits:
    """The bits `[high..=low]` (inclusive) of bit-vectors of the width `width`
    (`low <= high < width`)."""
    __match_args__ = ("high", "low", "width")
    def __init__(self, high: int, low: int, width: int) -> None: ...
    @property
    def high(self) -> int: ...
    @property
    def low(self) -> int: ...
    @property
    def width(self) -> int: ...


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
        def __init__(self, affine: Affine, batch: int) -> None: ...

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

    # the operations on the derivatives of reals (of the degree >= 1, else `ValueError`)
    class TAdd(LRA):
        def __init__(self, rows: int, cols: int, degree: int) -> None: ...

    class TSub(LRA):
        def __init__(self, rows: int, cols: int, degree: int) -> None: ...

    class TLinear(LRA):
        def __init__(self, affine: Affine, batch: int, degree: int) -> None: ...

    class TIte(LRA):
        def __init__(self, rows: int, cols: int, degree: int) -> None: ...

    class TId(LRA):
        def __init__(self, rows: int, cols: int, degree: int) -> None: ...

    class RealZerograd(LRA):  # unstable
        def __init__(self, rows: int, cols: int, degree: int) -> None: ...

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
        def __init__(self, affine: Affine, batch: int) -> None: ...

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
        def __init__(self, const: BVConst) -> None: ...

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
        def __init__(self, *, bits: Bits, rows: int, cols: int) -> None: ...

    class Extend(BV):
        def __init__(self, *, extra: int, width: int, rows: int, cols: int) -> None: ...

    class Uninterpreted(BV):
        def __init__(self, name: str, width: int, rows: int, cols: int) -> None: ...

    class Havoc(BV):
        def __init__(self, width: int, rows: int, cols: int) -> None: ...


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
    equality, hashing, and ordering all go through it, so a `Var` and its
    latched `Wire` are interchangeable e.g. as dictionary keys.
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
    """The variable's derivative wire: its sort is `Real(shape, 1)` for a `Real`
    variable, and `Zero(shape)` for the others."""
    ...


class Term:
    @staticmethod
    def function(
            itype: LRA | LIA | BV,
            write: Sequence[Wire | Var],
            read: Sequence[Wire | Var],
    ) -> Term: ...

    @staticmethod
    def constant(itype: LRA | LIA | BV, write: Sequence[Wire | Var]) -> Term: ...

    def __init__(
            self,
            itype: LRA | LIA | BV,
            write: Sequence[Wire | Var],
            read: Sequence[Wire | Var] | None = None,
    ) -> None: ...

    @property
    def write(self) -> Sequence[Wire]: ...

    @property
    def read(self) -> Sequence[Wire]: ...

    @property
    def itype(self) -> LRA | LIA | BV: ...

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
