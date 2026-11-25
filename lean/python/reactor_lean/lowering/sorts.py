"""Sorts and shapes of wires, the Lean types they print as, and the exact
number a constant holds (`decimal_fraction`)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import reactor.reactor as _ext
from reactor import Bool, Int, Real, Word, Wire

from ..errors import Refused

# A sort of `reactor.tensor.sort`: `Bool`, `Int`, `Real` or `Word`.
Sort = Any


@dataclass(frozen=True)
class MatShape:
    """A wire's shape as the `(m, n)` of the matrix it is: the one reading
    every encoding uses. A Sort's shape is always two-dimensional -- the
    constructor refuses any other -- so there is no 0-D or 1-D one to read
    differently."""

    m: int
    n: int

    @property
    def total(self) -> int:
        return self.m * self.n

    @property
    def is_scalar(self) -> bool:
        return self.total == 1


def wire_shape(wire: Wire) -> MatShape:
    """`wire`'s shape as the `(m, n)` of the matrix it is."""
    m, n = wire.dtype.shape
    return MatShape(m, n)


def lean_elem_type(sort: Sort) -> str:
    """The Lean type of one element of `sort`, ignoring its shape: `Bool`,
    `Int`, `Real` (proofs read the module's floats as reals) or
    `(BitVec w)`."""
    if isinstance(sort, Bool):
        return "Bool"
    if isinstance(sort, Int):
        return "Int"
    if isinstance(sort, Real):
        return "Real"
    if isinstance(sort, Word):
        return f"(BitVec {sort.width})"
    raise Refused(f"no Lean type for the sort {sort}")


def dtype_to_lean_type(wire: Wire, simple_types=False) -> str:
    """Map a Wire's Sort to its Lean type, `(Matrix (Fin m) (Fin n) t)`;
    with `simple_types`, a 1×1 wire is its bare element type `t`."""
    ty = lean_elem_type(wire.dtype)
    shape = wire_shape(wire)
    if shape.is_scalar and simple_types:
        return ty
    return f"(Matrix (Fin {shape.m}) (Fin {shape.n}) {ty})"


def mat_mk(entry: str, i: str = "_", j: str = "_") -> str:
    """The Lean matrix whose entry `(i, j)` is `entry`:
    `(Matrix.mk fun i j => entry)`.

    The codegen writes every matrix it builds from its entries as a
    `Matrix.mk` (ReactLib's), never as a bare `fun i j => ...`: `simp` and
    instance search do not see through Mathlib's `Matrix`, so to them a
    bare lambda is a function and no `Matrix` lemma or instance applies to
    it. Its size is not in the term, so the expression needs an expected
    type, which every emitted `let` and definition gives it.
    """
    return f"(Matrix.mk fun {i} {j} => {entry})"


# The theories an op comes from: the namespaces of `reactor.tensor` that hold
# operations, by their dotted names below it. `cast` is no theory of its own
# sorts: it is what crosses between the others' (`theory/src/tensor/cast.rs`).
# The structural operations, `Havoc`, `Skip` and `Zero`, belong to none, and
# get "".
THEORIES = ("bool", "int", "real", "word", "word.signed", "word.unsigned", "cast")


def _namespaces():
    """(theory, namespace) for every namespace holding operations: those of
    `reactor.tensor`, and their counterparts holding JAX arrays and either
    tensor, which hold the constants, products and affine maps."""
    for theory in THEORIES:
        ns = _ext.tensor
        for part in theory.split("."):
            ns = getattr(ns, part)
        yield theory, ns
    for backend in (_ext.jax, _ext.mixed):
        for theory in ("bool", "int", "real", "word"):
            yield theory, getattr(backend, theory)


# The class of every operation, to its theory. The classes of two theories
# share names (`tensor.int.Add`, `tensor.real.Add`), so the class, not its
# name, says which theory it is of.
_THEORY_OF: dict[type, str] = {
    cls: theory
    for theory, ns in _namespaces()
    for cls in vars(ns).values()
    if isinstance(cls, type)
}
_THEORY_OF.update({cls: "" for cls in (_ext.tensor.Havoc, _ext.tensor.Skip, _ext.tensor.Zero)})


def theory_of(itype) -> str:
    """The theory an op is of: one of `THEORIES`, or "" for the structural
    operations. Fails for an object that is no operation."""
    try:
        return _THEORY_OF[type(itype)]
    except KeyError:
        raise ValueError(f"not an operation of reactor.tensor: {itype!r}") from None


def itype_name(itype) -> str:
    """The name of an op's class, e.g. `tensor.int.Add([1, 1])` -> 'Add'."""
    return type(itype).__name__


def decimal_fraction(value) -> Fraction:
    """`value` as an exact fraction, a float read by its decimal form:
    `0.1` is 1/10, what the number was written as and what Lean makes of
    a decimal literal."""
    if isinstance(value, float):
        return Fraction(repr(value))
    return Fraction(value)
