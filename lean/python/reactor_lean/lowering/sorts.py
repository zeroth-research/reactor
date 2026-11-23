"""Sorts and shapes of wires, and the Lean types they print as."""

from __future__ import annotations

from dataclasses import dataclass

from reactor import BitVec, Bool, Int, Real, Sort, Wire


def dtype_shape(dt) -> list:
    """Shape of a Sort, across all element types.

    Bool/Int/Real carry the shape in field ``_0``; BitVec carries the bit-width
    in ``_0`` and the shape in ``_1``. main always uses 2-D shapes (a scalar is
    ``[1, 1]``).
    """
    if isinstance(dt, BitVec):
        return list(dt._1)
    return list(dt._0)


@dataclass(frozen=True)
class MatShape:
    """A wire's shape as the `(m, n)` of the matrix it is: the one reading
    every encoding uses."""

    m: int
    n: int

    @property
    def total(self) -> int:
        return self.m * self.n

    @property
    def is_scalar(self) -> bool:
        return self.total == 1


def sort_shape(sort) -> MatShape:
    """`sort`'s shape as the `(m, n)` of the matrix it is. A Sort's shape is
    always two-dimensional -- the constructor refuses any other -- so this
    is the shape itself, and every encoding reads it here rather than
    guessing at a 0-D or 1-D one that cannot occur."""
    m, n = dtype_shape(sort)
    return MatShape(m, n)


def wire_shape(wire: Wire) -> MatShape:
    """`sort_shape` of `wire`'s sort."""
    return sort_shape(wire.dtype)


def is_scalar_shape(shape: list) -> bool:
    """True if the shape denotes a single element ([], [1], [1, 1], ...)."""
    return all(d == 1 for d in shape)


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
    if isinstance(sort, BitVec):
        return f"(BitVec {sort._0})"
    raise ValueError(f"Unsupported Sort for Lean conversion: {sort}")


def lean_matrix_type(elem: str, m: int | str, n: int | str) -> str:
    """The Lean type of an `m × n` matrix of `elem`s: Mathlib's `Matrix`,
    indexed by `Fin`, parenthesized."""
    return f"(Matrix (Fin {m}) (Fin {n}) {elem})"


def dtype_to_lean_type(wire: Wire, simple_types=False) -> str:
    """Map a Wire's Sort to its Lean type, `(Matrix (Fin m) (Fin n) t)`;
    with `simple_types`, a 1×1 wire is its bare element type `t`."""

    dt = wire.dtype
    ty = lean_elem_type(dt)
    shape = sort_shape(dt)
    if shape.is_scalar and simple_types:
        return ty
    return lean_matrix_type(ty, shape.m, shape.n)


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


# The theories an op comes from. PyO3 exports each op as
# `<theory>_<variant>`: `LIA_Add`, `LRA_Real`, `BV_MatMul`, `Cast_BVToInt`.
# `Cast` is no theory of its own sorts: it is what crosses between the
# other three's (`theory/src/cast.rs`), and has rows of its own only.
THEORIES = ("LIA", "LRA", "BV", "Cast")


def theory_of(itype) -> str:
    """The theory an op came from: "LIA", "LRA", "BV" or "Cast"; an op
    with no such prefix belongs to no theory and gets "".
    """
    name = type(itype).__name__
    for theory in THEORIES:
        if name.startswith(theory + "_"):
            return theory
    return ""


def itype_name(itype) -> str:
    """Get the variant name of an op, e.g. LIA.Add() -> 'Add'."""
    name = type(itype).__name__
    theory = theory_of(itype)
    return name[len(theory) + 1:] if theory else name


# Constant op variants across theories (LIA.Int/Bool, LRA.Real/Bool,
# BV.Const). The element type is encoded in the variant name;
# scalar-vs-matrix is decided by the wire's shape.
CONST_VARIANTS = frozenset({"Int", "Real", "Bool", "Const"})


def is_constant_name(name: str) -> bool:
    """True if the variant name denotes a constant in any theory."""
    return name in CONST_VARIANTS


def is_scalar_tensor(wire: Wire) -> bool:
    """True if the wire carries a scalar Bool, Int or BitVec (not a matrix).

    Bool used to answer `True` for any shape, skipping the check the other
    two get, so a Bool *matrix* constant was never interned and reached
    `tensor_to_lean_inline`, which calls `.item()` on it.
    """
    dt = wire.dtype
    if isinstance(dt, (Bool, Int, BitVec)):
        return is_scalar_shape(dtype_shape(dt))
    return False


def is_scalar_wire(wire: Wire) -> bool:
    """True if the wire is a scalar (single element, any scalar dtype)."""
    return is_scalar_shape(dtype_shape(wire.dtype))


# ---------------------------------------------------------------------------
# Flat-scalar helpers — support multi-element (1 × n matrix) wires in the
# scalar / relational encoding by expanding each wire into n scalars.
# ---------------------------------------------------------------------------

def flat_element_type(wire: Wire) -> str:
    """Base Lean scalar type for one element of the wire (no matrix around it)."""
    return lean_elem_type(wire.dtype)
