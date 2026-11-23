"""A predicate in reactor's terms: the IR a module is written in.

`Formula` is what the native parser makes of a predicate: the terms that
compute it, in the order they were built, over the module's own wires --
the state after the round (`X` of each controlled variable) and the inputs
awaited in it and latched before it -- and the wire its value is on.
`Builder` makes them, one wire per distinct term: a subterm written twice
is built once, which is what lets the printer share it.

Every term but a matrix variable's is a scalar: `Get` reads one element of
a matrix wire, and whatever reads it after that reads a 1×1 `Matrix`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import torch

from reactor import Bool, Int, Real, Term, Wire, Word, tensor

from ..core.errors import Unreadable
from ..lowering.values import decimal_fraction

# The shape of every term a predicate builds.
SCALAR = [1, 1]


@dataclass(frozen=True)
class Formula:
    """`terms` compute the predicate onto `out`: each term reads only module
    wires and the wires written by the terms before it. `names` are what the
    module wires were called (`s0`, `e1`, ...), by wire id."""

    terms: tuple
    out: Wire
    names: dict = field(default_factory=dict)

    @property
    def sort(self):
        return self.out.dtype


def family(sort) -> str:
    """"Bool", "Int", "Real" or "(_ BitVec w)": a sort without its shape,
    spelled as SMT-LIB spells it."""
    match sort:
        case Bool(_):
            return "Bool"
        case Int(_):
            return "Int"
        case Real(_):
            return "Real"
        case Word(w, _):
            return f"(_ BitVec {w})"
    raise ValueError(f"no sort family for {sort}")


def element(sort):
    """The 1x1 sort of one element of `sort`."""
    match sort:
        case Bool(_):
            return Bool(SCALAR)
        case Int(_):
            return Int(SCALAR)
        case Real(_):
            return Real(SCALAR)
        case Word(w, _):
            return Word(w, SCALAR)
    raise ValueError(f"no element sort for {sort}")


class Builder:
    """Builds the terms of one formula, each distinct one once."""

    def __init__(self):
        self.terms: list = []
        self._memo: dict = {}

    def formula(self, out: Wire, names: "dict | None" = None) -> Formula:
        """The terms built, onto `out`; `names` maps each variable's name to
        its wire, as a `smtlib.Scope` holds them."""
        return Formula(tuple(self.terms), out,
                       {w.id: n for n, w in (names or {}).items()})

    def apply(self, op, reads: list, sort, key: object = None) -> Wire:
        """The wire `op` writes from `reads`, of `sort`: built the first time
        this op is applied to these wires, and the same wire after. The op is
        told by its class and sizes; `key` names it where those do not, as a
        constant's value."""
        if key is None:
            key = (type(op), *(getattr(op, a) for a in type(op).__match_args__))
            key = tuple(tuple(k) if isinstance(k, list) else k for k in key)
        memo_key = (key, tuple(w.id for w in reads))
        hit = self._memo.get(memo_key)
        if hit is not None:
            return hit
        out = Wire(sort)
        self.terms.append(Term(op, [out], reads) if reads else Term(op, [out]))
        self._memo[memo_key] = out
        return out

    # --- constants --------------------------------------------------------

    def boolean(self, value: bool) -> Wire:
        return self.apply(tensor.bool.Const(torch.tensor([[bool(value)]])), [],
                          Bool(SCALAR), key=("Bool", bool(value)))

    def integer(self, value: int) -> Wire:
        if not -(1 << 63) <= value < 1 << 63:
            raise Unreadable(f"`{value}` does not fit the 64 bits an Int "
                             f"constant of reactor's terms holds")
        return self.apply(tensor.int.Const(torch.tensor([[value]], dtype=torch.int64)),
                          [], Int(SCALAR), key=("Int", value))

    def real(self, value: Fraction) -> Wire:
        f = float(value)
        if decimal_fraction(f) != value:
            raise Unreadable(f"`{value}` is no float that reads back as "
                             f"itself, and a Real constant of reactor's "
                             f"terms is a float")
        return self.apply(tensor.real.Const(torch.tensor([[f]], dtype=torch.float64)),
                          [], Real(SCALAR), key=("Real", value))

    def bitvec(self, width: int, value: int) -> Wire:
        value %= 1 << width
        if not 0 < width <= 64 or value >= 1 << 63:
            raise Unreadable(f"`{value}` at {width} bits: a bit-vector constant "
                             f"of reactor's terms is 1 to 64 bits wide, and "
                             f"held in 64 signed ones")
        return self.apply(tensor.word.Const(torch.tensor([[value]], dtype=torch.int64), width),
                          [], Word(width, SCALAR), key=("BV", width, value))
