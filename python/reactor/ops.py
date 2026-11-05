"""Theory ops for the sorts of their operands.

The ops of the theories carry their parameters (the shapes of the matrices, the ranks of
derivatives, and the widths of bit-vectors), so that each op has a fixed boundary: the sorts
of the wires it reads and writes. The ops that apply to several kinds of values come in one
variant per kind (e.g., ``LRA.IdReal`` and ``LRA.IdBool``), and the ops on the derivatives of
reals are separate (e.g., ``LRA.TAdd``).

``op(theory, name, sort)`` builds the op ``name`` of ``theory`` that applies to operands of
the sort ``sort``, e.g.,

    op(LRA, "Add", Real([2, 3]))           # LRA.Add(2, 3)
    op(LRA, "Add", TReal([2, 3], 1))       # LRA.TAdd(2, 3, 1)
    op(LRA, "Id", Bool([2, 3]))            # LRA.IdBool(2, 3)
    op(BV, "ULt", BitVec(8, [2, 3]))       # BV.ULt(8, 2, 3)

The ops with other parameters are built directly (e.g., ``BV.MatMul(width, m, k, n)``),
or by ``linear()`` and ``bv_const()``.
"""

from .reactor import LRA, LIA, BV
from .sort import Bool, Int, Real, BitVec, TReal

# the ops of LRA and LIA that come in one variant per kind of values (e.g., `IdReal`)
_PER_KIND = ("Id", "Ite", "Transpose", "Uninterpreted")


def op(theory, name: str, sort, *args):
    """The op ``name`` of ``theory`` on operands of the sort ``sort`` (for ``Ite``, the sort
    of its branches; for ``Uninterpreted``, the sort it writes). ``args`` are the leading
    parameters of the op (e.g., the name of an uninterpreted value)."""
    match sort:
        case BitVec(width, [r, c]) if theory is BV:
            return _variant(theory, name, name, sort)(*args, width, r, c)
        case TReal([r, c], rank) if theory is LRA:
            return _variant(theory, f"T{name}", name, sort)(*args, r, c, rank)
        case Real([r, c]) if theory is LRA:
            kind = "Real"
        case Int([r, c]) if theory is LIA:
            kind = "Int"
        case Bool([r, c]) if theory is LRA or theory is LIA:
            kind = "Bool"
        case _:
            raise TypeError(f"{theory.__name__} has no {name} on {sort}")
    variant = f"{name}{kind}" if name in _PER_KIND else name
    return _variant(theory, variant, name, sort)(*args, r, c)


def linear(theory, weight, bias, sort):
    """The affine map ``x ↦ weight·x + bias`` of ``theory`` on ``x`` of the sort ``sort``."""
    match sort:
        case Real([_, batch]) if theory is LRA:
            return LRA.Linear(weight, bias, batch)
        case TReal([_, batch], rank) if theory is LRA:
            return LRA.TLinear(weight, bias, batch, rank)
        case Int([_, batch]) if theory is LIA:
            return LIA.Linear(weight, bias, batch)
    raise TypeError(f"{theory.__name__} has no Linear on {sort}")


def bv_const(tensor, sort):
    """The bit-vector constant ``tensor`` written to a wire of the sort ``sort``."""
    match sort:
        case BitVec(width, _):
            return BV.Const(tensor, width)
    raise TypeError(f"BV has no Const of {sort}")


def _variant(theory, variant: str, name: str, sort):
    cls = getattr(theory, variant, None)
    if cls is None:
        raise TypeError(f"{theory.__name__} has no {name} on {sort}")
    return cls
