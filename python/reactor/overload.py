"""Overloaded operations: the operation of each theory that a generic name, such as
``add``, denotes on a sort.

The operations of ``reactor.op`` are those of one theory on given sorts, as
``op.real.Add(m, n)`` on ``[m, n]`` real matrices. The frontends instead build terms
by a generic name and the sort their operands have, as Python's operators do; each
function here returns the operation of that name on that sort:

    add(Real([2, 3]))       # op.real.Add(2, 3)
    add(Real([2, 3], 1))    # op.real.TAdd(2, 3, 1), on rates of change
    add(Word(8, [2, 3]))    # op.word.Add(8, 2, 3)
    lt(Word(8, [1, 1]), signed=True)  # op.word.SLt(8, 1, 1)

The sort is that of the operands, or of the result where there are none (``const``,
``havoc``, ``zero``, ``uninterpreted``). A name with no operation on a sort raises
``TheoryError``, e.g. ``mul`` on rates of change, or ``matmul`` on words.
"""

from . import op
from .sort import Real, Int, Bool, Word, Zero


class TheoryError(Exception):
    """An operation cannot be expressed in the selected theory."""


def _none(name, sort):
    return TheoryError(f"no {name} on {sort}")


def _by_sort(name, sort, *, real=None, rate=None, int_=None, bool_=None, word=None):
    """The operation of ``name`` on ``sort``, from the constructor of each theory:
    ``real``, ``int_`` and ``bool_`` take ``(m, n)``, ``word`` takes ``(w, m, n)``, and
    ``rate``, on the rates of change of degree ``d > 0`` of reals, ``(m, n, d)``."""
    match sort:
        case Real([m, n], 0) if real is not None:
            return real(m, n)
        case Real([m, n], d) if d > 0 and rate is not None:
            return rate(m, n, d)
        case Int([m, n]) if int_ is not None:
            return int_(m, n)
        case Bool([m, n]) if bool_ is not None:
            return bool_(m, n)
        case Word(w, [m, n]) if word is not None:
            return word(w, m, n)
    raise _none(name, sort)


# --- every theory --------------------------------------------------------------------


def id_(sort):
    return _by_sort("Id", sort, real=op.real.Id, rate=op.real.TId, int_=op.int.Id,
                    bool_=op.bool.Id, word=op.word.Id)


def ite(sort):
    """If-then-else on branches of ``sort``; the guard is a single boolean."""
    return _by_sort("Ite", sort, real=op.real.Ite, rate=op.real.TIte, int_=op.int.Ite,
                    bool_=op.bool.Ite, word=op.word.Ite)


def havoc(sort):
    return _by_sort("Havoc", sort, real=op.real.Havoc, int_=op.int.Havoc,
                    bool_=op.bool.Havoc, word=op.word.Havoc)


def zero(sort):
    """The zero rate of change of ``sort``, the sort of a derivative wire: of degree
    ``d > 0`` for reals, and ``Zero`` for the discrete sorts."""
    match sort:
        case Zero():
            return op.bool.Zero()
        case Real([m, n], d):
            return op.real.Zero(m, n, d)
    raise _none("Zero", sort)


def const(tensor, sort):
    """The constant ``tensor``, written to a wire of ``sort``."""
    match sort:
        case Real(_, 0):
            return op.real.Const(tensor)
        case Int(_):
            return op.int.Const(tensor)
        case Bool(_):
            return op.bool.Const(tensor)
        case Word(w, _):
            return op.word.Const(tensor, w)
    raise _none("Const", sort)


def uninterpreted(name, sort):
    """The symbol ``name``, of unknown meaning, written to a wire of ``sort``."""
    match sort:
        case Real([m, n], 0):
            return op.real.Uninterpreted(name, m, n)
        case Int([m, n]):
            return op.int.Uninterpreted(name, m, n)
        case Bool([m, n]):
            return op.bool.Uninterpreted(name, m, n)
        case Word(w, [m, n]):
            return op.word.Uninterpreted(name, w, m, n)
    raise _none("Uninterpreted", sort)


# --- arithmetic ------------------------------------------------------------------------


def add(sort):
    return _by_sort("Add", sort, real=op.real.Add, rate=op.real.TAdd, int_=op.int.Add,
                    word=op.word.Add)


def sub(sort):
    return _by_sort("Sub", sort, real=op.real.Sub, rate=op.real.TSub, int_=op.int.Sub,
                    word=op.word.Sub)


def mul(sort):
    """Pointwise multiplication: non-linear, unlike ``linear``."""
    return _by_sort("Mul", sort, real=op.real.Mul, int_=op.int.Mul, word=op.word.Mul)


def div(sort, *, signed=False):
    """Division: real division, Euclidean division of integers, and the division of
    words read ``signed`` or unsigned."""
    return _by_sort("Div", sort, real=op.real.Div, int_=op.int.Div,
                    word=op.word.SDiv if signed else op.word.UDiv)


def mod(sort, *, signed=False):
    """The remainder of ``div``: Euclidean for integers; for words, of the sign of
    the divisor if ``signed`` (as Python's ``%``), and unsigned otherwise."""
    return _by_sort("Mod", sort, int_=op.int.Mod,
                    word=op.word.SMod if signed else op.word.URem)


def neg(sort):
    return _by_sort("Neg", sort, word=op.word.Neg)


def linear(a, b, sort):
    """The affine map ``x ↦ a·x + b``, on ``x`` of ``sort``."""
    match sort:
        case Real(_, 0):
            return op.real.Linear(a, b)
        case Real(_, d):
            return op.real.TLinear(a, b, d)
        case Int(_):
            return op.int.Linear(a, b)
    raise _none("Linear", sort)


def matmul(left, right):
    """The matrix product of matrices of the sorts ``left`` and ``right``."""
    match left, right:
        case Real([m, n], 0), Real([_, k], 0):
            return op.real.MatMul(m, n, k)
        case Int([m, n]), Int([_, k]):
            return op.int.MatMul(m, n, k)
    raise TheoryError(f"no MatMul on {left} and {right}")


def relu(sort):
    return _by_sort("ReLU", sort, real=op.real.ReLU, int_=op.int.ReLU)


def min_(sort):
    return _by_sort("Min", sort, real=op.real.Min, int_=op.int.Min)


def max_(sort):
    return _by_sort("Max", sort, real=op.real.Max, int_=op.int.Max)


def argmax(sort):
    return _by_sort("Argmax", sort, real=op.real.Argmax, int_=op.int.Argmax)


def transpose(sort):
    return _by_sort("Transpose", sort, real=op.real.Transpose, rate=op.real.TTranspose,
                    int_=op.int.Transpose)


# --- comparisons, which write booleans -------------------------------------------------


def _cmp(name, sort, real, int_, signed_word, unsigned_word, signed):
    return _by_sort(name, sort, real=real, int_=int_,
                    word=signed_word if signed else unsigned_word)


def lt(sort, *, signed=False):
    """``x < y``; on words, read ``signed`` or unsigned."""
    return _cmp("Lt", sort, op.real.Lt, op.int.Lt, op.word.SLt, op.word.ULt, signed)


def le(sort, *, signed=False):
    return _cmp("Le", sort, op.real.Le, op.int.Le, op.word.SLe, op.word.ULe, signed)


def gt(sort, *, signed=False):
    return _cmp("Gt", sort, op.real.Gt, op.int.Gt, op.word.SGt, op.word.UGt, signed)


def ge(sort, *, signed=False):
    return _cmp("Ge", sort, op.real.Ge, op.int.Ge, op.word.SGe, op.word.UGe, signed)


def eq(sort):
    """``x = y``; booleans have no equality: it is ``not_(xor_(x, y))``."""
    return _by_sort("Eq", sort, real=op.real.Eq, int_=op.int.Eq, word=op.word.Eq)


def ne(sort):
    """``x ≠ y``; booleans have no disequality: it is ``xor_(x, y)``."""
    return _by_sort("Ne", sort, real=op.real.Ne, int_=op.int.Ne, word=op.word.Ne)


# --- logic: on booleans, and bitwise on words ------------------------------------------


def and_(sort):
    return _by_sort("And", sort, bool_=op.bool.And, word=op.word.And)


def or_(sort):
    return _by_sort("Or", sort, bool_=op.bool.Or, word=op.word.Or)


def xor_(sort):
    return _by_sort("Xor", sort, bool_=op.bool.Xor, word=op.word.Xor)


def not_(sort):
    return _by_sort("Not", sort, bool_=op.bool.Not, word=op.word.Not)


# --- words -----------------------------------------------------------------------------


def extract(sort, high: int, low: int):
    """The bits ``high`` down to ``low``, both included, of words of ``sort``."""
    match sort:
        case Word(w, [m, n]):
            return op.word.Extract(high, low, w, m, n)
    raise _none("Extract", sort)


def extend(sort, extra: int, *, signed=False):
    """Words of ``sort`` extended by ``extra`` bits: with their sign bit if ``signed``,
    and with zeros otherwise."""
    match sort:
        case Word(w, [m, n]):
            ext = op.word.SignExtend if signed else op.word.ZeroExtend
            return ext(extra, w, m, n)
    raise _none("Extend", sort)
