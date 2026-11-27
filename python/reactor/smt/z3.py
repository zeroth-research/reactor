"""Translate terms to Z3 expressions.

Each wire value is a numpy array of Z3 expressions (the tensor, as a matrix). Dispatch is
by `match` on the op's class; reals and integers share cases. Words are Z3 bit-vectors of
their width, whose operations follow SMT-LIB2, as those of `reactor.eval` do.
"""

import z3 as _z3
import torch
from ..reactor import Havoc, Skip
from ..real import Real
from ..int import Int
from ..bool import Bool
from ..word import Word
from .. import real as _rl
from ..real import broadcast as _rb
from .. import int as _ii
from ..int import broadcast as _ib
from .. import bool as _bl
from ..bool import broadcast as _bb
from ..bool import reduce as _br
from .. import word as _ww
from ..word import broadcast as _wb
from ..word import signed as _ws
from ..word import unsigned as _wu
from .. import zero as _zz
import numpy as _np
import operator


__all__ = ["fresh", "interpret", "eval"]


def fresh(dtype, prefix):
    match dtype:
        case Real([n, m]):
            return _np.array([[_z3.FreshReal(f'{prefix}[{i},{j}]') for j in range(m)] for i in range(n)], dtype=object)
        case Bool([n, m]):
            return _np.array([[_z3.FreshBool(f'{prefix}[{i},{j}]') for j in range(m)] for i in range(n)], dtype=object)
        case Int([n, m]):
            return _np.array([[_z3.FreshInt(f'{prefix}[{i},{j}]') for j in range(m)] for i in range(n)], dtype=object)
        case Word(w, [n, m]):
            sort = _z3.BitVecSort(w)
            return _np.array([[_z3.FreshConst(sort, f'{prefix}[{i},{j}]') for j in range(m)] for i in range(n)], dtype=object)
        case _:
            raise NotImplementedError(f"fresh not implemented for dtype {dtype}")


def _zeros(dtype):
    match dtype:
        case Real([n, m]):
            return _np.full((n, m), _z3.RealVal(0), dtype=object)
        case Int([n, m]):
            return _np.full((n, m), _z3.IntVal(0), dtype=object)
        case Bool([n, m]):
            return _np.full((n, m), _z3.BoolVal(False), dtype=object)
        case Word(w, [n, m]):
            return _np.full((n, m), _z3.BitVecVal(0, w), dtype=object)
        case _:
            raise NotImplementedError(f"zero not implemented for dtype {dtype}")


def _argmax(x, val):
    """The index of the first maximal entry of the matrix `x`, counted row by row, written
    by `val` as a `[1, 1]` matrix: an entry is maximal when it is at least every other."""
    xs = list(x.flat)
    index = val(len(xs) - 1)
    for i in reversed(range(len(xs) - 1)):
        index = _z3.If(_z3.And([xs[i] >= y for y in xs]), val(i), index)
    return _np.array([[index]], dtype=object)


def _abs(x):
    """The absolute value of the word `x`, read in two's complement."""
    return _z3.If(x < 0, -x, x)


def interpret(itype):
    match itype:
        # structural
        case Skip():
            return lambda x: x
        case Havoc(dtype):
            return lambda: fresh(dtype, "HAVOC")
        case _zz.Zero():
            # the singleton sort is the rate of change of discrete values, which are
            # never integrated: nothing encodes it
            raise RuntimeError("Zero writes the singleton sort, which is never encoded")
        case _rl.Zero(Real(shape)):
            return lambda: _zeros(Real(shape))

        # constants
        case _bl.Truth(t):
            return lambda: _np.vectorize(_z3.BoolVal, otypes=[object])(t.detach().numpy())
        case _rl.Num(t):
            return lambda: _np.vectorize(_z3.RealVal, otypes=[object])(t.detach().numpy())
        case _ii.Num(t):
            return lambda: _np.vectorize(_z3.IntVal, otypes=[object])(t.detach().numpy())
        case _ww.Num(t, Word(w)):
            return lambda: _np.vectorize(lambda v: _z3.BitVecVal(int(v), w), otypes=[object])(t.detach().numpy())

        # arithmetic
        case _rl.Sub() | _ii.Sub():
            return _np.frompyfunc(operator.sub, 2, 1)
        case _rl.Add() | _ii.Add():
            return _np.frompyfunc(operator.add, 2, 1)
        case _rl.Mul() | _ii.Mul():
            return _np.frompyfunc(operator.mul, 2, 1)
        case _rl.Div() | _ii.Div():
            return _np.frompyfunc(operator.truediv, 2, 1)
        case _ii.Mod():
            return _np.frompyfunc(operator.mod, 2, 1)
        case _rl.MatMul() | _ii.MatMul():
            return lambda x, y: x @ y
        case _rl.Linear(a) | _ii.Linear(a):
            return lambda x: a.detach().numpy() @ x
        case _rl.Affine(weight, bias) | _ii.Affine(weight, bias):
            return lambda x: weight.detach().numpy() @ x + bias.detach().numpy()
        case _rl.Scale(c) | _ii.Scale(c):
            return lambda x: c.detach().numpy() * x

        # comparisons (produce Bool)
        case _rl.Eq() | _ii.Eq() | _bl.Eq():
            return _np.frompyfunc(operator.eq, 2, 1)
        case _rl.Ne() | _ii.Ne() | _bl.Ne():
            return _np.frompyfunc(operator.ne, 2, 1)
        case _rl.Lt() | _ii.Lt():
            return _np.frompyfunc(operator.lt, 2, 1)
        case _rl.Le() | _ii.Le():
            return _np.frompyfunc(operator.le, 2, 1)
        case _rl.Gt() | _ii.Gt():
            return _np.frompyfunc(operator.gt, 2, 1)
        case _rl.Ge() | _ii.Ge():
            return _np.frompyfunc(operator.ge, 2, 1)

        # logical
        case _bl.And():
            return _np.frompyfunc(lambda x, y: _z3.And(x, y), 2, 1)
        case _bl.Or():
            return _np.frompyfunc(lambda x, y: _z3.Or(x, y), 2, 1)
        case _bl.Not():
            return _np.frompyfunc(lambda x: _z3.Not(x), 1, 1)
        case _bl.Xor():
            return _np.frompyfunc(lambda x, y: _z3.Xor(x, y), 2, 1)
        case _br.And(axis):
            # the conjunction folded along the axis, which it keeps, of length 1
            return lambda x: _np.frompyfunc(_z3.And, 2, 1).reduce(x, axis=axis, keepdims=True)
        case _br.Or(axis):
            return lambda x: _np.frompyfunc(_z3.Or, 2, 1).reduce(x, axis=axis, keepdims=True)

        # control flow
        case (
            _rl.Ite()
            | _ii.Ite()
            | _bl.Ite()
            | _ww.Ite()
            | _rb.Ite()
            | _ib.Ite()
            | _bb.Ite()
            | _wb.Ite()
        ):
            # pointwise, or by a scalar condition, broadcast to the whole matrix
            return _np.frompyfunc(lambda c, tt, ff: _z3.If(c, tt, ff), 3, 1)

        # neural-ish / aggregate
        case _rl.Neg() | _ii.Neg():
            return _np.frompyfunc(lambda x: -x, 1, 1)
        case _rl.Abs() | _ii.Abs():
            return _np.frompyfunc(lambda x: _z3.If(x >= 0, x, -x), 1, 1)
        case _rl.ReLU() | _ii.ReLU():
            return _np.frompyfunc(lambda x: _z3.If(x > 0, x, 0), 1, 1)
        case _rl.Min() | _ii.Min():
            return _np.frompyfunc(lambda x, y: _z3.If(x <= y, x, y), 2, 1)
        case _rl.Max() | _ii.Max():
            return _np.frompyfunc(lambda x, y: _z3.If(x >= y, x, y), 2, 1)

        case _rl.Transpose() | _ii.Transpose():
            return _np.transpose
        case _rl.Argmax():
            return lambda x: _argmax(x, _z3.RealVal)
        case _ii.Argmax():
            return lambda x: _argmax(x, _z3.IntVal)

        # words: comparisons, unsigned by default, signed as Z3's own
        case _ww.Eq():
            return _np.frompyfunc(operator.eq, 2, 1)
        case _ww.Ne():
            return _np.frompyfunc(operator.ne, 2, 1)
        case _wu.Lt():
            return _np.frompyfunc(_z3.ULT, 2, 1)
        case _wu.Le():
            return _np.frompyfunc(_z3.ULE, 2, 1)
        case _wu.Gt():
            return _np.frompyfunc(_z3.UGT, 2, 1)
        case _wu.Ge():
            return _np.frompyfunc(_z3.UGE, 2, 1)
        case _ws.Lt():
            return _np.frompyfunc(operator.lt, 2, 1)
        case _ws.Le():
            return _np.frompyfunc(operator.le, 2, 1)
        case _ws.Gt():
            return _np.frompyfunc(operator.gt, 2, 1)
        case _ws.Ge():
            return _np.frompyfunc(operator.ge, 2, 1)

        # words: arithmetic modulo 2^w
        case _ww.Add():
            return _np.frompyfunc(operator.add, 2, 1)
        case _ww.Sub():
            return _np.frompyfunc(operator.sub, 2, 1)
        case _ww.Mul():
            return _np.frompyfunc(operator.mul, 2, 1)
        case _ww.Neg():
            return _np.frompyfunc(operator.neg, 1, 1)
        case _ws.Abs():
            return _np.frompyfunc(_abs, 1, 1)
        case _wu.Div():
            return _np.frompyfunc(_z3.UDiv, 2, 1)
        case _wu.Rem():
            return _np.frompyfunc(_z3.URem, 2, 1)
        case _ws.Div():
            # Z3's `/` on bit-vectors is the signed division
            return _np.frompyfunc(operator.truediv, 2, 1)
        case _ws.Rem():
            return _np.frompyfunc(_z3.SRem, 2, 1)
        case _ws.Mod():
            # Z3's `%` on bit-vectors is the signed modulus
            return _np.frompyfunc(operator.mod, 2, 1)

        # words: bitwise
        case _ww.And():
            return _np.frompyfunc(operator.and_, 2, 1)
        case _ww.Or():
            return _np.frompyfunc(operator.or_, 2, 1)
        case _ww.Xor():
            return _np.frompyfunc(operator.xor, 2, 1)
        case _ww.Not():
            return _np.frompyfunc(operator.invert, 1, 1)
        case _ww.Nand():
            return _np.frompyfunc(lambda x, y: ~(x & y), 2, 1)
        case _ww.Nor():
            return _np.frompyfunc(lambda x, y: ~(x | y), 2, 1)
        case _ww.Xnor():
            return _np.frompyfunc(lambda x, y: ~(x ^ y), 2, 1)

        # words: shifts, rotations and resizing
        case _ww.Shl():
            return _np.frompyfunc(operator.lshift, 2, 1)
        case _wu.Shr():
            return _np.frompyfunc(_z3.LShR, 2, 1)
        case _ws.Shr():
            # Z3's `>>` on bit-vectors is the arithmetic shift
            return _np.frompyfunc(operator.rshift, 2, 1)
        case _ww.RotateLeft(i):
            return _np.frompyfunc(lambda x: _z3.RotateLeft(x, i), 1, 1)
        case _ww.RotateRight(i):
            return _np.frompyfunc(lambda x: _z3.RotateRight(x, i), 1, 1)
        case _ww.Concat():
            return _np.frompyfunc(_z3.Concat, 2, 1)
        case _ww.Extract(i, j):
            return _np.frompyfunc(lambda x: _z3.Extract(i, j, x), 1, 1)
        case _ww.Repeat(i):
            return _np.frompyfunc(lambda x: _z3.RepeatBitVec(i, x), 1, 1)
        case _wu.Extend(i):
            return _np.frompyfunc(lambda x: _z3.ZeroExt(i, x), 1, 1)
        case _ws.Extend(i):
            return _np.frompyfunc(lambda x: _z3.SignExt(i, x), 1, 1)
        case _:
            raise NotImplementedError(f"itype: {itype}")


def eval(itype, read):
    """Translate a single op to Z3 expressions."""
    if not isinstance(read, list) or not all(isinstance(r, _np.ndarray) for r in read) or \
            not all(isinstance(a, _z3.AstRef) for r in read for a in r.flat):
        raise RuntimeError(f"read={read} must be a list of numpy arrays of z3 expressions")
    if not all(r.ndim == 2 for r in read):
        raise RuntimeError(
            f"read shapes {[r.shape for r in read]} must all be 2-dimensional to match "
            "their [n,m] sorts; a 1-D operand broadcasts incorrectly in matrix ops"
        )

    result = interpret(itype)(*read)
    return [result]
