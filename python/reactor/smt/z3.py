"""Translate terms to Z3 expressions.

Each wire value is a numpy array of Z3 expressions (the tensor, as a matrix). Dispatch is
by `match` on the op's class; reals and integers share cases. Words are Z3 bit-vectors of
their width, whose operations follow SMT-LIB2, as those of `reactor.eval` do.
"""

import z3 as _z3
import torch
from ..matrix import Havoc, Skip
from ..matrix.real import Real
from ..matrix.int import Int
from ..matrix.bool import Bool
from ..matrix.word import Word
import numpy as _np
import operator

from ..reactor import matrix
from ..zero import Zero


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
        case Zero():
            # the singleton sort is the rate of change of discrete values, which are
            # never integrated: nothing encodes it
            raise RuntimeError("Zero writes the singleton sort, which is never encoded")
        case matrix.real.Zero(Real(shape)):
            return lambda: _zeros(Real(shape))

        # constants
        case matrix.bool.Truth(t):
            return lambda: _np.vectorize(_z3.BoolVal, otypes=[object])(t.detach().numpy())
        case matrix.real.Num(t):
            return lambda: _np.vectorize(_z3.RealVal, otypes=[object])(t.detach().numpy())
        case matrix.int.Num(t):
            return lambda: _np.vectorize(_z3.IntVal, otypes=[object])(t.detach().numpy())
        case matrix.word.Num(t, Word(w)):
            return lambda: _np.vectorize(lambda v: _z3.BitVecVal(int(v), w), otypes=[object])(t.detach().numpy())

        # arithmetic
        case matrix.real.Sub() | matrix.int.Sub():
            return _np.frompyfunc(operator.sub, 2, 1)
        case matrix.real.Add() | matrix.int.Add():
            return _np.frompyfunc(operator.add, 2, 1)
        case matrix.real.Mul() | matrix.int.Mul():
            return _np.frompyfunc(operator.mul, 2, 1)
        case matrix.real.Div() | matrix.int.Div():
            return _np.frompyfunc(operator.truediv, 2, 1)
        case matrix.int.Mod():
            return _np.frompyfunc(operator.mod, 2, 1)
        case matrix.real.MatMul() | matrix.int.MatMul():
            return lambda x, y: x @ y
        case matrix.real.Linear(a) | matrix.int.Linear(a):
            return lambda x: a.detach().numpy() @ x
        case matrix.real.Affine(weight, bias) | matrix.int.Affine(weight, bias):
            return lambda x: weight.detach().numpy() @ x + bias.detach().numpy()
        case matrix.real.Scale(c) | matrix.int.Scale(c):
            return lambda x: c.detach().numpy() * x

        # comparisons (produce Bool)
        case matrix.real.Eq() | matrix.int.Eq() | matrix.bool.Eq():
            return _np.frompyfunc(operator.eq, 2, 1)
        case matrix.real.Ne() | matrix.int.Ne() | matrix.bool.Ne():
            return _np.frompyfunc(operator.ne, 2, 1)
        case matrix.real.Lt() | matrix.int.Lt():
            return _np.frompyfunc(operator.lt, 2, 1)
        case matrix.real.Le() | matrix.int.Le():
            return _np.frompyfunc(operator.le, 2, 1)
        case matrix.real.Gt() | matrix.int.Gt():
            return _np.frompyfunc(operator.gt, 2, 1)
        case matrix.real.Ge() | matrix.int.Ge():
            return _np.frompyfunc(operator.ge, 2, 1)

        # logical
        case matrix.bool.And():
            return _np.frompyfunc(lambda x, y: _z3.And(x, y), 2, 1)
        case matrix.bool.Or():
            return _np.frompyfunc(lambda x, y: _z3.Or(x, y), 2, 1)
        case matrix.bool.Not():
            return _np.frompyfunc(lambda x: _z3.Not(x), 1, 1)
        case matrix.bool.Xor():
            return _np.frompyfunc(lambda x, y: _z3.Xor(x, y), 2, 1)

        # control flow
        case (
            matrix.real.Ite()
            | matrix.int.Ite()
            | matrix.bool.Ite()
            | matrix.word.Ite()
            | matrix.real.broadcast.Ite()
            | matrix.int.broadcast.Ite()
            | matrix.bool.broadcast.Ite()
            | matrix.word.broadcast.Ite()
        ):
            # pointwise, or by a scalar condition, broadcast to the whole matrix
            return _np.frompyfunc(lambda c, tt, ff: _z3.If(c, tt, ff), 3, 1)

        # neural-ish / aggregate
        case matrix.real.Neg() | matrix.int.Neg():
            return _np.frompyfunc(lambda x: -x, 1, 1)
        case matrix.real.Abs() | matrix.int.Abs():
            return _np.frompyfunc(lambda x: _z3.If(x >= 0, x, -x), 1, 1)
        case matrix.real.ReLU() | matrix.int.ReLU():
            return _np.frompyfunc(lambda x: _z3.If(x > 0, x, 0), 1, 1)
        case matrix.real.Min() | matrix.int.Min():
            return _np.frompyfunc(lambda x, y: _z3.If(x <= y, x, y), 2, 1)
        case matrix.real.Max() | matrix.int.Max():
            return _np.frompyfunc(lambda x, y: _z3.If(x >= y, x, y), 2, 1)

        case matrix.real.Transpose() | matrix.int.Transpose():
            return _np.transpose
        case matrix.real.Argmax():
            return lambda x: _argmax(x, _z3.RealVal)
        case matrix.int.Argmax():
            return lambda x: _argmax(x, _z3.IntVal)

        # words: comparisons, unsigned by default, signed as Z3's own
        case matrix.word.Eq():
            return _np.frompyfunc(operator.eq, 2, 1)
        case matrix.word.Ne():
            return _np.frompyfunc(operator.ne, 2, 1)
        case matrix.word.unsigned.Lt():
            return _np.frompyfunc(_z3.ULT, 2, 1)
        case matrix.word.unsigned.Le():
            return _np.frompyfunc(_z3.ULE, 2, 1)
        case matrix.word.unsigned.Gt():
            return _np.frompyfunc(_z3.UGT, 2, 1)
        case matrix.word.unsigned.Ge():
            return _np.frompyfunc(_z3.UGE, 2, 1)
        case matrix.word.signed.Lt():
            return _np.frompyfunc(operator.lt, 2, 1)
        case matrix.word.signed.Le():
            return _np.frompyfunc(operator.le, 2, 1)
        case matrix.word.signed.Gt():
            return _np.frompyfunc(operator.gt, 2, 1)
        case matrix.word.signed.Ge():
            return _np.frompyfunc(operator.ge, 2, 1)

        # words: arithmetic modulo 2^w
        case matrix.word.Add():
            return _np.frompyfunc(operator.add, 2, 1)
        case matrix.word.Sub():
            return _np.frompyfunc(operator.sub, 2, 1)
        case matrix.word.Mul():
            return _np.frompyfunc(operator.mul, 2, 1)
        case matrix.word.Neg():
            return _np.frompyfunc(operator.neg, 1, 1)
        case matrix.word.signed.Abs():
            return _np.frompyfunc(_abs, 1, 1)
        case matrix.word.unsigned.Div():
            return _np.frompyfunc(_z3.UDiv, 2, 1)
        case matrix.word.unsigned.Rem():
            return _np.frompyfunc(_z3.URem, 2, 1)
        case matrix.word.signed.Div():
            # Z3's `/` on bit-vectors is the signed division
            return _np.frompyfunc(operator.truediv, 2, 1)
        case matrix.word.signed.Rem():
            return _np.frompyfunc(_z3.SRem, 2, 1)
        case matrix.word.signed.Mod():
            # Z3's `%` on bit-vectors is the signed modulus
            return _np.frompyfunc(operator.mod, 2, 1)

        # words: bitwise
        case matrix.word.And():
            return _np.frompyfunc(operator.and_, 2, 1)
        case matrix.word.Or():
            return _np.frompyfunc(operator.or_, 2, 1)
        case matrix.word.Xor():
            return _np.frompyfunc(operator.xor, 2, 1)
        case matrix.word.Not():
            return _np.frompyfunc(operator.invert, 1, 1)
        case matrix.word.Nand():
            return _np.frompyfunc(lambda x, y: ~(x & y), 2, 1)
        case matrix.word.Nor():
            return _np.frompyfunc(lambda x, y: ~(x | y), 2, 1)
        case matrix.word.Xnor():
            return _np.frompyfunc(lambda x, y: ~(x ^ y), 2, 1)

        # words: shifts, rotations and resizing
        case matrix.word.Shl():
            return _np.frompyfunc(operator.lshift, 2, 1)
        case matrix.word.unsigned.Shr():
            return _np.frompyfunc(_z3.LShR, 2, 1)
        case matrix.word.signed.Shr():
            # Z3's `>>` on bit-vectors is the arithmetic shift
            return _np.frompyfunc(operator.rshift, 2, 1)
        case matrix.word.RotateLeft(i):
            return _np.frompyfunc(lambda x: _z3.RotateLeft(x, i), 1, 1)
        case matrix.word.RotateRight(i):
            return _np.frompyfunc(lambda x: _z3.RotateRight(x, i), 1, 1)
        case matrix.word.Concat():
            return _np.frompyfunc(_z3.Concat, 2, 1)
        case matrix.word.Extract(i, j):
            return _np.frompyfunc(lambda x: _z3.Extract(i, j, x), 1, 1)
        case matrix.word.Repeat(i):
            return _np.frompyfunc(lambda x: _z3.RepeatBitVec(i, x), 1, 1)
        case matrix.word.unsigned.Extend(i):
            return _np.frompyfunc(lambda x: _z3.ZeroExt(i, x), 1, 1)
        case matrix.word.signed.Extend(i):
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
