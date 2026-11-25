"""Translate terms to Z3 expressions.

Each wire value is a numpy array of Z3 expressions (the tensor, as a matrix). Dispatch is
by `match` on the op's class; reals and integers share cases. Words are Z3 bit-vectors of
their width, whose operations follow SMT-LIB2, as those of `reactor.eval` do.
"""

import z3 as _z3
import torch
from ..tensor import Havoc, Skip
from ..tensor.sort import Real, Int, Bool, Word
import numpy as _np
import operator

from ..reactor import tensor, Zero


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
        case tensor.real.Zero(Real(shape)):
            return lambda: _zeros(Real(shape))

        # constants
        case tensor.bool.Const(t):
            return lambda: _np.vectorize(_z3.BoolVal, otypes=[object])(t.detach().numpy())
        case tensor.real.Const(t):
            return lambda: _np.vectorize(_z3.RealVal, otypes=[object])(t.detach().numpy())
        case tensor.int.Const(t):
            return lambda: _np.vectorize(_z3.IntVal, otypes=[object])(t.detach().numpy())
        case tensor.word.Const(t, Word(w)):
            return lambda: _np.vectorize(lambda v: _z3.BitVecVal(int(v), w), otypes=[object])(t.detach().numpy())

        # arithmetic
        case tensor.real.Sub() | tensor.int.Sub():
            return _np.frompyfunc(operator.sub, 2, 1)
        case tensor.real.Add() | tensor.int.Add():
            return _np.frompyfunc(operator.add, 2, 1)
        case tensor.real.Mul() | tensor.int.Mul():
            return _np.frompyfunc(operator.mul, 2, 1)
        case tensor.real.Div() | tensor.int.Div():
            return _np.frompyfunc(operator.truediv, 2, 1)
        case tensor.int.Mod():
            return _np.frompyfunc(operator.mod, 2, 1)
        case tensor.real.MatMul() | tensor.int.MatMul():
            return lambda x, y: x @ y
        case tensor.real.Linear(a) | tensor.int.Linear(a):
            return lambda x: a.detach().numpy() @ x
        case tensor.real.Affine(weight, bias) | tensor.int.Affine(weight, bias):
            return lambda x: weight.detach().numpy() @ x + bias.detach().numpy()
        case tensor.real.Scale(c) | tensor.int.Scale(c):
            return lambda x: c.detach().numpy() * x

        # comparisons (produce Bool)
        case tensor.real.Eq() | tensor.int.Eq() | tensor.bool.Eq():
            return _np.frompyfunc(operator.eq, 2, 1)
        case tensor.real.Ne() | tensor.int.Ne() | tensor.bool.Ne():
            return _np.frompyfunc(operator.ne, 2, 1)
        case tensor.real.Lt() | tensor.int.Lt():
            return _np.frompyfunc(operator.lt, 2, 1)
        case tensor.real.Le() | tensor.int.Le():
            return _np.frompyfunc(operator.le, 2, 1)
        case tensor.real.Gt() | tensor.int.Gt():
            return _np.frompyfunc(operator.gt, 2, 1)
        case tensor.real.Ge() | tensor.int.Ge():
            return _np.frompyfunc(operator.ge, 2, 1)

        # logical
        case tensor.bool.And():
            return _np.frompyfunc(lambda x, y: _z3.And(x, y), 2, 1)
        case tensor.bool.Or():
            return _np.frompyfunc(lambda x, y: _z3.Or(x, y), 2, 1)
        case tensor.bool.Not():
            return _np.frompyfunc(lambda x: _z3.Not(x), 1, 1)
        case tensor.bool.Xor():
            return _np.frompyfunc(lambda x, y: _z3.Xor(x, y), 2, 1)

        # control flow
        case tensor.real.Ite() | tensor.int.Ite() | tensor.bool.Ite() | tensor.word.Ite():
            return _np.frompyfunc(lambda c, tt, ff: _z3.If(c, tt, ff), 3, 1)

        # neural-ish / aggregate
        case tensor.real.Neg() | tensor.int.Neg():
            return _np.frompyfunc(lambda x: -x, 1, 1)
        case tensor.real.Abs() | tensor.int.Abs():
            return _np.frompyfunc(lambda x: _z3.If(x >= 0, x, -x), 1, 1)
        case tensor.real.ReLU() | tensor.int.ReLU():
            return _np.frompyfunc(lambda x: _z3.If(x > 0, x, 0), 1, 1)
        case tensor.real.Min() | tensor.int.Min():
            return _np.frompyfunc(lambda x, y: _z3.If(x <= y, x, y), 2, 1)
        case tensor.real.Max() | tensor.int.Max():
            return _np.frompyfunc(lambda x, y: _z3.If(x >= y, x, y), 2, 1)

        case tensor.real.Transpose() | tensor.int.Transpose():
            return _np.transpose
        case tensor.real.Argmax():
            return lambda x: _argmax(x, _z3.RealVal)
        case tensor.int.Argmax():
            return lambda x: _argmax(x, _z3.IntVal)

        # words: comparisons, unsigned by default, signed as Z3's own
        case tensor.word.Eq():
            return _np.frompyfunc(operator.eq, 2, 1)
        case tensor.word.Ne():
            return _np.frompyfunc(operator.ne, 2, 1)
        case tensor.word.unsigned.Lt():
            return _np.frompyfunc(_z3.ULT, 2, 1)
        case tensor.word.unsigned.Le():
            return _np.frompyfunc(_z3.ULE, 2, 1)
        case tensor.word.unsigned.Gt():
            return _np.frompyfunc(_z3.UGT, 2, 1)
        case tensor.word.unsigned.Ge():
            return _np.frompyfunc(_z3.UGE, 2, 1)
        case tensor.word.signed.Lt():
            return _np.frompyfunc(operator.lt, 2, 1)
        case tensor.word.signed.Le():
            return _np.frompyfunc(operator.le, 2, 1)
        case tensor.word.signed.Gt():
            return _np.frompyfunc(operator.gt, 2, 1)
        case tensor.word.signed.Ge():
            return _np.frompyfunc(operator.ge, 2, 1)

        # words: arithmetic modulo 2^w
        case tensor.word.Add():
            return _np.frompyfunc(operator.add, 2, 1)
        case tensor.word.Sub():
            return _np.frompyfunc(operator.sub, 2, 1)
        case tensor.word.Mul():
            return _np.frompyfunc(operator.mul, 2, 1)
        case tensor.word.Neg():
            return _np.frompyfunc(operator.neg, 1, 1)
        case tensor.word.signed.Abs():
            return _np.frompyfunc(_abs, 1, 1)
        case tensor.word.unsigned.Div():
            return _np.frompyfunc(_z3.UDiv, 2, 1)
        case tensor.word.unsigned.Rem():
            return _np.frompyfunc(_z3.URem, 2, 1)
        case tensor.word.signed.Div():
            # Z3's `/` on bit-vectors is the signed division
            return _np.frompyfunc(operator.truediv, 2, 1)
        case tensor.word.signed.Rem():
            return _np.frompyfunc(_z3.SRem, 2, 1)
        case tensor.word.signed.Mod():
            # Z3's `%` on bit-vectors is the signed modulus
            return _np.frompyfunc(operator.mod, 2, 1)

        # words: bitwise
        case tensor.word.And():
            return _np.frompyfunc(operator.and_, 2, 1)
        case tensor.word.Or():
            return _np.frompyfunc(operator.or_, 2, 1)
        case tensor.word.Xor():
            return _np.frompyfunc(operator.xor, 2, 1)
        case tensor.word.Not():
            return _np.frompyfunc(operator.invert, 1, 1)
        case tensor.word.Nand():
            return _np.frompyfunc(lambda x, y: ~(x & y), 2, 1)
        case tensor.word.Nor():
            return _np.frompyfunc(lambda x, y: ~(x | y), 2, 1)
        case tensor.word.Xnor():
            return _np.frompyfunc(lambda x, y: ~(x ^ y), 2, 1)

        # words: shifts, rotations and resizing
        case tensor.word.Shl():
            return _np.frompyfunc(operator.lshift, 2, 1)
        case tensor.word.unsigned.Shr():
            return _np.frompyfunc(_z3.LShR, 2, 1)
        case tensor.word.signed.Shr():
            # Z3's `>>` on bit-vectors is the arithmetic shift
            return _np.frompyfunc(operator.rshift, 2, 1)
        case tensor.word.RotateLeft(i):
            return _np.frompyfunc(lambda x: _z3.RotateLeft(x, i), 1, 1)
        case tensor.word.RotateRight(i):
            return _np.frompyfunc(lambda x: _z3.RotateRight(x, i), 1, 1)
        case tensor.word.Concat():
            return _np.frompyfunc(_z3.Concat, 2, 1)
        case tensor.word.Extract(i, j):
            return _np.frompyfunc(lambda x: _z3.Extract(i, j, x), 1, 1)
        case tensor.word.Repeat(i):
            return _np.frompyfunc(lambda x: _z3.RepeatBitVec(i, x), 1, 1)
        case tensor.word.unsigned.Extend(i):
            return _np.frompyfunc(lambda x: _z3.ZeroExt(i, x), 1, 1)
        case tensor.word.signed.Extend(i):
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
