"""Translate terms to Z3 expressions.

Each wire value is a numpy array of Z3 expressions (the tensor, as a matrix). Dispatch is
by `match` on the op's class; reals and integers share cases. Words (bit-vectors) are not
translated yet (their ops fall through to a clear error).
"""

import z3 as _z3
import torch
from ..reactor.tensor import Havoc, Skip
from ..sort import Real, Int, Bool
import numpy as _np
import operator

from ..reactor import tensor


def fresh(dtype, prefix):
    match dtype:
        case Real([n, m]):
            return _np.array([[_z3.FreshReal(f'{prefix}[{i},{j}]') for j in range(m)] for i in range(n)], dtype=object)
        case Bool([n, m]):
            return _np.array([[_z3.FreshBool(f'{prefix}[{i},{j}]') for j in range(m)] for i in range(n)], dtype=object)
        case Int([n, m]):
            return _np.array([[_z3.FreshInt(f'{prefix}[{i},{j}]') for j in range(m)] for i in range(n)], dtype=object)
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
        case _:
            raise NotImplementedError(f"zero not implemented for dtype {dtype}")


def interpret(itype):
    match itype:
        # structural
        case Skip():
            return lambda x: x
        case Havoc(dtype):
            return lambda: fresh(dtype, "HAVOC")
        case tensor.zero.Zero():
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
        case tensor.real.Linear(weight, bias) | tensor.int.Linear(weight, bias):
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
        case tensor.real.Ite() | tensor.int.Ite() | tensor.bool.Ite():
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
