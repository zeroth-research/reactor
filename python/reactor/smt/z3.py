"""Translate terms to Z3 expressions.

Each wire value is a numpy array of Z3 expressions (the tensor, as a matrix). Dispatch is
by `match` on the op's class; reals and integers share cases. Words (bit-vectors) are not
translated yet (their ops fall through to a clear error).
"""

import z3 as _z3
import torch
from ..reactor.tensor import Havoc, Skip, Zero
from ..arrays import as_numpy
from ..sort import Real, Int, Bool
import numpy as _np
import operator
from functools import reduce

from ..reactor import tensor
from ..reactor import jax as _jax
from ..reactor import mixed as _mixed


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
        case Skip(_):
            return lambda x: x
        case Havoc(dtype):
            return lambda: fresh(dtype, "HAVOC")
        case Zero(dtype):
            return lambda: _zeros(dtype)

        # constants
        case tensor.bool.Const(t) | _jax.bool.Const(t) | _mixed.bool.Const(t):
            return lambda: _np.vectorize(_z3.BoolVal, otypes=[object])(as_numpy(t))
        case tensor.real.Const(t) | _jax.real.Const(t) | _mixed.real.Const(t):
            return lambda: _np.vectorize(_z3.RealVal, otypes=[object])(as_numpy(t))
        case tensor.int.Const(t) | _jax.int.Const(t) | _mixed.int.Const(t):
            return lambda: _np.vectorize(_z3.IntVal, otypes=[object])(as_numpy(t))

        # arithmetic
        case tensor.real.Sub(_) | tensor.int.Sub(_):
            return _np.frompyfunc(operator.sub, 2, 1)
        case tensor.real.Add(_) | tensor.int.Add(_):
            return _np.frompyfunc(operator.add, 2, 1)
        case tensor.real.Mul(_) | tensor.int.Mul(_):
            return _np.frompyfunc(operator.mul, 2, 1)
        case tensor.real.Div(_) | tensor.int.Div(_):
            return _np.frompyfunc(operator.truediv, 2, 1)
        case tensor.int.Mod(_):
            return _np.frompyfunc(operator.mod, 2, 1)
        case tensor.real.MatMul(_) | tensor.int.MatMul(_):
            return lambda x, y: x @ y
        case tensor.real.Linear(weight, bias) | tensor.int.Linear(weight, bias) | \
                _jax.real.Linear(weight, bias) | _jax.int.Linear(weight, bias) | \
                _mixed.real.Linear(weight, bias) | _mixed.int.Linear(weight, bias):
            return lambda x: as_numpy(weight) @ x + as_numpy(bias)
        case tensor.real.Scale(c) | tensor.int.Scale(c) | _jax.real.Scale(c) | _jax.int.Scale(c) | \
                _mixed.real.Scale(c) | _mixed.int.Scale(c):
            return lambda x: as_numpy(c) * x

        # comparisons (produce Bool)
        case tensor.real.Eq(_) | tensor.int.Eq(_) | tensor.bool.Eq(_):
            return _np.frompyfunc(operator.eq, 2, 1)
        case tensor.real.Ne(_) | tensor.int.Ne(_) | tensor.bool.Ne(_):
            return _np.frompyfunc(operator.ne, 2, 1)
        case tensor.real.Lt(_) | tensor.int.Lt(_):
            return _np.frompyfunc(operator.lt, 2, 1)
        case tensor.real.Le(_) | tensor.int.Le(_):
            return _np.frompyfunc(operator.le, 2, 1)
        case tensor.real.Gt(_) | tensor.int.Gt(_):
            return _np.frompyfunc(operator.gt, 2, 1)
        case tensor.real.Ge(_) | tensor.int.Ge(_):
            return _np.frompyfunc(operator.ge, 2, 1)

        # logical
        case tensor.bool.And(_):
            return _np.frompyfunc(lambda x, y: _z3.And(x, y), 2, 1)
        case tensor.bool.Or(_):
            return _np.frompyfunc(lambda x, y: _z3.Or(x, y), 2, 1)
        case tensor.bool.Not(_):
            return _np.frompyfunc(lambda x: _z3.Not(x), 1, 1)
        case tensor.bool.Xor(_):
            return _np.frompyfunc(lambda x, y: _z3.Xor(x, y), 2, 1)
        case tensor.bool.All(_):
            return lambda x: _np.array([[_z3.And(*x.flat)]], dtype=object)
        case tensor.bool.Implies(_):
            return _np.frompyfunc(lambda x, y: _z3.Implies(x, y), 2, 1)

        # an entry of a matrix
        case tensor.real.Get(i, j, _) | tensor.int.Get(i, j, _) | tensor.bool.Get(i, j, _):
            return lambda x: x[i:i + 1, j:j + 1]

        # casts between the theories' sorts
        case tensor.cast.ToReal(_):
            return _np.frompyfunc(_z3.ToReal, 1, 1)
        case tensor.cast.ToInt(_):
            return _np.frompyfunc(_z3.ToInt, 1, 1)

        # control flow: the [1, 1] condition broadcasts over the branches
        case tensor.real.Ite(_) | tensor.int.Ite(_) | tensor.bool.Ite(_):
            return _np.frompyfunc(lambda c, tt, ff: _z3.If(c, tt, ff), 3, 1)

        # neural-ish / aggregate
        case tensor.real.Neg(_) | tensor.int.Neg(_):
            return _np.frompyfunc(lambda x: -x, 1, 1)
        case tensor.real.Abs(_) | tensor.int.Abs(_):
            return _np.frompyfunc(lambda x: _z3.If(x >= 0, x, -x), 1, 1)
        case tensor.real.ReLU(_) | tensor.int.ReLU(_):
            return _np.frompyfunc(lambda x: _z3.If(x > 0, x, 0), 1, 1)
        # the least and the greatest entry, folded row by row, as a [1, 1] matrix
        case tensor.real.Min(_) | tensor.int.Min(_):
            return lambda x: _np.array([[reduce(lambda m, y: _z3.If(m <= y, m, y), x.flat)]], dtype=object)
        case tensor.real.Max(_) | tensor.int.Max(_):
            return lambda x: _np.array([[reduce(lambda m, y: _z3.If(m >= y, m, y), x.flat)]], dtype=object)

        case tensor.real.Transpose(_) | tensor.int.Transpose(_):
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
