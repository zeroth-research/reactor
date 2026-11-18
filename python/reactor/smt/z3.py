"""Translate the operations of `reactor.op` to Z3 expressions.

Each wire value is a list of Z3 expressions (the flattened tensor). Dispatch is
by `match` on the op; reals and integers share cases. Words -> Z3 (BitVec) is not
implemented yet (word ops fall through to a clear error).
"""

import z3 as _z3
import torch
from .. import op
from ..sort import Real, Int, Bool
from ..tensor import as_numpy
import numpy as _np
import operator


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


def interpret(itype):
    match itype:
        case op.real.Id() | op.real.TId() | op.int.Id() | op.bool.Id():
            return lambda x: x
        case op.bool.Const(t):
            return lambda: _np.vectorize(_z3.BoolVal, otypes=[object])(as_numpy(t))
        case op.real.Const(t):
            return lambda: _np.vectorize(_z3.RealVal, otypes=[object])(as_numpy(t))
        case op.int.Const(t):
            return lambda: _np.vectorize(_z3.IntVal, otypes=[object])(as_numpy(t))
        case op.real.Havoc(m, n):
            return lambda: fresh(Real([m, n]), "HAVOC")
        case op.int.Havoc(m, n):
            return lambda: fresh(Int([m, n]), "HAVOC")
        case op.bool.Havoc(m, n):
            return lambda: fresh(Bool([m, n]), "HAVOC")

        # arithmetic
        case op.real.Sub() | op.real.TSub() | op.int.Sub():
            return _np.frompyfunc(operator.sub, 2, 1)
        case op.real.Add() | op.real.TAdd() | op.int.Add():
            return _np.frompyfunc(operator.add, 2, 1)
        case op.real.Linear(weight, bias) | op.real.TLinear(weight, bias, _) | op.int.Linear(weight, bias):
            return lambda x: as_numpy(weight) @ x + as_numpy(bias)

        # comparisons (produce Bool)
        case op.real.Eq() | op.int.Eq():
            return _np.frompyfunc(operator.eq, 2, 1)
        case op.real.Ne() | op.int.Ne():
            return _np.frompyfunc(operator.ne, 2, 1)
        case op.real.Lt() | op.int.Lt():
            return _np.frompyfunc(operator.lt, 2, 1)
        case op.real.Le() | op.int.Le():
            return _np.frompyfunc(operator.le, 2, 1)
        case op.real.Gt() | op.int.Gt():
            return _np.frompyfunc(operator.gt, 2, 1)
        case op.real.Ge() | op.int.Ge():
            return _np.frompyfunc(operator.ge, 2, 1)

        # logical
        case op.bool.And():
            return _np.frompyfunc(lambda x, y: _z3.And(x, y), 2, 1)
        case op.bool.Or():
            return _np.frompyfunc(lambda x, y: _z3.Or(x, y), 2, 1)
        case op.bool.Not():
            return _np.frompyfunc(lambda x: _z3.Not(x), 1, 1)
        case op.bool.Xor():
            return _np.frompyfunc(lambda x, y: _z3.Xor(x, y), 2, 1)

        # control flow
        case op.real.Ite() | op.real.TIte() | op.int.Ite() | op.bool.Ite():
            return _np.frompyfunc(lambda c, tt, ff: _z3.If(c, tt, ff), 3, 1)

        # neural-ish / aggregate
        case op.real.ReLU() | op.int.ReLU():
            return _np.frompyfunc(lambda x: _z3.If(x > 0., x, 0.), 1, 1)
        case op.real.Min() | op.int.Min():
            return _np.frompyfunc(lambda x, y: _z3.If(x <= y, x, y), 2, 1)
        case op.real.Max() | op.int.Max():
            return _np.frompyfunc(lambda x, y: _z3.If(x >= y, x, y), 2, 1)

        case op.real.Transpose() | op.real.TTranspose() | op.int.Transpose():
            return _np.transpose
        case (op.real.Uninterpreted(name) | op.int.Uninterpreted(name)
              | op.bool.Uninterpreted(name) | op.word.Uninterpreted(name)):
            raise RuntimeError(f"cannot translate uninterpreted '{name}' to Z3")
        case _:
            raise NotImplementedError(f"itype: {itype!r}")


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
