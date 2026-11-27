"""Tests for the Z3 translation of terms (`reactor.smt.z3`).

Every operation of the theories is translated on constant operands, the resulting Z3
expressions are simplified to values, and these are checked against the torch
interpreter `reactor.eval`, which follows the same SMT-LIB2 semantics.
"""

import pytest
import torch
import numpy as np
import z3

from reactor.real import Real
from reactor.int import Int
from reactor.bool import Bool
from reactor.word import Word
from reactor import real, word
import reactor as rr
from reactor import Skip
from reactor.eval import eval_itype
from reactor.smt import z3 as smt

R, I, B, W = Real([2, 2]), Int([2, 2]), Bool([2, 2]), Word(8, [2, 2])

signed, unsigned = word.signed, word.unsigned

# operands: reals exact in binary, integers of both signs, booleans of all pairs, and
# 8-bit words with high bits set (negative, read signed), zeros and shifts past the width
RX = torch.tensor([[1.5, -2.0], [0.25, 3.0]])
RY = torch.tensor([[0.5, 4.0], [-0.25, 3.0]])
IX = torch.tensor([[7, -7], [3, 0]])
IY = torch.tensor([[2, -2], [3, 5]])
BX = torch.tensor([[True, False], [True, False]])
BY = torch.tensor([[True, True], [False, False]])
WX = torch.tensor([[0xFB, 7], [0x80, 0]])
WY = torch.tensor([[2, 0], [0xFF, 3]])

A = torch.tensor([[1.0, -1.0], [0.5, 2.0], [0.0, 3.0]])
AI = torch.tensor([[1, -1], [2, 0], [0, 3]])

CASES = [
    # structural
    (Skip(R), [RX]),
    # the choice of whole matrices of each theory, by a scalar condition
    (real.broadcast.Ite([Bool([1, 1]), R, R]), [torch.tensor([[False]]), RX, RY]),
    (rr.int.broadcast.Ite([Bool([1, 1]), I, I]), [torch.tensor([[True]]), IX, IY]),
    (rr.bool.broadcast.Ite([Bool([1, 1]), B, B]), [torch.tensor([[True]]), BX, BY]),
    (word.broadcast.Ite([Bool([1, 1]), W, W]), [torch.tensor([[False]]), WX, WY]),
    (real.Zero(R), []),
    # constants, linear and affine maps
    (real.Num(RX), []),
    (rr.int.Num(IX), []),
    (rr.bool.Truth(BX), []),
    (word.Num(WX, W), []),
    (real.Linear(A, R), [RX]),
    (rr.int.Linear(AI, I), [IX]),
    (real.Affine(A, A, Real([2, 2])), [RX]),
    (rr.int.Affine(AI, AI, Int([2, 2])), [IX]),
    (real.Scale(RY, R), [RX]),
    (rr.int.Scale(IY, I), [IX]),
    # reals
    (real.Add([R, R]), [RX, RY]),
    (real.Sub([R, R]), [RX, RY]),
    (real.Mul([R, R]), [RX, RY]),
    (real.Div([R, R]), [RX, RY]),
    (real.Neg(R), [RX]),
    (real.Abs(R), [RX]),
    (real.ReLU(R), [RX]),
    (real.Min([R, R]), [RX, RY]),
    (real.Max([R, R]), [RX, RY]),
    (real.Argmax(R), [RX]),
    (real.Transpose(R), [RX]),
    (real.MatMul([R, R]), [RX, RY]),
    (real.Ite([B, R, R]), [BX, RX, RY]),
    (real.Eq([R, R]), [RX, RY]),
    (real.Ne([R, R]), [RX, RY]),
    (real.Lt([R, R]), [RX, RY]),
    (real.Le([R, R]), [RX, RY]),
    (real.Gt([R, R]), [RX, RY]),
    (real.Ge([R, R]), [RX, RY]),
    # integers
    (rr.int.Add([I, I]), [IX, IY]),
    (rr.int.Sub([I, I]), [IX, IY]),
    (rr.int.Mul([I, I]), [IX, IY]),
    (rr.int.Div([I, I]), [IX, IY]),
    (rr.int.Mod([I, I]), [IX, IY]),
    (rr.int.Neg(I), [IX]),
    (rr.int.Abs(I), [IX]),
    (rr.int.ReLU(I), [IX]),
    (rr.int.Min([I, I]), [IX, IY]),
    (rr.int.Max([I, I]), [IX, IY]),
    (rr.int.Argmax(I), [IX]),
    (rr.int.Transpose(I), [IX]),
    (rr.int.MatMul([I, I]), [IX, IY]),
    (rr.int.Ite([B, I, I]), [BX, IX, IY]),
    (rr.int.Eq([I, I]), [IX, IY]),
    (rr.int.Ne([I, I]), [IX, IY]),
    (rr.int.Lt([I, I]), [IX, IY]),
    (rr.int.Le([I, I]), [IX, IY]),
    (rr.int.Gt([I, I]), [IX, IY]),
    (rr.int.Ge([I, I]), [IX, IY]),
    # booleans
    (rr.bool.And([B, B]), [BX, BY]),
    (rr.bool.Or([B, B]), [BX, BY]),
    (rr.bool.Xor([B, B]), [BX, BY]),
    (rr.bool.Not(B), [BX]),
    (rr.bool.Eq([B, B]), [BX, BY]),
    (rr.bool.Ne([B, B]), [BX, BY]),
    (rr.bool.Ite([B, B, B]), [BX, BX, BY]),
    (rr.bool.reduce.And(0, B), [BX]),
    (rr.bool.reduce.And(1, B), [BX]),
    (rr.bool.reduce.Or(0, B), [BY]),
    (rr.bool.reduce.Or(1, B), [BY]),
    (rr.bool.reduce.And(1, Bool([2, 3])), [torch.tensor([[True, True, True], [True, False, True]])]),
    # words: comparisons
    (word.Eq([W, W]), [WX, WY]),
    (word.Ne([W, W]), [WX, WY]),
    (unsigned.Lt([W, W]), [WX, WY]),
    (unsigned.Le([W, W]), [WX, WY]),
    (unsigned.Gt([W, W]), [WX, WY]),
    (unsigned.Ge([W, W]), [WX, WY]),
    (signed.Lt([W, W]), [WX, WY]),
    (signed.Le([W, W]), [WX, WY]),
    (signed.Gt([W, W]), [WX, WY]),
    (signed.Ge([W, W]), [WX, WY]),
    # words: arithmetic, with division by zero
    (word.Add([W, W]), [WX, WY]),
    (word.Sub([W, W]), [WX, WY]),
    (word.Mul([W, W]), [WX, WY]),
    (word.Neg(W), [WX]),
    (signed.Abs(W), [WX]),
    (unsigned.Div([W, W]), [WX, WY]),
    (unsigned.Rem([W, W]), [WX, WY]),
    (signed.Div([W, W]), [WX, WY]),
    (signed.Rem([W, W]), [WX, WY]),
    (signed.Mod([W, W]), [WX, WY]),
    # words: bitwise
    (word.And([W, W]), [WX, WY]),
    (word.Or([W, W]), [WX, WY]),
    (word.Xor([W, W]), [WX, WY]),
    (word.Not(W), [WX]),
    (word.Nand([W, W]), [WX, WY]),
    (word.Nor([W, W]), [WX, WY]),
    (word.Xnor([W, W]), [WX, WY]),
    # words: shifts past the width, rotations and resizing
    (word.Shl([W, W]), [WX, WY]),
    (unsigned.Shr([W, W]), [WX, WY]),
    (signed.Shr([W, W]), [WX, WY]),
    (word.RotateLeft(3, W), [WX]),
    (word.RotateRight(3, W), [WX]),
    (word.Concat([W, W]), [WX, WY]),
    (word.Extract(6, 3, W), [WX]),
    (word.Repeat(2, W), [WX]),
    (unsigned.Extend(8, W), [WX]),
    (signed.Extend(8, W), [WX]),
    (word.Ite([B, W, W]), [BX, WX, WY]),
]


def _z3_value(sort, v):
    """The Z3 constant of the entry `v` of a matrix of `sort`."""
    match sort:
        case Real():
            return z3.RealVal(str(float(v)))
        case Int():
            return z3.IntVal(int(v))
        case Bool():
            return z3.BoolVal(bool(v))
        case Word(w):
            return z3.BitVecVal(int(v), w)


def _z3_matrix(sort, t):
    return np.array([[_z3_value(sort, v) for v in row] for row in t.tolist()], dtype=object)


def _python_value(e):
    """The value of the closed Z3 expression `e`."""
    e = z3.simplify(e)
    if z3.is_true(e) or z3.is_false(e):
        return z3.is_true(e)
    if z3.is_rational_value(e):
        return float(e.as_fraction())
    if z3.is_int_value(e) or z3.is_bv_value(e):
        return e.as_long()
    raise AssertionError(f"not a value: {e}")


def _dom(op, args):
    """The sorts that `op` reads, one per operand in `args`: none, when it reads nothing."""
    if not args:
        return []
    return op.dom if isinstance(op.dom, list) else [op.dom]


IDS = [f"{type(op).__module__.removeprefix('reactor.')}.{type(op).__name__}" for op, _ in CASES]


@pytest.mark.parametrize("op,args", CASES, ids=IDS)
def test_z3_agrees_with_eval(op, args):
    expected = eval_itype(op, args)[0]
    read = [_z3_matrix(s, t) for s, t in zip(_dom(op, args), args)]
    got = smt.eval(op, read)[0]
    got = np.vectorize(_python_value, otypes=[object])(got)
    assert list(got.shape) == list(expected.shape)
    assert list(got.flat) == pytest.approx(expected.flatten().tolist())
