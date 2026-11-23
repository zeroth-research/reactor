"""Two-bit counter reactive module fixture (BV encoding).

BV<1> state: b0, b1 are bits of a 2-bit counter, enable is external input.
Property: b0 = 0 ∧ b1 = 0 holds infinitely often (counter visits 00).
"""
import torch
from reactor import Wire, Term, Module, Bool, Word, Var, X, tensor


def module() -> Module:
    b0 = Var(Word(1, [1, 1]))
    b1 = Var(Word(1, [1, 1]))
    enable = Var(Word(1, [1, 1]))

    one = Wire(Word(1, [1, 1]))
    not_b0 = Wire(Word(1, [1, 1]))
    not_b1 = Wire(Word(1, [1, 1]))
    b0_and_enable = Wire(Word(1, [1, 1]))
    # a bit drives an `Ite` through its comparison with 1
    enabled = Wire(Bool([1, 1]))
    carry = Wire(Bool([1, 1]))

    init = [
        Term(tensor.word.Const(torch.tensor([[0]]), 1), [X(b0)]),
        Term(tensor.word.Const(torch.tensor([[0]]), 1), [X(b1)]),
    ]
    update = [
        Term(tensor.word.Const(torch.tensor([[1]]), 1), [one]),
        Term(tensor.word.Not(1, [1, 1]), [not_b0], [b0]),
        Term(tensor.word.Eq(1, [1, 1]), [enabled], [X(enable), one]),
        Term(tensor.word.Ite(1, [1, 1]), [X(b0)], [enabled, not_b0, b0]),
        Term(tensor.word.And(1, [1, 1]), [b0_and_enable], [b0, X(enable)]),
        Term(tensor.word.Eq(1, [1, 1]), [carry], [b0_and_enable, one]),
        Term(tensor.word.Not(1, [1, 1]), [not_b1], [b1]),
        Term(tensor.word.Ite(1, [1, 1]), [X(b1)], [carry, not_b1, b1]),
    ]
    return Module.sequential([b0, b1, enable], init, update)
