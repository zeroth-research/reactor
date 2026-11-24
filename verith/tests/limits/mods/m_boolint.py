"""Mixed Bool+Int state, built with explicit Terms.

b toggles between "counting up" and "counting down"; x walks 0..5..0.
"""
import torch
from reactor import Module, Term, Wire, Int, Bool, Var, X, tensor


def module() -> Module:
    b = Var(Bool([1, 1]))
    x = Var(Int([1, 1]))

    init = [
        Term(tensor.bool.Const(torch.tensor([[True]])), [X(b)]),
        Term(tensor.int.Const(torch.tensor([[0]])), [X(x)]),
    ]

    five, zero, one = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    lt5, gt0 = Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    up, dn = Wire(Int([1, 1])), Wire(Int([1, 1]))
    notb = Wire(Bool([1, 1]))
    x_if_b, b_if_b = Wire(Int([1, 1])), Wire(Bool([1, 1]))
    x_else, b_else = Wire(Int([1, 1])), Wire(Bool([1, 1]))

    update = [
        Term(tensor.int.Const(torch.tensor([[5]])), [five]),
        Term(tensor.int.Const(torch.tensor([[0]])), [zero]),
        Term(tensor.int.Const(torch.tensor([[1]])), [one]),
        Term(tensor.int.Lt([1, 1]), [lt5], [x, five]),
        Term(tensor.int.Gt([1, 1]), [gt0], [x, zero]),
        Term(tensor.int.Add([1, 1]), [up], [x, one]),
        Term(tensor.int.Sub([1, 1]), [dn], [x, one]),
        Term(tensor.bool.Not([1, 1]), [notb], [b]),
        # b: x' = x+1 while x<5, else flip b
        Term(tensor.int.Ite([1, 1]), [x_if_b], [lt5, up, x]),
        Term(tensor.bool.Ite([1, 1]), [b_if_b], [lt5, b, notb]),
        # not b: x' = x-1 while x>0, else flip b
        Term(tensor.int.Ite([1, 1]), [x_else], [gt0, dn, x]),
        Term(tensor.bool.Ite([1, 1]), [b_else], [gt0, b, notb]),
        Term(tensor.int.Ite([1, 1]), [X(x)], [b, x_if_b, x_else]),
        Term(tensor.bool.Ite([1, 1]), [X(b)], [b, b_if_b, b_else]),
    ]
    return Module.sequential([b, x], init, update)
