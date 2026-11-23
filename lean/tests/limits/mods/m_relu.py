"""LIA 1x1 with ReLU in the transition: x' = relu(x - 1), x0 = 5.

Counts down to 0 and stays there, so `x = 0` holds infinitely often. The
Lean side becomes `Max.max 0 (x - 1)`.
"""
import torch
from reactor import Module, Term, Wire, Int, Var, X, tensor


def module() -> Module:
    x = Var(Int([1, 1]))
    one = Wire(Int([1, 1]))
    diff = Wire(Int([1, 1]))
    init = [Term(tensor.int.Const(torch.tensor([[5]])), [X(x)])]
    update = [
        Term(tensor.int.Const(torch.tensor([[1]])), [one]),
        Term(tensor.int.Sub([1, 1]), [diff], [x, one]),
        Term(tensor.int.ReLU([1, 1]), [X(x)], [diff]),
    ]
    return Module.sequential([x], init, update)
