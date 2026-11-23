"""LRA: x steps down by 0.5 from 3.0, resetting at 0 — a non-integral Real state."""
import torch
from reactor import Module, Term, Wire, Real, Bool, Var, X, tensor


def module() -> Module:
    x = Var(Real([1, 1]))
    zero = Wire(Real([1, 1]))
    half = Wire(Real([1, 1]))
    three = Wire(Real([1, 1]))
    pos = Wire(Bool([1, 1]))
    dec = Wire(Real([1, 1]))
    init = [Term(tensor.real.Const(torch.tensor([[3.0]])), [X(x)])]
    update = [
        Term(tensor.real.Const(torch.tensor([[0.0]])), [zero]),
        Term(tensor.real.Const(torch.tensor([[0.5]])), [half]),
        Term(tensor.real.Const(torch.tensor([[3.0]])), [three]),
        Term(tensor.real.Gt([1, 1]), [pos], [x, zero]),
        Term(tensor.real.Sub([1, 1]), [dec], [x, half]),
        Term(tensor.real.Ite([1, 1]), [X(x)], [pos, dec, three]),
    ]
    return Module.sequential([x], init, update)
