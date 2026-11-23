"""LRA (Real) countdown without ReLU: x' = x - 1.0 while x > 0, else 5.0."""
import torch
from reactor import Module, Term, Wire, Real, Bool, Var, X, tensor


def module() -> Module:
    x = Var(Real([1, 1]))
    zero = Wire(Real([1, 1]))
    one = Wire(Real([1, 1]))
    five = Wire(Real([1, 1]))
    pos = Wire(Bool([1, 1]))
    dec = Wire(Real([1, 1]))
    init = [Term(tensor.real.Const(torch.tensor([[5.0]])), [X(x)])]
    update = [
        Term(tensor.real.Const(torch.tensor([[0.0]])), [zero]),
        Term(tensor.real.Const(torch.tensor([[1.0]])), [one]),
        Term(tensor.real.Const(torch.tensor([[5.0]])), [five]),
        Term(tensor.real.Gt([1, 1]), [pos], [x, zero]),
        Term(tensor.real.Sub([1, 1]), [dec], [x, one]),
        Term(tensor.real.Ite([1, 1]), [X(x)], [pos, dec, five]),
    ]
    return Module.sequential([x], init, update)
