"""LRA (Real) 1x1 with ReLU: x' = relu(x - 1.0), x0 = 5.0."""
import torch
from reactor import Module, Term, Wire, Real, Var, X, tensor


def module() -> Module:
    x = Var(Real([1, 1]))
    one = Wire(Real([1, 1]))
    diff = Wire(Real([1, 1]))
    init = [Term(tensor.real.Const(torch.tensor([[5.0]])), [X(x)])]
    update = [
        Term(tensor.real.Const(torch.tensor([[1.0]])), [one]),
        Term(tensor.real.Sub([1, 1]), [diff], [x, one]),
        Term(tensor.real.ReLU([1, 1]), [X(x)], [diff]),
    ]
    return Module.sequential([x], init, update)
