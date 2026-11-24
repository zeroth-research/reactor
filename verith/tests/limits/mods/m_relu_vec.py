"""LIA 3-vector with element-wise ReLU: v' = relu(v - 1), v0 = (3, 2, 1)."""
import torch
from reactor import Module, Term, Wire, Int, Var, X, tensor


def module() -> Module:
    v = Var(Int([3, 1]))
    ones = Wire(Int([3, 1]))
    diff = Wire(Int([3, 1]))
    init = [Term(tensor.int.Const(torch.tensor([[3], [2], [1]])), [X(v)])]
    update = [
        Term(tensor.int.Const(torch.ones((3, 1), dtype=torch.int64)), [ones]),
        Term(tensor.int.Sub([3, 1]), [diff], [v, ones]),
        Term(tensor.int.ReLU([3, 1]), [X(v)], [diff]),
    ]
    return Module.sequential([v], init, update)
