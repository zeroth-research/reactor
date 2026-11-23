"""LIA Argmax over a 3-vector into a 1x1 index state."""
import torch
from reactor import Module, Term, Wire, Int, Var, X, tensor


def module() -> Module:
    x = Var(Int([1, 1]))
    v = Wire(Int([3, 1]))
    init = [Term(tensor.int.Const(torch.tensor([[0]])), [X(x)])]
    # v = (2, x, 1): argmax is 1 while x > 2, else 0
    A = torch.tensor([[0], [1], [0]], dtype=torch.int64)
    b = torch.tensor([[2], [0], [1]], dtype=torch.int64)
    update = [
        Term(tensor.int.Linear(A, b), [v], [x]),
        Term(tensor.int.Argmax([3, 1]), [X(x)], [v]),
    ]
    return Module.sequential([x], init, update)
