"""LIA Max as a reduction: v = (x-1, 0), x' = max v — ReLU spelled Max."""
import torch
from reactor import Module, Term, Wire, Int, Var, X, tensor


def module() -> Module:
    x = Var(Int([1, 1]))
    v = Wire(Int([2, 1]))
    hi = Wire(Int([1, 1]))

    init = [Term(tensor.int.Const(torch.tensor([[5]])), [X(x)])]
    A = torch.tensor([[1], [0]], dtype=torch.int64)
    b = torch.tensor([[-1], [0]], dtype=torch.int64)
    update = [
        Term(tensor.int.Linear(A, b), [v], [x]),
        Term(tensor.int.Max([2, 1]), [hi], [v]),
        Term(tensor.Skip(Int([1, 1])), [X(x)], [hi]),
    ]
    return Module.sequential([x], init, update)
