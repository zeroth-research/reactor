"""LIA Min as a reduction: v = (x+1, 5), x' = min v — saturates at 5."""
import torch
from reactor import Module, Term, Wire, Int, Var, X, tensor


def module() -> Module:
    x = Var(Int([1, 1]))
    v = Wire(Int([2, 1]))
    lo = Wire(Int([1, 1]))

    init = [Term(tensor.int.Const(torch.tensor([[0]])), [X(x)])]
    A = torch.tensor([[1], [0]], dtype=torch.int64)
    b = torch.tensor([[1], [5]], dtype=torch.int64)
    update = [
        Term(tensor.int.Linear(A, b), [v], [x]),
        Term(tensor.int.Min([2, 1]), [lo], [v]),
        Term(tensor.Skip(Int([1, 1])), [X(x)], [lo]),
    ]
    return Module.sequential([x], init, update)
