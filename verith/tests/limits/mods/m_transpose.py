"""LIA Transpose — an op with no entry in the Lean expression tables."""
import torch
from reactor import Module, Term, Wire, Int, Var, X, tensor


def module() -> Module:
    v = Var(Int([1, 3]))
    t = Wire(Int([3, 1]))
    tt = Wire(Int([1, 3]))
    init = [Term(tensor.int.Const(torch.tensor([[1, 2, 3]])), [X(v)])]
    update = [
        Term(tensor.int.Transpose([1, 3]), [t], [v]),
        Term(tensor.int.Transpose([3, 1]), [tt], [t]),
        Term(tensor.Skip(Int([1, 3])), [X(v)], [tt]),
    ]
    return Module.sequential([v], init, update)
