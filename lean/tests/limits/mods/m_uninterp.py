"""LIA Havoc as a source (one write, no read): x' is an arbitrary Int each step."""
import torch
from reactor import Module, Term, Wire, Int, Var, X, tensor


def module() -> Module:
    x = Var(Int([1, 1]))
    y = Wire(Int([1, 1]))
    init = [Term(tensor.int.Const(torch.tensor([[0]])), [X(x)])]
    update = [
        Term(tensor.Havoc(Int([1, 1])), [y]),
        Term(tensor.Skip(Int([1, 1])), [X(x)], [y]),
    ]
    return Module.sequential([x], init, update)
