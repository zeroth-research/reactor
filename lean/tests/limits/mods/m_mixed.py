"""ctrl = [x : 1x1, v : 3x1] — wire indices and flat element slots disagree.

x accumulates v[0]; v counts (1,2,3) down on its head and resets at 0.
"""
import torch
from reactor import Module, Term, Wire, Int, Bool, Var, X, tensor


def module() -> Module:
    x = Var(Int([1, 1]))
    v = Var(Int([3, 1]))
    z11 = torch.zeros((1, 1), dtype=torch.int64)
    v0 = torch.tensor([[1], [2], [3]], dtype=torch.int64)

    init = [
        Term(tensor.int.Const(torch.zeros((1, 1), dtype=torch.int64)), [X(x)]),
        Term(tensor.int.Const(v0), [X(v)]),
    ]

    head, zc = Wire(Int([1, 1])), Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    reset, dec = Wire(Int([3, 1])), Wire(Int([3, 1]))
    row0 = torch.tensor([[1, 0, 0]], dtype=torch.int64)
    zero33 = torch.zeros((3, 3), dtype=torch.int64)
    bneg = torch.tensor([[-1], [0], [0]], dtype=torch.int64)
    update = [
        Term(tensor.int.Linear(row0, z11), [head], [v]),
        Term(tensor.int.Add([1, 1]), [X(x)], [x, head]),
        Term(tensor.int.Const(torch.zeros((1, 1), dtype=torch.int64)), [zc]),
        Term(tensor.int.Eq([1, 1]), [cond], [head, zc]),
        Term(tensor.int.Linear(zero33, v0), [reset], [v]),
        Term(tensor.int.Linear(torch.eye(3, dtype=torch.int64), bneg), [dec], [v]),
        Term(tensor.int.Ite([3, 1]), [X(v)], [cond, reset, dec]),
    ]
    return Module.sequential([x, v], init, update)
