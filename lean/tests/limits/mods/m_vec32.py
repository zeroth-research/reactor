"""32-wide state, countdown in slot 0 — the scaling case."""
import torch
from reactor import Module, Term, Wire, Int, Bool, Var, X, tensor

N = 32


def module() -> Module:
    s = Var(Int([N, 1]))
    z11 = torch.zeros((1, 1), dtype=torch.int64)

    init_vec = torch.zeros((N, 1), dtype=torch.int64)
    init_vec[0][0] = 100
    init = [Term(tensor.int.Const(init_vec), [X(s)])]

    x, zc = Wire(Int([1, 1])), Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    reset, dec = Wire(Int([N, 1])), Wire(Int([N, 1]))

    row0 = torch.zeros((1, N), dtype=torch.int64)
    row0[0][0] = 1
    diag_keep = torch.eye(N, dtype=torch.int64)
    diag_keep[0][0] = 0
    b100 = torch.zeros((N, 1), dtype=torch.int64)
    b100[0][0] = 100
    bneg = torch.zeros((N, 1), dtype=torch.int64)
    bneg[0][0] = -1

    update = [
        Term(tensor.int.Linear(row0, z11), [x], [s]),
        Term(tensor.int.Const(torch.tensor([[0]])), [zc]),
        Term(tensor.int.Eq([1, 1]), [cond], [x, zc]),
        Term(tensor.int.Linear(diag_keep, b100), [reset], [s]),
        Term(tensor.int.Linear(torch.eye(N, dtype=torch.int64), bneg), [dec], [s]),
        Term(tensor.int.Ite([N, 1]), [X(s)], [cond, reset, dec]),
    ]
    return Module.sequential([s], init, update)
