"""A long straight-line body: 64 chained additions before the countdown step.

Probes how far the *size* of a transition can go before elaboration or the
tactic script gives out, independently of state width.
"""
import torch
from reactor import Module, Term, Wire, Int, Bool, Var, X, tensor

DEPTH = 40


def module() -> Module:
    x = Var(Int([1, 1]))
    init = [Term(tensor.int.Const(torch.tensor([[100]])), [X(x)])]

    zero = Wire(Int([1, 1]))
    update = [Term(tensor.int.Const(torch.tensor([[0]])), [zero])]

    # acc = x + 0 + 0 + ... (DEPTH times): semantically the identity, but
    # syntactically a DEPTH-deep chain of Add terms.
    cur = x
    for _ in range(DEPTH):
        nxt = Wire(Int([1, 1]))
        update.append(Term(tensor.int.Add([1, 1]), [nxt], [cur, zero]))
        cur = nxt

    one = Wire(Int([1, 1]))
    hundred = Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    dec = Wire(Int([1, 1]))
    update += [
        Term(tensor.int.Const(torch.tensor([[1]])), [one]),
        Term(tensor.int.Const(torch.tensor([[100]])), [hundred]),
        Term(tensor.int.Eq([1, 1]), [cond], [cur, zero]),
        Term(tensor.int.Sub([1, 1]), [dec], [cur, one]),
        Term(tensor.int.Ite([1, 1]), [X(x)], [cond, hundred, dec]),
    ]
    return Module.sequential([x], init, update)
