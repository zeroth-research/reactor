"""ReLU plus an external input: x' = relu(x - e), x0 = 5, e supplied each step.

Sound only under a precondition `e >= 1`; without it the ranking does not
decrease. Used twice, with and without `--pre`.
"""
import torch
from reactor import Module, Term, Wire, Int, Var, X, tensor


def module() -> Module:
    x = Var(Int([1, 1]))
    e = Var(Int([1, 1]))
    diff = Wire(Int([1, 1]))
    init = [Term(tensor.int.Const(torch.tensor([[5]])), [X(x)])]
    update = [
        Term(tensor.int.Sub([1, 1]), [diff], [x, X(e)]),
        Term(tensor.int.ReLU([1, 1]), [X(x)], [diff]),
    ]
    return Module.sequential([x, e], init, update)
