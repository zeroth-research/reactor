"""Two independent 1x1 plants, each converging to 5 from either side.

`m_toward5` in two dimensions, so that |x-5| + |y-5| is a four-unit ReLU
Lyapunov function over a two-wire state.
"""
import torch
from reactor import Module, Term, Wire, Int, Bool, Var, X, tensor


def _leg(v, init_val):
    five, one = Wire(Int([1, 1])), Wire(Int([1, 1]))
    lo, hi = Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    inc, dec, up = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    init = [Term(tensor.int.Const(torch.tensor([[init_val]])), [X(v)])]
    update = [
        Term(tensor.int.Const(torch.tensor([[5]])), [five]),
        Term(tensor.int.Const(torch.tensor([[1]])), [one]),
        Term(tensor.int.Lt([1, 1]), [lo], [v, five]),
        Term(tensor.int.Gt([1, 1]), [hi], [v, five]),
        Term(tensor.int.Add([1, 1]), [inc], [v, one]),
        Term(tensor.int.Sub([1, 1]), [dec], [v, one]),
        Term(tensor.int.Ite([1, 1]), [up], [hi, dec, five]),
        Term(tensor.int.Ite([1, 1]), [X(v)], [lo, inc, up]),
    ]
    return init, update


def module() -> Module:
    x, y = Var(Int([1, 1])), Var(Int([1, 1]))
    ix, ux = _leg(x, 10)
    iy, uy = _leg(y, 0)
    return Module.sequential([x, y], ix + iy, ux + uy)
