"""LRA, two Real wires that converge independently and then stay put.

x: 3 -> 2 -> 1 -> 0 -> 0 ...      (x' = if x > 0 then x - 1 else 0)
y: 5 -> 4 -> 3 -> 2 -> 2 ...      (y' = if y > 2 then y - 1 else 2)

The natural invariant is a *conjunction of disjunctions* -- one range per
component -- and unlike `m_lra_two` the components do not cycle, so that
invariant is strong enough to carry a ranking. (In `m_lra_two` both components
cycle, and the product of their ranges admits a run that never reaches the
property, which makes the Buchi obligation false for *any* ranking.)
"""
import torch
from reactor import Module, Term, Wire, Real, Bool, Var, X, tensor


def module() -> Module:
    x = Var(Real([1, 1]))
    y = Var(Real([1, 1]))
    zero, one, two = Wire(Real([1, 1])), Wire(Real([1, 1])), Wire(Real([1, 1]))
    x_pos = Wire(Bool([1, 1]))
    y_big = Wire(Bool([1, 1]))
    x_dec = Wire(Real([1, 1]))
    y_dec = Wire(Real([1, 1]))

    init = [
        Term(tensor.real.Const(torch.tensor([[3.0]])), [X(x)]),
        Term(tensor.real.Const(torch.tensor([[5.0]])), [X(y)]),
    ]
    update = [
        Term(tensor.real.Const(torch.tensor([[0.0]])), [zero]),
        Term(tensor.real.Const(torch.tensor([[1.0]])), [one]),
        Term(tensor.real.Const(torch.tensor([[2.0]])), [two]),
        Term(tensor.real.Gt([1, 1]), [x_pos], [x, zero]),
        Term(tensor.real.Sub([1, 1]), [x_dec], [x, one]),
        Term(tensor.real.Ite([1, 1]), [X(x)], [x_pos, x_dec, zero]),
        Term(tensor.real.Gt([1, 1]), [y_big], [y, two]),
        Term(tensor.real.Sub([1, 1]), [y_dec], [y, one]),
        Term(tensor.real.Ite([1, 1]), [X(y)], [y_big, y_dec, two]),
    ]
    return Module.sequential([x, y], init, update)
