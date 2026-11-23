"""LRA, two Real wires: x walks 0.0 -> 3.0 in steps of 1.0 while y alternates.

Two independent components, so an invariant over it is naturally a conjunction
of disjunctions — the shape a single `constructor <;> linarith` cannot close.
"""
import torch
from reactor import Module, Term, Wire, Real, Bool, Var, X, tensor


def module() -> Module:
    x = Var(Real([1, 1]))
    y = Var(Real([1, 1]))
    zero, one, three = Wire(Real([1, 1])), Wire(Real([1, 1])), Wire(Real([1, 1]))
    two, five = Wire(Real([1, 1])), Wire(Real([1, 1]))
    lt3 = Wire(Bool([1, 1]))
    inc = Wire(Real([1, 1]))
    y_is_two = Wire(Bool([1, 1]))

    init = [
        Term(tensor.real.Const(torch.tensor([[0.0]])), [X(x)]),
        Term(tensor.real.Const(torch.tensor([[2.0]])), [X(y)]),
    ]
    update = [
        Term(tensor.real.Const(torch.tensor([[0.0]])), [zero]),
        Term(tensor.real.Const(torch.tensor([[1.0]])), [one]),
        Term(tensor.real.Const(torch.tensor([[3.0]])), [three]),
        Term(tensor.real.Const(torch.tensor([[2.0]])), [two]),
        Term(tensor.real.Const(torch.tensor([[5.0]])), [five]),
        Term(tensor.real.Lt([1, 1]), [lt3], [x, three]),
        Term(tensor.real.Add([1, 1]), [inc], [x, one]),
        # x: 0 -> 1 -> 2 -> 3 -> 0
        Term(tensor.real.Ite([1, 1]), [X(x)], [lt3, inc, zero]),
        # y toggles between 2.0 and 5.0
        Term(tensor.real.Eq([1, 1]), [y_is_two], [y, two]),
        Term(tensor.real.Ite([1, 1]), [X(y)], [y_is_two, five, two]),
    ]
    return Module.sequential([x, y], init, update)
