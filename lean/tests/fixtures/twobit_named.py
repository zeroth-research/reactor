"""`twobit_lia`, with names a claim may use: `lo` and `hi` for the bits,
`enable` for the input. Its claims and `twobit_lia`'s positional ones
generate the same project."""
import torch
from reactor import Wire, Term, Module, Bool, Var, X, tensor
from reactor_lean.inputs import Named


def module() -> Named:
    b0 = Var(Bool([1, 1]))
    b1 = Var(Bool([1, 1]))
    enable = Var(Bool([1, 1]))

    not_b0 = Wire(Bool([1, 1]))
    not_b1 = Wire(Bool([1, 1]))
    b0_and_enable = Wire(Bool([1, 1]))

    init = [
        Term(tensor.bool.Const(torch.tensor([[False]])), [X(b0)]),
        Term(tensor.bool.Const(torch.tensor([[False]])), [X(b1)]),
    ]
    update = [
        Term(tensor.bool.Not([1, 1]), [not_b0], [b0]),
        Term(tensor.bool.Ite([1, 1]), [X(b0)], [X(enable), not_b0, b0]),
        Term(tensor.bool.And([1, 1]), [b0_and_enable], [b0, X(enable)]),
        Term(tensor.bool.Not([1, 1]), [not_b1], [b1]),
        Term(tensor.bool.Ite([1, 1]), [X(b1)], [b0_and_enable, not_b1, b1]),
    ]
    module = Module.sequential([b0, b1, enable], init, update)
    return Named(module, {"lo": b0, "hi": b1, "enable": enable})


def stray() -> Named:
    """A name for a variable the module does not have."""
    named = module()
    return Named(named.module, {**named.names, "ghost": Var(Bool([1, 1]))})
