"""Three modules composed into one of three atoms: a counter, an
accumulator that awaits it, and a combinatorial output that awaits the
accumulator and the input.

- `x` counts 0, 1, 2, 3, 0, ..., and `wrap` says whether it came back to 0
  in the round: one atom controls both;
- `y` starts at the counter's first value and adds the counter's new value
  every round, so it awaits `x`;
- `z` is `y - e` in every round, the init one too: it awaits `y` and the
  input `e`.

`z` is declared first, so the module's state, `(z, x, wrap, y)`, is not in
the order its atoms run in: `x` and `wrap`, then `y`, then `z`. Only the
functional encoding composes atoms: `reactor --encodings functional`.
"""
import torch
from reactor import Bool, Int, Module, Term, Var, Wire, X, tensor
from reactor_lean.inputs import Named

S = [1, 1]


def _int(value: int, wire) -> Term:
    return Term(tensor.int.Const(torch.tensor([[value]])), [wire])


def counter(x: Var, wrap: Var) -> Module:
    one, four, zero = Wire(Int(S)), Wire(Int(S)), Wire(Int(S))
    succ, wrapped = Wire(Int(S)), Wire(Bool(S))
    init = [
        _int(0, X(x)),
        Term(tensor.bool.Const(torch.tensor([[False]])), [X(wrap)]),
    ]
    update = [
        _int(1, one), _int(4, four), _int(0, zero),
        Term(tensor.int.Add(S), [succ], [x, one]),
        Term(tensor.int.Eq(S), [wrapped], [succ, four]),
        Term(tensor.int.Ite(S), [X(x)], [wrapped, zero, succ]),
        Term(tensor.Skip(Bool(S)), [X(wrap)], [wrapped]),
    ]
    return Module.sequential([x, wrap], init, update)


def accumulator(y: Var, x: Var) -> Module:
    init = [Term(tensor.Skip(Int(S)), [X(y)], [X(x)])]
    update = [Term(tensor.int.Add(S), [X(y)], [y, X(x)])]
    return Module.sequential([y, x], init, update)


def difference(z: Var, y: Var, e: Var) -> Module:
    return Module.combinatorial(
        [z, y, e], [Term(tensor.int.Sub(S), [X(z)], [X(y), X(e)])])


def module() -> Named:
    z = Var(Int(S))
    x = Var(Int(S))
    wrap = Var(Bool(S))
    y = Var(Int(S))
    e = Var(Int(S))
    composed = Module.compose(difference(z, y, e), accumulator(y, x),
                              counter(x, wrap))
    return Named(composed, {"z": z, "x": x, "wrap": wrap, "y": y, "e": e})
