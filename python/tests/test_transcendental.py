"""Tests for the transcendental functions of reals (`reactor.tensor.real.Sqrt`,
`.Sin`, ..., `.Atan2`): their sorts, the torch evaluator, the Z3 encoding, the
builder, the `expr` DSL, sugar modules and the analyzer."""

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
import z3
from gymnasium import spaces

from reactor import Real, Int, TheoryError, Var, Wire, X, d
from reactor.builder import builder_for
from reactor.eval import eval_itype
from reactor.expr import collecting, expr
from reactor.gym import Env
from reactor.reactor import tensor
from reactor.smt import z3 as rz3
from reactor import sugar
from reactor.sugar import sin, cos, sqrt, atan2, tanh

UNARY = {
    "Sqrt": math.sqrt, "Exp": math.exp, "Log": math.log, "Sin": math.sin,
    "Cos": math.cos, "Tan": math.tan, "Asin": math.asin, "Acos": math.acos,
    "Atan": math.atan, "Sinh": math.sinh, "Cosh": math.cosh, "Tanh": math.tanh,
}
S = [2, 3]


def _real(values):
    return torch.tensor(values, dtype=torch.float64)


# ── the operations ────────────────────────────────────────────────


@pytest.mark.parametrize("name", [*UNARY, "Atan2"])
def test_an_op_is_sized_by_its_shape_and_matches_by_it(name):
    op = getattr(tensor.real, name)(S)
    assert op.shape == S
    assert str(op) == name
    match op:
        case tensor.real.Atan2(shape) | tensor.real.Sin(shape) | tensor.real.Sqrt(shape):
            assert shape == S
        case _:
            pass
    assert op == getattr(tensor.real, name)(S)
    assert op != getattr(tensor.real, name)([1, 1])


@pytest.mark.parametrize("name", UNARY)
def test_torch_evaluates_each_pointwise(name):
    x = _real([[0.1, 0.5, 0.9], [0.2, 0.3, 0.7]])  # in every domain
    (y,) = eval_itype(getattr(tensor.real, name)(S), [x])
    expected = _real([[UNARY[name](v) for v in row] for row in x.tolist()])
    torch.testing.assert_close(y, expected)


def test_torch_evaluates_atan2_in_every_quadrant():
    ys, xs = _real([[1.0, 1.0, -1.0], [-1.0, 0.0, 2.0]]), _real([[1.0, -1.0, -1.0], [1.0, -3.0, 0.0]])
    (a,) = eval_itype(tensor.real.Atan2(S), [ys, xs])
    expected = _real([[math.atan2(y, x) for y, x in zip(ry, rx)] for ry, rx in zip(ys.tolist(), xs.tolist())])
    torch.testing.assert_close(a, expected)


# ── Z3 ────────────────────────────────────────────────────────────


def _z3(op, *args):
    return rz3.eval(op, [np.array([[a]], dtype=object) for a in args])[0][0, 0]


def test_z3_square_root_is_exact():
    x = z3.Real("x")
    s = z3.Solver()
    s.add(x == 9, _z3(tensor.real.Sqrt([1, 1]), x) != 3)
    assert s.check() == z3.unsat


def test_z3_other_functions_are_one_uninterpreted_function_each():
    x, y = z3.Reals("x y")
    sin = tensor.real.Sin([1, 1])
    # equal arguments give equal values, whichever term they come from
    s = z3.Solver()
    s.add(x == y, _z3(sin, x) != _z3(sin, y))
    assert s.check() == z3.unsat
    # distinct functions are not confused
    s = z3.Solver()
    s.add(_z3(sin, x) != _z3(tensor.real.Cos([1, 1]), x))
    assert s.check() == z3.sat
    assert z3.eq(_z3(tensor.real.Atan2([1, 1]), y, x), rz3.transcendental("atan2", 2)(y, x))


# ── builder, expr and sugar ───────────────────────────────────────


def test_the_real_builder_builds_them_on_plain_reals_only():
    b = builder_for(Real)
    a = Wire(Real([1, 1]))
    assert b.sin(a).itype == tensor.real.Sin([1, 1])
    assert b.atan2(a, Wire(Real([1, 1]))).itype == tensor.real.Atan2([1, 1])
    with pytest.raises(TheoryError, match="degree 0"):
        b.cos(Wire(Real([1, 1], degree=1)))
    with pytest.raises(TheoryError, match="not supported"):
        builder_for(Int).sqrt(Wire(Int([1, 1])))


def test_expr_functions_build_and_evaluate():
    x = Var(Real([1, 1]))
    with collecting() as terms:
        e = expr(x)
        y = atan2(sin(e), cos(e)) + sqrt(e * e) - tanh(e)
    state = {x: _real([[0.4]])}
    for t in terms:
        for w, r in zip(t.write, eval_itype(t.itype, [state[w] for w in t.read])):
            state[w] = r
    assert float(state[y.wire]) == pytest.approx(0.4 + 0.4 - math.tanh(0.4))


def test_expr_functions_refuse_rates_of_change():
    with collecting():
        rate = expr(Wire(Real([1, 1], degree=1)))
        with pytest.raises(TypeError, match="degree 0"):
            sin(rate)


def _rate(rate: float, e):
    """The rate of change `rate · e` of the plain real `e`: the product with a
    constant of degree 1, which `*` would make a `Scale` of degree 0."""
    from reactor import Term
    from reactor.expr import _emit, _wrap

    c = expr(_real([[rate]]), sort=Real([1, 1], degree=1))
    term = Term(tensor.real.Mul([1, 1], degrees=[1, 0]), [Wire(Real([1, 1], degree=1))],
                [c.wire, e.wire])
    _emit(term)
    return _wrap(term.write[0])


def test_a_pendulum_flows_with_a_sine():
    """d(θ) = ω, d(ω) = -g sin θ: a differential module that needs `Sin`."""

    class Pendulum(sugar.Module):
        def init(self):
            return 0.3, 0.0

        def delay(self, theta, omega):
            return _rate(1.0, omega), _rate(-9.81, sin(theta))

    theta, omega = Var(Real([1, 1])), Var(Real([1, 1]))
    m = Pendulum(ctrl=(theta, omega))
    (atom,) = m.atoms
    assert {d(theta), d(omega)} <= set(atom.delay.write())
    assert any(t.itype == tensor.real.Sin([1, 1]) for t in atom.delay)


# ── the analyzer ──────────────────────────────────────────────────


class _Trig(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(1,))
        self.x = 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0.5
        return self.x, 0.0, False, False

    def step(self, action):
        self.x = math.atan2(math.sin(self.x), math.cos(self.x)) + math.sqrt(self.x * self.x) * 0.1
        return self.x, 0.0, False, False


def test_the_analyzer_translates_math_functions():
    e = Env(_Trig(), attrs=Real([1, 1]), interpret=True)
    e.reset()
    e.step(0)
    assert e.x == pytest.approx(0.5 + 0.05)
