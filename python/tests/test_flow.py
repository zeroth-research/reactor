"""Tests of running the delay blocks: `execute_delay`, `flowing`, `rates` and
`execute_flow` of `reactor.eval`, which integrate a differential module."""

import math

import pytest
import torch

from reactor import Real, Var, X, d
from reactor import sugar
from reactor.eval import execute_delay, execute_flow, execute_init, flowing, rates
from reactor.reactor import tensor


def _f64(v):
    return torch.tensor([[v]], dtype=torch.float64)


def _decay(k: float):
    """d(x) = -k·x, from x = 1."""

    class Decay(sugar.Module):
        def init(self):
            return 1.0

        def delay(self, x):
            from reactor import Term, Wire
            from reactor.expr import _emit, _wrap, expr

            c = expr(_f64(-k), sort=Real([1, 1], degree=1))
            term = Term(tensor.real.Mul([1, 1], degrees=[1, 0]), [Wire(Real([1, 1], degree=1))],
                        [c.wire, x.wire])
            _emit(term)
            return _wrap(term.write[0])

    x = Var(Real([1, 1]))
    return Decay(ctrl=(x,)), x


def _start(module):
    state = {}
    execute_init(state, module.atoms)
    for v in module.ctrl:
        if X(v) in state:
            state[v] = state[X(v)].to(torch.float64)
    return state


def test_the_delay_block_writes_the_rates():
    m, x = _decay(2.0)
    state = _start(m)
    execute_delay(state, m.atoms)
    assert float(state[d(x)]) == -2.0
    assert flowing(m.atoms) == [x]
    assert [float(r) for r in rates({x: _f64(0.5)}, m.atoms)] == [-1.0]


@pytest.mark.parametrize("method, tol", [("euler", 1e-2), ("rk4", 1e-9)])
def test_a_decay_flows_to_its_closed_form(method, tol):
    m, x = _decay(2.0)
    state = _start(m)
    for _ in range(100):
        execute_flow(state, m.atoms, 0.01, method)
    assert float(state[x]) == pytest.approx(math.exp(-2.0), abs=tol)


def test_it_flows_as_the_casadi_step():
    cs = pytest.importorskip("casadi")
    import numpy as np

    from reactor.casadi import differential, rk4

    x, u = cs.SX.sym("x", 2), cs.SX.sym("u")
    f = cs.Function("f", [x, u], [cs.vertcat(x[1], -4 * x[0] * x[0] * x[1] - x[0] + cs.sin(u))])
    cm = differential(f, init=[1.0, 0.0])
    state = _start(cm.module)
    (uvar,) = cm.inputs[0]
    step, xc = rk4(f, 0.01), np.array([1.0, 0.0])
    for k in range(200):
        state[uvar] = _f64(0.3 * math.sin(0.1 * k))
        execute_flow(state, cm.module.atoms, 0.01)
        xc = np.array(step(xc, 0.3 * math.sin(0.1 * k))).ravel()
    assert [float(state[v]) for v in cm.state] == xc.tolist()  # the same float64 operations


def test_a_hybrid_module_holds_its_discrete_part_while_it_flows():
    class Clock(sugar.Module):
        def init(self, t):
            return 0.0

        def update(self, x, t):
            return x

        def delay(self, x, t):
            return sugar.d(t) + sugar.d(t)

    x, t = Var(Real([1, 1])), Var(Real([1, 1]))
    m = Clock(ctrl=(x,), extl=(t,))
    assert flowing(m.atoms) == [x]


def test_an_unknown_method_is_refused():
    m, _ = _decay(1.0)
    with pytest.raises(ValueError, match="unknown method"):
        execute_flow(_start(m), m.atoms, 0.1, "midpoint")


# ── Runner: the compiled evaluation ──────────────────────────────


def _oscillator():
    cs = pytest.importorskip("casadi")
    x, u = cs.SX.sym("x", 2), cs.SX.sym("u")
    return cs.Function("f", [x, u], [cs.vertcat(x[1], -4 * x[0] * x[0] * x[1] - x[0] + cs.sin(u))])


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_a_runner_flows_exactly_as_execute_flow(method):
    from reactor.casadi import differential
    from reactor.eval import Runner

    cm = differential(_oscillator(), init=[1.0, 0.0])
    (uvar,) = cm.inputs[0]
    state, runner = _start(cm.module), Runner(cm.module)
    runner.init()
    assert runner.flowing == flowing(cm.module.atoms)
    for k in range(50):
        state[uvar] = runner[uvar] = _f64(math.sin(0.1 * k))
        execute_flow(state, cm.module.atoms, 0.01, method)
        runner.flow(0.01, method)
    assert all(torch.equal(state[v], runner[v]) for v in cm.state)


def test_a_runner_steps_exactly_as_execute_update():
    from reactor.casadi import rk4, sequential
    from reactor.eval import Runner, execute_update

    cm = sequential(rk4(_oscillator(), 0.01), init=[1.0, 0.0])
    (uvar,) = cm.inputs[0]
    state, runner = _start(cm.module), Runner(cm.module)
    runner.init()
    for k in range(50):
        state[uvar] = runner[uvar] = _f64(math.sin(0.1 * k))
        execute_update(state, cm.module.atoms)
        for v in cm.state:
            state[v] = state[X(v)]
        runner.update()
    assert all(torch.equal(state[v], runner[v]) for v in cm.state)
    assert [float(r) for r in Runner(cm.module).rates()] == []  # a sequential module does not flow


def test_a_runner_refuses_a_wire_the_module_does_not_use():
    from reactor.eval import Runner

    m, _ = _decay(1.0)
    with pytest.raises(KeyError):
        Runner(m)[Var(Real([1, 1]))] = _f64(0.0)
