"""Tests for reactor.casadi: CasADi -> Reactor translation and its validation."""

import math

import numpy as np
import pytest
import torch

cs = pytest.importorskip("casadi")

from reactor import Real, X, d
from reactor.eval import execute_init, execute_update
from reactor.casadi import (
    UnsupportedOp,
    check_equivalence,
    check_numeric,
    differential,
    euler,
    rk4,
    sequential,
    to_terms,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _mixed():
    """Linear arithmetic, constant folding, comparisons and every piecewise op."""
    x, u = cs.SX.sym("x", 2), cs.SX.sym("u")
    mass = 0.0434
    y = cs.vertcat(
        3 * x[0] + 0.5 * x[1] - u / 4,
        cs.if_else(u > x[0], x[1], -x[1]),
        cs.fmax(x[0], 1.5),
        2.0,  # constant output
        cs.fabs(x[1] - u),
        cs.sign(u),
        cs.fmin(x[0], x[1]) * (1 / mass) + math.sin(0.3) * u,  # folded constants
        cs.logic_and(x[0] < 0, u >= 0),  # boolean output as 0/1
    )
    return cs.Function("f", [x, u], [y, x[0] - x[1]], ["x", "u"], ["y", "z"])


def _run(cm, steps, inputs):
    """Simulate a CasadiModule with Reactor's torch evaluator."""
    state = {}
    execute_init(state, cm.module.atoms)
    for v in cm.state:
        state[v] = state[X(v)].clone()
    trace = [[float(state[v]) for v in cm.state]]
    for k in range(steps):
        for group, vals in zip(cm.inputs, inputs(k)):
            for v, val in zip(group, vals):
                state[v] = torch.tensor([[val]], dtype=torch.float64)
        execute_update(state, cm.module.atoms)
        for v in cm.state:
            state[v] = state[X(v)].clone()
        trace.append([float(state[v]) for v in cm.state])
    return np.array(trace)


def _hover_linearization(dt=0.002):
    """Euler step of a quadrotor linearized at hover, via cs.jacobian.

    State p, v, rpy, ω (12) and motor thrusts (4); input: thrust commands (4).
    The nonlinear model is polynomial (thrust rotated by the small-angle
    attitude, gyroscopic term), so it translates with products of variables;
    the linearization translates into the linear fragment only.
    """
    m, g, L, tau = 0.0434, 9.81, 0.035, 0.05
    J = [25e-6, 28e-6, 49e-6]
    k_yaw = 0.006
    p, v, rpy, w = (cs.SX.sym(n, 3) for n in ("p", "v", "rpy", "w"))
    F, u = cs.SX.sym("F", 4), cs.SX.sym("u", 4)
    T = cs.sum1(F)
    phi, theta = rpy[0], rpy[1]
    acc = cs.vertcat(theta * T / m, -phi * T / m, T / m - g)
    tau_b = cs.vertcat(L * (F[1] + F[2] - F[0] - F[3]), L * (F[2] + F[3] - F[0] - F[1]),
                       k_yaw * (F[0] - F[1] + F[2] - F[3]))
    wdot = cs.vertcat(*[(tau_b[i] - (w[(i + 1) % 3] * w[(i + 2) % 3]) * (J[(i + 2) % 3] - J[(i + 1) % 3])) / J[i]
                        for i in range(3)])
    xs = cs.vertcat(p, v, rpy, w, F)
    xdot = cs.vertcat(v, acc, w, wdot, (u - F) / tau)
    f = cs.Function("quad", [xs, u], [xdot], ["x", "u"], ["xdot"])

    x0 = np.concatenate([np.zeros(12), np.full(4, m * g / 4)])
    u0 = np.full(4, m * g / 4)
    A = np.array(cs.DM(cs.Function("A", [xs, u], [cs.jacobian(xdot, xs)])(x0, u0)))
    B = np.array(cs.DM(cs.Function("B", [xs, u], [cs.jacobian(xdot, u)])(x0, u0)))
    c = np.array(cs.DM(f(x0, u0))).reshape(-1)  # zero at hover, kept for generality
    dx, du = cs.SX.sym("dx", 16), cs.SX.sym("du", 4)
    lin = cs.Function("quad_lin", [dx, du], [cs.mtimes(cs.DM(A), dx) + cs.mtimes(cs.DM(B), du) + cs.DM(c)],
                      ["x", "u"], ["xdot"])
    return f, euler(lin, dt)


# ── translation ──────────────────────────────────────────────────────────────


def test_mixed_function_matches_numerically():
    assert check_numeric(_mixed(), 300, low=-3.0, high=3.0) == 0.0


def test_mixed_function_is_proved_equivalent():
    result = check_equivalence(_mixed())
    assert result.proved, str(result)
    assert len(result.entries) == 9  # 8 entries of y, 1 of z


def test_constant_inputs_partially_evaluate():
    x, u = cs.SX.sym("x"), cs.SX.sym("u")
    f = cs.Function("f", [x, u], [x * u + 1])  # nonlinear in (x, u), linear once u is fixed
    from reactor.casadi import translate, as_real
    from reactor.expr import collecting, expr
    from reactor import Var

    xv = Var(Real([1, 1]))
    with collecting() as terms:
        (out,) = translate(f, [expr(xv), 2.5])
        y = as_real(out[0])
    # 2.5 * x + 1: one Scale, one constant, one Add
    assert sorted(type(t.itype).__name__ for t in terms) == ["Add", "Const", "Scale"]
    state = {xv: torch.tensor([[4.0]], dtype=torch.float64)}
    from reactor.eval import eval_itype

    for t in terms:
        for w, r in zip(t.write, eval_itype(t.itype, [state[w] for w in t.read])):
            state[w] = r
    assert float(state[y.wire]) == 11.0


def _ops(terms):
    return {type(t.itype).__name__ for t in terms}


def test_polynomial_and_rational_functions_translate():
    x = cs.SX.sym("x", 3)
    f = cs.Function("f", [x], [cs.vertcat(
        x[0] * x[1] - 3 * x[2] ** 2,     # products, an integer power
        x[0] ** 5 - x[1] ** -2,           # repeated squaring, a negative power
        x[0] / x[1] + 1 / x[2],           # division by variables, the inverse
        cs.fabs(x[2]) * -x[0],            # Abs, Neg
    )])
    t = to_terms(f)
    assert {"Mul", "Div", "Abs", "Neg"} <= _ops(t.terms)
    assert check_numeric(f, 200, low=0.5, high=2.0, terms=t) < 1e-12
    result = check_equivalence(f, terms=t)
    assert result.proved, str(result)


def test_transcendental_of_variable_is_rejected():
    x = cs.SX.sym("x")
    with pytest.raises(UnsupportedOp, match="sin"):
        to_terms(cs.Function("f", [x], [cs.sin(x)]))
    with pytest.raises(UnsupportedOp, match="sqrt"):
        to_terms(cs.Function("f", [x], [cs.sqrt(x)]))
    with pytest.raises(UnsupportedOp, match="pow"):
        to_terms(cs.Function("f", [x], [x**1.5]))


def test_division_by_a_constant_stays_linear():
    x = cs.SX.sym("x")
    t = to_terms(cs.Function("f", [x], [x / 3]))
    assert _ops(t.terms) == {"Scale"}


def test_mx_function_is_expanded():
    x = cs.MX.sym("x", 3)
    A = cs.DM([[1, 2, 0], [0, -1, 4]])
    f = cs.Function("f", [x], [cs.mtimes(A, x)])
    assert check_numeric(f) < 1e-12
    assert check_equivalence(f).proved


def test_sparse_output_structural_zeros():
    x = cs.SX.sym("x", 2)
    y = cs.SX(3, 1)
    y[0] = x[0]
    y[2] = 2 * x[1]  # y[1] is a structural zero
    f = cs.Function("f", [x], [y])
    t = to_terms(f)
    assert len(t.outputs[0]) == 3
    assert check_numeric(f, terms=t) == 0.0
    assert check_equivalence(f, terms=t).proved


# ── the miter catches wrong translations ─────────────────────────────────────


def test_miter_finds_strict_vs_nonstrict_comparison():
    x, u = cs.SX.sym("x"), cs.SX.sym("u")
    right = cs.Function("f", [x, u], [cs.if_else(u > x, x, -x)])
    wrong = cs.Function("g", [x, u], [cs.if_else(u >= x, x, -x)])
    result = check_equivalence(right, terms=to_terms(wrong))
    assert not result.proved
    (entry,) = result.failures()
    assert entry.status == "counterexample"
    cex = entry.counterexample
    assert cex["i0[0]"] == cex["i1[0]"] and cex["i0[0]"] != 0  # only u == x != 0 differs
    # random sampling essentially never hits u == x
    assert check_numeric(right, 1000, terms=to_terms(wrong)) == 0.0


def test_miter_finds_wrong_coefficient():
    x = cs.SX.sym("x")
    right = cs.Function("f", [x], [0.1 * x])
    wrong = cs.Function("g", [x], [0.1000001 * x])
    assert not check_equivalence(right, terms=to_terms(wrong)).proved


def test_miter_respects_assumptions():
    z3 = pytest.importorskip("z3")
    x = cs.SX.sym("x")
    right = cs.Function("f", [x], [cs.fabs(x)])
    same_if_positive = cs.Function("g", [x], [x])
    xs = [[z3.Real("x")]]

    assert not check_equivalence(right, terms=to_terms(same_if_positive), inputs=xs).proved
    assert check_equivalence(
        right, terms=to_terms(same_if_positive), inputs=xs, assumptions=[xs[0][0] >= 0]
    ).proved


# ── modules ──────────────────────────────────────────────────────────────────


def test_sequential_module_simulates_like_casadi():
    x, u = cs.SX.sym("x", 2), cs.SX.sym("u")
    f = cs.Function("osc", [x, u], [cs.vertcat(x[1], -4 * x[0] - 0.3 * x[1] + cs.fmin(u, 0.5))])
    step = rk4(f, 0.05)
    cm = sequential(step, init=[1.0, 0.0])
    assert len(cm.state) == 2 and len(cm.inputs) == 1 and len(cm.inputs[0]) == 1
    assert set(cm.module.ctrl) == set(cm.state)

    us = [math.sin(0.3 * k) for k in range(40)]
    trace = _run(cm, 40, lambda k: [[us[k]]])
    xk = np.array([1.0, 0.0])
    for k in range(40):
        xk = np.array(step(xk, us[k])).reshape(-1)
        np.testing.assert_allclose(trace[k + 1], xk, rtol=0, atol=1e-12)


def test_sequential_unconstrained_init_is_havoc():
    x = cs.SX.sym("x")
    cm = sequential(cs.Function("f", [x], [x + 1]), init=[None])
    assert "Havoc" in str(cm.module)


def test_quadrotor_nonlinear_and_linearization_proved():
    f, step = _hover_linearization()
    tf = to_terms(f)
    assert "Mul" in _ops(tf.terms)
    assert check_numeric(f, 50, terms=tf, low=-0.5, high=0.5) < 1e-9
    result = check_equivalence(f, terms=tf)
    assert result.proved, str(result)

    t = to_terms(step)
    assert "Mul" not in _ops(t.terms) and "Div" not in _ops(t.terms)
    assert check_numeric(step, 50, terms=t) < 1e-12
    result = check_equivalence(step, terms=t)
    assert result.proved, str(result)
    cm = sequential(step, init=[0.0] * 16)
    assert len(cm.state) == 16 and len(cm.inputs[0]) == 4


def test_differential_module_keeps_the_dynamics():
    x, u = cs.SX.sym("x", 2), cs.SX.sym("u")
    f = cs.Function("f", [x, u], [cs.vertcat(x[1], -x[0] * x[0] * x[1] + u, 0.0)[:2]])
    cm = differential(f, init=[1.0, None])
    (atom,) = cm.module.atoms
    rates = {d(v) for v in cm.state}
    assert rates <= set(atom.delay.write())
    assert all(w.dtype == Real([1, 1], degree=1) for w in rates)
    # each rate is the product of the rate 1 with the translated entry
    muls = [t for t in atom.delay if type(t.itype).__name__ == "Mul"]
    assert any(list(t.itype.degrees) == [1, 0] for t in muls)


def test_differential_constant_rate():
    x = cs.SX.sym("x", 2)
    f = cs.Function("f", [x], [cs.vertcat(1.5, -x[0])])
    cm = differential(f, init=[0.0, 0.0])
    consts = [t for t in cm.module.atoms[0].delay if type(t.itype).__name__ == "Const"]
    assert any(t.itype.degree == 1 and float(t.itype.tensor) == 1.5 for t in consts)
