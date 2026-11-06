"""Tests for `sugar.Module` — the subclass-a-Module DSL front-end.

A `sugar.Module` subclass *is* a base Module: pass `theory` and the `ctrl`(/`extl`)
variables and override `init`/`update`/`delay`, and the module is built in the constructor.
These tests build small modules and step them through `reactor.eval` (mirroring test_eval),
plus check the ctrl/extl partition and the config surface.
"""

import pytest

from reactor import LIA, Module, Sort, Wire, sugar, Int, Var, Real, LRA, X as _X, d as _d, NonLinearError
from reactor.sugar import ite, d, X
from reactor.eval import execute_init, execute_update

INT = Int([1, 1])
Real1 = Real([1, 1])


def _pair():
    """A fresh variable (test helper)."""
    return Var(INT)


# --- stepping helpers (same shape as test_eval) -----------------------------


def _init(m):
    state = {}
    execute_init(state, m.atoms)
    return state


def _latch(m, state):
    return {var: state[_X(var)] for var in m.ctrl}


def _update(m, state):
    execute_update(state, m.atoms)
    return state


def _trace(m, steps, view):
    """The value of `view` after init and after each of `steps` updates."""
    state = _init(m)
    out = [state[view].item()]
    for _ in range(steps):
        state = _update(m, _latch(m, state))
        out.append(state[view].item())
    return out


# --- counter (closed, single var, no extl) ----------------------------------


class Counter(sugar.Module):
    def init(self):
        return 0

    def update(self, cnt):
        return cnt + 1


def test_counter_is_a_closed_base_module():
    m = Counter(theory=LIA, ctrl=(_pair(),))
    assert isinstance(m, Module)
    assert m.closed()


def test_counter_counts():
    x = _pair()
    m = Counter(theory=LIA, ctrl=(x,))
    print(m)
    # (_x_lat, x_nxt) = x
    assert _trace(m, 5, _X(x)) == [0, 1, 2, 3, 4, 5]


# --- multi-var with ite and an unchanged var --------------------------------


class Bounded(sugar.Module):
    def init(self):
        return 0, 3

    def update(self, x, cap):
        return ite(x < cap, x + 1, x), cap  # x climbs to cap, cap unchanged


def test_multivar_ite_and_hold():
    cap, x = _pair(), _pair()
    m = Bounded(theory=LIA, ctrl=(x, cap))
    assert m.closed()
    assert _trace(m, 5, _X(x)) == [0, 1, 2, 3, 3, 3]
    assert _trace(m, 5, _X(cap)) == [3, 3, 3, 3, 3, 3]


# --- extl variables become external inputs (open module) --------------------


class Gate(sugar.Module):
    def init(self, _y):
        return 0

    def update(self, x, y):
        return ite(x < y, x + 1, y)


def test_extl_is_external_and_module_is_open():
    x, y = _pair(), _pair()  # x controlled, y external
    m = Gate(theory=LIA, ctrl=(x,), extl=(y,))
    assert m.open()
    assert len(list(m.ctrl)) == 1
    assert len(list(m.extl)) == 1


# --- config surface ---------------------------------------------------------


def test_builds_from_wire_pairs_passed_directly():
    class C(sugar.Module):
        def init(self):
            return 0

        def update(self, ctrl):
            return ctrl + 1

    m = C(theory=LIA, ctrl=(_pair(),))
    assert m.closed()


def test_missing_config_raises():
    class Bad(sugar.Module):
        def update(self, ctrl):
            return ctrl

    with pytest.raises(TypeError):
        Bad()  # no theory / ctrl


def test_missing_init_raises():
    class NoInit(sugar.Module):
        def update(self, ctrl):
            return ctrl + 1

    # init is required — every ctrl variable needs an initial value
    # not true anymore - this is a jump module
    # with pytest.raises(TypeError):

    m = NoInit(theory=LIA, ctrl=(_pair(),))
    assert m.closed()


def test_subexpression_shared_across_returns():
    # a guard reused in several returned values must be emitted once, not once per
    # value (else its wire is "written more than once")
    class Shared(sugar.Module):
        def init(self):
            return 0, 0

        def update(self, x, y):
            loop = x < y
            return ite(loop, x + 1, x), ite(loop, y, y - 1)

    m = Shared(theory=LIA, ctrl=(_pair(), _pair()))
    assert m.closed()


def test_simple_clock():
    class Simple(sugar.Module):
        def init(self, t):
            return 0

        def update(self, x, t):
            return ite(X(t) == 1, 0, x)

        def delay(self, x, t):
            return 1 * d(t)

    x = Var(Real1)
    t = Var(Real1)
    m = Simple(ctrl=(x,), extl=(t,), theory=LRA)


def test_differential_sugar_module():
    """init + delay (no update) builds a differential module: the update is
    the synthesised skip, and the delay drives the derivative wires."""

    class Drift(sugar.Module):
        def init(self, t):
            return 0

        def delay(self, x, t):
            return d(t) + d(t)  # dx = 2 dt: a drifting clock

    x = Var(Real1)
    t = Var(Real1)
    m = Drift(ctrl=(x,), extl=(t,), theory=LRA)

    assert isinstance(m, Module)
    assert m.open()  # t is inferred external
    assert m.intf == [x]
    assert m.extl == [t]

    a = m.atoms[0]
    # the atom controls x and awaits t; the delay drives d(x), the default skip
    # update keeps x, and discards the next value of t, which only the delay
    # uses (as d(t))
    assert a.ctrl == [x] and a.wait == [t] and a.read == [x, t]
    assert [w.dtype for w in a.delay.write] == [_d(x).dtype]
    assert len(a.update) == 1 + 2


def test_hybrid_sugar_module():
    """init + update + delay builds a hybrid module: all three blocks are
    explicit, mixing a discrete reset with continuous drift."""

    class Clock(sugar.Module):
        def init(self, t):
            return 0

        def update(self, x, t):
            return ite(X(t) == 1, 0, x)  # discrete reset when t' hits 1

        def delay(self, x, t):
            return 1 * d(t)  # continuous drift with t

    x = Var(Real1)
    t = Var(Real1)
    m = Clock(ctrl=(x,), extl=(t,), theory=LRA)

    assert isinstance(m, Module)
    assert m.open()
    assert m.intf == [x]
    assert m.extl == [t]

    a = m.atoms[0]
    # every block is explicit and drives its value of x
    assert [w.dtype for w in a.init.write] == [_X(x).dtype]
    assert [w.dtype for w in a.update.write] == [_X(x).dtype]
    assert [w.dtype for w in a.delay.write] == [_d(x).dtype]
    # the update takes the latched x and t, and the next value of t (the
    # reset guard)
    assert len(a.update.read) == 3 and a.wait == [t]


def test_invalid_nonlinear():
    class Simple(sugar.Module):
        def init(self, t):
            return 0

        def update(self, x, t):
            return ite(X(t) == 1, 0, x)

        def delay(self, x, t):
            return x * d(t)

    x = Var(Real1)
    t = Var(Real1)
    with pytest.raises(NonLinearError):
        Simple(ctrl=(x,), extl=(t,), theory=LRA)


def test_useless_module():
    class Useless(sugar.Module):
        def init(self, i):
            return ()

        def update(self, i):
            return ()

    i = Var(Real1)
    m = Useless(ctrl=(), extl=(i,), theory=LRA)
    # no control: the declared external is the only variable, awaited though unused
    assert m.extl == [i]
    assert len(m.intf) + len(m.prvt) == 0


def test_private():
    class DigitalDelay(sugar.Module):
        def update(self, x, _o, i):
            return (i, x)

    i = Var(Real1)
    o = Var(Real1)
    x = Var(Real1)
    m = DigitalDelay(ctrl=(x, o), extl=(i,), hide=frozenset([x]), theory=LRA)

    assert i in m.extl
    assert o in m.intf
    assert x in m.prvt


def test_compose_hiding_coupled_variables():
    class Producer(sugar.Module):
        def init(self, _w):
            return (0, 0)

        def update(self, x, y, w):
            return (w, y + 1)

    class Consumer(sugar.Module):
        def init(self, _y, _w):
            return 0

        def update(self, _z, y, w):
            return y + w

    x, y, z, w = (Var(INT) for _ in range(4))

    # m1 controls x and y, reading the shared input w; m2 reads y and w
    # and controls z: y couples the two, w is external on both sides
    m1 = Producer(theory=LIA, ctrl=(x, y), extl=(w,))
    m2 = Consumer(theory=LIA, ctrl=(z,), extl=(y, w))

    assert y in m1.intf and y in m2.extl
    assert w in m1.extl and w in m2.extl

    # hide exactly the coupling variables: interface on one side, external
    # on the other (in either direction) — not the shared external w
    def coupled(v):
        return (v in m1.intf and v in m2.extl) or (v in m1.extl and v in m2.intf)

    m = Module.compose(m1, m2, hide=coupled)

    assert m.prvt == [y]
    assert m.intf == [x, z]
    assert m.extl == [w]

    # the coupling took place regardless: hiding only restricts visibility
    assert y not in m.obs and y not in m.extl
