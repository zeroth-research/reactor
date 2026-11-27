"""Tests for `sugar.Module` — the subclass-a-Module DSL front-end.

A `sugar.Module` subclass *is* a base Module: pass the `ctrl`(/`extl`) variables and
override `init`/`update`, and the sequential module is built in the constructor.
These tests build small modules and step them through `reactor.eval` (mirroring test_eval),
plus check the ctrl/extl partition and the config surface.
"""

import pytest
import torch

import reactor
from reactor import Module, Wire, sugar, Var, T, TheoryError
from reactor.bool import Bool
from reactor.int import Int
from reactor.real import Real
from reactor.enum import Enum
from reactor.event import Event
from reactor.sugar import ite, d, X, Zero, expr, recv, Havoc
from reactor import Skip
from reactor.zero import Real0
from reactor.eval import eval_itype

INT = Int([1, 1])
Real1 = Real([1, 1])


def _pair():
    """A fresh (latched, next) wire pair (test helper)."""
    return Var(INT)


# --- stepping helpers (same shape as test_eval) -----------------------------


def _run_block(m, state, get_block):
    for a in m.atoms:
        for t in get_block(a):
            read = [state[w] for w in t.read]
            out_sort = t.write[0].dtype if len(t.write) else None
            state.update(zip(t.write, eval_itype(t.itype, read, out_sort)))
    return state


def _init(m):
    return _run_block(m, {}, lambda a: a.init)


def _latch(m, state):
    return {var: state[reactor.X(var)] for var in m.ctrl}


def _update(m, state):
    return _run_block(m, state, lambda a: a.update)


def _trace(m, steps, wire):
    """Latched value of `wire` after init and after each of `steps` updates."""
    state = _init(m)
    out = [state[wire].item()]
    for _ in range(steps):
        state = _update(m, _latch(m, state))
        out.append(state[wire].item())
    return out


# --- counter (closed, single var, no extl) ----------------------------------


class Counter(sugar.Module):
    def init(self):
        return 0

    def update(self, cnt):
        return cnt + 1


def test_counter_is_a_closed_base_module():
    m = Counter(ctrl=(_pair(),))
    assert isinstance(m, Module)
    assert m.closed()


def test_counter_counts():
    x = _pair()
    m = Counter(ctrl=(x,))
    print(m)
    # (_x_lat, x_nxt) = x
    assert _trace(m, 5, reactor.X(x)) == [0, 1, 2, 3, 4, 5]


# --- multi-var with ite and an unchanged var --------------------------------


class Bounded(sugar.Module):
    def init(self):
        return 0, 3

    def update(self, x, cap):
        return ite(x < cap, x + 1, x), cap  # x climbs to cap, cap unchanged


def test_multivar_ite_and_hold():
    cap, x = _pair(), _pair()
    m = Bounded(ctrl=(x, cap))
    assert m.closed()
    assert _trace(m, 5, reactor.X(x)) == [0, 1, 2, 3, 3, 3]
    assert _trace(m, 5, reactor.X(cap)) == [3, 3, 3, 3, 3, 3]


# --- extl variables become external inputs (open module) --------------------


class Gate(sugar.Module):
    def init(self, _y):
        return 0

    def update(self, x, y):
        return ite(x < y, x + 1, y)


def test_extl_is_external_and_module_is_open():
    x, y = _pair(), _pair()  # x controlled, y external
    m = Gate(ctrl=(x,), extl=(y,))
    assert m.open()
    assert len(list(m.ctrl)) == 1
    assert len(list(m.extl)) == 1


# --- a scalar condition chooses between matrices ----------------------------


A4 = torch.arange(16.0).reshape(4, 4)
B4 = -A4 - 1


class Choose(sugar.Module):
    def update(self, m, x, y):
        return ite(x < y, expr(A4, sort=Real), expr(B4, sort=Real))


def test_ite_by_a_scalar_condition_chooses_between_matrices():
    # x and y are scalars, [1, 1], and the branches [4, 4] matrices: the condition
    # chooses the whole matrix
    m, x, y = Var(Real([4, 4])), Var(Real1), Var(Real1)
    module = Choose(ctrl=(m,), extl=(x, y))
    for xv, yv, expected in [(1.0, 2.0, A4), (2.0, 1.0, B4)]:
        state = {m: torch.zeros(4, 4), x: torch.tensor([[xv]]), y: torch.tensor([[yv]])}
        assert torch.equal(_update(module, state)[reactor.X(m)], expected)


P4 = torch.eye(4, dtype=torch.bool)
Q4 = ~P4


class ChooseBool(sugar.Module):
    def update(self, m, c):
        return ite(c.any(), expr(P4, sort=Bool), expr(Q4, sort=Bool))


def test_ite_by_a_reduced_condition_chooses_between_boolean_matrices():
    # the same choice of whole matrices, between boolean ones: the [1, 1] condition is
    # whether any entry of the boolean matrix c holds, reduced along both axes
    m, c = Var(Bool([4, 4])), Var(Bool([2, 3]))
    module = ChooseBool(ctrl=(m,), extl=(c,))
    some = torch.tensor([[False, False, False], [False, True, False]])
    none = torch.zeros(2, 3, dtype=torch.bool)
    for cv, expected in [(some, P4), (none, Q4)]:
        state = {m: torch.zeros(4, 4, dtype=torch.bool), c: cv}
        assert torch.equal(_update(module, state)[reactor.X(m)], expected)


# --- config surface ---------------------------------------------------------


def test_builds_from_wire_pairs_passed_directly():
    class C(sugar.Module):
        def init(self):
            return 0

        def update(self, ctrl):
            return ctrl + 1

    m = C(ctrl=(_pair(),))
    assert m.closed()


def test_missing_config_raises():
    class Bad(sugar.Module):
        def update(self, ctrl):
            return ctrl

    with pytest.raises(ValueError, match="update expects"):
        Bad()  # no ctrl


def test_missing_init_raises():
    class NoInit(sugar.Module):
        def update(self, ctrl):
            return ctrl + 1

    # init is required — every ctrl variable needs an initial value
    # not true anymore - this is a jump module
    # with pytest.raises(TypeError):

    m = NoInit(ctrl=(_pair(),))
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

    m = Shared(ctrl=(_pair(), _pair()))
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
    m = Simple(ctrl=(x,), extl=(t,))


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
    m = Drift(ctrl=(x,), extl=(t,))

    assert isinstance(m, Module)
    assert m.open()  # t is inferred external
    assert m.intf == [x]
    assert m.extl == [t]

    a = m.atoms[0]
    # the delay drives d(x); the synthesised skip update drives X(x)
    assert reactor.d(x) in a.delay.write()
    assert reactor.X(x) in a.update.write()
    assert len(a.update) == 1


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
    m = Clock(ctrl=(x,), extl=(t,))

    assert isinstance(m, Module)
    assert m.open()
    assert m.intf == [x]
    assert m.extl == [t]

    a = m.atoms[0]
    # every block is explicit and drives its wire of x
    assert reactor.X(x) in a.init.write()
    assert reactor.X(x) in a.update.write()
    assert reactor.d(x) in a.delay.write()
    # the update reads the reset guard from the input's next value
    assert reactor.X(t) in a.update.read()


class _Mixed(sugar.Module):
    """A real variable drifting with a real input, and a discrete counter."""

    def init(self, t):
        return 0, 0

    def update(self, x, n, t):
        return x, n + 1

    def delay(self, x, n, t):
        return 1 * d(t), Zero()  # the counter does not change in time: its rate is zero


def test_delay_of_mixed_real_and_discrete_variables():
    x, n, t = Var(Real1), Var(Int([1, 1])), Var(Real1)
    a = _Mixed(ctrl=(x, n), extl=(t,)).atoms[0]
    writer = {w: term for term in a.delay for w in term.write}
    # the counter's rate of change is the singleton: `Zero`, reading nothing, copied
    # into d(n) at the singleton sort
    skip = writer[reactor.d(n)]
    assert skip.itype == Skip(Real0())
    zero = writer[skip.read[0]]
    assert zero.itype == Zero() and len(zero.read) == 0
    # the real's rate of change is a real of degree 1, copied at that sort
    skip = writer[reactor.d(x)]
    assert skip.itype == Skip(T(Real([1, 1])))
    assert skip.read[0].dtype == T(Real([1, 1]))


@pytest.mark.parametrize("rate", [0, 1, False, 0.0])
def test_delay_of_a_discrete_variable_is_zero_only(rate):
    class Bad(_Mixed):
        def delay(self, x, n, t):
            return 1 * d(t), rate

    x, n, t = Var(Real1), Var(Int([1, 1])), Var(Real1)
    with pytest.raises(TypeError, match="discrete"):
        Bad(ctrl=(x, n), extl=(t,))


def test_delay_literal_takes_the_degree_of_the_rate():
    class Steady(sugar.Module):
        def init(self):
            return 0

        def delay(self, x):
            return 1  # x' = 1: a constant rate of change, of degree 1

    x = Var(Real1)
    a = Steady(ctrl=(x,)).atoms[0]
    sorts = [w.dtype for term in a.delay for w in term.write]
    assert sorts and all(s == T(Real([1, 1])) for s in sorts)


def test_useless_module():
    class Useless(sugar.Module):
        def init(self, i):
            return ()

        def update(self, i):
            return ()

    i = Var(Real1)
    m = Useless(ctrl=(), extl=(i,))
    assert m.closed()  # unused externals and no control -> closed module
    assert len(m.extl) + len(m.intf) + len(m.prvt) == 0


def test_private():
    class DigitalDelay(sugar.Module):
        def update(self, x, _o, i):
            return (i, x)

    i = Var(Real1)
    o = Var(Real1)
    x = Var(Real1)
    m = DigitalDelay(ctrl=(x, o), extl=(i,), hide=frozenset([x]))

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
    m1 = Producer(ctrl=(x, y), extl=(w,))
    m2 = Consumer(ctrl=(z,), extl=(y, w))

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


# --- enumerations and events -------------------------------------------------

LIGHT = Enum({"on", "off"})


def _light(state, s):
    """The value of the light `s` written in `state`, by its name."""
    return LIGHT.values[state[reactor.X(s)].item()]


def _happens(m, steps, e):
    """Whether the event `e` happens in each of `steps` updates of `m`: whether its next
    value differs from its latched one."""
    state, out = _init(m), []
    for _ in range(steps):
        state = _update(m, _latch(m, state))
        out.append(bool(state[reactor.X(e)] != state[e]))
    return out


class Switch(sugar.Module):
    """A light, switched over whenever the button is pressed."""

    def init(self, _press):
        return "off"

    def update(self, s, press):
        return ite(recv(press), ite(s == "off", "on", "off"), s)


class Button(sugar.Module):
    """A button pressed in every other step: whenever its phase is up."""

    def init(self):
        # the initial state of an event is arbitrary
        return Havoc, False

    def update(self, press, up):
        # shifted by its phase, the event is sent whenever the phase is up
        return press ^ up, ~up


def test_a_light_switches_when_its_button_is_pressed():
    s, press = Var(LIGHT), Var(Event())
    m = Switch(ctrl=(s,), extl=(press,))
    state = _init(m)
    trace = [_light(state, s)]
    # the environment presses the button by flipping its event
    value = torch.tensor([[False]])
    for pressed in [False, True, True, False, True]:
        state = _latch(m, state)
        state[press] = value
        value = ~value if pressed else value
        state[reactor.X(press)] = value
        state = _update(m, state)
        trace.append(_light(state, s))
    assert trace == ["off", "off", "on", "off", "off", "on"]


def test_a_button_sends_its_event_by_true_and_not_by_false():
    press, up = Var(Event()), Var(Bool([1, 1]))
    m = Button(ctrl=(press, up))
    assert _happens(m, 5, press) == [False, True, False, True, False]


def test_a_light_follows_the_button_it_is_composed_with():
    s, press, up = Var(LIGHT), Var(Event()), Var(Bool([1, 1]))
    m = Module.compose(Button(ctrl=(press, up)), Switch(ctrl=(s,), extl=(press,)))
    assert m.closed()
    state = _init(m)
    trace = [_light(state, s)]
    for _ in range(6):
        state = _update(m, _latch(m, state))
        trace.append(_light(state, s))
    # pressed in every other step, so on, then off, every two steps
    assert trace == ["off", "off", "on", "on", "off", "off", "on"]


def test_an_event_is_never_sent_by_a_module_with_no_update():
    class Never(sugar.Module):
        def init(self):
            return Havoc

    e = Var(Event())
    # its initial state means nothing, and the update of a module with none keeps it,
    # so that it never flips
    assert _happens(Never(ctrl=(e,)), 3, e) == [False, False, False]



def test_an_event_is_initialised_arbitrarily_not_by_a_constant():
    class Start(sugar.Module):
        def init(self):
            return True

    # an event has no constants: its initial state, never observed, is arbitrary
    with pytest.raises(TypeError):
        Start(ctrl=(Var(Event()),))
