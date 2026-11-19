"""LTL formulas as automaton modules through `zrth.omega`: a safety formula's running
verdict follows the trace, init reads the initial marking, liveness carries its acceptance
condition, and the automaton composes with the net it watches."""

import pytest

from zrth import SPN
from zrth import Module as compose
from zrth.omega import Automaton, F, G, U, X
from zrth.petri import Arc, Place, Transition
from zrth.simulate import simulate
from zrth.sugar import Module


def birth_death(initial=0):
    bth, dth = Arc("production"), Arc("consumption")
    birth = Transition(rate=2.0, outputs=[bth])
    death = Transition(rate=1.0, inputs=[dth])
    queue = Place(initial=initial, inputs=[bth], outputs=[dth])
    return compose(birth, death, queue), queue


def bounded(n):
    return G(~(n >= 3))


def reaches(n):
    return F(n >= 3)


def running_verdict(counts, bound=3):
    ok, out = True, []
    for c in counts:
        ok = ok and c < bound
        out.append(ok)
    return out


def test_a_safety_formula_is_the_running_verdict_over_the_trace():
    net, queue = birth_death()
    monitor = Automaton(bounded, extl=(queue.count,))
    assert monitor.text == "G(!(p0))"
    assert monitor.acceptance == "Inf(0)"
    assert len(monitor.accepting) == 1
    run = simulate(compose(net, monitor), horizon=20.0, seed=1)
    counts = run.series(queue.count)
    assert max(counts) >= 3
    assert run.series(monitor.accepting[0]) == running_verdict(counts)


def test_init_reads_the_initial_marking():
    net, queue = birth_death(initial=5)
    monitor = Automaton(bounded, extl=(queue.count,))
    run = simulate(compose(net, monitor), horizon=1.0, seed=1)
    assert run.series(monitor.accepting[0])[0] is False


def test_a_co_safety_formula_turns_true_once_and_stays():
    net, queue = birth_death()
    monitor = Automaton(reaches, extl=(queue.count,))
    run = simulate(compose(net, monitor), horizon=20.0, seed=1)
    verdict = run.series(monitor.accepting[0])
    first = next(i for i, c in enumerate(run.series(queue.count)) if c >= 3)
    assert verdict == [i >= first for i in range(len(verdict))]


def test_liveness_without_a_buchi_automaton_gets_parity_acceptance():
    def settles(n):
        return F(G(n == 0))

    net, queue = birth_death()
    monitor = Automaton(settles, extl=(queue.count,))
    assert monitor.acceptance == "Fin(0)"
    assert monitor.spot.is_deterministic()
    run = simulate(compose(net, monitor), horizon=5.0, seed=2)
    assert run.series(monitor.accepting[0]) == [c != 0 for c in run.series(queue.count)]


def test_atoms_are_shared_by_identity_and_formulas_combine():
    def both(n):
        full = n >= 3
        return G(~full) | (U(~full, X(full)) & F(full))

    net, queue = birth_death()
    monitor = Automaton(both, extl=(queue.count,))
    assert monitor.text == "(G(!(p0))) | (((!(p0)) U (X(p0))) & (F(p0)))"


def test_one_formula_watches_two_queues():
    net_a, a = birth_death()
    net_b, b = birth_death()
    assert Automaton(bounded, extl=(a.count,)).extl != Automaton(bounded, extl=(b.count,)).extl


def test_the_automaton_composes_with_the_net_it_watches():
    net, queue = birth_death()
    monitor = Automaton(bounded, extl=(queue.count,))
    assert set(monitor.extl) == {queue.count}
    assert set(monitor.ctrl) == {monitor.q, *monitor.accepting}
    whole = compose(net, monitor)
    assert set(whole.extl) == set(net.extl)


def test_an_atom_must_be_a_bool_expression():
    def wrong(n):
        return G(n + 1)

    net, queue = birth_death()
    with pytest.raises(TypeError, match="Bool"):
        Automaton(wrong, extl=(queue.count,))


def test_a_controller_reads_the_verdict_latched_in_the_loop():
    bth, dth, gate = Arc("production"), Arc("consumption"), Arc("enabling")
    birth = Transition(rate=2.0, inputs=[gate], outputs=[bth])
    death = Transition(rate=1.0, inputs=[dth])
    queue = Place(initial=0, inputs=[bth], outputs=[dth])
    monitor = Automaton(bounded, extl=(queue.count,))
    ok = monitor.accepting[0]

    class Stop(Module):
        def init(self, ok):
            return True

        def next(self, open, ok):
            return ok

    stop = Stop(theory=SPN, ctrl=(gate.enabled,), extl=(ok,))
    run = simulate(compose(birth, death, queue, monitor, stop), horizon=30.0, seed=1)
    counts, verdict = run.series(queue.count), run.series(ok)
    first = verdict.index(False)
    assert counts[first] == 3
    assert all(b <= a for a, b in zip(counts[first + 1 :], counts[first + 2 :]))
    assert counts[-1] == 0
