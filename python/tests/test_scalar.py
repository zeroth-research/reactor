"""The scalar theories, `reactor.enum` and `reactor.event`: their sorts, and their
operations, built and matched as those of the matrix theories, from the parameters their
sorts do not determine, then their domain, and optionally their codomain."""

import pytest
import reactor
from reactor import Havoc, Skip, T, Term, Wire
from reactor.bool import Bool
from reactor.zero import Real0
from reactor.enum import Enum, Sym, Into
from reactor.event import Event, Set, Reset
import reactor.enum as enum_
import reactor.event as event_

COND = Bool([1, 1])


def test_an_enumeration_is_the_set_of_its_values():
    e = Enum({"idle", "busy"})
    assert e == Enum(["busy", "idle"])
    assert e.values == ["busy", "idle"]
    assert str(e) == 'Enum(["busy", "idle"])'
    match e:
        case Enum(values):
            assert values == ["busy", "idle"]
    # any iterable of distinct strings, in any order
    assert Enum(("busy", "idle")) == Enum({"idle", "busy"}) == Enum(frozenset(e.values)) == e
    assert Enum(v for v in ["idle", "busy"]) == e
    with pytest.raises(ValueError):
        Enum(["idle", "busy", "idle"])
    with pytest.raises(ValueError):
        Enum(set())
    # a string is not the iterable of its letters
    with pytest.raises(TypeError):
        Enum("idle")
    with pytest.raises(TypeError):
        Enum({"idle", 1})
    with pytest.raises(TypeError):
        Enum(3)


def test_the_union_of_enumerations_holds_the_values_of_both():
    a, b = Enum({"idle", "busy"}), Enum({"busy", "done"})
    assert a | b == b | a == Enum({"idle", "busy", "done"})
    assert a | a == a


def test_a_value_is_in_an_enumeration_when_it_is_one_of_its_constants():
    e = Enum({"idle", "busy"})
    assert "idle" in e and "busy" in e
    assert "done" not in e
    # a value that is not a string is never in it
    assert 1 not in e and None not in e


def test_events_have_no_parameters():
    assert Event() == Event() and str(Event()) == "Event()"
    match Event():
        case Event():
            pass
        case _:
            pytest.fail("an event matches by nothing")


def test_the_scalar_sorts_are_discrete():
    assert T(Enum({"a"})) == Real0() and Enum({"a"}).tangent() == Real0()
    assert T(Event()) == Real0() and Event().tangent() == Real0()


def test_each_sort_is_in_the_namespace_of_its_theory():
    assert Enum.__module__ == enum_.__name__ == "reactor.enum"
    assert Event.__module__ == event_.__name__ == "reactor.event"
    assert reactor.enum is enum_ and reactor.event is event_


def test_a_constant_is_built_from_its_value_and_codomain():
    e = Enum({"idle", "busy"})
    op = Sym("idle", e)
    assert (op.value, op.cod, str(op)) == ("idle", e, "idle")
    assert not hasattr(op, "dom")
    match op:
        case Sym(value, Enum(values)):
            assert (value, values) == ("idle", ["busy", "idle"])
    with pytest.raises(ValueError):
        Sym("done", e)


def test_into_needs_a_codomain_holding_the_values_of_its_domain():
    small, large = Enum({"a", "b"}), Enum({"a", "b", "c"})
    op = Into(small, large)
    assert (op.dom, op.cod) == (small, large)
    match op:
        case Into(d, c):
            assert (d, c) == (small, large)
    with pytest.raises(ValueError):
        Into(large, small)


def test_the_comparisons_of_enumerations_write_a_boolean():
    e = Enum({"idle", "busy"})
    for cls in [enum_.Eq, enum_.Ne]:
        op = cls([e, e])
        assert (op.dom, op.cod) == ([e, e], COND)
        assert cls([e, e], COND) == op
        with pytest.raises(ValueError):
            cls([e, e], Bool([2, 2]))
        with pytest.raises(ValueError):
            cls([e, Enum({"idle"})])


def test_the_choice_of_enumerations_reads_a_scalar_condition():
    e = Enum({"idle", "busy"})
    op = enum_.Ite([COND, e, e])
    assert (op.dom, op.cod) == ([COND, e, e], e)
    match op:
        case enum_.Ite([Bool(shape), x, y], c):
            assert (list(shape), x, y, c) == ([1, 1], e, e, e)
    with pytest.raises(ValueError):
        enum_.Ite([Bool([2, 2]), e, e])


def test_the_operations_on_events_have_a_fixed_boundary():
    ev = Event()
    cases = [
        (Set, [], ev),
        (Reset, [], ev),
        (event_.Test, ev, COND),
        (event_.Ite, [COND, ev, ev], ev),
    ]
    for cls, dom, cod in cases:
        op = cls()
        assert (op.dom, op.cod) == (dom, cod)
        # the boundary, given, must be the fixed one
        assert cls(dom) == cls(dom, cod) == op
        with pytest.raises(ValueError):
            cls(cod=Real0())
        match op:
            case cls(d, c):
                assert (d, c) == (dom, cod)


def test_the_scalar_sorts_and_operations_go_through_terms():
    e = Enum({"idle", "busy"})
    for op, read, write in [
        (Sym("idle", e), [], [Wire(e)]),
        (enum_.Eq([e, e]), [Wire(e), Wire(e)], [Wire(COND)]),
        (Set(), [], [Wire(Event())]),
        (event_.Test(), [Wire(Event())], [Wire(COND)]),
        (Havoc(e), [], [Wire(e)]),
        (Skip(Event()), [Wire(Event())], [Wire(Event())]),
    ]:
        term = Term(op, write, read) if read else Term(op, write)
        assert term.itype == op
