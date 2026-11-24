"""`parse.native`: the Python predicate syntax read into reactor's terms,
and a run's predicates read by the native parser.

A predicate in the syntax builds the terms its SMT-LIB form builds, and one
wrong in it is refused in its own terms.
"""

from __future__ import annotations

import pytest

from reactor_lean.errors import PredicateRefused, Unreadable
from reactor_lean.lowering.wires import ModuleWires
from reactor_lean.parse import load_module
from reactor_lean.lowering.sorts import itype_name, theory_of
from reactor_lean.parse import dsl
from reactor_lean.parse.formula import Builder
from reactor_lean.parse.native import (
    NativeParser, namespace, parse_predicate, scope_of,
)
from reactor_lean.parse.smtlib import Elaborator
from testdata import MODS


def _wires(name: str) -> ModuleWires:
    return ModuleWires.of(load_module(MODS / f"{name}.py"))


def _shape(formula, wires: ModuleWires):
    """`formula` as nested tuples: each term's op and payload over its
    reads, and a variable by its name."""
    names = {w.id: n for n, w in scope_of(wires).names.items()}
    by_wire = {t.write[0].id: t for t in formula.terms}

    def go(w):
        if w.id in names:
            return names[w.id]
        t = by_wire[w.id]
        name = itype_name(t.itype)
        payload = (t.itype.tensor.tolist() if name == "Const"
                   else (t.itype.row, t.itype.col) if name == "Get" else None)
        return (f"{theory_of(t.itype)}.{name}", payload, *[go(r) for r in t.read])

    return go(formula.out)


@pytest.mark.parametrize(
    "mod, python, smtlib",
    [
        ("m_countdown", "s[0][0] == 0", "(= s0 0)"),
        ("m_countdown", " s[0][0] == 0", "(= s0 0)"),
        ("m_countdown", "Ite(s[0][0] == 0, 0, s[0][0])", "(ite (= s0 0) 0 s0)"),
        (
            "m_boolint",
            "And(s1[0][0] >= 0, s1[0][0] <= 5, Implies(s0[0][0], s1[0][0] <= 5))",
            "(and (>= s1 0) (<= s1 5) (=> s0 (<= s1 5)))",
        ),
        (
            "m_relu_vec",
            "s[1, 0] + 1 > s[2][0]",
            "(> (+ ((_ tuple.select 1) s0) 1) ((_ tuple.select 2) s0))",
        ),
        ("m_countdown", "0 <= s[0][0] <= 5", "(and (>= s0 0) (<= s0 5))"),
        ("m_boolint", "not s0[0][0] and s1[0][0] == 0", "(and (not s0) (= s1 0))"),
        ("m_boolint", "s1[0][0] if s0[0][0] else 0", "(ite s0 s1 0)"),
        ("m_boolint", "s0[0][0] == True", "(= s0 true)"),
        ("m_countdown", "-s[0][0] * 2 < 1", "(< (* (- s0) 2) 1)"),
    ],
)
def test_the_python_syntax_builds_the_smtlib_it_means(mod, python, smtlib):
    wires = _wires(mod)
    assert (_shape(parse_predicate(wires, python), wires)
            == _shape(parse_predicate(wires, smtlib), wires))


@pytest.mark.parametrize(
    "src",
    [
        "().__class__",
        "s0.__class__",
        "__import__('os')",
        "[s0 for s0 in ()]",
        "lambda: 0",
        "Ite(s0 == 0, 0, e=1)",
        "And(*(s0,))",
        "s0(0)",
        "s[s0][0]",
    ],
)
def test_python_outside_the_syntax_is_not_read(src):
    wires = _wires("m_countdown")
    ns = namespace(Elaborator(scope_of(wires), Builder()), wires)
    with pytest.raises((dsl.NotPredicate, SyntaxError)):
        dsl.evaluate(src, ns)


def test_a_bare_numeral_still_falls_through_to_smtlib():
    f = parse_predicate(_wires("m_countdown"), "0")
    op = f.terms[-1].itype
    assert (theory_of(op), itype_name(op)) == ("int", "Const")


@pytest.mark.parametrize(
    "src, why",
    [
        ("s1[0][0] + s0[0][0] > 0", "arithmetic"),
        ("s1[5][0] > 0", "out of bounds"),
        ("Ite(s0[0][0], 1)", "missing"),
    ],
)
def test_the_syntax_says_what_is_wrong_with_a_predicate_in_it(src, why):
    with pytest.raises(Unreadable, match=f"is predicate syntax.*{why}"):
        parse_predicate(_wires("m_boolint"), src)


def test_a_fault_in_the_syntax_itself_is_not_a_parse_error(monkeypatch):
    import reactor_lean.parse.native as native

    def broken(src, ns):
        raise AttributeError("a bug")

    monkeypatch.setattr(native.dsl, "evaluate", broken)
    with pytest.raises(AttributeError, match="a bug"):
        NativeParser(load_module(MODS / "m_countdown.py")).parse(
            "s[0][0] == 0", "property")


# ══════════════════════════════════════════════════════════════════════
# A run's predicates
# ══════════════════════════════════════════════════════════════════════


def test_a_predicate_is_read_once():
    parser = NativeParser(load_module(MODS / "m_countdown.py"))
    assert parser.parse("(> s0 0)", "invariant") is parser.parse("(> s0 0)", "invariant")


def test_an_unreadable_predicate_is_refused_by_its_flag():
    parser = NativeParser(load_module(MODS / "m_countdown.py"))
    with pytest.raises(PredicateRefused, match="--invariant: cannot read `\\(\\+ s0`"):
        parser.parse("(+ s0", "invariant")
    with pytest.raises(PredicateRefused, match="--safety: cannot read"):
        parser.parse("(> s9 0)", "property", kind="safety")


def test_a_predicate_of_the_wrong_sort_is_refused_by_its_flag():
    parser = NativeParser(load_module(MODS / "m_countdown.py"))
    with pytest.raises(PredicateRefused, match="--ranking: must have sort Int, got Bool"):
        parser.parse("(> s0 0)", "ranking", want="Int")
    with pytest.raises(PredicateRefused, match="--invariant: must have sort Bool, got Int"):
        parser.parse("s0", "invariant")
