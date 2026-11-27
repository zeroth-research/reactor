"""`parse.predicates`: the Python predicate syntax read into reactor's
terms, and a run's predicates read.

A predicate in the syntax builds the terms its SMT-LIB form builds, and one
wrong in it is refused in its own terms.
"""

from __future__ import annotations

import pytest

from reactor_lean.errors import PredicateRefused, Unreadable
from reactor_lean.lowering.elements import ModuleWires
from reactor_lean.parse import load
from reactor_lean.lowering.sorts import itype_name, theory_of
from reactor_lean.parse import dsl
from reactor_lean.parse.formula import Builder
from reactor_lean.parse.predicates import (
    Certificate, Claim, namespace, parse_predicate, read_predicates, scope_of,
)
from reactor_lean.parse.smtlib import Elaborator
from testdata import MODS


def _wires(name: str) -> ModuleWires:
    return ModuleWires.of(load(MODS / f"{name}.py").module)


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
        ("m_countdown", "s0[0][0] == 0", "(= s0 0)"),
        ("m_countdown", " s0[0][0] == 0", "(= s0 0)"),
        ("m_countdown", "Ite(s0[0][0] == 0, 0, s0[0][0])", "(ite (= s0 0) 0 s0)"),
        (
            "m_boolint",
            "And(s1[0][0] >= 0, s1[0][0] <= 5, Implies(s0[0][0], s1[0][0] <= 5))",
            "(and (>= s1 0) (<= s1 5) (=> s0 (<= s1 5)))",
        ),
        (
            "m_relu_vec",
            "s0[1, 0] + 1 > s0[2][0]",
            "(> (+ ((_ tuple.select 1) s0) 1) ((_ tuple.select 2) s0))",
        ),
        ("m_countdown", "0 <= s0[0][0] <= 5", "(and (>= s0 0) (<= s0 5))"),
        ("m_boolint", "not s0[0][0] and s1[0][0] == 0", "(and (not s0) (= s1 0))"),
        ("m_boolint", "s1[0][0] if s0[0][0] else 0", "(ite s0 s1 0)"),
        ("m_boolint", "s0[0][0] == True", "(= s0 true)"),
        ("m_countdown", "True", "true"),
        ("m_countdown", "False", "false"),
        ("m_countdown", "s0[0][0] > 0 or True", "(or (> s0 0) true)"),
        ("m_countdown", "-s0[0][0] * 2 < 1", "(< (* (- s0) 2) 1)"),
        # A number the syntax computes is the constant it is.
        ("m_countdown", "10 - 3", "7"),
        ("m_countdown", "2.5 * 2", "5.0"),
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
        "s0[s0][0]",
    ],
)
def test_python_outside_the_syntax_is_not_read(src):
    wires = _wires("m_countdown")
    ns = namespace(Elaborator(scope_of(wires), Builder()), wires)
    with pytest.raises((dsl.NotPredicate, SyntaxError)):
        dsl.evaluate(src, ns)


def test_a_bare_numeral_is_an_int_constant():
    f = parse_predicate(_wires("m_countdown"), "0")
    op = f.terms[-1].itype
    assert (theory_of(op), itype_name(op)) == ("int", "Const")


@pytest.mark.parametrize(
    "src, why",
    [
        ("s1[0][0] + s0[0][0] > 0", "arithmetic"),
        ("s1[5][0] > 0", "out of bounds"),
        ("Ite(s0[0][0], 1)", "missing"),
        ("s1[0, 0, 0] > 0", "two indices, not 3"),
        ("s1[0][0] < 1e999", "no number"),
    ],
)
def test_the_syntax_says_what_is_wrong_with_a_predicate_in_it(src, why):
    with pytest.raises(Unreadable, match=f"is predicate syntax.*{why}"):
        parse_predicate(_wires("m_boolint"), src)


@pytest.mark.parametrize(
    "src, why",
    [
        ("s0[0][0] / 2 > 0", "`s0\\[0\\]\\[0\\] / 2` is not predicate syntax"),
        ("s0[0][0] ** 2 > 0", "`s0\\[0\\]\\[0\\] \\*\\* 2` is not predicate syntax"),
        ("s9[0][0] > 0", "`s9` is not declared"),
    ],
)
def test_python_the_syntax_does_not_read_is_refused_in_its_terms(src, why):
    """Not handed on to the SMT-LIB reader, which refused `/` as "text
    after the expression"."""
    with pytest.raises(Unreadable, match=why):
        parse_predicate(_wires("m_countdown"), src)


@pytest.mark.parametrize("bug", [AttributeError, ValueError])
def test_a_fault_in_the_syntax_itself_is_not_a_parse_error(monkeypatch, bug):
    """A `ValueError` too: `Unreadable` is one, and so is a bug."""
    import reactor_lean.parse.predicates as predicates

    def broken(src, ns):
        raise bug("a bug")

    monkeypatch.setattr(predicates.dsl, "evaluate", broken)
    with pytest.raises(bug, match="a bug") as raised:
        read_predicates(load(MODS / "m_countdown.py").module,
                        Claim(prp="s0[0][0] == 0"))
    assert type(raised.value) is bug    # not a refusal, which is a ValueError


# ══════════════════════════════════════════════════════════════════════
# A run's predicates
# ══════════════════════════════════════════════════════════════════════

COUNTDOWN = MODS / "m_countdown.py"


def test_one_assumption_of_both_blocks_is_read_once():
    q = read_predicates(load(COUNTDOWN).module,
                        Claim(init_assume="(>= s0 0)", update_assume="(>= s0 0)"))
    assert q.init_assume is q.update_assume


def test_an_assumption_is_refused_by_the_flag_it_came_from():
    """`--extl-assume` is both blocks'; a block's own is its own flag's."""
    m = load(COUNTDOWN).module
    with pytest.raises(PredicateRefused, match="^--extl-assume: must have sort Bool"):
        read_predicates(m, Claim(init_assume="(+ s0 1)", update_assume="(+ s0 1)"))
    with pytest.raises(PredicateRefused, match="^--extl-assume-update: must have sort Bool"):
        read_predicates(m, Claim(init_assume="(>= s0 0)", update_assume="(+ s0 1)"))


def test_an_unreadable_predicate_is_refused_by_its_flag():
    m = load(COUNTDOWN).module
    with pytest.raises(PredicateRefused, match="--invariant: cannot read `\\(\\+ s0`"):
        read_predicates(m, Claim(), Certificate(inv="(+ s0"))
    with pytest.raises(PredicateRefused, match="--safety: cannot read"):
        read_predicates(m, Claim(prp="(> s9 0)", kind="safety"))


def test_a_predicate_of_the_wrong_sort_is_refused_by_its_flag():
    m = load(COUNTDOWN).module
    with pytest.raises(PredicateRefused, match="--ranking: must have sort Int, got Bool"):
        read_predicates(m, Claim(), Certificate(ranking="(> s0 0)"))
    with pytest.raises(PredicateRefused, match="--invariant: must have sort Bool, got Int"):
        read_predicates(m, Claim(), Certificate(inv="s0"))
