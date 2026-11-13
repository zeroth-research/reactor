"""`parse.smtlib`: SMT-LIB predicates read into reactor's terms, no solver.

Held to cvc5's parser: each predicate is read by both, the native reading
is encoded back to cvc5 by the op table's `smt` column, and cvc5 is asked
whether the two can differ. What only one of them reads is listed, with
why.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from reactor import BV, LIA, LRA, BitVec, Bool, Int, Module, Real, Term, Var, X
from reactor.lean.core.errors import Unreadable
from reactor.lean.lowering.sorts import itype_name
from reactor.lean.parse.formula import family
from reactor.lean.parse.smtlib import Scope, read_smtlib


def _mixed() -> Module:
    """One state variable of each sort -- Int, Bool, Real, a 4-bit vector
    and a 2x2 Int matrix (`s0`..`s4`) -- and an Int input (`e0`, `el0`)."""
    i, b, r = Var(Int([1, 1])), Var(Bool([1, 1])), Var(Real([1, 1]))
    v, m, e = Var(BitVec(4, [1, 1])), Var(Int([2, 2])), Var(Int([1, 1]))
    init = [
        Term(LIA.Int(torch.tensor([[0]])), [X(i)]),
        Term(LIA.Bool(torch.tensor([[True]])), [X(b)]),
        Term(LRA.Real(torch.tensor([[0.0]])), [X(r)]),
        Term(BV.Const(torch.tensor([[0]])), [X(v)]),
        Term(LIA.Int(torch.zeros(2, 2, dtype=torch.int64)), [X(m)]),
    ]
    update = [
        Term(LIA.Id(), [X(i)], [X(e)]),
        Term(LIA.Id(), [X(b)], [b]),
        Term(LRA.Id(), [X(r)], [r]),
        Term(BV.Id(), [X(v)], [v]),
        Term(LIA.Id(), [X(m)], [m]),
    ]
    return Module.sequential([i, b, r, v, m, e], init, update)


MODULE = _mixed()


def _scope(module=MODULE) -> Scope:
    names = {}
    for prefix, wires in (("s", [X(v) for v in module.ctrl]),
                          ("e", [X(v) for v in module.extl]),
                          ("el", list(module.extl))):
        for k, w in enumerate(wires):
            names[f"{prefix}{k}"] = w
    return Scope(names)


def _ops(src: str) -> list[str]:
    return [itype_name(t.itype) for t in read_smtlib(_scope(), src).terms]


# ══════════════════════════════════════════════════════════════════════
# Held to cvc5
# ══════════════════════════════════════════════════════════════════════

AGREED = [
    "(> s0 0)",
    "(and (> s0 0) s1 (< s2 1.5))",
    "(or s1 (not s1) (= s0 e0))",
    "(=> s1 (>= s0 2) (< s0 9))",
    "(= s0 e0 3)",
    "(distinct s0 1 2)",
    "(< s0 e0 5)",
    "(- s0 1 2)",
    "(+ s0 (* 2 s0) (- 3))",
    "(* s0 s0)",
    "(div s0 2)",
    "(= (mod s0 2) 0)",
    "(< s2 2)",
    "(< s2 s0)",
    "(+ s2 0.5)",
    "(= s2 (to_real s0))",
    "(= (to_int s2) s0)",
    "(> (ubv_to_int s3) 3)",
    "(= s3 5)",
    "(distinct 5 s3)",
    "(= s3 -3)",
    "(= s3 #b0101)",
    "(= s3 (_ bv5 4))",
    "(bvult s3 #x5)",
    "(= (bvadd s3 #b0001) #b0000)",
    "(ite s1 s0 e0)",
    "(ite s1 s2 1.0)",
    "(ite s1 s3 #b0000)",
    "(ite s1 true (> s0 0))",
    "(let ((x (+ s0 1)) (y 2)) (> (* x x) y))",
    "(let ((s0 1)) (let ((s0 (+ s0 1)) (y s0)) (= s0 (+ y 1))))",
    "(> ((_ tuple.select 3) s4) ((_ tuple.select 0) s4))",
    "(xor s1 (> s0 0))",
    "(not (not s1))",
    "(and s1)",
    "(- 5)",
    "(- 0.5)",
    "(< s2 (- 5))",
    "(>= s0 -1)",
    "(>= s2 -0.5)",
    "(= s1 true)",
    "(= s1 (> s0 0))",
    "(distinct s1 true)",
    "(> s0 0) ; a comment",
    "0",
    "2.5",
]


@pytest.mark.parametrize("src", AGREED)
def test_the_native_reading_is_cvc5s(src):
    cvc5 = pytest.importorskip("cvc5")
    from reactor.lean.smt.encode import translate_terms
    from reactor.lean.smt.module import ModuleSMT
    from reactor.lean.smt.predicates import PredicateParser, parse_predicate

    tm = cvc5.TermManager()
    env = PredicateParser(ModuleSMT(tm=tm, module=MODULE))
    msmt = env.msmt
    binds = {}
    for wires, consts in ((msmt.ctrl_next, env.state_vars),
                          (msmt.extl_next, env.extl_next_vars),
                          (msmt.extl_latched, env.extl_latched_vars)):
        binds.update({w.id: c for w, c in zip(wires, consts)})

    want = parse_predicate(env, src)
    f = read_smtlib(_scope(), src)
    got = translate_terms(tm, list(f.terms), binds)[f.out.id]
    solver = cvc5.Solver(tm)
    solver.setLogic("ALL")
    solver.assertFormula(tm.mkTerm(cvc5.Kind.DISTINCT, want, got))
    assert solver.checkSat().isUnsat(), f"{src}: native {got}, cvc5 {want}"


# ══════════════════════════════════════════════════════════════════════
# What it builds
# ══════════════════════════════════════════════════════════════════════


def test_a_subterm_written_twice_is_built_once():
    """As cvc5 hash-conses, so that the printer can share what repeats."""
    once = _ops("(+ s0 1)")
    twice = _ops("(* (+ s0 1) (+ s0 1))")
    assert len(twice) == len(once) + 1, twice


def test_a_chain_is_built_as_cvc5_builds_it():
    assert _ops("(and s1 s1 s1)").count("And") == 2
    assert _ops("(< s0 1 2)") == ["Int", "Lt", "Int", "Lt", "And"]
    assert _ops("(distinct s0 1 2)").count("Ne") == 3


def test_a_numeral_takes_the_sort_it_meets():
    """An Int numeral against a Real is a Real constant, not a cast one; a
    variable is cast."""
    assert "IntToReal" not in _ops("(< s2 2)")
    assert "Real" in _ops("(< s2 2)")
    assert "IntToReal" in _ops("(< s2 s0)")
    assert _ops("(- 5)") == ["Int", "Neg"]
    assert _ops("(< s2 (- 5))")[:2] == ["Real", "Neg"]


def test_a_bit_vector_comparison_is_read_as_a_bool():
    f = read_smtlib(_scope(), "(= s3 5)")
    assert family(f.sort) == "Bool"
    assert _ops("(= s3 5)") == ["Const", "Eq", "BitToBool"]


@pytest.mark.parametrize("src, why", [
    ("(> s0 0) (< s0 5)", "text after the expression"),
    ("(> s0 0))", "text after the expression"),
    ("(> s0", "no `\\)` to close it"),
    ("", "empty"),
    ("  ; only a comment", "empty"),
    ("()", "no term"),
    ("(> s9 0)", "`s9` is not declared"),
    ("(> s0 007)", "`007` is no numeral"),
    ("(+ s0 s1)", "arithmetic"),
    ("(ite s1 s2 1)", "branches are Real and Int"),
    ("(= s3 (_ bv5 8))", "compares \\(_ BitVec 4\\) with \\(_ BitVec 8\\)"),
    ("(abs s0)", "`abs` is no function"),
    ("(/ s2 2)", "`/` is no function"),
    ("(not s0 s1)", "takes 1 argument"),
    ("(mod s0 1 2)", "takes 2 arguments"),
    ("(div s2 2)", "integer division"),
    ("(to_int s0)", "reads a Real"),
    ("((_ tuple.select 4) s4)", "has no element 4"),
    ("((_ tuple.select 0) s0)", "no matrix variable"),
    ("s4", "one element at a time"),
    ("(< s2 0.1234567890123456789)", "no float"),
    ("(! s1 :named p)", "annotation"),
])
def test_what_it_cannot_read_it_says_why(src, why):
    with pytest.raises(Unreadable, match=why):
        f = read_smtlib(_scope(), src)
        # A matrix variable alone parses, and is refused when settled.
        assert f


def test_reading_loads_no_solver():
    """The point of the native parser: reading a predicate imports no cvc5."""
    code = ("import sys, torch\n"
            "from reactor.lean.parse import smtlib\n"
            "print('cvc5' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout
    assert out.strip() == "False"
