"""`parse.smtlib`: SMT-LIB predicates read into reactor's terms, no solver.

Each of `READ` is read; what it does not read is refused with why.
"""

from __future__ import annotations


import pytest
import torch

from reactor import Bool, Int, Module, Real, Term, Var, Word, X, tensor
from reactor_lean.errors import Unreadable
from reactor_lean.lowering.sorts import itype_name, theory_of
from reactor_lean.parse.formula import family
from reactor_lean.lowering.elements import ModuleWires
from reactor_lean.parse.native import parse_predicate, scope_of
from reactor_lean.parse.smtlib import Scope, read_smtlib


def _mixed() -> Module:
    """One state variable of each sort -- Int, Bool, Real, a 4-bit vector
    and a 2x2 Int matrix (`s0`..`s4`) -- and an Int input (`e0`, `el0`)."""
    i, b, r = Var(Int([1, 1])), Var(Bool([1, 1])), Var(Real([1, 1]))
    v, m, e = Var(Word(4, [1, 1])), Var(Int([2, 2])), Var(Int([1, 1]))
    init = [
        Term(tensor.int.Const(torch.tensor([[0]])), [X(i)]),
        Term(tensor.bool.Const(torch.tensor([[True]])), [X(b)]),
        Term(tensor.real.Const(torch.tensor([[0.0]])), [X(r)]),
        Term(tensor.word.Const(torch.tensor([[0]]), 4), [X(v)]),
        Term(tensor.int.Const(torch.zeros(2, 2, dtype=torch.int64)), [X(m)]),
    ]
    update = [
        Term(tensor.Skip(Int([1, 1])), [X(i)], [X(e)]),
        Term(tensor.Skip(Bool([1, 1])), [X(b)], [b]),
        Term(tensor.Skip(Real([1, 1])), [X(r)], [r]),
        Term(tensor.Skip(Word(4, [1, 1])), [X(v)], [v]),
        Term(tensor.Skip(Int([2, 2])), [X(m)], [m]),
    ]
    return Module.sequential([i, b, r, v, m, e], init, update)


MODULE = _mixed()


def _scope() -> Scope:
    return scope_of(ModuleWires.of(MODULE))


def _ops(src: str) -> list[str]:
    """The ops `src` is read into, each named with its theory (`int.Lt`)."""
    return [f"{theory_of(t.itype)}.{itype_name(t.itype)}"
            for t in read_smtlib(_scope(), src).terms]


# ══════════════════════════════════════════════════════════════════════
# What it reads
# ══════════════════════════════════════════════════════════════════════

READ = [
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
    "(ite s1 s0 e0)",
    "(ite s1 s2 1.0)",
    "(ite s1 s3 #b0000)",
    "(ite s1 true (> s0 0))",
    "(let ((x (+ s0 1)) (y 2)) (> (* x x) y))",
    "(let ((s0 1)) (let ((s0 (+ s0 1)) (y s0)) (= s0 (+ y 1))))",
    "(> ((_ tuple.select 3) s4) ((_ tuple.select 0) s4))",
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


@pytest.mark.parametrize("src", READ)
def test_it_reads(src):
    assert read_smtlib(_scope(), src).out is not None


# ══════════════════════════════════════════════════════════════════════
# What it builds
# ══════════════════════════════════════════════════════════════════════


def test_a_subterm_written_twice_is_built_once():
    """So that the printer can share what repeats."""
    once = _ops("(+ s0 1)")
    twice = _ops("(* (+ s0 1) (+ s0 1))")
    assert len(twice) == len(once) + 1, twice


def test_a_chain_is_built_as_a_conjunction_of_its_links():
    assert _ops("(and s1 s1 s1)").count("bool.And") == 2
    assert _ops("(< s0 1 2)") == [
        "int.Const", "int.Lt", "int.Const", "int.Lt", "bool.And"]
    assert _ops("(distinct s0 1 2)").count("int.Ne") == 3


def test_a_numeral_takes_the_sort_it_meets():
    """An Int numeral against a Real is a Real constant, not a cast one; a
    variable is cast."""
    assert "cast.ToReal" not in _ops("(< s2 2)")
    assert "real.Const" in _ops("(< s2 2)")
    assert "cast.ToReal" in _ops("(< s2 s0)")
    assert _ops("(- 5)") == ["int.Const", "int.Neg"]
    assert _ops("(< s2 (- 5))")[:2] == ["real.Const", "real.Neg"]


def test_a_bit_vector_comparison_is_read_as_a_bool():
    f = read_smtlib(_scope(), "(= s3 5)")
    assert family(f.sort) == "Bool"
    assert _ops("(= s3 5)") == ["word.Const", "word.Eq"]


@pytest.mark.parametrize("src", ["(s3[0][0] & 1) == 0", "~s3[0][0] == 0", "-s3[0][0] == 0"])
def test_the_python_syntax_reads_no_bit_vector_operation_either(src):
    """`&`, `~` and `-` on a word were `bvand`, `bvnot` and `bvneg`, which
    no predicate has a Lean form for; they are the Bool and arithmetic
    operators, which refuse a word."""
    with pytest.raises(Unreadable, match="wants a Bool|is arithmetic"):
        parse_predicate(ModuleWires.of(MODULE), src)


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
    ("(xor s1 (> s0 0))", "`xor` is no function"),
    ("(bvult s3 #x5)", "`bvult` is no function"),
    ("(= (bvadd s3 #b0001) #b0000)", "`bvadd` is no function"),
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
