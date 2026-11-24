"""Tests for the predicate printer, `reactor_lean.proof.from_ir`, which
prints what the native parser read.

Covers the `min`/`max` peephole and the sharing pass. There is no `max` kind to translate
from, so a ReLU unit reaches the translator as `(ite (>= e 0) e 0)`. Left as
an `ite` each unit costs the certificate a `split_ifs` branch, and `hrank`
mentions the ranking twice, so a k-unit net fans one goal out into 2^(2k) --
which is what made a wide net exhaust the heartbeat budget rather than fail
to be provable. `omega` handles `min`/`max` over `Int` with no split at all.
"""

import pytest
import torch

from reactor import Module, Term, Var, X, Int, Real, tensor
from reactor_lean.proof.render import LeanCertificate
from reactor_lean.core.certificate import Certificate, Claim
from reactor_lean.core.errors import PredicateRefused
from reactor_lean.parse.native import read_predicates
from reactor_lean.cli.certificate import render
from reactor_lean.proof.from_ir import ir_to_lean_nat
from testdata import dense, relu


def _int_module() -> Module:
    """One 1x1 integer wire; enough to give `s0` a sort."""
    x = Var(Int([1, 1]))
    return Module.sequential(
        [x],
        [Term(tensor.int.Const(torch.tensor([[0]])), [X(x)])],
        [Term(tensor.Skip(Int([1, 1])), [X(x)], [x])],
    )


def _real_module() -> Module:
    x = Var(Real([1, 1]))
    return Module.sequential(
        [x],
        [Term(tensor.real.Const(torch.tensor([[0.0]])), [X(x)])],
        [Term(tensor.Skip(Real([1, 1])), [X(x)], [x])],
    )


def _lean(module, **cert: str) -> LeanCertificate:
    """`cert`'s predicates printed as the certificate prints them."""
    return render(module, Claim(), Certificate(**cert))


def _inv(module, smt: str) -> str:
    return _lean(module, inv=smt).inv


@pytest.mark.parametrize(
    "smt,expected",
    [
        # a ReLU unit, in the four ways a comparison can be written
        ("(ite (>= s0 0) s0 0)", "max"),
        ("(ite (> s0 0) s0 0)", "max"),
        ("(ite (<= s0 0) 0 s0)", "max"),
        ("(ite (< s0 0) 0 s0)", "max"),
        # branches swapped relative to the comparison: that is a min
        ("(ite (>= s0 0) 0 s0)", "min"),
        ("(ite (<= s0 0) s0 0)", "min"),
        # not just against zero
        ("(ite (>= s0 5) s0 5)", "max"),
    ],
)
def test_comparison_ite_becomes_min_or_max(smt, expected):
    lean = _inv(_int_module(), f"(= {smt} 0)")
    assert expected in lean, lean
    assert "if " not in lean, f"still branching: {lean}"


@pytest.mark.parametrize(
    "smt",
    [
        # branches are not the comparison's operands
        "(ite (>= s0 0) 1 2)",
        # only one branch matches
        "(ite (>= s0 0) s0 1)",
        # the condition is not a comparison at all
        "(ite (= s0 0) s0 0)",
    ],
)
def test_other_ites_are_left_alone(smt):
    lean = _inv(_int_module(), f"(= {smt} 0)")
    assert "if " in lean, f"should still be an ite: {lean}"
    assert "max" not in lean and "min" not in lean


def test_real_keeps_the_ite():
    """`linarith`, which is what a Real goal gets, has no min/max support.

    Folding there would take away the `split_ifs` branch that is the only way
    through, so the peephole is Int-only.
    """
    lean = _inv(_real_module(), "(= (ite (>= s0 0.0) s0 0.0) 0.0)")
    assert "if " in lean, lean
    assert "max" not in lean


def test_a_whole_relu_layer_folds():
    """Every unit of a layer folds, so nothing is left for `split_ifs`."""
    units = " ".join(
        f"(* {w} (ite (>= (+ s0 (- {b})) 0) (+ s0 (- {b})) 0))"
        for w, b in ((3, 0), (2, 1), (1, 2))
    )
    lean = _rank(_int_module(), f"(+ {units})")
    assert lean.count("max") == 3, lean
    assert "if " not in lean, lean


# ─────────────────────────────────────────────────────────────────────
# Sharing
#
# A subterm written twice is built once, so a predicate arrives as a DAG and
# the printer used to expand it into a tree. For a dense net whose hidden layer feeds the next
# that expansion is exponential in depth.
# ─────────────────────────────────────────────────────────────────────


def _rank(module, smt: str) -> str:
    return _lean(module, ranking=smt).ranking


def test_a_deep_net_is_emitted_once_per_unit_not_once_per_path():
    """Four layers of two units: 8 units, but 30 `max` when expanded."""
    lean = _rank(_int_module(), dense(4))
    assert lean.count("(max ") == 8, lean
    # every repeated subterm gets a name, pre-activations included
    assert lean.count("let ") == 14


def _ranking(smt: str):
    """`smt` read as a ranking: the term, the state wires, and the printer
    that prints it."""
    q = read_predicates(_int_module(), Claim(), Certificate(ranking=smt))
    return q.ranking, q.wires.ctrl_next, ir_to_lean_nat


def test_the_same_net_without_sharing_shows_what_it_costs():
    term, wires, nat = _ranking(dense(4))
    plain = nat(term, wires, share=False)
    assert plain.count("(max ") == 30
    assert "let " not in plain


def test_sharing_never_makes_the_output_longer():
    for smt in ("s0", "(+ s0 1)", relu("s0"), dense(1), dense(2), dense(5)):
        term, wires, nat = _ranking(smt)
        assert len(nat(term, wires, share=True)) <= len(
            nat(term, wires, share=False)
        ), smt


def test_a_shallow_net_gets_no_bindings():
    """One layer repeats nothing worth naming, so no `let` appears."""
    assert "let " not in _rank(_int_module(), dense(1))


def test_propositions_are_never_bound():
    """`split_ifs` / `casesm*` / `not_and_or` all match on this structure."""
    shared_cond = "(and (>= s0 3) (or (>= s0 3) (<= s0 9)))"
    lean = _inv(_int_module(), shared_cond)
    assert "let " not in lean, lean
    assert lean.count("≥ (3 : Int)") == 2


def test_ground_subterms_are_never_bound():
    """Binding `(- 1)` costs more text than repeating it."""
    lean = _rank(_int_module(), "(+ (* s0 (- 1)) (* s0 (- 1)))")
    assert "let u0 := (- 1)" not in lean, lean


def test_an_int_literal_against_a_real_var_becomes_a_real_literal():
    """The `0` of `(= s0 0)` is a Real when `s0` is Real."""
    lean = _inv(_real_module(), "(= s0 0)")
    assert "(0 : Real)" in lean, lean
    assert "to_real" not in lean, lean


def test_a_non_literal_to_real_is_coerced_from_int():
    """The `Argmax` index an LRA module carries on a Real wire arrives this way."""
    lean = _inv(_real_module(), "(= s0 (to_real (to_int s0)))")
    assert ": Int) : Real)" in lean, lean
    assert "⌊" in lean, lean


def test_euclidean_division_and_modulus_go_across_unchanged():
    """SMT-LIB's `div`/`mod` are Euclidean and so are Lean 4's `/` and `%` on
    `Int` -- `(-7 : Int) / 2` is `-4` where truncating division gives `-3` --
    so the pair renders as the operators rather than being refused.

    An LLM reaches for them on any program whose invariant is a congruence:
    `m_step2` steps by two, its invariant is `(= (mod s0 2) 0)`, and the rank
    that goes with it counts in halves. `div` having no case turned such a
    certificate into a traceback."""
    lean = _inv(_int_module(), "(and (= (mod s0 2) 0) (= (div s0 2) 3))")
    assert "% (2 : Int)" in lean and "/ (2 : Int)" in lean, lean


def test_distinct_is_every_pair_distinct():
    """`(distinct a b c)` is one term of three arguments, and the printer
    once took the first two: `a ≠ b` alone, a weaker predicate than the one
    given."""
    lean = _inv(_int_module(), "(distinct s0 1 2)")
    assert lean.count("≠") == 3, lean
    assert "((1 : Int) ≠ (2 : Int))" in lean, lean
    assert _inv(_int_module(), "(distinct s0 1)") == "fun s => ((s 0 0) ≠ (1 : Int))"


def test_a_kind_with_no_lean_form_is_refused_by_the_flag_it_came_from():
    """A refusal, not a bare `ValueError`: `Refused` is what the pipeline
    turns into an `error:` line, and an answer with no Lean form is the
    commonest way to reach it. Reactor's terms have no `abs`, so the parser
    refuses it."""
    with pytest.raises(PredicateRefused, match="--invariant: .*`abs` is no function"):
        _inv(_int_module(), "(>= (abs s0) 0)")
