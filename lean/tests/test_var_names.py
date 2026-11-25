"""The variables' source names in the certificate's Lean: `reactor_lean.proof.var_names`."""

import pytest
from reactor import Bool, Real, Var, X

from reactor_lean.inputs import Names
from reactor_lean.proof.var_names import VarNames, lean_ident


@pytest.mark.parametrize("name, ident", [
    ("px", "px"), ("w_vz", "w_vz"), ("x'", "x'"),
    ("a-b", "«a-b»"), ("x.y", "«x.y»"), ("2x", "«2x»"), ("fun", "«fun»"),
])
def test_a_name_is_a_lean_identifier(name, ident):
    assert lean_ident(name) == ident


def test_nothing_named_is_no_names():
    assert VarNames.of(None) is None
    assert VarNames.of(Names((None, None), (None,))) is None


def test_a_predicate_reads_named_variables_through_their_abbreviations():
    names = VarNames.of(Names(("px", None, "init"), ("w", None)))
    assert names.rename() == {"s0": "px", "s2": "Var.init", "e0": "w", "el0": "w.latched"}
    assert names.simp_names() == ["Var.px", "Var.init", "Var.w", "Var.w.latched"]


def test_the_abbreviations_read_the_positions():
    ctrl = [X(Var(Real([1, 1]))), X(Var(Real([1, 2]))), X(Var(Bool([1, 1])))]
    extl = [X(Var(Real([1, 1]))), X(Var(Real([1, 1])))]
    names = VarNames.of(Names(("px", "v", None), (None, "a-b")))
    text = names.definitions(ctrl, extl, "S", "E")
    assert text.startswith("namespace Var\nnoncomputable section")
    assert "@[simp] abbrev px (s : S) := s.1 0 0" in text
    assert "@[simp] abbrev v (s : S) := s.2.1\n" in text  # a vector: the matrix itself
    assert "@[simp] abbrev «a-b» (e : E) := e.2.2 0 0" in text
    assert "@[simp] abbrev «a-b».latched (e : E) := e.1.2 0 0" in text
    assert text.endswith("end\nend Var")
