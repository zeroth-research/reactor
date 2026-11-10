"""The SMT-LIB `verith.common.smtlib` writes predicates out as.

The scripts the routes hand a solver outside this process are pinned with
the routes, in `test_route_scripts.py`.
"""

from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════════════════════
# Predicates written out
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("coeffs, const, want", [
    ([0], 0, "0"),                                  # nothing at all
    ([0], 5, "5"),                                  # a constant alone
    ([1], 0, "s0"),                                 # a bare column
    ([-1], 0, "(- s0)"),
    ([2], 0, "(* 2 s0)"),
    ([1, -1], 3, "(+ 3 s0 (- s1))"),
    ([1, 0, 2], -1, "(+ (- 1) s0 (* 2 s2))"),       # a zero coefficient drops
])
def test_an_affine_row_prints_as_one_term(coeffs, const, want):
    from verith.common.smtlib import affine_smt

    assert affine_smt(const, coeffs, ["s0", "s1", "s2"][:len(coeffs)]) == want


def test_an_affine_form_over_the_reals_writes_its_literals_as_decimals():
    from verith.common.smtlib import affine_smt, smt_real

    assert affine_smt(1, [2, 1], ["x", "y"], lit=smt_real) == \
        "(+ 1.0 (* 2.0 x) y)"


def test_the_same_fact_twice_is_one_conjunct():
    """A property seeded as a candidate that the lattice also proposes survives
    as both, and the certificate should carry it once."""
    from verith.common.smtlib import conjunction_smt

    assert conjunction_smt(["(<= s0 100)", "(>= s0 0)", "(<= s0 100)"]) == \
        "(and (<= s0 100) (>= s0 0))"
    assert conjunction_smt(["(<= s0 100)", "true"]) == "(<= s0 100)"
    assert conjunction_smt([]) == "true"
