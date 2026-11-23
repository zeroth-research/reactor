"""The claim and the certificate read against the module
(`reactor_lean.smt.queries`), and printed as Lean from that one parse.
"""

import pytest

from reactor import Module, Var, X, LIA, Int
from reactor.analyzer import convert_method
from reactor_lean.core.certificate import Certificate, Claim, LeanAnswer
from reactor_lean.core.errors import Refused
from reactor_lean.proof.render import render_certificate
from reactor_lean.smt.queries import predicate_terms

pytest.importorskip("cvc5")


def countdown() -> Module:
    """`x = 100`, then `x-1` each step, back to 100 at zero."""

    def init():
        return 100

    def update(old_x):
        if old_x == 0:
            return 100
        return old_x - 1

    s = Var(Int([1, 1]))
    return Module.sequential(
        [s],
        convert_method(init, {}, [X(s)], theory=LIA),
        convert_method(update, {"old_x": s}, [X(s)], theory=LIA),
    )


CLAIM = Claim(prp="(= s0 0)")
LEAN = LeanAnswer(
    inv="fun s => 0 <= s 0 0 ∧ s 0 0 <= 100",
    ranking="fun s => (s 0 0).toNat",
)


def _relu(e: str) -> str:
    return f"(ite (>= {e} 0) {e} 0)"


def _dense(layers: int) -> str:
    a, b = "s0", "s0"
    for _ in range(layers):
        a, b = _relu(f"(+ {a} {b})"), _relu(f"(+ {a} (- {b}))")
    return f"(+ {a} {b})"


def test_an_unparseable_predicate_is_refused_by_the_flag_it_came_from():
    bad = Certificate(inv="(this is not smt")
    with pytest.raises(Refused, match="--invariant: cannot read"):
        predicate_terms(countdown(), Claim(), bad)


def test_translation_can_be_asked_not_to_share():
    """`share=False` prints every subterm in full -- for reading; the
    certificate itself is printed shared."""
    q = predicate_terms(countdown(), Claim(), Certificate(ranking=_dense(4)))
    shared = render_certificate(q, share=True).ranking
    plain = render_certificate(q, share=False).ranking
    assert "let " in shared
    assert "let " not in plain



def test_a_lean_answer_is_printed_as_it_was_given():
    """An answer that is Lean already is printed as it is; the property
    beside it is not."""
    defs = render_certificate(predicate_terms(countdown(), CLAIM), verbatim=LEAN)
    assert (defs.inv, defs.ranking) == (LEAN.inv, LEAN.ranking)
    assert defs.prp == "fun s => ((s 0 0) = (0 : Int))"



def test_a_predicate_that_does_not_parse_is_refused_by_its_flag():
    with pytest.raises(Refused, match="--ranking"):
        predicate_terms(countdown(), CLAIM, Certificate(ranking="(+ s0"))
