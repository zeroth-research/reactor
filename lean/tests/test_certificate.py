"""The certificate types (`reactor_lean.core.certificate`)."""

from dataclasses import FrozenInstanceError

import pytest

from reactor_lean.core.certificate import Certificate, Claim


def test_the_types_are_values():
    """A route is handed these, and handing one back changed is `replace`."""
    for value, field in ((Claim(prp="(= s0 0)"), "prp"),
                         (Certificate(inv="true"), "inv")):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, "x")


def test_the_kind_is_the_proof_rule():
    assert Claim().kind == "buchi" and not Claim().is_safety
    assert Claim(prp="(<= s0 9)", kind="safety").is_safety


def test_an_empty_certificate_has_neither_predicate():
    assert Certificate().empty
    assert Certificate(inv="", ranking="").empty
    assert not Certificate(inv="(>= s0 0)").empty
    assert not Certificate(ranking="s0").empty
