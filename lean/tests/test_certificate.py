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


def test_a_claim_is_buchi_unless_it_says_safety():
    assert Claim().kind == "buchi"
