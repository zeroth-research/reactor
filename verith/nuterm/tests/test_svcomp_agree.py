"""The SV-COMP benchmarks' SMT encoding takes the rounds the module takes.

Each benchmark's encoded round, with the round's state and inputs as
literals, which cvc5's rewriter evaluates without deciding anything, against
the round `reactor.eval` takes from the same state (`_agree`). The Lean side
is `verith/tests/lean/test_lean_svcomp_agree.py`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cvc5")

from nuterm.tests._agree import CORPUS, Skip, disagreements_smt  # noqa: E402


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_the_smt_encoding_takes_the_rounds_the_benchmark_takes(name):
    try:
        bad = disagreements_smt(CORPUS[name]())
    except Skip as e:
        pytest.skip(str(e))
    assert not bad, "\n".join(bad)
