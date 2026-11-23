"""The proofs `verith.proofs` writes, pinned byte for byte.

For a fixed set of modules and certificates -- the ones whose definitions
`reactor`'s `lean/tests/test_pipeline_golden.py` pins -- the proofs as
`Certificate/Proofs.lean` holds them and as a `--cert-file` certificate
inlines them, compared with the snapshots under `tests/golden/pipeline/`;
set `REACTOR_UPDATE_GOLDEN=1` to rewrite them, and only for a change to the
output that is intended.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from reactor import Module, Term, Var, Wire, X, LIA, LRA, Int, Real, Bool
from reactor_lean.codegen.context import LeanContext
from reactor_lean.core.certificate import Certificate, Claim, LeanAnswer
from reactor_lean.inputs.python import load_module
from reactor_lean.proof.render import (
    generate_proofs_lean,
    generate_standalone_cert_lean,
    render_certificate,
)
from verith.common.encoding import predicate_terms
from verith.common.limits import SmtBudget
from verith.proofs.facts import solver_hints
from verith.proofs.scripts import print_hints, proof_scripts

pytest.importorskip("cvc5")

GOLDEN = Path(__file__).parent / "golden" / "pipeline"
# The fixtures are `reactor`'s, in the `python/` project beside this one.
FIXTURES = Path(__file__).resolve().parent / "fixtures"
UPDATE = os.environ.get("REACTOR_UPDATE_GOLDEN") == "1"


def _check(name: str, text: str) -> None:
    path = GOLDEN / name
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"no snapshot {path}; run with REACTOR_UPDATE_GOLDEN=1"
    assert text == path.read_text(), f"{name} differs from its snapshot"


def _fixture(name: str) -> Module:
    return load_module(str(FIXTURES / f"{name}.py"))


def _vector_counter() -> Module:
    """3x1 state over a 2x1 input, as `tests/lean/conftest.py` builds
    it."""
    state = Var(Int([3, 1]))
    extl = Var(Int([2, 1]))
    zero31 = torch.zeros((3, 1), dtype=torch.int64)
    zero11 = torch.zeros((1, 1), dtype=torch.int64)
    a = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [Term(LIA.Linear(a, zero31), [X(state)], [X(extl)])]
    x, y, z = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    x_lt_y, x_lt_z, cond = Wire(Bool([1, 1])), Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    yes, no = Wire(Int([3, 1])), Wire(Int([3, 1]))
    row = lambda i: torch.tensor([[int(j == i) for j in range(3)]], dtype=torch.int64)
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    diag_yz = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int64)
    update = [
        Term(LIA.Linear(row(0), zero11), [x], [state]),
        Term(LIA.Linear(row(1), zero11), [y], [state]),
        Term(LIA.Linear(row(2), zero11), [z], [state]),
        Term(LIA.Lt(), [x_lt_y], [x, y]),
        Term(LIA.Lt(), [x_lt_z], [x, z]),
        Term(LIA.Or(), [cond], [x_lt_y, x_lt_z]),
        Term(LIA.Linear(torch.eye(3, dtype=torch.int64), e1), [yes], [state]),
        Term(LIA.Linear(diag_yz, zero31), [no], [state]),
        Term(LIA.Ite(), [X(state)], [cond, yes, no]),
    ]
    return Module.sequential([state, extl], init, update)


def _real_countdown() -> Module:
    x = Var(Real([1, 1]))
    one = Wire(Real([1, 1]))
    return Module.sequential(
        [x],
        [Term(LRA.Real(torch.tensor([[5.0]])), [X(x)])],
        [
            Term(LRA.Real(torch.tensor([[1.0]])), [one]),
            Term(LRA.Sub(), [X(x)], [x, one]),
        ],
    )


def _relu(e: str) -> str:
    return f"(ite (> {e} 0) {e} 0)"


def _dense(layers: int) -> str:
    """A ranking whose printed form shares subterms behind `let`."""
    a, b = "s0", "s0"
    for _ in range(layers):
        a, b = _relu(f"(+ {a} {b})"), _relu(f"(+ {a} (- {b}))")
    return f"(+ {a} {b})"


COUNTDOWN_INV = "(and (>= s0 0) (<= s0 100))"

# (name, module, claim, certificate, ask cvc5 for hints)
PROOFS = [
    ("bare", lambda: _fixture("counter"), None, None, False),
    ("countdown", lambda: _fixture("svcomp_countdown"), Claim(prp="(= s0 0)"),
     Certificate(COUNTDOWN_INV, "(ite (= s0 0) 0 s0)"), False),
    ("countdown_safe", lambda: _fixture("svcomp_countdown"),
     Claim(prp="(<= s0 100)", kind="safety"), Certificate(COUNTDOWN_INV),
     False),
    ("property_only", lambda: _fixture("svcomp_countdown"),
     Claim(prp="(= s0 0)"), Certificate(), False),
    ("twovars", lambda: _fixture("svcomp_twovars"), Claim(prp="(= s0 s1)"),
     Certificate("(and (>= s0 0) (<= s0 s1) (= s1 10))",
                 "(ite (= s0 s1) 0 (- s1 s0))"), False),
    ("vector_counter_pre", _vector_counter,
     Claim(prp="s[0][0] == 0",
           init_pre="And(e[0][0] >= 0, e[1][0] >= 0)",
           update_pre="e[0][0] >= 0"),
     Certificate(
         "And(s[0][0] >= 0, Or(s[0][0] <= s[1][0], s[0][0] <= s[2][0]))",
         "Ite(s[0][0] == 0, 0, (Ite(s[1][0] >= s[2][0], s[1][0], "
         "s[2][0]) - s[0][0]) + 1)"), False),
    ("shared_let", lambda: _fixture("svcomp_countdown"), Claim(prp="(= s0 0)"),
     Certificate(COUNTDOWN_INV, _dense(3)), False),
    ("hints_settle_a_branch", lambda: _fixture("svcomp_countdown"),
     Claim(prp="(= s0 0)"),
     Certificate(COUNTDOWN_INV, "(ite (<= s0 200) s0 999)"), True),
    ("nonlinear_hints", lambda: _fixture("svcomp_countdown"),
     Claim(prp="(= s0 0)"), Certificate(COUNTDOWN_INV, "(* s0 s0)"), True),
    ("real", _real_countdown, Claim(prp="(<= s0 0.0)"),
     Certificate("(<= s0 5.0)", "(to_int s0)"), False),
    ("real_safety", _real_countdown, Claim(prp="(<= s0 5.0)", kind="safety"),
     Certificate("(<= s0 5.0)"), False),
]


def _proofs(module: Module, claim: "Claim | None", cert: "Certificate | None",
            hints: bool, verbatim: "LeanAnswer | None" = None) -> str:
    """Proofs.lean and the standalone certificate, as one text."""
    lean = q = printed = None
    if claim is not None:
        q = predicate_terms(module, claim, cert)
        lean = render_certificate(q, verbatim=verbatim)
        if hints:
            printed = print_hints(solver_hints(q, SmtBudget()),
                                  q.msmt.ctrl_next)
    proofs = proof_scripts(LeanContext(module), lean, q, printed, verbatim)
    return "\n".join([
        "=== Certificate/Proofs.lean",
        generate_proofs_lean(lean, proofs),
        "=== standalone",
        generate_standalone_cert_lean(module, lean, proofs),
    ])


@pytest.mark.parametrize("name,make,claim,cert,hints", PROOFS,
                         ids=[c[0] for c in PROOFS])
def test_proofs_are_unchanged(name, make, claim, cert, hints):
    _check(f"proofs-{name}.golden", _proofs(make(), claim, cert, hints))


def test_proofs_of_a_nonlinear_lean_answer_are_unchanged():
    """`--infer ai` hands back Lean with no term behind it, so the facts
    cannot speak for its ranking; a product in it still brings `nlinarith`
    into the plan."""
    answer = LeanAnswer("fun s => (s 0 0) ≥ 0",
                        "fun s => ((s 0 0) * (s 0 0)).toNat")
    _check("proofs-lean_answer_nonlinear.golden",
           _proofs(_fixture("svcomp_countdown"), Claim(prp="(= s0 0)"),
                   None, False, verbatim=answer))
