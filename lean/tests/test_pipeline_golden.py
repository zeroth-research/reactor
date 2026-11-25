"""The certificate files and `reactor` transcripts, pinned byte for byte.

Two layers: the codegen entry points on a fixed set of modules and
predicates, and whole in-process `reactor` runs (stdout and the two certificate
files). Both are compared with the snapshots under
`tests/golden/pipeline/`; set `REACTOR_UPDATE_GOLDEN=1` to rewrite them, and
only for a change to the output that is intended.

The certificate's obligations are proved by `sorry` here.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
import torch

from reactor import Module, Term, Var, Wire, X, Int, Real, Bool, tensor
from reactor_lean.proof.render import (
    generate_certificate_lean,
    generate_data_lean,
    generate_module_lean,
    generate_proofs_lean,
    generate_standalone_cert_lean,
)
from reactor_lean.parse import Certificate, Claim
from reactor_lean.proof.render import lean_certificate
from reactor_lean.codegen.context import LeanContext
from reactor_lean.project.lake_cache import LAKE_CACHE_ENV
from testdata import FIXTURES, TESTS, dense, fixture

GOLDEN = TESTS / "golden" / "pipeline"
UPDATE = os.environ.get("REACTOR_UPDATE_GOLDEN") == "1"


@pytest.fixture(autouse=True)
def _no_lake_cache(monkeypatch):
    """A run against a Lake package cache says so, naming where the cache
    is: the snapshots are of the generator, not of the machine."""
    monkeypatch.delenv(LAKE_CACHE_ENV, raising=False)


def _check(name: str, text: str) -> None:
    path = GOLDEN / name
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"no snapshot {path}; run with REACTOR_UPDATE_GOLDEN=1"
    assert text == path.read_text(), f"{name} differs from its snapshot"


# ══════════════════════════════════════════════════════════════════════════
# Codegen
# ══════════════════════════════════════════════════════════════════════════


def _vector_counter() -> Module:
    """3x1 state over a 2x1 input, as `tests/lean/conftest.py` builds it."""
    state = Var(Int([3, 1]))
    extl = Var(Int([2, 1]))
    zero31 = torch.zeros((3, 1), dtype=torch.int64)
    zero11 = torch.zeros((1, 1), dtype=torch.int64)
    a = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [Term(tensor.int.Linear(a, zero31), [X(state)], [X(extl)])]
    x, y, z = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    x_lt_y, x_lt_z, cond = Wire(Bool([1, 1])), Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    yes, no = Wire(Int([3, 1])), Wire(Int([3, 1]))
    row = lambda i: torch.tensor([[int(j == i) for j in range(3)]], dtype=torch.int64)
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    diag_yz = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int64)
    update = [
        Term(tensor.int.Linear(row(0), zero11), [x], [state]),
        Term(tensor.int.Linear(row(1), zero11), [y], [state]),
        Term(tensor.int.Linear(row(2), zero11), [z], [state]),
        Term(tensor.int.Lt([1, 1]), [x_lt_y], [x, y]),
        Term(tensor.int.Lt([1, 1]), [x_lt_z], [x, z]),
        Term(tensor.bool.Or([1, 1]), [cond], [x_lt_y, x_lt_z]),
        Term(tensor.int.Linear(torch.eye(3, dtype=torch.int64), e1), [yes], [state]),
        Term(tensor.int.Linear(diag_yz, zero31), [no], [state]),
        Term(tensor.int.Ite([3, 1]), [X(state)], [cond, yes, no]),
    ]
    return Module.sequential([state, extl], init, update)


def _real_countdown() -> Module:
    x = Var(Real([1, 1]))
    one = Wire(Real([1, 1]))
    return Module.sequential(
        [x],
        [Term(tensor.real.Const(torch.tensor([[5.0]])), [X(x)])],
        [
            Term(tensor.real.Const(torch.tensor([[1.0]])), [one]),
            Term(tensor.real.Sub([1, 1]), [X(x)], [x, one]),
        ],
    )


COUNTDOWN_INV = "(and (>= s0 0) (<= s0 100))"

# (name, module, claim, certificate).
CODEGEN = [
    ("bare", lambda: fixture("counter"), None, None),
    ("countdown", lambda: fixture("svcomp_countdown"), Claim(prp="(= s0 0)"),
     Certificate(COUNTDOWN_INV, "(ite (= s0 0) 0 s0)")),
    ("countdown_safe", lambda: fixture("svcomp_countdown"),
     Claim(prp="(<= s0 100)", kind="safety"), Certificate(COUNTDOWN_INV)),
    ("property_only", lambda: fixture("svcomp_countdown"),
     Claim(prp="(= s0 0)"), Certificate()),
    ("twovars", lambda: fixture("svcomp_twovars"), Claim(prp="(= s0 s1)"),
     Certificate("(and (>= s0 0) (<= s0 s1) (= s1 10))",
                 "(ite (= s0 s1) 0 (- s1 s0))")),
    ("vector_counter_pre", _vector_counter,
     Claim(prp="s0[0][0] == 0",
           init_pre="And(e0[0][0] >= 0, e0[1][0] >= 0)",
           update_pre="e0[0][0] >= 0"),
     Certificate(
         "And(s0[0][0] >= 0, Or(s0[0][0] <= s0[1][0], s0[0][0] <= s0[2][0]))",
         "Ite(s0[0][0] == 0, 0, (Ite(s0[1][0] >= s0[2][0], s0[1][0], "
         "s0[2][0]) - s0[0][0]) + 1)")),
    ("shared_let", lambda: fixture("svcomp_countdown"), Claim(prp="(= s0 0)"),
     Certificate(COUNTDOWN_INV, dense(3, cmp=">"))),
    ("real", _real_countdown, Claim(prp="(<= s0 0.0)"),
     Certificate("(<= s0 5.0)", "(to_int s0)")),
    ("real_safety", _real_countdown, Claim(prp="(<= s0 5.0)", kind="safety"),
     Certificate("(<= s0 5.0)")),
]


def _codegen(module: Module, claim: "Claim | None", cert: "Certificate | None") -> str:
    """Data.lean, Module.lean, Proofs.lean, Certificate.lean and the
    standalone certificate, as one text."""
    lean = lean_certificate(module, claim, cert or Certificate())
    ctx = LeanContext(module)
    return "\n".join([
        "=== Certificate/Data.lean",
        generate_data_lean(ctx, lean),
        "=== Certificate/Module.lean",
        generate_module_lean(),
        "=== Certificate/Proofs.lean",
        generate_proofs_lean(lean),
        "=== Certificate/Certificate.lean",
        generate_certificate_lean(ctx, lean),
        "=== standalone",
        generate_standalone_cert_lean(module, lean),
    ])


@pytest.mark.parametrize("name,make,claim,cert", CODEGEN,
                         ids=[c[0] for c in CODEGEN])
def test_codegen_is_unchanged(name, make, claim, cert):
    _check(f"codegen-{name}.golden", _codegen(make(), claim, cert))



# ══════════════════════════════════════════════════════════════════════════
# `reactor`, in process
# ══════════════════════════════════════════════════════════════════════════

COUNTER = "counter"
COUNTDOWN = "svcomp_countdown"
BOUNDED = "svcomp_bounded_input"

# (name, fixture, arguments). `-o` and `-p` are added.
RUNS = [
    ("bare", COUNTER, []),
    ("supplied", COUNTER, [
        "--buchi", "(= (X @intf.0) 0)",
        "--invariant", "(and (>= (X @intf.0) 0) (<= (X @intf.0) 9))",
        "--ranking", "(ite (<= (X @intf.0) 200) (ite (= (X @intf.0) 0) 0 (- 10 (X @intf.0))) 999)",
    ]),
    ("supplied_pre", BOUNDED, [
        "--safety", "(>= (X @intf.0) 0)", "--pre", "(>= (X @extl.0) 0)",
        "--invariant", "(>= (X @intf.0) 0)",
    ]),
    ("cert_file", COUNTER, [
        "--buchi", "(= (X @intf.0) 0)",
        "--invariant", "(and (>= (X @intf.0) 0) (<= (X @intf.0) 9))",
        "--ranking", "(ite (= (X @intf.0) 0) 0 (- 10 (X @intf.0)))",
        "--cert-file", "{out}/Cert.lean",
    ]),
    ("unreadable_property", COUNTER, ["--buchi", "(= (X @intf.0)"]),
]

def _runs() -> list:
    return [pytest.param(name, fixture, args, id=name)
            for name, fixture, args in RUNS]


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _normalise(text: str, out: Path) -> str:
    text = _ANSI.sub("", text)
    text = text.replace(str(out), "<OUT>").replace(str(FIXTURES), "<FIX>")
    text = re.sub(r"#\d+", "#N", text)                     # wire ids
    text = re.sub(r"\s+\d+ ms\b", " N ms", text)            # verdict timings
    text = re.sub(r"\b\d+(\.\d+)? s\b", "N s", text)        # route timings
    return text


def _transcript(monkeypatch, capsys, tmp_path, fixture, args) -> str:
    from reactor_lean.cli import main

    out = tmp_path / "out"
    argv = [a.replace("{out}", str(out)) for a in args]
    monkeypatch.setattr(sys, "argv", [
        "reactor", str(FIXTURES / f"{fixture}.py"), *argv,
        "-o", str(out), "-p", "G",
    ])
    try:
        main()
        status = "exit 0"
    except SystemExit as e:
        status = f"exit {e.code}"
    captured = capsys.readouterr()
    project = out / "G"

    def read(rel: str) -> str:
        p = project / rel
        return p.read_text() if p.exists() else "(none)\n"

    parts = [
        f"=== {status}",
        "=== stdout", captured.out,
        "=== stderr", captured.err,
        "=== Certificate/Data.lean", read("Certificate/Data.lean"),
        "=== Certificate/Module.lean", read("Certificate/Module.lean"),
        "=== Certificate/Proofs.lean", read("Certificate/Proofs.lean"),
        "=== Certificate/Certificate.lean", read("Certificate/Certificate.lean"),
    ]
    if (out / "Cert.lean").exists():
        parts += ["=== Cert.lean", (out / "Cert.lean").read_text()]
    return _normalise("\n".join(parts), out)


@pytest.mark.parametrize("name,fixture,args", _runs())
def test_reactor_run_is_unchanged(name, fixture, args, monkeypatch, capsys,
                                 tmp_path):
    _check(f"reactor-{name}.golden",
           _transcript(monkeypatch, capsys, tmp_path, fixture, args))
