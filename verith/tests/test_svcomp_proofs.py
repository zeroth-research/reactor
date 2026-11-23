"""SV-COMP style termination certificates, proved and built.

Each test takes a reactive module and a correct invariant and ranking
function, generates its Lean 4 project with the proofs `verith` writes, and
builds it: `zeroth_hammer` closes every obligation, `hrank` included.
(`lean/tests/test_lean_svcomp.py` generates the same projects, unproved.)

Slow: requires lake, ReactLib (`$REACTLIB`) and, to not fetch and build
Mathlib per project, the Lake package cache (`$REACTOR_LAKE_CACHE`):
    just vr-run pytest tests/test_svcomp_proofs.py -m slow -v
"""

from pathlib import Path

import pytest

from reactor_lean.codegen.context import LeanContext
from reactor_lean.core.certificate import Certificate, Claim
from reactor_lean.inputs.python import load_module
from reactor_lean.pipeline.certificate import render
from reactor_lean.proof.render import certificate_files
from reactor_lean.project.scaffold import create_project
from verith.proofs import LAKEFILE, hammer_files, prove
from verith.proofs.build import build_certificate

TESTS = Path(__file__).resolve().parent
OUTPUT_DIR = TESTS / "LeanTests"


def _proved(name, fixture, prp, inv, ranking):
    """The project of `fixture`'s module with the certificate, proved."""
    module = load_module(TESTS / "fixtures" / f"{fixture}.py")
    claim = Claim(prp=prp)
    cert = Certificate(inv=inv, ranking=ranking)
    ctx = LeanContext(module)
    lean_cert = render(module, claim, cert)
    proofs = prove(module, ctx, claim, cert, lean_cert)
    return create_project(
        output_dir=OUTPUT_DIR,
        module=module,
        project_name=name,
        certificate={**certificate_files(ctx, lean_cert, proofs),
                     **hammer_files()},
        lakefile=LAKEFILE,
    )


@pytest.mark.slow
def test_countdown_proofs():
    build_certificate(_proved(
        "SvcompCountdown", "svcomp_countdown",
        prp="(= s0 0)",
        inv="(and (>= s0 0) (<= s0 100))",
        ranking="(ite (= s0 0) 0 s0)",
    ))


@pytest.mark.slow
def test_twovars_proofs():
    build_certificate(_proved(
        "SvcompTwovars", "svcomp_twovars",
        prp="(= s0 s1)",
        inv="(and (>= s0 0) (<= s0 s1) (= s1 10))",
        ranking="(ite (= s0 s1) 0 (- s1 s0))",
    ))


@pytest.mark.slow
def test_collatz_bounded_proofs():
    build_certificate(_proved(
        "SvcompCollatz", "svcomp_collatz_bounded",
        prp="(= s0 1)",
        inv="(and (>= s0 1) (<= s0 8))",
        ranking="(ite (= s0 1) 0 (- s0 1))",
    ))
