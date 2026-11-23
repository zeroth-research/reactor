"""SV-COMP style termination verification tests.

Each test creates a reactive module, provides a correct invariant and ranking
function, and generates a Lean 4 project. With the proofs `verith` writes
(the `verith/` project beside this one; the slow tests are skipped without
it), the zeroth_hammer should close all proof obligations including hrank.

To verify proofs actually close (slow, requires lake + mathlib):
    .venv/bin/python -m pytest tests/test_lean_svcomp.py -m slow -v
"""

import subprocess
import pytest

from reactor_lean.codegen.context import LeanContext
from reactor_lean.project.scaffold import create_project
from reactor_lean.pipeline.certificate import render
from reactor_lean.proof.render import certificate_files
from reactor_lean.core.certificate import Certificate, Claim
from testdata import TESTS, fixture

OUTPUT_DIR = TESTS / "LeanTests"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _gen(name, module, prp, inv, ranking, pre=None, proved=False):
    """Generate a Lean project with certificate and return the project dir;
    `proved`, with the proofs `verith` writes for it."""
    claim = Claim(prp=prp, init_pre=pre, update_pre=pre)
    cert = Certificate(inv=inv, ranking=ranking)
    ctx = LeanContext(module)
    lean_cert = render(module, claim, cert)
    if not proved:
        return create_project(
            output_dir=OUTPUT_DIR,
            module=module,
            project_name=name,
            certificate=certificate_files(ctx, lean_cert),
        )
    pytest.importorskip("verith")
    from verith.proofs import LAKEFILE, hammer_files, prove

    proofs = prove(module, ctx, claim, cert, lean_cert)
    return create_project(
        output_dir=OUTPUT_DIR,
        module=module,
        project_name=name,
        certificate={**certificate_files(ctx, lean_cert, proofs),
                     **hammer_files()},
        lakefile=LAKEFILE,
    )


def _lake_build(project_dir):
    """Run lake build Certificate and check for sorry/errors."""
    r = subprocess.run(
        ["lake", "update"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, f"lake update failed: {r.stderr[-300:]}"

    r = subprocess.run(
        ["lake", "build", "Certificate"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    sorry_lines = [
        l for l in r.stdout.split("\n") if "sorry" in l and "Certificate/" in l
    ]
    error_lines = [l for l in r.stdout.split("\n") if l.startswith("error:")]
    assert r.returncode == 0, r"lake build failed:\n" + "\n".join(error_lines[:5])
    assert len(sorry_lines) == 0, "Certificate has sorry:\n" + "\n".join(sorry_lines)


# ──────────────────────────────────────────────────────────────
# Generation tests (fast — just check project is created)
# ──────────────────────────────────────────────────────────────


def test_countdown_generates():
    m = fixture("svcomp_countdown")
    p = _gen(
        "SvcompCountdown",
        m,
        prp="(= s0 0)",
        inv="(and (>= s0 0) (<= s0 100))",
        ranking="(ite (= s0 0) 0 s0)",
    )
    assert (p / "Certificate" / "Certificate.lean").exists()


def test_twovars_generates():
    m = fixture("svcomp_twovars")
    p = _gen(
        "SvcompTwovars",
        m,
        prp="(= s0 s1)",
        inv="(and (>= s0 0) (<= s0 s1) (= s1 10))",
        ranking="(ite (= s0 s1) 0 (- s1 s0))",
    )
    assert (p / "Certificate" / "Certificate.lean").exists()


def test_collatz_bounded_generates():
    m = fixture("svcomp_collatz_bounded")
    p = _gen(
        "SvcompCollatz",
        m,
        prp="(= s0 1)",
        inv="(and (>= s0 1) (<= s0 8))",
        ranking="(ite (= s0 1) 0 (- s0 1))",
    )
    assert (p / "Certificate" / "Certificate.lean").exists()


def test_nested_generates():
    m = fixture("svcomp_nested")
    p = _gen(
        "SvcompNested",
        m,
        prp="(and (= s0 0) (= s1 0))",
        inv="(and (>= s0 0) (<= s0 3) (>= s1 0) (<= s1 3))",
        ranking="(ite (and (= s0 0) (= s1 0)) 0 (+ (* (- 3 s0) 4) (- 3 s1)))",
    )
    assert (p / "Certificate" / "Certificate.lean").exists()


def test_gcd_generates():
    m = fixture("svcomp_gcd")
    p = _gen(
        "SvcompGcd",
        m,
        prp="(= s0 s1)",
        inv="(and (>= s0 1) (>= s1 1))",
        ranking="(ite (= s0 s1) 0 (+ (- s0 1) (- s1 1)))",
    )
    assert (p / "Certificate" / "Certificate.lean").exists()


# ──────────────────────────────────────────────────────────────
# Proof verification tests (slow — require lake + mathlib)
# ──────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_countdown_proofs():
    m = fixture("svcomp_countdown")
    p = _gen(
        "SvcompCountdown",
        m,
        prp="(= s0 0)",
        inv="(and (>= s0 0) (<= s0 100))",
        ranking="(ite (= s0 0) 0 s0)",
        proved=True,
    )
    _lake_build(p)


@pytest.mark.slow
def test_twovars_proofs():
    m = fixture("svcomp_twovars")
    p = _gen(
        "SvcompTwovars",
        m,
        prp="(= s0 s1)",
        inv="(and (>= s0 0) (<= s0 s1) (= s1 10))",
        ranking="(ite (= s0 s1) 0 (- s1 s0))",
        proved=True,
    )
    _lake_build(p)


@pytest.mark.slow
def test_collatz_bounded_proofs():
    m = fixture("svcomp_collatz_bounded")
    p = _gen(
        "SvcompCollatz",
        m,
        prp="(= s0 1)",
        inv="(and (>= s0 1) (<= s0 8))",
        ranking="(ite (= s0 1) 0 (- s0 1))",
        proved=True,
    )
    _lake_build(p)
