"""SV-COMP style termination certificates.

Each test creates a reactive module, provides a correct invariant and ranking
function, and generates a Lean 4 project whose certificate states them.
"""

from reactor_lean.codegen.context import LeanContext
from reactor_lean.project.scaffold import create_project
from reactor_lean.cli.certificate import render
from reactor_lean.proof.render import certificate_files
from reactor_lean.parse import Certificate, Claim
from testdata import TESTS, fixture

OUTPUT_DIR = TESTS / "LeanTests"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _gen(name, module, prp, inv, ranking, pre=None):
    """Generate a Lean project with certificate and return the project dir."""
    claim = Claim(prp=prp, init_pre=pre, update_pre=pre)
    cert = Certificate(inv=inv, ranking=ranking)
    ctx = LeanContext(module)
    lean_cert = render(module, claim, cert)
    return create_project(
        output_dir=OUTPUT_DIR,
        module=module,
        project_name=name,
        certificate=certificate_files(ctx, lean_cert),
    )


# ──────────────────────────────────────────────────────────────
# Generation tests (check the project is created)
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
