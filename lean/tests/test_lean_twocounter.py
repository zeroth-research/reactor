"""Tests for the Module-to-Lean4 functional converter."""

from reactor_lean.codegen.context import LeanContext
from reactor_lean.project.scaffold import create_project
from reactor_lean.cli.certificate import render
from reactor_lean.proof.render import certificate_files
from reactor_lean.core.certificate import Certificate, Claim
from testdata import TESTS, fixture


def test_twobitcounter_generates_lean():
    m = fixture("twobit_lia")

    _ = create_project(
        output_dir=TESTS / "LeanTests",
        module=m,
        project_name="TwoBitCertificate",
        executable=True,
        # P: b0 is false. ranking = 2*b1 + b0 - 3, each bit read as 0 or 1.
        certificate=certificate_files(LeanContext(m), render(
            m,
            Claim(prp="(not s0)"),
            Certificate(ranking="(- (+ (ite s1 1 0) (ite s1 1 0) (ite s0 1 0)) 3)"),
        )),
    )
