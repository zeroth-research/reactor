"""
Lean compilation tests for the tests/lean/ lake project.

Fast tests (default):
    just lean-test tests/lean/                  # checks generated files exist

Slow tests (require lake, ReactLib and the Lake package cache):
    just lean-run pytest tests/lean/ -m slow    # runs lake build

Targets verified by the slow tests:
  Certs              — the encodings and the scalar Argmax (generated)
  ManualTests        — the semantics of Relational.Mat's reductions (manually written)

And a generated project with an executable, built and run.
"""
import subprocess
import tempfile
from pathlib import Path

import pytest

import testdata
from lake import lake_build
from testdata import LAKE_PROJECT


# ──────────────────────────────────────────────────────────────
# Fast tests — just verify file presence after fixture generation
# ──────────────────────────────────────────────────────────────


def test_generated_argmax_scalar_mirrors_matrix_form(generate_lean_files):
    """The emitted scalar Argmax must seed from s0 and compare strictly.

    `argmax1d_scalar_n_eq` proves the unrolled form equal to
    `Relational.Mat.argmax_1d` by unfolding both, so a neutral seed or a
    non-strict comparison here would leave that proof passing against a
    definition that disagrees with `torch.argmax`.
    """
    src = (LAKE_PROJECT / "Certs" / "ArgmaxScalar.lean").read_text()
    assert "let b0 : Nat × Int := (0, s0)" in src, "scalar Argmax is not seeded with element 0"
    assert "(0, (0 : Int))" not in src, "scalar Argmax still seeds a neutral value"
    assert ".2 < s1" in src, "scalar Argmax does not use a strict comparison"
    assert ".2 ≤ s" not in src, "scalar Argmax still uses a non-strict comparison"


# ──────────────────────────────────────────────────────────────
# Slow tests — full lake build
# ──────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_manual_tests_build(generate_lean_files):
    """ManualTests compiles — Relational.Mat's reductions mean what the
    examples there say they do."""
    r = lake_build("ManualTests")
    assert r.returncode == 0, (
        f"lake build ManualTests failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "ManualTests" in l]
    assert not sorry_lines, "ManualTests proof used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_argmax_scalar_equiv_build(generate_lean_files):
    """Certs/ArgmaxScalar.lean compiles — each `argmax1d_scalar_n_eq` proof
    goes through, which is what ties the generated unrolled form to
    `Relational.Mat.argmax_1d`."""
    r = lake_build("Certs.ArgmaxScalar")
    assert r.returncode == 0, (
        f"lake build Certs.ArgmaxScalar failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert not sorry_lines, "ArgmaxScalar proof used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
@pytest.mark.parametrize(
    "fixture,stdin,expected",
    [
        # LIA: x starts at 0, increments each step, resets at 10
        ("counter", "x\nx\nx\n", ["0", "1", "2"]),
        # BV: two-bit counter with enable=1 counts 00, 10, 01, 11
        ("twobit", "1\n1\n1\n", ["0x0#1", "0x0#1", "0x1#1", "0x0#1", "0x0#1", "0x1#1"]),
    ],
)
def test_generated_executable_builds_and_runs(
    generate_lean_files, fixture, stdin, expected
):
    """`reactor -x` end to end: generate a project, build `main`, run it.

    The only coverage the executable path has. Its three generated pieces
    have to agree — `update`'s curried signature, the component order in
    `showCtrl`/`parseExtl`, and each wire's element type — and none of that
    is visible without elaborating the result.

    Built against the Lake package cache, as `reactor` writes it with
    `$REACTOR_LAKE_CACHE` set, so the run does not fetch Mathlib; skipped
    without one.
    """
    from reactor_lean.codegen.context import LeanContext
    from reactor_lean.project.lake_cache import LAKE_CACHE_ENV, lake_cache
    from reactor_lean.project.lakefile import REACTLIB_ENV, reactlib_checkout
    from reactor_lean.project.scaffold import create_project
    from reactor_lean.proof.render import certificate_files

    if lake_cache() is None:
        pytest.skip(f"no Lake package cache: set ${LAKE_CACHE_ENV}")
    if reactlib_checkout() is None:
        pytest.skip(f"no ReactLib checkout: set ${REACTLIB_ENV}")

    ctx = LeanContext(testdata.fixture(fixture))
    with tempfile.TemporaryDirectory() as tmp:
        project = create_project(
            Path(tmp), ctx, "Rea", executable=True,
            certificate=certificate_files(ctx))

        build = subprocess.run(
            ["lake", "build", "main"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert build.returncode == 0, (
            "lake build main failed.\n"
            f"stdout:\n{build.stdout[-2000:]}\nstderr:\n{build.stderr[-800:]}"
        )

        run = subprocess.run(
            [str(project / ".lake" / "build" / "bin" / "main")],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert run.returncode == 0, f"executable failed: {run.stderr[-500:]}"
        assert run.stdout.split() == expected, (
            f"unexpected trace: {run.stdout.split()}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["Scalar", "Vec6", "Counter", "Vec32", "RealHalf"])
def test_scalar_encoding_builds(generate_lean_files, name):
    """The scalar encoding elaborates alongside the functional one.

    `product_type_scalar` flattens a multi-element wire into one component
    per element, and the body and the `_scalar_eq` proofs have to follow. No
    certificate in Certs/ carries a Scalar section, so without this the
    encoding was never compiled at all — it had four type errors per
    matrix-state module.
    """
    r = lake_build(f"Certs.ScalarEnc{name}")
    assert r.returncode == 0, (
        f"lake build Certs.ScalarEnc{name} failed.\n"
        f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-800:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert not sorry_lines, f"ScalarEnc{name} used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", ["Scalar", "Vec6", "Counter", "Vec32", "RealHalf", "TwoVars", "Mixed"]
)
def test_relational_encoding_builds(generate_lean_files, name):
    """ScalarRel elaborates on top of the scalar encoding.

    `effect_i` is per ctrl *wire*; the state tuple it is related to is per
    *element*. Projecting component `i` of that tuple therefore put a
    `Matrix (Fin 6) (Fin 1) Int` against an `Int x ... x Int` for any multi-element wire, and
    `Mixed` is the case where the offsets differ too. The chain that has teeth
    here is `TransRel_scalar_eq` into `TransRel_func_eq`: it ties the sliced
    relation back to the functional `update` through `pack`/`unpack`, so a
    slice at the wrong offset does not typecheck rather than proving a wrong
    equation. No certificate in Certs/ carries a ScalarRel section, so
    without this it was not compiled at all.

    The encodings are separate modules (see `conftest.py`), as they are in a
    generated project. Concatenated into one file they elaborate more than
    they do apart -- a `match` gets one auxiliary matcher per module -- and
    that difference alone was enough to hide a broken `effect_i_eq` from
    this test.
    """
    r = lake_build(f"Certs.RelEnc{name}")
    assert r.returncode == 0, (
        f"lake build Certs.RelEnc{name} failed.\n"
        f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-800:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert not sorry_lines, f"RelEnc{name} used sorry:\n" + "\n".join(sorry_lines)
