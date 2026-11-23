"""
Lean compilation tests for the tests/lean/ lake project: the hammer, and
the certificates `verith` proves.

Fast tests (default):
    just vr-test tests/lean/                  # checks generated files exist

Slow tests (require lake, ReactLib and the Lake package cache):
    just vr-run pytest tests/lean/ -m slow    # runs lake build

Targets verified by the slow tests:
  ZerothHammerTests  — individual tactic examples (manually written)
  ZerothHammer       — standalone zeroth_hammer tactic (generated)
  Certs              — self-contained module certificates (generated)
  ManualTests        — zeroth_hammer on standalone goal shapes (manually written)
"""
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────
# Fast tests — just verify file presence after fixture generation
# ──────────────────────────────────────────────────────────────


def test_generated_files_present(generate_lean_files):
    """ZerothHammer.lean and Certs/*.lean exist after generation fixture runs."""
    assert (PROJECT / "ZerothHammer.lean").exists(), "ZerothHammer.lean not generated"
    for name in ("Countdown", "CountdownSafe", "TwoVars", "Collatz"):
        path = PROJECT / "Certs" / f"{name}.lean"
        assert path.exists(), f"Certs/{name}.lean not generated"


def test_the_bridges_are_generated(generate_lean_files):
    """The `--fbk-proveit` bridges."""
    for name in ("BridgeCountdown", "BridgeTwoVars"):
        path = PROJECT / "Certs" / f"{name}.lean"
        assert path.exists(), f"Certs/{name}.lean not generated"


def test_the_learned_certificate_is_generated(generate_lean_files):
    """`--infer nuterm` writes one too, so the `Certs` build elaborates a
    ranking function of the shape the learner produces. It needs cvc5 to
    render its SMT-LIB as Lean, so its absence is a skip, not a failure."""
    pytest.importorskip("cvc5")
    path = PROJECT / "Certs" / "LearnedCountdown.lean"
    assert path.exists(), "Certs/LearnedCountdown.lean not generated"
    src = path.read_text()
    assert "sorry" not in src, "the learned certificate left an obligation open"
    assert "if " in src, "the ranking function should carry the net's ReLU units"


# ──────────────────────────────────────────────────────────────
# Slow tests — full lake build
# ──────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_zeroth_hammer_proofs(generate_lean_files, lake_build):
    """ZerothHammerTests.lean compiles — every example in the file type-checks."""
    r = lake_build("ZerothHammerTests")
    assert r.returncode == 0, (
        f"lake build ZerothHammerTests failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert "sorry" not in r.stdout and "warning: declaration uses 'sorry'" not in r.stderr, (
        "Build succeeded but some proof used sorry."
    )


@pytest.mark.slow
def test_manual_tests_build(generate_lean_files, lake_build):
    """ManualTests/Basic.lean compiles — zeroth_hammer closes each standalone example."""
    r = lake_build("ManualTests")
    assert r.returncode == 0, (
        f"lake build ManualTests failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "ManualTests" in l]
    assert not sorry_lines, "ManualTests proof used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_the_learned_certificate_closes_in_lean(generate_lean_files, lake_build):
    """`Certs/LearnedCountdown.lean` compiles.

    The certificate `--infer nuterm` proved for itself is one `zeroth_hammer`
    closes too -- `hrank` included, over a ranking function that is a sum of
    ReLU units rather than the single `ite` every hand-written certificate in
    this suite uses."""
    pytest.importorskip("cvc5")
    r = lake_build("Certs.LearnedCountdown")
    assert r.returncode == 0, (
        f"lake build Certs.LearnedCountdown failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert not sorry_lines, "learned certificate used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_playground_build(generate_lean_files, lake_build):
    """Playground.lean compiles — proof-pattern experiments and regression tests."""
    r = lake_build("Playground")
    assert r.returncode == 0, (
        f"lake build Playground failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Playground" in l]
    assert not sorry_lines, "Playground proof used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_hugecounter_build(generate_lean_files, lake_build):
    """Certs/HugeCounter.lean: a 32-wide state whose transitions are 32-wide
    contractions. Pre-contraction keeps this tractable where a dense matrix
    literal + symbolic sum-expansion blows the heartbeat budget."""
    r = lake_build("Certs.HugeCounter")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.HugeCounter failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "HugeCounter certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_bigcounter_build(generate_lean_files, lake_build):
    """Certs/BigCounter.lean: a 6-vector state whose every transition is a 6-wide
    MatMul contraction — exercises Fin.sum_univ_succ expansion at scale."""
    r = lake_build("Certs.BigCounter")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.BigCounter failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "BigCounter certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_counter_build(generate_lean_files, lake_build):
    """Certs/Counter.lean: a 3×1 vector-state module with LIA.Linear transitions
    and tuple-select (s[i][j]) predicates — zeroth_hammer closes all obligations."""
    r = lake_build("Certs.Counter")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Counter failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "Counter certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_countdown_build(generate_lean_files, lake_build):
    """Certs/Countdown.lean: zeroth_hammer closes all proof obligations."""
    r = lake_build("Certs.Countdown")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Countdown failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert not sorry_lines, "Countdown certificate has sorry:\n" + "\n".join(sorry_lines)


def test_generated_safety_cert_has_no_ranking(generate_lean_files):
    """`Certs/CountdownSafe.lean` is the `--safety` shape: `rule_globally`
    over an invariant that implies `P`, with no ranking function defined or
    named anywhere -- a `def ranking := sorry` would be a `sorry` in a file
    whose point is that it has none."""
    src = (PROJECT / "Certs" / "CountdownSafe.lean").read_text()
    assert "rule_globally" in src and "theorem inv_imp_P" in src
    assert "rule_buchi" not in src
    assert "ranking" not in src and "sorry" not in src


@pytest.mark.slow
@pytest.mark.parametrize("name", ["Countdown", "TwoVars"])
def test_fbk_bridge_builds(generate_lean_files, lake_build, name):
    """`Certs/Bridge*.lean`: the proof that the `--fbk-proveit` model is the
    module.

    The certificate that route installs is about the *model*; this is the
    file that carries it back. Two independent translators are on the two
    sides of `bridge_k` -- `smt.encode` + cvc5 against
    `translate_terms_scalar` -- so nothing here closes by `rfl` alone, and
    a translation defect in either shows up as an unclosed goal.
    """
    r = lake_build(f"Certs.Bridge{name}")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Bridge{name} failed.\n"
        f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, f"Bridge{name} used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_countdownsafe_build(generate_lean_files, lake_build):
    """Certs/CountdownSafe.lean: the safety certificate elaborates, which is
    what says `rule_globally lts P hP` is applied to the right things --
    `hP` transfers the invariant to `P` through `inv_imp_P`."""
    r = lake_build("Certs.CountdownSafe")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.CountdownSafe failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "CountdownSafe certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_twovars_build(generate_lean_files, lake_build):
    """Certs/TwoVars.lean: zeroth_hammer closes all proof obligations."""
    r = lake_build("Certs.TwoVars")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.TwoVars failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert not sorry_lines, "TwoVars certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_collatz_build(generate_lean_files, lake_build):
    """Certs/Collatz.lean: zeroth_hammer closes all proof obligations."""
    r = lake_build("Certs.Collatz")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Collatz failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert not sorry_lines, "Collatz certificate has sorry:\n" + "\n".join(sorry_lines)


