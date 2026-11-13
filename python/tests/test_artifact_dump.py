"""What a run leaves in `artifacts/`: the README, and the SMT it asked.

The point of the directory is that a reader can check the run without the run.
What `--pre-check` leaves there -- an obligation that can be handed to a
*different* solver, from the file alone, and give the same answer -- is
`verith`'s, and tested in `verith/tests/test_precheck_artifacts.py`.
Everything here is about the README staying an honest index of what is
there, and about the predicates a run was given.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from reactor.lean.artifacts import ArtifactStore

PY_DIR = Path(__file__).parent.parent
MODULE = PY_DIR / "tests" / "limits" / "mods" / "m_countdown.py"

pytest.importorskip("cvc5")


# ──────────────────────────────────────────────────────────────────────
# The README
# ──────────────────────────────────────────────────────────────────────


def _store(tmp_path, **kw):
    kw.setdefault("producer", "test")
    return ArtifactStore(dir=tmp_path / "artifacts", **kw)


def test_what_a_component_said_is_what_the_readme_says(tmp_path):
    """A component describes its own file as it writes it; nothing else knows
    enough to. The README is where those sentences end up, under the name of
    the file each is about."""
    store = _store(tmp_path)
    store.encoded("system", "system", "(check-sat)", what="The module, encoded.")
    store.put("inv", "(>= s0 0)", status="too_weak", what="A candidate invariant.",
              why="inductive, but it does not imply the property")

    readme = (tmp_path / "artifacts" / "README.md").read_text()
    assert "## `system.smt2`" in readme and "The module, encoded." in readme
    assert "A candidate invariant." in readme
    assert "does not imply the property" in readme, "the `why` is not carried"
    assert "*status: too_weak*" in readme, "a candidate's status is not flagged"
    # ... but a question the run asked has no status worth a line.
    assert "*status: encoded*" not in readme


def test_the_readme_is_the_index_and_not_a_log(tmp_path):
    """It is rendered from `index.json` rather than appended to, so a file a
    later run rewrites replaces its own section instead of gaining a second
    one. A workspace that is reused across runs would otherwise grow a README
    that is mostly history."""
    store = _store(tmp_path)
    for _ in range(3):
        store.encoded("system", "system", "(check-sat)", what="The module, encoded.")

    readme = (tmp_path / "artifacts" / "README.md").read_text()
    assert readme.count("## `system.smt2`") == 1, readme
    assert len(list((tmp_path / "artifacts").glob("system*"))) == 1


def test_a_candidate_accumulates_and_a_question_does_not(tmp_path):
    """Two namings, because they are two things: an indefinite number of runs
    may each propose an invariant, and every one of them is worth keeping;
    there is one `system.smt2`, and it is looked for by that name."""
    store = _store(tmp_path)
    store.put("inv", "(>= s0 0)", what="one")
    store.put("inv", "(>= s0 1)", what="another")
    store.encoded("system", "system", "(check-sat)", what="the module")

    names = sorted(p.name for p in (tmp_path / "artifacts").glob("*.smt*"))
    assert names == ["inv-0001-test.smt2", "inv-0002-test.smt2", "system.smt2"]


# ──────────────────────────────────────────────────────────────────────
# The SMT
# ──────────────────────────────────────────────────────────────────────


def _reactor(out: Path, *args) -> str:
    r = subprocess.run(
        ["uv", "run", "reactor", str(MODULE), "--buchi", "(= (X @intf.0) 0)",
         "--invariant", "(and (>= (X @intf.0) 0) (<= (X @intf.0) 100))", "--ranking", "(X @intf.0)",
         *args, "-o", str(out), "-p", "Rea"],
        cwd=PY_DIR, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_the_predicates_are_kept_as_the_text_they_were_given_as():
    """`.smt` is one predicate and not a script -- what a flag was handed,
    before anything encoded it. The README says which is which."""
    with tempfile.TemporaryDirectory() as tmp:
        _reactor(Path(tmp))
        art = Path(tmp) / "Rea" / "artifacts"
        assert (art / "property.smt").read_text().strip() == "(= s0 0)"
        assert (art / "ranking.smt").read_text().strip() == "s0"
        assert "is not a script" in (art / "README.md").read_text()


# ──────────────────────────────────────────────────────────────────────
# The answer, written back onto the question
# ──────────────────────────────────────────────────────────────────────


def _obligations(art: Path) -> dict:
    """`{name: (status, why)}` for every obligation the index describes."""
    import json

    return {
        e["name"].removeprefix("obligation-").removesuffix(".smt2"):
            (e["status"], e["why"])
        for e in json.loads((art / "index.json").read_text())
        if e["role"] == "obligation"
    }


def test_resolve_answers_a_question_without_rewriting_it(tmp_path):
    """The file is the question either way; only the metadata moves."""
    store = _store(tmp_path)
    a = store.encoded("obligation", "obligation-hrank", "(check-sat)",
                      what="The `hrank` obligation, negated.")
    assert a.status == "encoded"

    store.resolve(a, status="refuted", why="counterexample: s0 = 1")
    entry = _obligations(tmp_path / "artifacts")["hrank"]
    assert entry == ("refuted", "counterexample: s0 = 1")
    assert (tmp_path / "artifacts" / "obligation-hrank.smt2").read_text() \
        == "(check-sat)", "the question was rewritten, not merely answered"


def test_resolve_is_silent_about_a_question_never_asked(tmp_path):
    """An answer about an obligation that was never written is not an
    error: the pre-check has to run identically whatever was written."""
    store = _store(tmp_path)
    store.encoded("obligation", "obligation-hrank", "(check-sat)", what="x")
    store.resolve("obligation-nosuch.smt2", status="refuted", why="...")
    assert set(_obligations(tmp_path / "artifacts")) == {"hrank"}
