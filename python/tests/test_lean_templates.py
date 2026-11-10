"""The templates, and the Lean library the files they render import.

`templates/` holds the Jinja templates of the generated project's files and
no Lean sources: the library those files import -- `Relational.*`, and
`LeanAI` -- is reactor-lean's, and the project's lakefile requires it. A
`.lean` file left under `templates/` would be read by nothing, so editing it
would silently do nothing, which is what the first two tests exist to
prevent. (`templates/` once carried two byte-identical sets of the Core
files, and edits to the unread one did nothing for months.)
"""

from pathlib import Path

import pytest

from reactor.lean.core.errors import Refused
from reactor.lean.project.lakefile import (
    REACTOR_LEAN_ENV,
    REACTOR_LEAN_PACKAGES,
    REACTOR_LEAN_URL,
    generate_lakefile,
    reactor_lean_checkout,
    with_reactor_lean,
)
from reactor.lean.project.templates import TEMPLATES_DIR, PROJECT_TEMPLATES_DIR

FIXTURES = Path(__file__).parent / "fixtures"


def test_the_templates_hold_no_lean_sources():
    """The Lean library is reactor-lean's; a copy here can never be emitted."""
    stray = sorted(str(p.relative_to(TEMPLATES_DIR))
                   for p in TEMPLATES_DIR.rglob("*.lean"))
    assert not stray, (
        "these are read by nothing, so editing them has no effect -- the "
        f"Lean library the project imports is in reactor-lean: {stray}"
    )


def test_every_project_template_is_rendered():
    """A `.j2` that no `render()` call names drifts from the code that prints it."""
    lean_pkg = TEMPLATES_DIR.parent
    src = "\n".join(p.read_text() for p in lean_pkg.rglob("*.py"))
    unused = sorted(
        rel
        for p in PROJECT_TEMPLATES_DIR.rglob("*.j2")
        if f'"{(rel := p.relative_to(TEMPLATES_DIR).as_posix())}"' not in src
    )
    assert not unused, f"these templates are rendered by nothing: {unused}"


# ══════════════════════════════════════════════════════════════════════
#  The lakefile requires reactor-lean: from git, or from $REACTOR_LEAN
# ══════════════════════════════════════════════════════════════════════

def _checkout(root: Path) -> Path:
    for p in REACTOR_LEAN_PACKAGES:
        (root / p).mkdir(parents=True)
        (root / p / "lakefile.toml").write_text("")
    return root


def _require(lakefile: str, name: str) -> str:
    """The `[[require]]` table named `name`, up to the next blank line."""
    block = lakefile.split(f'[[require]]\nname = "{name}"\n', 1)
    assert len(block) == 2, f"the lakefile does not require {name}"
    return block[1].split("\n\n", 1)[0]


def test_the_lakefile_requires_each_package_from_its_directory_of_the_repo(
        monkeypatch):
    monkeypatch.delenv(REACTOR_LEAN_ENV, raising=False)
    lakefile = generate_lakefile("P")
    for p in REACTOR_LEAN_PACKAGES:
        req = _require(lakefile, p)
        assert f'git = "{REACTOR_LEAN_URL}"' in req
        assert f'subDir = "{p}"' in req
        assert "path =" not in req


def test_reactor_lean_set_requires_the_checkout_by_path(monkeypatch, tmp_path):
    root = _checkout(tmp_path / "rl")
    monkeypatch.setenv(REACTOR_LEAN_ENV, str(root))
    lakefile = generate_lakefile("P")
    for p in REACTOR_LEAN_PACKAGES:
        assert _require(lakefile, p) == f'path = "{root.resolve() / p}"'
    assert REACTOR_LEAN_URL not in lakefile


def test_a_reactor_lean_that_is_not_a_checkout_is_refused_before_the_project(
        monkeypatch, tmp_path):
    """Otherwise the lakefile points at nothing, and lake says so only when
    it is asked to build, long after the project was written."""
    from reactor.lean.pipeline import main

    monkeypatch.setenv(REACTOR_LEAN_ENV, str(tmp_path))
    with pytest.raises(Refused, match="is not a reactor-lean checkout"):
        reactor_lean_checkout()
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "reactor", str(FIXTURES / "counter.py"),
        "--safety", "(>= (X @intf.0) 0)", "-o", str(out), "-p", "P"])
    with pytest.raises(SystemExit, match="is not a reactor-lean checkout"):
        main()
    assert not out.exists()


def test_a_copied_manifest_gets_the_checkout_as_its_own(tmp_path):
    """`lake build` reads a path dependency's `dir` from the manifest, not
    the lakefile, so a manifest copied from elsewhere must be repointed."""
    root = _checkout(tmp_path / "rl")
    stale = {"name": "Relational", "type": "path", "dir": "../../elsewhere"}
    kept = {"name": "cslib", "type": "git"}
    out = with_reactor_lean({"version": "1.1.0", "packages": [kept, stale]}, root)
    assert out["version"] == "1.1.0"
    assert out["packages"][0] == kept
    dirs = {e["name"]: e["dir"] for e in out["packages"] if e["type"] == "path"}
    assert dirs == {p: str(root / p) for p in REACTOR_LEAN_PACKAGES}


# ══════════════════════════════════════════════════════════════════════
#  The reductions have to stay computable under the certificate's `simp`
# ══════════════════════════════════════════════════════════════════════

REDUCTIONS = ("matMin", "matMax", "argmax_1d", "argmax")


def _mat_source() -> str:
    checkout = reactor_lean_checkout()
    if checkout is None:
        pytest.skip(f"no reactor-lean checkout: set ${REACTOR_LEAN_ENV}")
    return (checkout / "Relational" / "Relational" / "Mat.lean").read_text()


def test_the_reductions_enumerate_with_ofFn_not_finRange():
    """`List.finRange` is what made these unprovable for a year.

    `simp_mat` carries `List.ofFn_succ` and `List.ofFn_zero` and no lemma
    that computes `List.finRange`, so a `finRange` fold stays stuck in the
    goal at every concrete shape and every closer fails on it; `be70a43`
    moved them to `List.ofFn`. Rewriting one of these back to `finRange`
    would look like a simplification and would silently cost twelve
    certificates.
    """
    src = _mat_source()
    for name in REDUCTIONS:
        body = src.split(f"def {name} ", 1)[1].split("\ndef ", 1)[0]
        assert "List.ofFn" in body, f"{name} no longer enumerates with List.ofFn"
        assert "List.finRange" not in body, (
            f"{name} enumerates with List.finRange, which `simp_mat` cannot "
            f"compute (fixed in be70a43)"
        )


def _counter_ctx():
    from reactor.lean.codegen.context import LeanContext
    from reactor.lean.inputs.python import load_module

    fixture = Path(__file__).parent / "fixtures" / "counter.py"
    return LeanContext(load_module(fixture))


def test_the_certificate_simp_set_unfolds_them():
    """Computing the fold is no use if the definition never opens."""
    from reactor.lean.proof.render import generate_certificate_lean

    cert = generate_certificate_lean(_counter_ctx())
    simp_mat = [ln for ln in cert.splitlines()
                if "simp [" in ln and "MatAdd_apply" in ln]
    assert simp_mat, "the simp_mat macro moved"
    for name in REDUCTIONS:
        assert name in simp_mat[0], f"{name} is not in simp_mat"
