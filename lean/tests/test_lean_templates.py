"""The templates, and the Lean library the files they render import.

`templates/` holds the Jinja templates of the generated project's files and
no Lean sources: the library those files import -- `Relational.*`, and
`LeanAI` -- is ReactLib's, and the project's lakefile requires it. A
`.lean` file left under `templates/` would be read by nothing, so editing it
would silently do nothing, which is what the first two tests exist to
prevent. (`templates/` once carried two byte-identical sets of the Core
files, and edits to the unread one did nothing for months.)
"""

import json
import re
from pathlib import Path

import pytest

from reactor_lean.core.errors import Refused
from reactor_lean.project.lake_cache import (
    LAKE_CACHE_ENV,
    cached_manifest,
    lake_cache,
)
from reactor_lean.project.lakefile import (
    FIXED_REQUIRES,
    REACTLIB_ENV,
    REACTLIB_PACKAGES,
    REACTLIB_REV,
    REACTLIB_URL,
    Require,
    generate_lakefile,
    project_requires,
    reactlib_checkout,
)
from reactor_lean.project.scaffold import write_lakefile
from reactor_lean.project.templates import TEMPLATES_DIR, PROJECT_TEMPLATES_DIR
from testdata import FIXTURES


def test_the_templates_hold_no_lean_sources():
    """The Lean library is ReactLib's; a copy here can never be emitted."""
    stray = sorted(str(p.relative_to(TEMPLATES_DIR))
                   for p in TEMPLATES_DIR.rglob("*.lean"))
    assert not stray, (
        "these are read by nothing, so editing them has no effect -- the "
        f"Lean library the project imports is in ReactLib: {stray}"
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
#  The lakefile requires ReactLib: from git, or from $REACTLIB
# ══════════════════════════════════════════════════════════════════════

def _checkout(root: Path) -> Path:
    for p in REACTLIB_PACKAGES:
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
    monkeypatch.delenv(REACTLIB_ENV, raising=False)
    lakefile = generate_lakefile("P")
    for p in REACTLIB_PACKAGES:
        req = _require(lakefile, p)
        assert f'git = "{REACTLIB_URL}"' in req
        assert f'subDir = "{p}"' in req
        assert "path =" not in req


def test_reactlib_set_requires_the_checkout_by_path(monkeypatch, tmp_path):
    root = _checkout(tmp_path / "rl")
    monkeypatch.setenv(REACTLIB_ENV, str(root))
    lakefile = generate_lakefile("P")
    for p in REACTLIB_PACKAGES:
        assert _require(lakefile, p) == f'path = "{root.resolve() / p}"'
    assert REACTLIB_URL not in lakefile


def test_a_reactlib_that_is_not_a_checkout_is_refused_before_the_project(
        monkeypatch, tmp_path):
    """Otherwise the lakefile points at nothing, and lake says so only when
    it is asked to build, long after the project was written."""
    from reactor_lean.pipeline import main

    monkeypatch.setenv(REACTLIB_ENV, str(tmp_path))
    with pytest.raises(Refused, match="is not a ReactLib checkout"):
        reactlib_checkout()
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "reactor", str(FIXTURES / "counter.py"),
        "--safety", "(>= (X @intf.0) 0)", "-o", str(out), "-p", "P"])
    with pytest.raises(SystemExit, match="is not a ReactLib checkout"):
        main()
    assert not out.exists()


# ══════════════════════════════════════════════════════════════════════
#  The project builds against a Lake package cache, with $REACTOR_LAKE_CACHE
# ══════════════════════════════════════════════════════════════════════

def _git(name: str, rev: str, input_rev: str = "") -> dict:
    return {"type": "git", "name": name, "rev": rev,
            "inputRev": input_rev or rev, "inherited": False}


def _cache(root: Path, *extra: dict) -> Path:
    """A cache holding the fixed packages, Mathlib (as Cslib requires it),
    `extra`, and a path entry of the project it was made from."""
    (root / "packages").mkdir(parents=True)
    (cslib,) = FIXED_REQUIRES
    packages = [
        _git("cslib", "990e65a", cslib.rev),
        {**_git("mathlib", "5ed2965", "v4.34.0"), "inherited": True},
        {"type": "path", "name": "ReactLib", "dir": "../../elsewhere"},
        *extra,
    ]
    (root / "lake-manifest.json").write_text(json.dumps(
        {"version": "1.2.0", "name": "Made", "packagesDir": ".lake/packages",
         "packages": packages}))
    return root


def test_no_cache_is_none_and_a_directory_that_is_no_cache_is_refused(
        monkeypatch, tmp_path):
    monkeypatch.delenv(LAKE_CACHE_ENV, raising=False)
    assert lake_cache() is None
    monkeypatch.setenv(LAKE_CACHE_ENV, str(tmp_path))
    with pytest.raises(Refused, match="no packages or lake-manifest.json"):
        lake_cache()
    _cache(tmp_path / "c")
    monkeypatch.setenv(LAKE_CACHE_ENV, str(tmp_path / "c"))
    assert lake_cache() == (tmp_path / "c").resolve()


def test_the_manifest_holds_the_cached_packages_and_the_projects_paths(
        monkeypatch, tmp_path):
    """The cache's git packages, the transitive ones too; not the path
    package of the project the cache was made from, which is the
    project's own; and an entry for each path require this project has."""
    cache = _cache(tmp_path / "c")
    root = _checkout(tmp_path / "rl")
    monkeypatch.setenv(REACTLIB_ENV, str(root))
    m = cached_manifest(cache, project_requires(), name="P",
                        project=tmp_path / "P")
    assert m["name"] == "P" and m["version"] == "1.2.0"
    git = [e["name"] for e in m["packages"] if e["type"] == "git"]
    assert git == ["cslib", "mathlib"]
    dirs = {e["name"]: e["dir"] for e in m["packages"] if e["type"] == "path"}
    assert dirs == {p: str(root.resolve() / p) for p in REACTLIB_PACKAGES}


def test_a_git_require_the_cache_does_not_hold_is_refused(monkeypatch,
                                                          tmp_path):
    """Without `$REACTLIB`, ReactLib's packages are git requires, which a
    cache made against a checkout does not hold."""
    cache = _cache(tmp_path / "c")
    monkeypatch.delenv(REACTLIB_ENV, raising=False)
    with pytest.raises(Refused, match="does not hold ReactLib"):
        cached_manifest(cache, project_requires(), name="P", project=tmp_path)
    held = _cache(tmp_path / "held", *(_git(p, REACTLIB_REV)
                                       for p in REACTLIB_PACKAGES))
    m = cached_manifest(held, project_requires(), name="P", project=tmp_path)
    assert {e["name"] for e in m["packages"]} >= set(REACTLIB_PACKAGES)


def test_a_package_cached_at_another_revision_is_refused(tmp_path):
    """Lake would fetch the asked-for revision over the one every other
    project builds against."""
    cache = _cache(tmp_path / "c")
    newer = (Require("cslib", scope="leanprover", rev="v4.35.0"),)
    with pytest.raises(Refused, match="holds cslib at v4.34.0"):
        cached_manifest(cache, newer, name="P", project=tmp_path)


def test_the_lakefile_written_with_a_cache_brings_the_manifest_and_packages(
        monkeypatch, tmp_path):
    """`write_lakefile` is where a project's lakefile is written, first or
    again, so whatever writes one builds against the cache."""
    cache = _cache(tmp_path / "c")
    monkeypatch.setenv(REACTLIB_ENV, str(_checkout(tmp_path / "rl")))
    monkeypatch.setenv(LAKE_CACHE_ENV, str(cache))
    project = tmp_path / "P"
    project.mkdir()
    for _ in range(2):              # and again, over the first
        write_lakefile(project, "P", executable=False)
    link = project / ".lake" / "packages"
    assert link.is_symlink() and link.resolve() == (cache / "packages").resolve()
    manifest = json.loads((project / "lake-manifest.json").read_text())
    assert manifest["name"] == "P"


def test_a_project_with_packages_of_its_own_keeps_them(monkeypatch, tmp_path):
    """One an earlier run built without the cache: its packages are not
    the cache's to delete, and its manifest goes with them."""
    cache = _cache(tmp_path / "c")
    monkeypatch.setenv(REACTLIB_ENV, str(_checkout(tmp_path / "rl")))
    monkeypatch.setenv(LAKE_CACHE_ENV, str(cache))
    own = tmp_path / "P" / ".lake" / "packages" / "mathlib"
    own.mkdir(parents=True)
    (tmp_path / "P" / "lake-manifest.json").write_text("its own")
    write_lakefile(tmp_path / "P", "P", executable=False)
    assert own.is_dir()
    assert (tmp_path / "P" / "lake-manifest.json").read_text() == "its own"


def test_a_bad_cache_is_refused_before_the_project(monkeypatch, tmp_path):
    from reactor_lean.pipeline import main

    monkeypatch.setenv(LAKE_CACHE_ENV, str(tmp_path))
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "reactor", str(FIXTURES / "counter.py"),
        "--safety", "(>= (X @intf.0) 0)", "-o", str(out), "-p", "P"])
    with pytest.raises(SystemExit, match="is not a Lake package cache"):
        main()
    assert not out.exists()


# ══════════════════════════════════════════════════════════════════════
#  The reductions have to stay computable under the certificate's `simp`
# ══════════════════════════════════════════════════════════════════════

REDUCTIONS = ("matMin", "matMax", "argmax_1d", "argmax")


def _mat_source() -> str:
    checkout = reactlib_checkout()
    if checkout is None:
        pytest.skip(f"no ReactLib checkout: set ${REACTLIB_ENV}")
    return (checkout / "ReactLib" / "Relational" / "Mat.lean").read_text()


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
    from reactor_lean.codegen.context import LeanContext
    from reactor_lean.inputs.python import load_module

    fixture = FIXTURES / "counter.py"
    return LeanContext(load_module(fixture))


def test_the_certificate_is_one_text_for_each_kind_of_property():
    """What changes with the module and its certificate is in `Data.lean`,
    and in `Proofs.lean` once a prover has proved it, so either can be
    rewritten -- by `reactor` or by anything else -- without the
    certificate, or `Module.lean`. Unproved, `Proofs.lean` states the
    obligations of the property's kind and nothing else."""
    from reactor_lean.codegen.context import LeanContext
    from reactor_lean.core.certificate import Certificate, Claim
    from reactor_lean.inputs.python import load_module
    from reactor_lean.project.names import (
        CERTIFICATE, DATA, MODULE, PROOFS, cert_file,
    )
    from reactor_lean.pipeline.certificate import render
    from reactor_lean.proof.render import certificate_files

    counter = load_module(FIXTURES / "counter.py")
    twobit = load_module(FIXTURES / "twobit.py")
    variants = {kind: [] for kind in ("buchi", "safety")}
    for m in (counter, twobit):
        for kind in variants:
            variants[kind].append(certificate_files(
                LeanContext(m), render(m, Claim(kind=kind))))
    variants["buchi"].append(certificate_files(
        LeanContext(counter), render(
            counter, Claim(prp="(= s0 0)"),
            Certificate(inv="(>= s0 0)", ranking="(ite (= s0 0) 0 s0)"))))

    for kind, files in variants.items():
        certs = {f[cert_file(CERTIFICATE)] for f in files}
        assert len(certs) == 1, f"the {kind} certificate changes with the module"
        proofs = {f[cert_file(PROOFS)] for f in files}
        assert len(proofs) == 1, f"the unproved {kind} proofs change"
        data = {f[cert_file(DATA)] for f in files}
        assert len(data) > 1, "the test's certificates are all alike"
    for name in (CERTIFICATE, PROOFS):
        texts = [variants[kind][0][cert_file(name)] for kind in variants]
        assert texts[0] != texts[1]
    every = [f for files in variants.values() for f in files]
    assert len({f[cert_file(MODULE)] for f in every}) == 1


def test_each_obligation_is_stated_alike_in_both_files():
    """`Certificate.lean` takes each obligation's proof from `Proofs.lean`
    with `exact`, which checks the statements agree; they are written to,
    so that the check is a formality rather than a unification."""
    from reactor_lean.core.certificate import Claim
    from reactor_lean.project.names import CERTIFICATE, PROOFS, cert_file
    from reactor_lean.pipeline.certificate import render
    from reactor_lean.proof.render import certificate_files

    ctx = _counter_ctx()
    for kind, names in (("buchi", ("init_inv", "step_inv", "hrank")),
                        ("safety", ("init_inv", "step_inv", "inv_imp_P"))):
        files = certificate_files(ctx, render(ctx.module, Claim(kind=kind)))
        cert, proofs = files[cert_file(CERTIFICATE)], files[cert_file(PROOFS)]
        for name in names:
            stated = re.search(rf"^theorem {name} (.*?) := by\n  exact {name}_proof$",
                               cert, re.M | re.S)
            assert stated, f"{kind}: `{name}` is not taken from `{name}_proof`"
            again = re.search(rf"^theorem {name}_proof (.*?) := by$", proofs, re.M | re.S)
            assert again, f"{kind}: `Proofs.lean` does not state `{name}_proof`"
            assert stated.group(1) == again.group(1), name
