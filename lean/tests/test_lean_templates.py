"""The templates, and the Lean library the files they render import.

`templates/` holds the Jinja templates of the generated project's files and
no Lean sources: the library those files import -- `Relational.*`, and
`LeanAI` -- is ReactLib's, and the project's lakefile requires it. A
`.lean` file left under `templates/` would be read by nothing, so editing it
would silently do nothing, which is what the first two tests exist to
prevent. (`templates/` once carried two byte-identical sets of the Core
files, and edits to the unread one did nothing for months.)
"""

import re
from pathlib import Path

import pytest

from reactor_lean.core.errors import Refused
from reactor_lean.project.lakefile import (
    REACTLIB_ENV,
    REACTLIB_PACKAGES,
    REACTLIB_URL,
    generate_lakefile,
    reactlib_checkout,
    with_reactlib,
)
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


def test_a_copied_manifest_gets_the_checkout_as_its_own(tmp_path):
    """`lake build` reads a path dependency's `dir` from the manifest, not
    the lakefile, so a manifest copied from elsewhere must be repointed."""
    root = _checkout(tmp_path / "rl")
    stale = {"name": "ReactLib", "type": "path", "dir": "../../elsewhere"}
    kept = {"name": "cslib", "type": "git"}
    out = with_reactlib({"version": "1.1.0", "packages": [kept, stale]}, root)
    assert out["version"] == "1.1.0"
    assert out["packages"][0] == kept
    dirs = {e["name"]: e["dir"] for e in out["packages"] if e["type"] == "path"}
    assert dirs == {p: str(root / p) for p in REACTLIB_PACKAGES}


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
