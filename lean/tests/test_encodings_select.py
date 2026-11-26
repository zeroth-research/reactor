"""`--encodings`, which of the module's encodings a run writes; and a module
of several atoms, which the functional encoding alone composes.

`fixtures/composed.py` is the module of several atoms: three, its state
`(z, x, wrap, y)` in another order than its atoms run in. Its rounds are
held to the interpreter with every other fixture's, by
`test_encodings_agree.py` (SMT) and `lean/test_lean_agree.py` (Lean).
"""

from __future__ import annotations

import re

import pytest

from reactor_lean.codegen.context import LeanContext
from reactor_lean.codegen.encodings import ModuleToLean4
from reactor_lean.core.encodings import ENCODING_NAMES, FUNCTIONAL
from reactor_lean.core.errors import Refused
from reactor_lean.parse import PARSERS
from reactor_lean.project.scaffold import ENCODINGS, create_project, selected_encodings
from reactor_lean.proof.render import certificate_files
from reactor_cli import reactor
from testdata import FIXTURES, fixture

COMPOSED = str(FIXTURES / "composed.py")
COUNTER = str(FIXTURES / "counter.py")


def _names(rows) -> tuple[str, ...]:
    return tuple(e.name for e in rows)


def _system_files(project) -> list[str]:
    return sorted(p.name for p in (project / "System").glob("*.lean"))


# ── the selection ───────────────────────────────────────────────────────────


def test_the_names_are_the_rows():
    """The command line names the encodings without importing the table;
    the two lists are one."""
    assert _names(ENCODINGS) == ENCODING_NAMES
    assert ENCODINGS[0].name == FUNCTIONAL and ENCODINGS[0].suffix == ""


def test_a_selection_brings_the_functional_encoding_and_what_it_imports():
    ctx = LeanContext(fixture("counter"))
    assert _names(selected_encodings(ctx)) == ENCODING_NAMES
    assert _names(selected_encodings(ctx, ())) == (FUNCTIONAL,)
    assert _names(selected_encodings(ctx, ("relational",))) == (FUNCTIONAL, "relational")
    assert _names(selected_encodings(ctx, ("scalar-relational",))) == (
        FUNCTIONAL, "scalar", "scalar-relational")


def test_an_encoding_there_is_none_of_is_refused():
    with pytest.raises(Refused, match="no encoding 'tensor'"):
        selected_encodings(LeanContext(fixture("counter")), ("tensor",))


# ── several atoms ───────────────────────────────────────────────────────────


def test_a_module_of_several_atoms_has_the_functional_encoding_alone():
    ctx = LeanContext(fixture("composed"))
    assert len(ctx.atoms) == 3
    assert _names(selected_encodings(ctx, (FUNCTIONAL,))) == (FUNCTIONAL,)
    with pytest.raises(Refused) as refused:
        selected_encodings(ctx)
    said = str(refused.value)
    assert said.startswith("--encodings: the module has 3 atoms")
    assert "no relational, scalar, scalar-relational encoding" in said
    assert "`--encodings functional`" in said
    # and every encoding written over the one atom refuses it, too
    with pytest.raises(Refused, match="3 atoms"):
        ModuleToLean4(ctx).to_lean_scalar()


def test_each_atom_is_a_function_and_the_module_composes_them():
    lean = ModuleToLean4(fixture("composed")).to_lean_functional()
    assert re.findall(r"^namespace (\S+)", lean, re.M) == ["Atom0", "Atom1", "Atom2"]
    # Atom 1 (`y`) awaits atom 0's first variable (`x`), and atom 2 (`z`)
    # atom 1's, in both blocks; and the state is read off them in the
    # module's order, `(z, x, wrap, y)`, not the atoms'.
    for block, args in (("init", "extl_n"), ("update", "ctrl extl_l extl_n")):
        assert (f"  let a0 := Atom0.{block} {args}\n"
                f"  let a1 := Atom1.{block} {args} a0.1\n"
                f"  let a2 := Atom2.{block} {args} a1\n"
                f"  (a2, a0.1, a0.2, a1)\n") in lean


# ── a project, and `reactor` ────────────────────────────────────────────────


def test_a_project_carries_the_encodings_selected_and_no_other(tmp_path):
    module = fixture("counter")
    files = certificate_files(LeanContext(module))
    project = create_project(tmp_path, module, "Rea", certificate=files)
    assert len(_system_files(project)) == len(ENCODINGS)

    # Again into the same `-o`, with fewer: the others' files go.
    create_project(tmp_path, module, "Rea", certificate=files,
                   encodings=("scalar",))
    assert _system_files(project) == ["Scalar.lean", "System.lean"]
    root = (project / "System.lean").read_text()
    assert re.findall(r"^import (\S+)", root, re.M) == ["System.System",
                                                        "System.Scalar"]


def test_reactor_writes_the_encodings_selected(tmp_path):
    r = reactor(COUNTER, "--encodings", "functional", "-o", str(tmp_path),
                "-p", "C")
    assert r.returncode == 0, r.stderr
    assert _system_files(tmp_path / "C") == ["System.lean"]


def test_reactor_refuses_a_module_of_several_atoms_before_writing(tmp_path):
    r = reactor(COMPOSED, "-o", str(tmp_path), "-p", "C")
    assert r.returncode == 1
    assert "`--encodings functional`" in r.stderr
    assert not (tmp_path / "C").exists()


@pytest.mark.parametrize("parser", PARSERS)
def test_reactor_writes_a_module_of_several_atoms_in_the_functional_encoding(
        tmp_path, parser):
    """Its claim read by either parser: the SMT encoding composes the atoms
    as well, by running their blocks one after another."""
    inv = "(and (>= (X x) 0) (<= (X x) 3) (>= (X y) 0))"
    r = reactor(COMPOSED, "--encodings", "functional", "--parser", parser,
                "--safety", "(<= (X x) 3)", "--invariant", inv,
                "-o", str(tmp_path), "-p", "C")
    assert r.returncode == 0, r.stderr
    project = tmp_path / "C"
    assert _system_files(project) == ["System.lean"]
    assert "namespace Atom2" in (project / "System" / "System.lean").read_text()
    data = (project / "Certificate" / "Data.lean").read_text()
    assert re.search(r"^def inv : .* := fun s => ", data, re.M)


def test_a_cert_file_has_the_encodings_selected_beside_it(tmp_path):
    r = reactor(COMPOSED, "--encodings", "functional",
                "--cert-file", str(tmp_path / "Cert.lean"))
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Cert.lean"]
    assert "namespace Atom0" in (tmp_path / "Cert.lean").read_text()

    r = reactor(COUNTER, "--encodings", "scalar",
                "--cert-file", str(tmp_path / "One" / "Cert.lean"))
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (tmp_path / "One").iterdir()) == [
        "Cert.lean", "CertScalar.lean"]


@pytest.mark.parametrize("value,said", [
    ("tensor", "no encoding 'tensor'"),
    ("scalar,,tensor", "no encoding 'tensor'"),
    ("", "names no encoding"),
])
def test_reactor_refuses_an_encoding_there_is_none_of(tmp_path, value, said):
    r = reactor(COUNTER, "--encodings", value, "-o", str(tmp_path))
    assert r.returncode == 2
    assert said in r.stderr
