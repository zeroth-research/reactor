"""`--encodings`, which of the module's encodings a run writes."""

from __future__ import annotations

import re

import pytest

from reactor_lean.codegen.context import LeanContext
from reactor_lean.core.encodings import ENCODING_NAMES, FUNCTIONAL
from reactor_lean.core.errors import Refused
from reactor_lean.project.scaffold import ENCODINGS, create_project, selected_encodings
from reactor_lean.proof.render import certificate_files
from reactor_cli import reactor
from testdata import FIXTURES, fixture

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
    assert _names(selected_encodings()) == ENCODING_NAMES
    assert _names(selected_encodings(())) == (FUNCTIONAL,)
    assert _names(selected_encodings(("circuit",))) == (FUNCTIONAL, "circuit")
    assert _names(selected_encodings(("scalar-relational",))) == (
        FUNCTIONAL, "scalar", "scalar-relational")


def test_an_encoding_there_is_none_of_is_refused():
    with pytest.raises(Refused, match="no encoding 'tensor'"):
        selected_encodings(("tensor",))


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


def test_a_cert_file_has_the_encodings_selected_beside_it(tmp_path):
    r = reactor(COUNTER, "--encodings", "functional",
                "--cert-file", str(tmp_path / "Cert.lean"))
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Cert.lean"]

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
