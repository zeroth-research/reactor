"""`reactor_lean.project.names` is what the project is written under: every
module the generated Lean imports from the project is a file the project
has."""

import re

from reactor_cli import reactor
from reactor_lean.cli import lake_module_prefix
from reactor_lean.codegen.context import LeanContext
from reactor_lean.parse import load
from reactor_lean.project.names import CERT_LIB, SYSTEM_LAYOUT
from reactor_lean.project.scaffold import create_project
from reactor_lean.proof.render import certificate_files
from testdata import FIXTURES


def _own_imports(text: str) -> list[str]:
    mods = re.findall(r"^import (\S+)", text, re.M)
    return [m for m in mods
            if m.startswith(SYSTEM_LAYOUT.prefix) or m.startswith(f"{CERT_LIB}.")]


def test_every_project_module_imported_is_a_file_the_project_has(tmp_path):
    ctx = LeanContext(load(FIXTURES / "counter.py").module)
    project = create_project(tmp_path, ctx, "Rea",
                             certificate=certificate_files(ctx))
    texts = {
        stem: (project / CERT_LIB / f"{stem}.lean").read_text()
        for stem in ("Module", "Proofs", "Certificate")
    }
    texts["root"] = (project / f"{CERT_LIB}.lean").read_text()
    for who, text in texts.items():
        mods = _own_imports(text)
        assert mods or who == "root", f"{who} imports nothing of the project"
        for mod in mods:
            path = project / (mod.replace(".", "/") + ".lean")
            assert path.exists(), f"{who} imports {mod}, which is not written"


def test_the_lakefile_builds_the_certificate_library(tmp_path):
    ctx = LeanContext(load(FIXTURES / "counter.py").module)
    project = create_project(tmp_path, ctx, "Rea",
                             certificate=certificate_files(ctx))
    lakefile = (project / "lakefile.toml").read_text()
    assert f'name = "{CERT_LIB}"' in lakefile
    assert re.search(rf'defaultTargets = \[[^\]]*"{CERT_LIB}"', lakefile)


def test_a_cert_file_in_a_lake_project_imports_by_its_module_name(tmp_path):
    """`--cert-file` writes its sibling encodings next to it, and they
    import the certificate and each other by the module names Lake gives
    them: under a lakefile two directories up, `Certs.Sub.C`."""
    (tmp_path / "lakefile.toml").write_text("")
    cert = tmp_path / "Certs" / "Sub" / "C.lean"
    r = reactor(str(FIXTURES / "counter.py"), "--cert-file", str(cert))
    assert r.returncode == 0, r.stderr
    siblings = sorted(cert.parent.glob("C?*.lean"))
    assert siblings
    for path in siblings:
        for mod in re.findall(r"^import (\S+)", path.read_text(), re.M):
            if not mod.startswith("Relational."):
                assert mod.startswith("Certs.Sub.C"), (path.name, mod)
                assert (tmp_path / (mod.replace(".", "/") + ".lean")).exists()


def test_the_module_prefix_is_the_path_from_the_lakefile(tmp_path):
    assert lake_module_prefix(tmp_path) == ""
    (tmp_path / "lakefile.lean").write_text("")
    assert lake_module_prefix(tmp_path) == ""
    assert lake_module_prefix(tmp_path / "A" / "B") == "A.B."
