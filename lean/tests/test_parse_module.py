"""The module file: `reactor_lean.parse.module`."""

import pytest

from reactor import Module
from reactor_lean import parse
from testdata import FIXTURES


def test_a_python_file_loads():
    path = FIXTURES / "counter.py"
    problem = parse.load(path)
    assert isinstance(problem.module, Module)
    assert problem.source == path.read_text()


def test_a_missing_file_is_named():
    with pytest.raises(parse.InputError, match="nope.py"):
        parse.load(FIXTURES / "nope.py")


def test_a_file_that_is_no_python_is_refused(tmp_path):
    path = tmp_path / "m.txt"
    path.write_text("def module(): ...\n")
    with pytest.raises(parse.InputError, match="no Python file"):
        parse.load(path)


def test_a_callable_the_file_does_not_define_is_refused():
    with pytest.raises(parse.InputError, match="must define `stray`"):
        parse.load(FIXTURES / "counter.py", "stray")


def test_the_project_keeps_the_input(tmp_path):
    """`dbg/` holds the input the module was read from."""
    from reactor_lean.codegen.context import LeanContext
    from reactor_lean.project.scaffold import create_project
    from reactor_lean.proof.render import certificate_files

    problem = parse.load(FIXTURES / "counter.py")
    ctx = LeanContext(problem.module)
    project = create_project(tmp_path, ctx, "Rea", source=problem.source,
                             certificate=certificate_files(ctx))
    assert (project / "dbg" / "system.py").read_text() == problem.source
