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


def test_a_file_that_looks_itself_up_loads(tmp_path):
    """`dataclasses` looks a class's module up in `sys.modules` when its
    annotations are strings: the file is there while it runs, and gone
    after."""
    import sys

    path = tmp_path / "m.py"
    path.write_text("from __future__ import annotations\n"
                    "from dataclasses import dataclass\n"
                    + (FIXTURES / "counter.py").read_text()
                    + "\n\n@dataclass\nclass Params:\n    bound: int = 10\n")
    assert isinstance(parse.load(path).module, Module)
    assert all(getattr(m, "__file__", None) != str(path)
               for m in list(sys.modules.values()))


@pytest.mark.parametrize("src, why", [
    ("def module():\n    pass\n", "returned `NoneType`, not a `Module`"),
    ((FIXTURES / "counter.py").read_text() + "\nmodule = module()\n",
     "is a `Module`, not a function returning the module"),
])
def test_a_callable_that_makes_no_module_is_refused(tmp_path, src, why):
    """What is wrong with the file, not an error of its own code: a
    `module()` that forgets its `return`, a `module` that is no callable."""
    path = tmp_path / "m.py"
    path.write_text(src)
    with pytest.raises(parse.InputError, match=why):
        parse.load(path)


def test_a_named_environment_is_refused(tmp_path):
    """Only a `Module` has variables a name can be given to; an
    environment has them once `reactor.Env` made it one."""
    path = tmp_path / "m.py"
    path.write_text("from reactor_lean.parse import Named\n"
                    "def module():\n"
                    "    return Named(object(), {})\n")
    with pytest.raises(parse.InputError, match="takes a `Module`, not `object`"):
        parse.load(path)


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
