"""The input adapters: `reactor_lean.inputs`."""

from pathlib import Path

import pytest

from reactor import Module
from reactor_lean import inputs

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_a_python_file_loads_through_its_adapter():
    path = FIXTURES / "counter.py"
    problem = inputs.load(path)
    assert isinstance(problem.module, Module)
    assert problem.source == inputs.SourceText("python", path, path.read_text(), ".py")


def test_a_missing_file_is_named():
    with pytest.raises(inputs.InputError, match="nope.py"):
        inputs.load(FIXTURES / "nope.py")


def test_an_unknown_format_is_refused_with_the_known_ones():
    with pytest.raises(KeyError, match="have: python"):
        inputs.adapter_by_name("cobol")


def test_the_project_keeps_the_input_under_its_formats_suffix(tmp_path):
    """`dbg/` holds the input the module was read from, named for its
    format rather than assumed to be Python."""
    from reactor_lean.codegen.context import LeanContext
    from reactor_lean.project.scaffold import create_project
    from reactor_lean.proof.render import certificate_files

    module = inputs.load(FIXTURES / "counter.py").module
    source = inputs.SourceText("text", tmp_path / "m.in", "a module\n", ".txt")
    project = create_project(tmp_path, module, "Rea", source=source,
                             certificate=certificate_files(LeanContext(module)))
    assert (project / "dbg" / "system.txt").read_text() == "a module\n"
    assert not (project / "dbg" / "system.py").exists()


def test_an_input_option_its_format_requires_has_to_be_given():
    """As a route's required option is: a format that cannot read its input
    without an option says so at parse time, not halfway through loading."""
    import argparse
    from dataclasses import replace

    from reactor_lean.cli.checks import check_input_options
    from reactor_lean.core.options import Opt

    fmt = replace(inputs.adapter_by_name("python"),
                  options=(Opt(("--needed",), {}, required=True),))
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        check_input_options(parser, argparse.Namespace(needed=None), fmt)
    check_input_options(parser, argparse.Namespace(needed="x"), fmt)
