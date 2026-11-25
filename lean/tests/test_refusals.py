"""What is wrong with the input is one `error:` line, not a traceback."""

import pytest

from reactor_cli import reactor
from testdata import FIXTURES

COUNTER = str(FIXTURES / "counter.py")


def _refused(r) -> str:
    assert r.returncode != 0
    assert "Traceback" not in r.stderr, r.stderr
    assert r.stderr.startswith("error: "), r.stderr
    return r.stderr


def test_a_module_of_two_atoms(tmp_path):
    path = tmp_path / "two.py"
    path.write_text(
        "from reactor import Module\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('c', {COUNTER!r})\n"
        "c = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(c)\n"
        "def module():\n"
        "    return Module.compose(c.module(), c.module())\n"
    )
    assert "2 atoms" in _refused(reactor(str(path), "-o", str(tmp_path)))


def test_an_op_with_no_lean_form_writes_nothing(tmp_path):
    """Even when nothing reads what the op writes: the refusal comes before
    the project directory does."""
    path = tmp_path / "havoc.py"
    path.write_text(
        "import torch\n"
        "from reactor import Int, Module, Term, Var, Wire, X, tensor\n"
        "from reactor.tensor import Havoc, Skip\n"
        "def module():\n"
        "    x, y = Var(Int([1, 1])), Wire(Int([1, 1]))\n"
        "    init = [Term(tensor.int.Const(torch.tensor([[0]])), [X(x)])]\n"
        "    update = [Term(Havoc(Int([1, 1])), [y]),\n"
        "              Term(Skip(Int([1, 1])), [X(x)], [x])]\n"
        "    return Module.sequential([x], init, update)\n"
    )
    out = tmp_path / "out"
    assert "`Havoc`" in _refused(reactor(str(path), "-o", str(out)))
    assert not out.exists()


_REAL_STATE = """\
import torch
from reactor import Module, Real, Term, Var, X, tensor
def module():
    s = Var(Real([1, 1]))
    init = [Term(tensor.real.Const(torch.tensor([[0.5]])), [X(s)])]
    return Module.sequential([s], init, [Term(tensor.Skip(Real([1, 1])), [X(s)], [s])])
"""

_REAL_BETWEEN_INTS = """\
import torch
from reactor import Int, Module, Real, Term, Var, Wire, X, tensor
def module():
    s, r = Var(Int([1, 1])), Wire(Real([1, 1]))
    init = [Term(tensor.int.Const(torch.tensor([[1]])), [X(s)])]
    update = [Term(tensor.cast.ToReal([1, 1]), [r], [s]),
              Term(tensor.cast.ToInt([1, 1]), [X(s)], [r])]
    return Module.sequential([s], init, update)
"""


@pytest.mark.parametrize("src", [_REAL_STATE, _REAL_BETWEEN_INTS],
                         ids=["state", "between_ints"])
def test_an_executable_that_computes_with_reals_writes_nothing(tmp_path, src):
    """`init` and `update` are `noncomputable` whenever a `Real` is in the
    module, read or printed or not: the refusal used to come after the
    project (Real state), or not at all (a Real between two Ints), and
    `lake build main` failed."""
    path = tmp_path / "reals.py"
    path.write_text(src)
    out = tmp_path / "out"
    assert "--executable: the module computes with `Real`" in _refused(
        reactor(str(path), "-x", "-o", str(out)))
    assert not out.exists()


def test_a_cert_file_sibling_that_cannot_be_written(tmp_path):
    (tmp_path / "CertRel.lean").mkdir()   # where the Rel encoding goes
    err = _refused(reactor(COUNTER, "--cert-file", str(tmp_path / "Cert.lean")))
    assert err.startswith("error: --cert-file: ")


def test_a_let_in_a_claim(tmp_path):
    err = _refused(reactor(COUNTER, "--buchi", "(= (X @intf.0) 0)",
                           "--invariant", "(let ((y (X @intf.0))) (>= y 0))",
                           "-o", str(tmp_path)))
    assert "--invariant: `let` is not supported in a claim" in err
