"""The module encoding: the module as a `Relational.Module` (`System/Module.lean`).

What it says of the module -- its variables and their types, each atom's
sets and functions, the module's own sets, the order the atoms run in -- read
off the text. That the proofs it carries go through is Lean's to say:
`lean/test_lean_module.py` elaborates it for every module of the corpus, and
the goldens pin it for a module of one atom and one of three.
"""

from __future__ import annotations

import os
import re

import pytest
import torch
from reactor import Int, Module, Term, Var, X, tensor
from reactor_lean.codegen.encodings import ModuleToLean4
from testdata import TESTS, fixture

# The snapshots beside the pipeline's (`test_pipeline_golden.py`), rewritten
# the same way: `REACTOR_UPDATE_GOLDEN=1`.
GOLDEN = TESTS / "golden" / "pipeline"


def _check(name: str, text: str) -> None:
    path = GOLDEN / name
    if os.environ.get("REACTOR_UPDATE_GOLDEN") == "1":
        path.write_text(text)
        return
    assert path.exists(), f"no snapshot {path}; run with REACTOR_UPDATE_GOLDEN=1"
    assert text == path.read_text(), f"{name} differs from its snapshot"


def _lean(module) -> str:
    return ModuleToLean4(module).to_lean_module()


def _def(lean: str, name: str) -> str:
    """The right-hand side of the one-line `def name : Finset Name := ...`."""
    m = re.search(rf"^def {name} : Finset Name := (.*)$", lean, re.MULTILINE)
    assert m, name
    return m.group(1)


def _cases(lean: str, head: str) -> dict[str, str]:
    """The `| .n => rhs` cases of the definition that `head` starts."""
    block = lean.split(head, 1)[1].split("\n\n", 1)[0]
    return dict(re.findall(r"^  \| \.(\w+) => (.*)$", block, re.MULTILINE))


@pytest.mark.parametrize("name", ["counter", "composed"])
def test_module_encoding_is_unchanged(name):
    _check(f"encoding-module-{name}.golden", _lean(fixture(name)))


def test_the_variables_are_named_by_place_and_typed():
    """`(z, x, wrap, y)` and the input `e`: `s0..s3`, `e0`."""
    lean = _lean(fixture("composed"))
    assert "  | s0 | s1 | s2 | s3 | e0\n" in lean
    assert _cases(lean, "abbrev Dom : Name → Type") == {
        "s0": "(Matrix (Fin 1) (Fin 1) Int)",
        "s1": "(Matrix (Fin 1) (Fin 1) Int)",
        "s2": "(Matrix (Fin 1) (Fin 1) Bool)",
        "s3": "(Matrix (Fin 1) (Fin 1) Int)",
        "e0": "(Matrix (Fin 1) (Fin 1) Int)",
    }


def test_each_atom_awaits_reads_and_writes_what_it_does():
    """The counter (`x`, `wrap`) reads `x`; the accumulator (`y`) reads `y`
    and awaits `x`; the difference (`z`) awaits `y` and the input."""
    lean = _lean(fixture("composed"))
    sets = {(i, s): _def(lean, f"atom{i}_{s}")
            for i in range(3) for s in ("await", "read", "write")}
    assert sets == {
        (0, "await"): "∅",
        (0, "read"): "{.s1}",
        (0, "write"): "{.s1, .s2}",
        (1, "await"): "{.s1}",
        (1, "read"): "{.s3}",
        (1, "write"): "{.s3}",
        (2, "await"): "{.s3, .e0}",
        (2, "read"): "∅",
        (2, "write"): "{.s0}",
    }
    # The atom's functions are the functional encoding's; what it neither
    # reads nor awaits is `default`.
    assert ("(fun r a => atom1_out (_root_.Atom1.update (default, default, "
            "default, (r ⟨Name.s3, by decide⟩)) default default "
            "(a ⟨Name.s1, by decide⟩)))") in lean
    # Its rounds, said in those functions.
    assert ("    atom2.Update s s' ↔ (s' .s0) = _root_.Atom2.update (ctrlOf s) "
            "(extlOf s) (extlOf s') (s' .s3) :=") in lean
    assert "  atoms := [atom0, atom1, atom2]\n" in lean


def test_the_atoms_run_in_the_order_of_rank():
    """An input is of place 0, and an atom's variables of one more than its
    place in the module's order."""
    lean = _lean(fixture("composed"))
    assert _cases(lean, "def rank : Name → Nat") == {
        "s0": "3", "s1": "1", "s2": "1", "s3": "2", "e0": "0"}


def test_a_module_of_one_atom_is_its_init_and_update():
    """The atom writes every state variable, in `ctrl` order, and its
    functions are the module's `init` and `update`."""
    lean = _lean(fixture("svcomp_twovars"))
    assert "  atoms := [atom0]\n" in lean
    assert "_root_.init " in lean and "_root_.update " in lean
    assert "Atom0." not in lean
    assert _def(lean, "atom0_write") == "{.s0, .s1}"
    assert ("    atom0.Init s ↔ (s .s0, s .s1) = _root_.init (extlOf s) :="
            in lean)
    # one atom: the round is the atom's, with nothing to compose
    assert "rintro" not in lean.split("theorem initRound_iff", 1)[1].split(
        "theorem updateRound_iff", 1)[0]


def test_a_hidden_variable_is_private():
    """`hide` makes a state variable private: it is in `priv`, not `intf`,
    and still written by its atom."""
    S = [1, 1]
    x, y = Var(Int(S)), Var(Int(S))
    zero = Term(tensor.int.Const(torch.tensor([[0]])), [X(x)])
    m = Module.compose(
        Module.sequential([y, x], [Term(tensor.Skip(Int(S)), [X(y)], [X(x)])],
                          [Term(tensor.int.Add(S), [X(y)], [y, X(x)])]),
        Module.sequential([x], [zero], [Term(tensor.Skip(Int(S)), [X(x)], [x])]),
        hide=[x])
    lean = _lean(m)
    names = {v.id: f"s{i}" for i, v in enumerate(m.ctrl)}
    hidden, shown = names[x.id], names[y.id]
    assert f"  intf := {{.{shown}}}\n" in lean
    assert f"  priv := {{.{hidden}}}\n" in lean
    assert "  extl := ∅\n" in lean
