"""`generate_main_lean` must emit a Main.lean that matches what it calls.

The executable project is generated from two places that have
to agree: `atom_to_lean_functional` emits `init`/`update`, and
`generate_main_lean` emits the IO loop that drives them. Nothing checked
that agreement, so the loop drifted from the signatures it calls.
"""

import torch
import pytest

from reactor import Term, Module, Int, Bool, Word, Real, Var, X, tensor
from reactor_lean.errors import Refused
from reactor_lean.project.main_lean import check_runnable, generate_main_lean
from reactor_lean.codegen.context import LeanContext
from reactor_lean.codegen.encodings import ModuleToLean4


def _module_with_extl():
    """State plus a genuine external input, so extl_l/extl_n are non-trivial."""
    s = Var(Int([1, 1]))
    e = Var(Int([1, 1]))
    init = [Term(tensor.int.Linear(torch.tensor([[1]]), torch.tensor([[0]])), [X(s)], [X(e)])]
    update = [Term(tensor.int.Add([1, 1]), [X(s)], [s, X(e)])]
    return Module.sequential([s, e], init, update)


def _module_without_extl():
    s = Var(Int([1, 1]))
    return Module.sequential(
        [s],
        [Term(tensor.int.Const(torch.tensor([[0]])), [X(s)])],
        [Term(tensor.Skip(Int([1, 1])), [X(s)], [s])],
    )


def _two_ctrl_module():
    """Two ctrl components, to expose ordering in the show/parse helpers."""
    a = Var(Int([1, 1]))
    b = Var(Int([1, 1]))
    init = [
        Term(tensor.int.Const(torch.tensor([[1]])), [X(a)]),
        Term(tensor.int.Const(torch.tensor([[2]])), [X(b)]),
    ]
    update = [Term(tensor.Skip(Int([1, 1])), [X(a)], [a]), Term(tensor.Skip(Int([1, 1])), [X(b)], [b])]
    return Module.sequential([a, b], init, update)


@pytest.mark.parametrize(
    "make", [_module_with_extl, _module_without_extl, _two_ctrl_module]
)
def test_main_applies_update_curried(make):
    """`update` takes (ctrl) (extl_l) (extl_n) — three groups, not one tuple."""
    module = make()
    functional = ModuleToLean4(LeanContext(module)).to_lean_functional()
    assert "def update (ctrl:" in functional, "signature shape changed; update this test"

    main = generate_main_lean(module)
    assert "update state extlPrev extl" in main, (
        "Main.lean must apply update's three parameter groups separately"
    )
    assert "update (state" not in main, "update is called with a merged tuple"


@pytest.mark.parametrize(
    "make", [_module_with_extl, _module_without_extl, _two_ctrl_module]
)
def test_main_threads_the_previous_external_input(make):
    """`extl_l` is the previous step's input, so the loop has to carry it.

    It previously had no source at all in `main`.
    """
    main = generate_main_lean(make())
    assert "let mut extlPrev := extl0" in main, "extl_l is never initialised"
    assert "extlPrev := extl" in main, "extl_l is never advanced"


# ──────────────────────────────────────────────────────────────
# Component order: the literals must match `product_type`
# ──────────────────────────────────────────────────────────────


def _heterogeneous_module():
    """Bool then Int ctrl, so a swapped order is a type error, not a silent one."""
    b = Var(Bool([1, 1]))
    n = Var(Int([1, 1]))
    init = [
        Term(tensor.bool.Const(torch.tensor([[True]])), [X(b)]),
        Term(tensor.int.Const(torch.tensor([[7]])), [X(n)]),
    ]
    update = [Term(tensor.Skip(Bool([1, 1])), [X(b)], [b]), Term(tensor.Skip(Int([1, 1])), [X(n)], [n])]
    return Module.sequential([b, n], init, update)


def test_show_ctrl_destructures_in_declaration_order():
    """`v{i}` must bind ctrl[i]: the formatter per slot is chosen from its type.

    `product_type` and `product_accessor` both order components as declared, but
    the pattern was built reversed, so every slot got the wrong formatter.
    """
    main = generate_main_lean(_heterogeneous_module())
    assert "let (v0, v1) := v" in main
    assert "let (v1, v0) := v" not in main, "destructuring is reversed"
    # and the ctrl type ascription is in the same order
    assert "showCtrl (v : (Matrix (Fin 1) (Fin 1) Bool) × (Matrix (Fin 1) (Fin 1) Int))" in main


def test_parse_extl_builds_the_tuple_in_declaration_order():
    """Same for the parsed input tuple, against `parseExtl`'s return type."""
    e0 = Var(Bool([1, 1]))
    e1 = Var(Int([1, 1]))
    s = Var(Int([1, 1]))
    init = [Term(tensor.int.Const(torch.tensor([[0]])), [X(s)])]
    update = [Term(tensor.int.Ite([1, 1]), [X(s)], [X(e0), X(e1), s])]
    module = Module.sequential([s, e0, e1], init, update)

    main = generate_main_lean(module)
    assert "pure (e0, e1)" in main
    assert "pure (e1, e0)" not in main, "parsed tuple is reversed"


def test_single_component_needs_no_tuple():
    """One wire: the value itself, matching `product_accessor`'s total==1 case."""
    main = generate_main_lean(_module_without_extl())
    assert "let v0 := v" in main
    assert "let (v0) := v" not in main


# ──────────────────────────────────────────────────────────────
# Element types: IO must match the wire's own sort
# ──────────────────────────────────────────────────────────────


def test_main_imports_a_module_that_exists():
    """`init`/`update` land in System/System.lean, under the `System` lib root.

    The import was built as `{project_name}.{module_name}`, which named no
    file the generator writes.
    """
    main = generate_main_lean(_module_without_extl())
    assert main.splitlines()[0] == "import System"
    assert "import Rea.Rea" not in main


def test_bool_ctrl_is_shown_as_bool():
    """`dtype_to_lean_type` always returns a `Matrix ...`, so the old string
    comparison against "Bool" never matched and every element became Int."""
    main = generate_main_lean(_heterogeneous_module())
    assert "showMatrix 1 1 showBool v0" in main, "Bool ctrl is not shown as Bool"
    assert "showMatrix 1 1 toString v1" in main, "Int ctrl is not shown as Int"


def test_bitvec_extl_is_parsed_as_bitvec():
    """A BitVec input wire must build a BitVec array, not an Int one."""
    b = Var(Word(8, [1, 1]))
    s = Var(Word(8, [1, 1]))
    init = [Term(tensor.word.Const(torch.tensor([[0]]), 8), [X(s)])]
    update = [Term(tensor.Skip(Word(8, [1, 1])), [X(s)], [X(b)])]
    module = Module.sequential([s, b], init, update)

    main = generate_main_lean(module)
    assert "Array (BitVec 8)" in main, "BitVec input parsed into an Int array"
    assert "BitVec.ofInt 8 v" in main, "BitVec input is not converted"
    assert "Matrix (Fin 1) (Fin 1) (BitVec 8)" in main
    assert "Array Int" not in main


def test_real_io_is_refused_with_a_reason():
    """Lean's `Real` is noncomputable: no parsing, no printing, and no
    `init` or `update` to run, so say so -- before `Main.lean` is
    generated, which `cli.main` asks first."""
    r = Var(Real([1, 1]))
    module = Module.sequential(
        [r],
        [Term(tensor.real.Const(torch.tensor([[0.0]])), [X(r)])],
        [Term(tensor.Skip(Real([1, 1])), [X(r)], [r])],
    )
    with pytest.raises(Refused, match="noncomputable"):
        check_runnable(LeanContext(module))
    check_runnable(LeanContext(_heterogeneous_module()))


def test_show_mat_is_generic_over_its_element():
    """The shower takes the element renderer, so it is not pinned to Int."""
    main = generate_main_lean(_heterogeneous_module())
    assert "def showMatrix {t : Type} (m n : Nat) (f : t → String)" in main
    assert "(mat : Fin m → Fin n → Int)" not in main
