"""Unit tests for the reflected `Linear` (pre-contraction) codegen.

These are fast, Lean-free checks that the Python side emits the correct list
literals for an Int/Real `Linear` op: right orientation (row-major
`A[i][l]`, not transposed), signs, bias (of the output's shape), non-square
shapes, and element formatting (Int/Real). They complement the lake-verified
`Relational.Mat.matVecAffine_eq`, which checks the *Lean* side reduces to the intended
affine map — here we guard the *generator* that feeds it.
"""

import pytest
import torch

from reactor import Word, Bool, Wire, Term, Int, Real, tensor
from reactor_lean.codegen.literals import linear_list_literals
from reactor_lean.codegen.terms import _linear_expr


def _int_linear(A, B, out_rows):
    """A `Linear(A, B)` term over Int with out=[out_rows,1], in=[A.cols,1]."""
    in_cols = A.shape[1]
    out_w = Wire(Int([out_rows, 1]))
    x_w = Wire(Int([in_cols, 1]))
    return Term(tensor.int.Linear(A, B), [out_w], [x_w]), x_w


def test_identity_with_bias():
    A = torch.eye(3, dtype=torch.int64)
    B = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    term, _ = _int_linear(A, B, 3)
    out_m, a_lit, b_lit, ty = linear_list_literals(term)
    assert out_m == 3
    assert a_lit == "([[(1 : Int), (0 : Int), (0 : Int)], [(0 : Int), (1 : Int), (0 : Int)], [(0 : Int), (0 : Int), (1 : Int)]] : List (List Int))"
    assert b_lit == "([(1 : Int), (0 : Int), (0 : Int)] : List Int)"
    assert ty == "Int"


def test_orientation_is_row_major():
    # Asymmetric, non-square: a transpose bug would change both shape and values.
    A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)  # out=2, in=3
    B = torch.zeros((2, 1), dtype=torch.int64)
    term, _ = _int_linear(A, B, 2)
    out_m, a_lit, b_lit, _ = linear_list_literals(term)
    assert out_m == 2
    # rows are A[i][:] in order — NOT the transpose [[1,4],[2,5],[3,6]]
    assert a_lit == "([[(1 : Int), (2 : Int), (3 : Int)], [(4 : Int), (5 : Int), (6 : Int)]] : List (List Int))"
    assert b_lit == "([(0 : Int), (0 : Int)] : List Int)"


def test_negative_coefficients():
    A = torch.tensor([[-1, 2], [0, -3]], dtype=torch.int64)
    B = torch.tensor([[-5], [7]], dtype=torch.int64)
    term, _ = _int_linear(A, B, 2)
    _, a_lit, b_lit, _ = linear_list_literals(term)
    assert a_lit == "([[(-1 : Int), (2 : Int)], [(0 : Int), (-3 : Int)]] : List (List Int))"
    assert b_lit == "([(-5 : Int), (7 : Int)] : List Int)"


def test_zero_bias_becomes_zeros():
    """The bias has the output's shape: an empty one is refused, and a zero
    one prints one zero per output row."""
    A = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)  # out=3, in=2
    with pytest.raises(ValueError, match="B of shape"):
        tensor.int.Linear(A, torch.zeros((0, 0), dtype=torch.int64))
    zero = torch.zeros((3, 1), dtype=torch.int64)
    term, _ = _int_linear(A, zero, 3)
    out_m, a_lit, b_lit, _ = linear_list_literals(term)
    assert out_m == 3
    assert a_lit == "([[(0 : Int), (0 : Int)], [(1 : Int), (0 : Int)], [(0 : Int), (1 : Int)]] : List (List Int))"
    assert b_lit == "([(0 : Int), (0 : Int), (0 : Int)] : List Int)"  # one zero per output row


def test_real_lra_formatting():
    A = torch.tensor([[1.5, 0.0]], dtype=torch.float64)  # out=1, in=2
    B = torch.tensor([[2.0]], dtype=torch.float64)
    out_w = Wire(Real([1, 1]))
    x_w = Wire(Real([2, 1]))
    term = Term(tensor.real.Linear(A, B), [out_w], [x_w])
    out_m, a_lit, b_lit, ty = linear_list_literals(term)
    assert out_m == 1
    assert ty == "Real"
    assert a_lit == "([[((3 : Real) / 2), (0 : Real)]] : List (List Real))"
    assert b_lit == "([(2 : Real)] : List Real)"


def test_linear_expr_emits_matvecaffine():
    A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    B = torch.tensor([[5], [0]], dtype=torch.int64)
    term, x_w = _int_linear(A, B, 2)
    expr = _linear_expr(term, {x_w.id: "extl_n"})
    assert expr == (
        "(matVecAffine 2 ([[(1 : Int), (2 : Int), (3 : Int)], [(4 : Int), (5 : Int), (6 : Int)]] : List (List Int)) "
        "([(5 : Int), (0 : Int)] : List Int) extl_n)"
    )


def test_linear_expr_uses_read_wire_accessor():
    A = torch.eye(2, dtype=torch.int64)
    B = torch.zeros((2, 1), dtype=torch.int64)
    term, x_w = _int_linear(A, B, 2)
    # whatever accessor the caller bound for the read wire is threaded through as X
    assert "ctrl.2.1" in _linear_expr(term, {x_w.id: "ctrl.2.1"})


# ──────────────────────────────────────────────────────────────
# Bool matrix constants
# ──────────────────────────────────────────────────────────────


def test_bool_matrix_constant_is_interned_not_inlined():
    """A Bool matrix constant must become a top-level def, not `.item()`."""
    import torch
    from reactor import Term, Module, Bool, Var, X
    from reactor_lean import LeanContext, ModuleToLean4

    flags = Var(Bool([2, 3]))
    data = torch.tensor([[True, False, True], [False, True, False]])
    module = Module.sequential(
        [flags],
        [Term(tensor.bool.Const(data), [X(flags)])],
        [Term(tensor.Skip(Bool([2, 3])), [X(flags)], [flags])],
    )
    lean = ModuleToLean4(LeanContext(module)).to_lean_functional()
    assert "Matrix (Fin 2) (Fin 3) Bool" in lean, "the Bool matrix constant was not emitted"


@pytest.mark.parametrize("shape, data", [
    ([1, 1], [[1.5]]),
    ([2, 1], [[1.5], [2.0]]),
])
def test_a_real_constant_is_noncomputable(shape, data):
    """`1.5` is printed `((3 : Real) / 2)`, and a `def` that divides Reals
    does not compile: every hybrid benchmark with a fractional constant was
    `BUILD-FAIL`. `Certs.ScalarEncRealHalf` builds one under `just test-lean`."""
    import torch
    from reactor import Term, Module, Var, X
    from reactor_lean import LeanContext, ModuleToLean4

    x = Var(Real(shape))
    module = Module.sequential(
        [x],
        [Term(tensor.real.Const(torch.tensor(data)), [X(x)])],
        [Term(tensor.Skip(Real(shape)), [X(x)], [x])],
    )
    lean = ModuleToLean4(LeanContext(module)).to_lean_functional()
    defs = [l for l in lean.splitlines() if " def c" in l]
    assert defs and all("noncomputable def" in l for l in defs), defs


# ──────────────────────────────────────────────────────────────
# BitVec literals
# ──────────────────────────────────────────────────────────────


def test_bitvec_literal_uses_of_int():
    """`BitVec.ofNat w (-3)` has no `Neg ℕ` instance and does not elaborate.

    Latent rather than live: a word `Const` holds unsigned values and words
    have no `Linear`, so nothing
    currently feeds a negative here. `ofInt` closes the trap and agrees with
    `ofNat` on non-negatives.
    """
    from reactor import Wire, Word
    from reactor_lean.codegen.literals import lean_literal

    dt = Wire(Word(8, [1, 1])).dtype
    assert lean_literal(dt, -3) == "(BitVec.ofInt 8 (-3))"
    assert lean_literal(dt, 3) == "(BitVec.ofInt 8 (3))"
    assert "ofNat" not in lean_literal(dt, -1)


def test_non_negative_bitvec_constant_still_emits():
    """The reachable case must be unchanged by the switch to `ofInt`."""
    import torch
    from reactor import Term, Module, Word, Var, X
    from reactor_lean import LeanContext, ModuleToLean4

    v = Var(Word(8, [1, 1]))
    module = Module.sequential(
        [v],
        [Term(tensor.word.Const(torch.tensor([[5]]), 8), [X(v)])],
        [Term(tensor.Skip(Word(8, [1, 1])), [X(v)], [v])],
    )
    lean = ModuleToLean4(LeanContext(module)).to_lean_functional()
    assert "BitVec.ofInt 8 (5)" in lean
    assert "BitVec.ofNat" not in lean


@pytest.mark.parametrize("sort, value, lean", [
    (Int([1, 1]), 3, "(3 : Int)"),
    (Int([1, 1]), -3, "(-3 : Int)"),
    (Real([1, 1]), 2.0, "(2 : Real)"),
    (Real([1, 1]), -0.5, "((-1 : Real) / 2)"),
    (Real([1, 1]), 0.1, "((1 : Real) / 10)"),
    (Real([1, 1]), 1e-07, "((1 : Real) / 10000000)"),
    (Bool([1, 1]), True, "true"),
    (Word(8, [1, 1]), -3, "(BitVec.ofInt 8 (-3))"),
])
def test_one_printer_spells_every_literal(sort, value, lean):
    """Ints ascribed whatever their sign, Reals exact -- a float read by its
    decimal form."""
    from reactor_lean.codegen.literals import lean_literal

    assert lean_literal(sort, value) == lean


