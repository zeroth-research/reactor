"""Tests for the theory in Python: the sorts, the operations of `reactor.op`, the
logics, and the overloaded operations of `reactor.overload`."""

import pytest
import torch

from reactor import (Atom, Bool, BV, Int, LIA, LRA, Module, Real, Sort, Term, Var, Wire,
                     Word, X, Zero, d, op)
from reactor import overload as ov
from reactor.eval import eval_itype
from reactor.expr import collecting, expr
from reactor.op.real import Add


# --- sorts ---------------------------------------------------------------------------


def test_sorts_take_their_fields_apart():
    match Real([2, 3]):
        case Real([m, n], degree):
            assert (m, n, degree) == (2, 3, 0)
    match Word(8, [2, 3]):
        case Word(w, [m, n]):
            assert (w, m, n) == (8, 2, 3)
    assert Real([2, 3], 1).degree == 1 and Int([2, 3]).shape == [2, 3]


def test_sorts_compare_and_hash_by_value():
    assert Real([2, 3]) == Real([2, 3]) and Real([2, 3]) != Real([2, 3], 1)
    assert Word(8, [1, 1]) != Word(16, [1, 1])
    assert len({Int([1, 1]), Int([1, 1]), Bool([1, 1]), Zero(), Zero()}) == 3
    assert all(isinstance(s, Sort) for s in (Real([1, 1]), Word(1, [1, 1]), Zero()))


def test_sorts_print_as_in_rust_and_repr_as_their_constructor():
    assert str(Real([2, 3], 1)) == "T^1(Real(2, 3))"
    assert str(Word(8, [2, 3])) == "Word<8>(2, 3)"
    assert repr(Real([2, 3], 1)) == "Real([2, 3], 1)"
    assert repr(Word(8, [2, 3])) == "Word(8, [2, 3])"
    assert repr(Zero()) == "Zero()"


def test_words_have_at_least_one_bit():
    with pytest.raises(ValueError, match="at least one bit"):
        Word(0, [1, 1])


def test_derivative_wires_have_the_tangent_sort():
    assert d(Var(Real([2, 3]))).dtype == Real([2, 3], 1)
    assert X(Var(Real([2, 3]))).dtype == Real([2, 3])
    for sort in (Int([1, 1]), Bool([1, 1]), Word(8, [1, 1])):
        assert d(Var(sort)).dtype == Zero()


# --- operations ----------------------------------------------------------------------


def test_operations_take_their_fields_apart():
    match op.word.Extract(7, 4, 8, 2, 3):
        case op.word.Extract(i, j, w, m, n):
            assert (i, j, w, m, n) == (7, 4, 8, 2, 3)
    assert op.real.TAdd(2, 3, 1).d == 1
    assert op.real.Uninterpreted("f", 1, 1).symbol == "f"
    a, b = torch.eye(2), torch.zeros(2, 1)
    match op.real.Linear(a, b):
        case op.real.Linear(x, y):
            assert torch.equal(x, a) and torch.equal(y, b)


def test_operations_are_importable_from_their_module():
    assert Add is op.real.Add
    assert op.word.SDiv.__module__ == "reactor.op.word"


def test_operations_print_as_in_rust_and_repr_as_their_constructor():
    assert str(op.word.ULt(8, 1, 1)) == "ULt"
    assert str(op.word.Extract(7, 4, 8, 1, 1)) == "Extract[7:4]"
    assert repr(op.real.Add(2, 3)) == "real.Add(2, 3)"
    assert repr(op.bool.Zero()) == "bool.Zero()"


def test_terms_hold_operations_of_every_theory():
    for itype, sort in [
        (op.real.Havoc(1, 2), Real([1, 2])),
        (op.int.Havoc(1, 2), Int([1, 2])),
        (op.bool.Havoc(1, 2), Bool([1, 2])),
        (op.word.Havoc(8, 1, 2), Word(8, [1, 2])),
    ]:
        term = Term.constant(itype, [Wire(sort)])
        assert type(term.itype) is type(itype) and repr(term.itype) == repr(itype)


def test_terms_reject_what_is_not_an_operation():
    with pytest.raises(TypeError, match="expected an operation"):
        Term.constant(Real([1, 1]), [Wire(Real([1, 1]))])


def test_terms_read_and_write_the_sorts_of_their_operation():
    r, b = Real([2, 3]), Bool([2, 3])
    Term(op.real.Add(2, 3), [Wire(r)], [Wire(r), Wire(r)])
    with pytest.raises(Exception, match=r"expected Add to read Real\(2, 3\) ⊗ Real\(2, 3\), "
                                        r"got Bool\(2, 3\) ⊗ Bool\(2, 3\)"):
        Term(op.real.Add(2, 3), [Wire(r)], [Wire(b), Wire(b)])
    with pytest.raises(Exception, match="expected Add to read"):
        Term(op.real.Add(2, 3), [Wire(r)], [Wire(r)])
    with pytest.raises(Exception, match=r"expected \[\[1\]\] to write Real\(1, 1\), got Real\(7, 7\)"):
        Term.constant(op.real.Const(torch.tensor([[1.0]])), [Wire(Real([7, 7]))])
    # a rate of change is not a value
    x = Var(Real([2, 3]))
    with pytest.raises(Exception, match=r"got T\^1\(Real\(2, 3\)\) ⊗ T\^1\(Real\(2, 3\)\)"):
        Term(op.real.Add(2, 3), [Wire(r)], [d(x), d(x)])
    Term(op.real.TAdd(2, 3, 1), [Wire(Real([2, 3], 1))], [d(x), d(x)])


def test_operations_compare_by_value():
    assert op.real.Add(2, 3) == op.real.Add(2, 3) and op.real.Add(2, 3) != op.real.Add(3, 2)
    assert op.word.SDiv(8, 1, 1) != op.word.UDiv(8, 1, 1)
    term = Term.constant(op.int.Havoc(1, 2), [Wire(Int([1, 2]))])
    assert term.itype == op.int.Havoc(1, 2)


def test_tensor_operations_hold_tensors_of_their_theory():
    with pytest.raises(TypeError, match="floating-point"):
        op.real.Const(torch.tensor([[1]]))
    with pytest.raises(TypeError, match="integer"):
        op.int.Linear(torch.eye(2), torch.zeros(2, 1))
    with pytest.raises(TypeError, match="boolean"):
        op.bool.Const(torch.tensor([[1]]))
    with pytest.raises(ValueError, match=r"\[m, k\]"):
        op.real.TLinear(torch.eye(2), torch.zeros(3, 1), 1)


def test_tensor_operations_are_operations_of_their_theory_module():
    for theory, classes in [
        ("real", (op.real.Const, op.real.Linear, op.real.TLinear)),
        ("int", (op.int.Const, op.int.Linear)),
        ("bool", (op.bool.Const,)),
        ("word", (op.word.Const,)),
    ]:
        for cls in classes:
            assert cls.__module__ == f"reactor.op.{theory}"
    match op.word.Const(torch.tensor([[5]]), 8):
        case op.word.Const(tensor, width):
            assert (tensor.item(), width) == (5, 8)
    match op.real.TLinear(torch.eye(2), torch.zeros(2, 1), 1):
        case op.real.TLinear(a, b, d):
            assert torch.equal(a, torch.eye(2)) and b.shape == (2, 1) and d == 1
    assert repr(op.real.Const(torch.tensor([[1.0]]))) == "real.Const(tensor([[1.]]))"
    assert repr(op.word.Const(torch.tensor([[5]]), 8)) == "word.Const(tensor([[5]]), 8)"


def test_lists_of_rows_are_matrices_held_as_lists():
    assert op.int.Const([[1, 2]]).tensor == [[1, 2]]
    assert op.bool.Const([[True], [False]]).tensor == [[True], [False]]
    assert repr(op.real.Const([[1.5, 2]])) == "real.Const([[1.5, 2.0]])"
    with pytest.raises(TypeError, match="floating-point"):
        op.real.Const([[1, 2]])
    with pytest.raises(ValueError, match=r"rows of lengths \[1, 2\]"):
        op.int.Const([[1], [2, 3]])
    (value,) = eval_itype(op.int.Const([[4]]), [])
    assert torch.equal(value, torch.tensor([[4]]))


def test_terms_share_the_tensors_of_their_operations():
    a = torch.eye(2)
    term = Term(op.real.Linear(a, torch.zeros(2, 1)), [Wire(Real([2, 1]))], [Wire(Real([2, 1]))])
    match term.itype:
        case op.real.Linear(weight, _):
            assert weight.data_ptr() == a.data_ptr()


def test_logics_contain_their_operations():
    assert LRA.contains(op.real.Add(1, 1)) and LRA.contains(op.bool.And(1, 1))
    assert not LRA.contains(op.real.Mul(1, 1))
    assert not LIA.contains(op.real.Add(1, 1))
    assert BV.contains(op.word.SDiv(8, 1, 1)) and BV.contains(op.bool.Not(1, 1))


# --- the implicit blocks, on the sorts of every theory ---------------------------------


@pytest.mark.parametrize("sort", [Real([1, 2]), Int([1, 2]), Bool([1, 2]), Word(8, [1, 2])])
def test_held_variables_write_the_zero_of_their_derivative(sort):
    x = Var(sort)
    atom = Atom.hold([x])
    (zero,) = atom.delay
    assert zero.write == [d(x)]
    expected = op.real.Zero(1, 2, 1) if isinstance(sort, Real) else op.bool.Zero()
    assert repr(zero.itype) == repr(expected)
    (skip,) = atom.update
    assert repr(skip.itype) == repr(ov.id_(sort))


# --- overloaded operations ---------------------------------------------------------------


def test_overloads_pick_the_operation_of_the_sort():
    assert repr(ov.add(Real([2, 3]))) == "real.Add(2, 3)"
    assert repr(ov.add(Real([2, 3], 1))) == "real.TAdd(2, 3, 1)"
    assert repr(ov.add(Int([2, 3]))) == "int.Add(2, 3)"
    assert repr(ov.add(Word(8, [2, 3]))) == "word.Add(8, 2, 3)"
    assert repr(ov.lt(Word(8, [1, 1]), signed=True)) == "word.SLt(8, 1, 1)"
    assert repr(ov.and_(Bool([1, 1]))) == "bool.And(1, 1)"
    assert repr(ov.and_(Word(8, [1, 1]))) == "word.And(8, 1, 1)"
    assert repr(ov.zero(Zero())) == "bool.Zero()"
    assert repr(ov.ite(Real([1, 1], 2))) == "real.TIte(1, 1, 2)"


def test_overloads_reject_what_the_sort_has_not():
    with pytest.raises(ov.TheoryError, match="no Mul on T\\^1"):
        ov.mul(Real([1, 1], 1))
    with pytest.raises(ov.TheoryError, match="no Eq on Bool"):
        ov.eq(Bool([1, 1]))
    with pytest.raises(ov.TheoryError, match="no MatMul"):
        ov.matmul(Word(8, [1, 1]), Word(8, [1, 1]))


# --- evaluation of what the old theories lacked ------------------------------------------


def _eval(itype, *values):
    return eval_itype(itype, [torch.tensor([[v]]) for v in values])[0].item()


def test_eval_reads_signed_words_in_twos_complement():
    assert _eval(op.word.ULt(8, 1, 1), 255, 1) is False
    assert _eval(op.word.SLt(8, 1, 1), 255, 1) is True  # -1 < 1
    assert _eval(op.word.SDiv(8, 1, 1), 250, 2) == 253  # -6 / 2 = -3
    assert _eval(op.word.SignExtend(8, 8, 1, 1), 255) == 0xFFFF


def test_eval_of_word_bit_operations():
    assert _eval(op.word.Extract(7, 4, 8, 1, 1), 0xAB) == 0xA
    assert _eval(op.word.Concat(4, 4, 1, 1), 0xA, 0xB) == 0xAB
    assert _eval(op.word.RotateLeft(1, 8, 1, 1), 0x81) == 0x03
    assert _eval(op.word.Repeat(2, 4, 1, 1), 0xA) == 0xAA


def test_eval_of_euclidean_integer_division():
    assert _eval(op.int.Div(1, 1), -7, 2) == -4
    assert _eval(op.int.Mod(1, 1), -7, 2) == 1
    assert _eval(op.int.Div(1, 1), 7, -2) == -3
    assert _eval(op.int.Mod(1, 1), 7, -2) == 1


# --- frontends ---------------------------------------------------------------------------


def test_boolean_equality_is_built_from_xor():
    b = Bool([1, 1])
    with collecting() as terms:
        x, y = expr(Var(b), theory=LIA), expr(Var(b), theory=LIA)
        x == y
    assert [type(t.itype) for t in terms] == [op.bool.Xor, op.bool.Not]
    with collecting() as terms:
        x != y
    assert [type(t.itype) for t in terms] == [op.bool.Xor]


def test_rates_of_change_build_rate_operations():
    t = Var(Real([1, 1]))
    with collecting() as terms:
        dt = expr(d(t), theory=LRA)
        dt + dt
        2 * dt
    assert [type(t.itype) for t in terms] == [op.real.TAdd, op.real.Const, op.real.TLinear]
