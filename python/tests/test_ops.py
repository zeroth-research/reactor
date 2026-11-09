import pytest
import torch
from reactor import LRA, LIA, BV, Affine, BVConst, Bits, Bool, Int, Real, BitVec, Zero, Term, Wire
from reactor.ops import op, linear, bv_const


def _params(o) -> tuple:
    """The parameters of the op ``o`` (the fields of its variant)."""
    return tuple(getattr(o, f) for f in o.__match_args__)


def test_op_carries_the_parameters_of_the_sort():
    cases = [
        (op(LRA, "Add", Real([2, 3])), LRA.Add, (2, 3)),
        (op(LRA, "Lt", Real([2, 3])), LRA.Lt, (2, 3)),
        (op(LRA, "Not", Bool([2, 3])), LRA.Not, (2, 3)),
        (op(LIA, "ReLU", Int([1, 4])), LIA.ReLU, (1, 4)),
        (op(BV, "ULt", BitVec(8, [2, 3])), BV.ULt, (8, 2, 3)),
        (op(BV, "Id", BitVec(8, [2, 3])), BV.Id, (8, 2, 3)),
    ]
    for o, cls, params in cases:
        assert isinstance(o, cls) and _params(o) == params


def test_op_per_kind_and_on_derivatives():
    # the ops on several kinds of values come in one variant per kind
    assert isinstance(op(LRA, "Id", Real([1, 1])), LRA.IdReal)
    assert isinstance(op(LRA, "Id", Bool([1, 1])), LRA.IdBool)
    assert isinstance(op(LIA, "Ite", Int([1, 1])), LIA.IteInt)
    assert isinstance(op(LIA, "Transpose", Bool([2, 3])), LIA.TransposeBool)
    u = op(LRA, "Uninterpreted", Real([1, 2]), "f")
    assert isinstance(u, LRA.UninterpretedReal) and _params(u) == ("f", 1, 2)
    # the ops on derivatives of reals are separate, and carry the degree
    o = op(LRA, "Add", Real([2, 3], 1))
    assert isinstance(o, LRA.TAdd) and _params(o) == (2, 3, 1)
    assert isinstance(op(LRA, "Id", Real([1, 1], 2)), LRA.TId)


def test_op_missing():
    with pytest.raises(TypeError, match="LRA has no Lt on"):
        op(LRA, "Lt", Real([1, 1], 1))
    with pytest.raises(TypeError, match="LIA has no Add on"):
        op(LIA, "Add", Real([1, 1]))
    with pytest.raises(TypeError, match="LRA has no Id on"):
        op(LRA, "Id", Zero([1, 1]))
    with pytest.raises(TypeError, match="BV has no ReLU on"):
        op(BV, "ReLU", BitVec(8, [1, 1]))


def test_linear_and_bv_const():
    a, b = torch.zeros(2, 3), torch.zeros(2, 1)
    o = linear(LRA, a, b, Real([3, 4]))
    assert isinstance(o, LRA.Linear) and _params(o)[1] == 4
    # the constants are the tensors themselves
    match o:
        case LRA.Linear(Affine(weight, bias)):
            assert weight.data_ptr() == a.data_ptr() and bias.data_ptr() == b.data_ptr()
    o = linear(LRA, a, b, Real([3, 1], 2))
    assert isinstance(o, LRA.TLinear) and _params(o)[1:] == (1, 2)
    o = bv_const(torch.tensor([[1]]), BitVec(8, [1, 1]))
    assert isinstance(o, BV.Const) and isinstance(_params(o)[0], BVConst)
    assert _params(o)[0].width == 8


def test_parameters_are_checked():
    # the ill-formed parameters raise when the op is built
    with pytest.raises(ValueError, match="expected a 2D tensor, got 1D"):
        LRA.Real(torch.zeros(3))
    with pytest.raises(ValueError, match="invalid zero value"):
        LRA.TAdd(1, 1, 0)
    with pytest.raises(ValueError, match="`A`: expected a 2D tensor, got 1D"):
        Affine(torch.zeros(3), torch.empty(0))
    with pytest.raises(ValueError, match=r"`B` must be empty or a column vector \[2x1\]"):
        linear(LRA, torch.zeros(2, 3), torch.zeros(2, 2), Real([3, 1]))
    with pytest.raises(ValueError, match="tensor values do not fit in 8 bits"):
        bv_const(torch.tensor([[256]]), BitVec(8, [1, 1]))
    with pytest.raises(ValueError, match=r"high \(8\) is out of range for width 8"):
        Bits(8, 0, 8)
    # an empty constant fits into any width
    o = bv_const(torch.zeros((0, 3), dtype=torch.int64), BitVec(8, [0, 3]))
    Term(o, [Wire(BitVec(8, [0, 3]))])
    # the literals are the tensors themselves
    t = torch.zeros(2, 3)
    match LRA.Real(t):
        case LRA.Real(u):
            assert u.data_ptr() == t.data_ptr()
    assert BV.BitSelect(bits=Bits(7, 4, 8), rows=1, cols=1).bits == Bits(7, 4, 8)


def test_terms_check_the_boundary():
    r, b = Real([1, 1]), Bool([1, 1])
    Term(op(LRA, "Add", r), [Wire(r)], [Wire(r), Wire(r)])
    with pytest.raises(Exception, match="expected `Real\\(1, 1\\) ⊗ Real\\(1, 1\\) -> Real\\(1, 1\\)`"):
        Term(op(LRA, "Add", r), [Wire(b)], [Wire(r), Wire(r)])
