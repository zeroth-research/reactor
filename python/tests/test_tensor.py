"""The operations of `reactor.tensor` match positionally their parameters, then their
domain, unless they read nothing, and their codomain; a single sort comes bare, and
several as a list."""

import torch
from reactor.tensor import Havoc, Skip, real, int as int_, bool as bool_, word, zero
from reactor.tensor.sort import Real, Int, Bool, Word, Zero


def test_binary_operations_match_their_domain_as_a_list():
    match real.Mul([2, 3], [1, 0]):
        case real.Mul([Real(s, degree=p), Real(_, degree=q)], Real(_, degree=r)):
            assert (list(s), p, q, r) == ([2, 3], 1, 0, 1)
        case _:
            raise AssertionError("no match")
    assert real.MatMul([[2, 3], [3, 4]]).cod == Real([2, 4])


def test_unary_operations_match_their_domain_bare():
    op = real.Transpose([2, 3], 1)
    assert op.dom == Real([2, 3], 1)
    assert op.cod == Real([3, 2], 1)
    assert int_.Lt([2, 2]).cod == Bool([2, 2])
    match word.signed.Extend(3, 5, [1, 1]):
        case word.signed.Extend(i, Word(w), Word(v)):
            assert (i, w, v) == (3, 5, 8)
        case _:
            raise AssertionError("no match")


def test_parameters_come_first():
    assert word.Extract.__match_args__ == ("i", "j", "dom", "cod")
    assert word.RotateLeft.__match_args__ == ("i", "dom", "cod")
    assert real.Add.__match_args__ == ("dom", "cod")
    assert real.Linear.__match_args__ == ("a", "b", "dom", "cod")
    assert real.Scale.__match_args__ == ("tensor", "dom", "cod")
    match real.Linear(torch.ones(4, 2), torch.zeros(4, 3)):
        case real.Linear(a, _, Real(s), Real(t)):
            assert a.shape == (4, 2)
            assert (list(s), list(t)) == ([2, 3], [4, 3])
        case _:
            raise AssertionError("no match")


def test_choice_reads_its_condition_and_branches():
    assert real.Ite([2, 2]).dom == [Bool([2, 2]), Real([2, 2]), Real([2, 2])]
    assert bool_.Ite([1, 1]).dom == [Bool([1, 1])] * 3
    assert word.Ite(8, [1, 1]).cod == Word(8, [1, 1])


def test_nullary_operations_have_no_domain():
    for op in [
        real.Zero([1, 2]),
        real.Const(torch.ones(1, 2)),
        int_.Const(torch.ones(1, 2, dtype=torch.int64)),
        bool_.Const(torch.ones(1, 2, dtype=torch.bool)),
        word.Const(torch.ones(1, 2, dtype=torch.int64), 4),
        Havoc(Int([1, 2])),
        zero.Zero(),
    ]:
        assert type(op).__match_args__[-1] == "cod"
        assert "dom" not in type(op).__match_args__
        assert not hasattr(op, "dom")
    assert real.Const(torch.ones(1, 2), 1).cod == Real([1, 2], 1)
    assert word.Const(torch.ones(1, 2, dtype=torch.int64), 4).cod == Word(4, [1, 2])
    assert Havoc(Int([1, 2])).cod == Int([1, 2])
    assert zero.Zero().cod == Zero()


def test_skip_reads_and_writes_its_sort():
    op = Skip(Word(8, [1, 1]))
    assert (op.dom, op.cod) == (Word(8, [1, 1]), Word(8, [1, 1]))
