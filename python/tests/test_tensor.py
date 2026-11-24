"""The operations of `reactor.tensor` are built from the parameters their sorts do not
determine, then their domain, and optionally their codomain, which must agree; those that
read nothing are built from their codomain. They match positionally the same way, their
domain left out when they read nothing; a single sort comes bare, and several as a list."""

import pytest
import torch
from reactor.tensor import Havoc, Skip, real, int as int_, bool as bool_, word, zero
from reactor.tensor.sort import Real, Int, Bool, Word, Zero


def test_binary_operations_take_and_match_their_domain_as_a_list():
    x, y = Real([2, 3], 1), Real([2, 3], 0)
    match real.Mul([x, y]):
        case real.Mul([Real(s, degree=p), Real(degree=q)], Real(degree=r)):
            assert (list(s), p, q, r) == ([2, 3], 1, 0, 1)
        case _:
            raise AssertionError("no match")
    assert real.MatMul([Real([2, 3]), Real([3, 4])]).cod == Real([2, 4])


def test_unary_operations_take_and_match_their_domain_bare():
    op = real.Transpose(Real([2, 3], 1))
    assert op.dom == Real([2, 3], 1)
    assert op.cod == Real([3, 2], 1)
    assert int_.Lt([Int([2, 2]), Int([2, 2])]).cod == Bool([2, 2])
    match word.signed.Extend(3, Word(5, [1, 1])):
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
    assert word.Extract(5, 2, Word(8, [1, 1])).cod == Word(4, [1, 1])
    match real.Linear(torch.ones(4, 2), torch.zeros(4, 3), Real([2, 3])):
        case real.Linear(a, _, Real(s), Real(t)):
            assert a.shape == (4, 2)
            assert (list(s), list(t)) == ([2, 3], [4, 3])
        case _:
            raise AssertionError("no match")


def test_choice_reads_its_condition_and_branches():
    dom = [Bool([2, 2]), Real([2, 2]), Real([2, 2])]
    assert real.Ite(dom).dom == dom
    assert bool_.Ite([Bool([1, 1])] * 3).dom == [Bool([1, 1])] * 3
    assert word.Ite([Bool([1, 1]), Word(8, [1, 1]), Word(8, [1, 1])]).cod == Word(8, [1, 1])
    with pytest.raises(ValueError, match="condition"):
        real.Ite([Real([2, 2])] * 3)


def test_the_codomain_is_optional_and_checked():
    x = Real([2, 3], 1)
    assert real.Add([x, x], x) == real.Add([x, x])
    assert real.Add([x, x], cod=x) == real.Add([x, x])
    with pytest.raises(ValueError, match="writes"):
        real.Add([x, x], Real([2, 3]))
    with pytest.raises(ValueError, match="writes"):
        int_.Lt([Int([1, 1])] * 2, Bool([2, 2]))
    with pytest.raises(TypeError):
        int_.Lt([Int([1, 1])] * 2, Int([1, 1]))
    with pytest.raises(TypeError):
        real.Add(cod=x)


def test_domains_of_no_instance_raise():
    with pytest.raises(ValueError):
        real.Add([Real([2, 3]), Real([3, 2])])
    with pytest.raises(ValueError):
        real.MatMul([Real([2, 3]), Real([2, 3])])
    with pytest.raises(ValueError):
        word.Extract(8, 2, Word(8, [1, 1]))
    with pytest.raises(ValueError):
        real.Scale(torch.ones(2, 3), Real([3, 2]))


def test_nullary_operations_take_and_match_their_codomain_only():
    for op in [
        real.Zero(Real([1, 2], 1)),
        real.Const(torch.ones(1, 2), Real([1, 2])),
        int_.Const(torch.ones(1, 2, dtype=torch.int64), Int([1, 2])),
        bool_.Const(torch.ones(1, 2, dtype=torch.bool), Bool([1, 2])),
        word.Const(torch.ones(1, 2, dtype=torch.int64), Word(4, [1, 2])),
        Havoc(Int([1, 2])),
        zero.Zero(),
    ]:
        assert type(op).__match_args__[-1] == "cod"
        assert "dom" not in type(op).__match_args__
        assert not hasattr(op, "dom")
    assert real.Const(torch.ones(1, 2), Real([1, 2], 1)).cod == Real([1, 2], 1)
    assert word.Const(torch.ones(1, 2, dtype=torch.int64), Word(4, [1, 2])).cod == Word(4, [1, 2])
    assert Havoc(Int([1, 2])).cod == Int([1, 2])
    assert zero.Zero().cod == Zero()
    with pytest.raises(TypeError):
        real.Const(torch.ones(1, 2))
    with pytest.raises(ValueError):
        real.Const(torch.ones(1, 2), Real([2, 1]))


def test_skip_reads_and_writes_its_sort():
    op = Skip(Word(8, [1, 1]))
    assert (op.dom, op.cod) == (Word(8, [1, 1]), Word(8, [1, 1]))
    assert Skip(Word(8, [1, 1]), Word(8, [1, 1])) == op
