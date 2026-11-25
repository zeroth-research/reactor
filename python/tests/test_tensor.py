"""The operations of `reactor.tensor` are built from the parameters their sorts do not
determine, then their domain, and optionally their codomain, which must agree; those that
read nothing are built from their codomain. They match positionally the same way, their
domain left out when they read nothing; a single sort comes bare, and several as a list."""

import pytest
import torch
import reactor
from reactor import Zero
from reactor.sort import ZeroSort
from reactor.tensor import Havoc, Skip, real, int as int_, bool as bool_, word
from reactor.tensor.sort import Real, Int, Bool, Word
from reactor import T


def test_binary_operations_take_and_match_their_domain_as_a_list():
    x, y = T(Real([2, 3])), Real([2, 3])
    match real.Mul([x, y]):
        case real.Mul([Real(s, degree=p), Real(degree=q)], Real(degree=r)):
            assert (list(s), p, q, r) == ([2, 3], 1, 0, 1)
        case _:
            raise AssertionError("no match")
    assert real.MatMul([Real([2, 3]), Real([3, 4])]).cod == Real([2, 4])


def test_unary_operations_take_and_match_their_domain_bare():
    op = real.Transpose(T(Real([2, 3])))
    assert op.dom == T(Real([2, 3]))
    assert op.cod == T(Real([3, 2]))
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
    assert real.Affine.__match_args__ == ("a", "b", "dom", "cod")
    assert real.Scale.__match_args__ == ("tensor", "dom", "cod")
    assert word.Extract(5, 2, Word(8, [1, 1])).cod == Word(4, [1, 1])
    match real.Affine(torch.ones(4, 2), torch.zeros(4, 3), Real([2, 3])):
        case real.Affine(a, _, Real(s), Real(t)):
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
    x = T(Real([2, 3]))
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
        real.Zero(T(Real([1, 2]))),
        real.Const(torch.ones(1, 2), Real([1, 2])),
        int_.Const(torch.ones(1, 2, dtype=torch.int64), Int([1, 2])),
        bool_.Const(torch.ones(1, 2, dtype=torch.bool), Bool([1, 2])),
        word.Const(torch.ones(1, 2, dtype=torch.int64), Word(4, [1, 2])),
        Havoc(Int([1, 2])),
        Zero(),
    ]:
        assert type(op).__match_args__[-1] == "cod"
        assert "dom" not in type(op).__match_args__
        assert not hasattr(op, "dom")
    assert real.Const(torch.ones(1, 2), T(Real([1, 2]))).cod == T(Real([1, 2]))
    assert word.Const(torch.ones(1, 2, dtype=torch.int64), Word(4, [1, 2])).cod == Word(4, [1, 2])
    assert Havoc(Int([1, 2])).cod == Int([1, 2])
    assert Zero().cod == ZeroSort()
    with pytest.raises(ValueError):
        real.Const(torch.ones(1, 2), Real([2, 1]))


def test_the_codomain_of_constants_is_optional_when_their_tensor_determines_it():
    assert real.Const(torch.ones(1, 2)).cod == Real([1, 2])
    assert real.Const(torch.ones(1, 2), T(Real([1, 2]))).cod == T(Real([1, 2]))
    assert int_.Const(torch.ones(1, 2, dtype=torch.int64)).cod == Int([1, 2])
    assert bool_.Const(torch.ones(1, 2, dtype=torch.bool)).cod == Bool([1, 2])
    # the tensor of words does not determine their width
    with pytest.raises(TypeError):
        word.Const(torch.ones(1, 2, dtype=torch.int64))


def test_skip_reads_and_writes_its_sort():
    op = Skip(Word(8, [1, 1]))
    assert (op.dom, op.cod) == (Word(8, [1, 1]), Word(8, [1, 1]))
    assert Skip(Word(8, [1, 1]), Word(8, [1, 1])) == op


def test_the_domain_is_optional_when_the_parameters_determine_it():
    ints = torch.ones(2, 3, dtype=torch.int64)
    assert int_.Scale(ints).dom == int_.Scale(ints, Int([2, 3])).dom == Int([2, 3])
    assert int_.Scale(ints, cod=Int([2, 3])).cod == Int([2, 3])
    with pytest.raises(ValueError):
        int_.Scale(ints, cod=Int([3, 2]))
    a, b = torch.ones(4, 2, dtype=torch.int64), torch.zeros(4, 3, dtype=torch.int64)
    assert int_.Affine(a, b).dom == Int([2, 3])
    assert int_.Affine(a, b).cod == Int([4, 3])
    # the tensors of reals determine all but their degree, 0 unless given
    assert real.Scale(torch.ones(2, 3)).dom == Real([2, 3])
    assert real.Scale(torch.ones(2, 3), T(Real([2, 3]))).cod == T(Real([2, 3]))
    assert real.Affine(torch.ones(4, 2), torch.zeros(4, 3)).dom == Real([2, 3])
    assert real.Affine(torch.ones(4, 2), torch.zeros(4, 3)).cod == Real([4, 3])
    with pytest.raises(ValueError):
        real.Scale(torch.ones(2, 3), cod=T(Real([2, 3])))


def test_the_singleton_sort_and_its_operation_belong_to_no_theory():
    # the sort is shared by all theories, and the operation writing its only value is a
    # special case, at the top level
    assert Zero().cod == ZeroSort()
    assert ZeroSort.__module__ == "reactor.sort"
    # an operation like any other, built with no arguments: no instance stands for it
    assert isinstance(Zero(), Zero) and Zero() == Zero()
    assert Zero.__module__ == "reactor"
    assert not hasattr(reactor, "ZeroType") and not hasattr(reactor, "ZeroOp")
    assert not hasattr(reactor, "ZeroSort") and not hasattr(reactor, "zero")
    assert not hasattr(reactor.tensor, "zero") and not hasattr(reactor.tensor.sort, "Zero")
    # the sorts are reached through their namespaces, not at the top level
    assert not any(hasattr(reactor, s) for s in ["Real", "Int", "Bool", "Word"])
