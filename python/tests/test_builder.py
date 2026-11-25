"""Tests for `reactor.builder` (the term builder used by the gym / SMV / analyzer front-ends)."""

import pytest
import torch

from reactor import T, Wire, TheoryError, tensor
from reactor.tensor.sort import Real, Int
from reactor.builder import builder_for, RealTermBuilder
from reactor.eval import eval_itype


@pytest.mark.parametrize("sort,dtype", [
    (Real, torch.float32),
    (Int, torch.int64),
])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_mul_multiplies_pointwise(sort, dtype, n):
    builder = builder_for(sort)
    x = Wire(sort([n, 1]))
    const = builder.const(torch.full((n, 1), 3, dtype=dtype))

    term = builder.mul(const, x)

    vals = torch.arange(1, n + 1, dtype=dtype).reshape(n, 1)
    out = eval_itype(term.itype, [torch.full((n, 1), 3, dtype=dtype), vals], term.write[0].dtype)[0]
    assert out.flatten().tolist() == [3 * v for v in range(1, n + 1)]


@pytest.mark.parametrize("sort,dtype", [(Real, torch.float32), (Int, torch.int64)])
def test_const_takes_the_numeric_sort(sort, dtype):
    term = builder_for(sort).const(torch.tensor([[3]]))
    assert term.write[0].dtype == sort([1, 1])
    assert term.itype.tensor.dtype == dtype


def _real(degree):
    """The `[2, 3]` real matrices of `degree`: `T` applied `degree` times to plain ones."""
    sort = Real([2, 3])
    for _ in range(degree):
        sort = T(sort)
    return sort


@pytest.mark.parametrize("p,q", [(1, 1), (1, 0), (0, 1)])
def test_mul_adds_the_degrees_of_reals(p, q):
    builder = builder_for(Real)
    x, y = Wire(_real(p)), Wire(_real(q))

    term = builder.mul(x, y)

    assert term.itype == tensor.real.Mul([_real(p), _real(q)])
    assert term.write[0].dtype == _real(p + q)


def test_transpose_keeps_the_degree_of_reals():
    term = RealTermBuilder().transpose(Wire(T(Real([2, 3]))))
    assert term.write[0].dtype == T(Real([3, 2]))


def test_linear_keeps_the_degree_of_reals():
    terms = RealTermBuilder().linear(Wire(T(Real([1, 2]))), torch.ones(3, 2), torch.zeros(3))
    assert [t.write[0].dtype for t in terms] == [T(Real([2, 1])), T(Real([3, 1])), T(Real([1, 3]))]


def test_add_keeps_the_degree_of_reals():
    builder = builder_for(Real)
    x, y = Wire(T(Real([2, 3]))), Wire(T(Real([2, 3])))

    term = builder.add(x, y)

    assert term.itype == tensor.real.Add([T(Real([2, 3]))] * 2)
    assert term.write[0].dtype == T(Real([2, 3]))


@pytest.mark.parametrize("op", [RealTermBuilder.add, RealTermBuilder.sub])
def test_reals_of_different_degrees_do_not_add_up(op):
    x, y = Wire(T(Real([2, 3]))), Wire(Real([2, 3]))

    with pytest.raises(TheoryError):
        op(RealTermBuilder(), x, y)


@pytest.mark.parametrize("sort,dtype", [
    (Real, torch.float32),
    (Int, torch.int64),
])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_mul_scales_a_column_vector(sort, dtype, n):
    builder = builder_for(sort)
    # `linear` maps row vectors, as `nn.Linear` does: transpose in, Linear, transpose out
    x = Wire(sort([1, n]))

    terms = builder.linear(x, 3 * torch.eye(n, dtype=dtype))

    env = {x: torch.arange(1, n + 1, dtype=dtype).reshape(1, n)}
    for t in terms:
        outs = eval_itype(t.itype, [env[w] for w in t.read], t.write[0].dtype)
        env.update(zip(t.write, outs))
    out = env[terms[-1].write[0]]
    assert out.flatten().tolist() == [3 * v for v in range(1, n + 1)]


def test_neg_keeps_the_degree_of_reals():
    x = Wire(T(Real([1, 1])))
    term = RealTermBuilder().neg(x)
    assert term.itype.degree == 1
    assert term.write[0].dtype == T(Real([1, 1]))


def test_builder_takes_the_degree_of_reals():
    b = RealTermBuilder()
    x, dx = Wire(Real([1, 1])), Wire(T(Real([1, 1])))
    assert b.relu(dx).itype.degree == 1
    assert b.lt(dx, dx).itype.degree == 1
    assert b.div(dx, x).write[0].dtype == T(Real([1, 1]))
    with pytest.raises(TheoryError):
        b.div(x, dx)  # a negative degree is not defined
    with pytest.raises(TheoryError):
        b.lt(x, dx)  # reals of different degrees are not comparable
