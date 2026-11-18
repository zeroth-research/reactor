"""Tests for `reactor.builder` (the term builder used by the gym / SMV / analyzer front-ends)."""

import pytest
import torch

from reactor import Real, Int, Wire
from reactor.builder import TermBuilder
from reactor.eval import eval_itype


@pytest.mark.parametrize("sort,dtype", [
    (Real, torch.float32),
    (Int, torch.int64),
])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_mul_multiplies_pointwise(sort, dtype, n):
    builder = TermBuilder(number=sort)
    x = Wire(sort([n, 1]))
    const = builder.const(torch.full((n, 1), 3, dtype=dtype))

    term = builder.mul(const, x)

    vals = torch.arange(1, n + 1, dtype=dtype).reshape(n, 1)
    out = eval_itype(term.itype, [torch.full((n, 1), 3, dtype=dtype), vals], term.write[0].dtype)[0]
    assert out.flatten().tolist() == [3 * v for v in range(1, n + 1)]


@pytest.mark.parametrize("sort,dtype", [(Real, torch.float32), (Int, torch.int64)])
def test_const_takes_the_numeric_sort(sort, dtype):
    term = TermBuilder(number=sort).const(torch.tensor([[3]]))
    assert term.write[0].dtype == sort([1, 1])
    assert term.itype.tensor.dtype == dtype
