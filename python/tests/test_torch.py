"""Tests for reactor.torch.Module: interface sorts selected by `number=` (Real/Int/Word)
and weight-dtype validation."""

import pytest
import torch
import torch.nn as nn

from reactor import X
from reactor.matrix.sort import Real, Int, Word, Bool
from reactor.torch import Module


class Net(nn.Module):
    """Linear -> ReLU -> Linear (obs_size=2, qval=1)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _to_int_weights(net, dtype=torch.int64):
    """Replace a net's Linear weights/biases with integer tensors (values rounded)."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight = nn.Parameter(m.weight.detach().round().to(dtype), requires_grad=False)
            m.bias = nn.Parameter(m.bias.detach().round().to(dtype), requires_grad=False)
    return net


# --- interface sorts follow the numeric sort ---------------------------------------


def _sorts(m):
    return list(m.extl)[0].dtype, list(m.intf)[0].dtype


def test_default_number_is_real():
    e, i = _sorts(Module(Net()))
    assert isinstance(e, Real) and isinstance(i, Real)


def test_real_is_real():
    e, i = _sorts(Module(Net(), number=Real))
    assert isinstance(e, Real) and isinstance(i, Real)


def test_int_is_int():
    e, i = _sorts(Module(_to_int_weights(Net()), number=Int))
    assert isinstance(e, Int) and isinstance(i, Int)


def test_word_unsupported_for_nn():
    # words have no Transpose/Linear ops, so a neural module can't compile to them (yet).
    with pytest.raises(NotImplementedError, match="need real or integer numbers"):
        Module(_to_int_weights(Net()), number=Word)


# --- weight dtype must match the numeric sort (used as-is, no coercion) ------------


def test_float_weights_under_int_raise():
    with pytest.raises(TypeError, match="integer"):
        Module(Net(), number=Int)


def test_int_weights_under_real_raise():
    with pytest.raises(TypeError, match="floating-point"):
        Module(_to_int_weights(Net()), number=Real)


# --- combinatorial (V(s')) vs sequential (V(s)) ------------------------------


def _read_ids(m):
    return {w.id for w in m.atoms[0].read}


def _wait_ids(m):
    return {X(w).id for w in m.atoms[0].wait}


def test_default_is_sequential_reads_latched_input():
    # V(s): the sequential atom reads the latched input.
    m = Module(Net())
    extl = list(m.extl)[0]
    assert extl.id in _read_ids(m)
    assert X(extl).id not in _wait_ids(m)


def test_combinatorial_reads_next_only():
    # V(s'): memoryless — does not read the latched input, only awaits the next one.
    m = Module(Net(), combinatorial=True)
    extl = list(m.extl)[0]
    assert extl.id not in _read_ids(m)
    assert X(extl).id in _wait_ids(m)


def test_sequential_with_int_builds():
    m = Module(_to_int_weights(Net()), number=Int)
    extl = list(m.extl)[0]
    assert isinstance(extl.dtype, Int)
    assert extl.id in _read_ids(m)
