"""The terms, atoms and modules over torch tensors and JAX arrays alike, `reactor.mixed`:
the operations of one module, or even one affine map, may hold either, and evaluate
through torch. Skipped unless JAX is installed (the `jax` dependency group)."""

import numpy as np
import pytest
import torch
import z3

import reactor.jax as rjax
import reactor.mixed as mixed
from reactor import Int, Real, Var, Wire, X
from reactor import tensor
from reactor.eval import eval_itype, execute_init, execute_update
from reactor.smt.z3 import interpret

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_operations_hold_either_tensor_as_given():
    a, b = jnp.eye(2), torch.ones(2, 1)
    linear = mixed.real.Linear(a, b)
    match linear:
        case mixed.real.Linear(held_a, held_b):
            assert held_a is a
            assert isinstance(held_b, torch.Tensor) and torch.equal(held_b, b)
    term = mixed.Term(linear, [Wire(Real([2, 1]))], [Wire(Real([2, 1]))])
    assert isinstance(term.itype, mixed.real.Linear)


def test_either_tensor_is_checked():
    # integers are reals too, kept as given, but not booleans
    mixed.real.Const(jnp.array([[1]]))
    mixed.real.Const(torch.tensor([[1]]))
    with pytest.raises(ValueError, match="floating-point or integer elements, got bool"):
        mixed.real.Const(jnp.array([[True]]))
    with pytest.raises(ValueError, match="floating-point elements, got int32"):
        mixed.real.Scale(jnp.array([[1]]))
    with pytest.raises(ValueError, match="floating-point elements, got Int64"):
        mixed.real.Scale(torch.tensor([[1]]))
    for vector in (jnp.zeros(3), torch.zeros(3)):
        with pytest.raises(ValueError, match="expected a matrix"):
            mixed.real.Const(vector)
    with pytest.raises(ValueError, match=r"\[m, k\]"):
        mixed.real.Linear(jnp.eye(2), torch.zeros(3, 1))
    with pytest.raises(TypeError, match="torch.Tensor or a jax.Array"):
        mixed.real.Const(np.zeros((1, 1)))


def test_a_module_mixes_jax_and_torch_operations():
    # a counter starting at a JAX 0 and counting by a JAX map of a torch 1
    x = Var(Int([1, 1]))
    one, two = Wire(Int([1, 1])), Wire(Int([1, 1]))
    init = [mixed.Term(mixed.int.Const(jnp.array([[0]])), [X(x)])]
    update = [
        mixed.Term(mixed.int.Const(torch.tensor([[1]])), [one]),
        mixed.Term(mixed.int.Linear(jnp.array([[2]]), torch.tensor([[0]])), [two], [one]),
        mixed.Term(tensor.int.Add([1, 1]), [X(x)], [x, two]),
    ]
    counter = mixed.Module.sequential([x], init, update)

    state = {}
    execute_init(state, counter.atoms)
    for _ in range(3):
        state[x] = state[X(x)]
        execute_update(state, counter.atoms)
    assert state[X(x)].dtype is torch.int64
    assert state[X(x)].item() == 6

    # the terms of the other packages are not those of `mixed`
    jax_init = [rjax.Term(rjax.int.Const(jnp.array([[0]])), [X(x)])]
    with pytest.raises(TypeError, match="expected the terms of `reactor.mixed`"):
        mixed.Module.sequential([x], jax_init, update)


def test_an_operation_evaluates_with_either_tensor():
    linear = mixed.real.Linear(jnp.eye(2), torch.ones(2, 1))
    (out,) = eval_itype(linear, [torch.tensor([[1.0], [2.0]])])
    assert torch.equal(out, torch.tensor([[2.0], [3.0]]))


def test_a_scale_of_rates_evaluates_with_either_tensor():
    for c in (torch.tensor([[2.0]]), jnp.array([[2.0]])):
        scale = mixed.real.Scale(c, degree=1)
        assert scale.degree == 1
        (out,) = eval_itype(scale, [torch.tensor([[3.0]])])
        assert torch.equal(out, torch.tensor([[6.0]]))


def test_mixed_operations_export_to_z3():
    (value,) = interpret(mixed.int.Const(jnp.array([[3]])))().flat
    assert value.as_long() == 3
    linear = interpret(mixed.int.Linear(torch.tensor([[2]]), jnp.array([[1]])))
    (value,) = linear(np.array([[z3.IntVal(4)]], dtype=object)).flat
    assert z3.simplify(value).as_long() == 9
    for c in (torch.tensor([[3]]), jnp.array([[3]])):
        (value,) = interpret(mixed.int.Scale(c))(np.array([[z3.IntVal(4)]], dtype=object)).flat
        assert z3.simplify(value).as_long() == 12
