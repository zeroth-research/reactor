"""JAX arrays alongside torch tensors in the operations: held as given, checked as
torch tensors are, and evaluated through torch. Skipped unless JAX is installed (the
`jax` dependency group)."""

import pytest
import torch

from reactor import Int, Module, Real, Term, Var, Wire, X, op
from reactor.eval import eval_itype, execute_init, execute_update
from reactor.smt.z3 import interpret

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_operations_hold_jax_arrays_as_given():
    a = jnp.array([[1.0, 2.0]])
    const = op.real.Const(a)
    assert const.tensor is a
    term = Term.constant(const, [Wire(Real([1, 2]))])
    match term.itype:
        case op.real.Const(tensor):
            assert tensor is a
    assert str(const) == "[[1, 2]]"


def test_jax_arrays_are_checked_as_torch_tensors_are():
    with pytest.raises(TypeError, match="floating-point"):
        op.real.Const(jnp.array([[1]]))
    with pytest.raises(TypeError, match="integer"):
        op.int.Linear(jnp.eye(2), jnp.zeros((2, 1)))
    with pytest.raises(ValueError, match="matrix"):
        op.real.Const(jnp.zeros(3))
    with pytest.raises(ValueError, match=r"\[m, k\]"):
        op.real.Linear(jnp.eye(2), jnp.zeros((3, 1)))


def test_an_operation_mixes_jax_and_torch():
    linear = op.real.Linear(jnp.eye(2), torch.ones(2, 1))
    match linear:
        case op.real.Linear(a, b):
            assert isinstance(a, jax.Array) and isinstance(b, torch.Tensor)
    (out,) = eval_itype(linear, [torch.tensor([[1.0], [2.0]])])
    assert torch.equal(out, torch.tensor([[2.0], [3.0]]))


def test_a_module_mixes_jax_and_torch_constants():
    # a counter starting at a JAX 0 and counting by a torch 1
    x = Var(Int([1, 1]))
    one = Wire(Int([1, 1]))
    init = [Term(op.int.Const(jnp.array([[0]])), [X(x)])]
    update = [
        Term(op.int.Const(torch.tensor([[1]])), [one]),
        Term(op.int.Add(1, 1), [X(x)], [x, one]),
    ]
    counter = Module.sequential([x], init, update)

    state = {}
    execute_init(state, counter.atoms)
    for _ in range(3):
        state[x] = state[X(x)]
        execute_update(state, counter.atoms)
    assert state[X(x)].item() == 3


def test_jax_constants_export_to_z3():
    (value,) = interpret(op.int.Const(jnp.array([[3]])))().flat
    assert value.as_long() == 3
